# 第 15 章：多级流水线与 warp specialization

> 范围标签：`[SM90+/SM100+][Experimental]`
>
> 前置章节：第 11 章 SMEM、第 12 章同步、第 13 章 async copy、第 14 章 TMA
>
> 配套代码：[`15_pipeline/`](../code/15_pipeline/README.md)

## 15.1 本章要回答的问题

前两章已经能异步搬运一个 tile，但都是“发出后很快等待”的单 stage 程序。本章要回答：

1. 怎样证明 producer 不会覆盖 consumer 仍在读取的 stage？
2. stage index、phase 和 iteration count 怎样组成循环状态？
3. prologue、steady state、tail 分别完成什么协议义务？
4. warp specialization 改变的是控制流、参与者，还是硬件并发？
5. `CooperativeGroup`、`Agent`、`PipelineState` 与手写 mbarrier 怎样对应？
6. TMA load、TMA store、TMA→UMMA、UMMA→async 为什么需要不同 pipeline？
7. stage 更多为什么可能更慢？
8. 正确结果、真实 overlap 和性能提升应怎样分别验证？

## 15.2 Pipeline 不是“多个 buffer”的别名

多分配几份 SMEM 只产生 storage。完整 pipeline 还必须定义：

```text
Pipeline
  = stage storage
  + ownership state
  + producer completion
  + consumer completion
  + circular advancement
  + prologue/tail lifetime
```

每个 stage 至少经历：

```text
EMPTY
  -- producer acquire --> PRODUCER_OWNED
  -- producer commit  --> FULL
  -- consumer wait    --> CONSUMER_OWNED
  -- consumer release --> EMPTY(next generation)
```

少掉任何一条边，stage reuse 都可能在某次调度下覆盖尚未消费的数据。

## 15.3 Full/empty 是双向 ownership transfer

本章两个示例都给每个 stage 配置两条 mbarrier：

| Barrier | Signal sender | Waiter | 含义 |
|---|---|---|---|
| `full[s]` | producer/TMA engine | consumer | stage `s` 的当前 generation 已填满 |
| `empty[s]` | consumer | producer | stage `s` 的上一 generation 已读完，可重用 |

`full` 只阻止 consumer 读得太早；`empty` 只阻止 producer 覆盖得太早。只保留 full barrier 的程序，在单 tile 或 producer 较慢时可能通过，但不是闭合的循环 buffer 协议。

## 15.4 Stage index 与 phase

对 `S` 个 stages，第 `k` 个 tile 使用：

```text
stage(k) = k mod S
phase(k) = floor(k / S) mod 2
```

例如 `S=3`：

| k | stage | phase | 说明 |
|---:|---:|---:|---|
| 0 | 0 | 0 | 第一轮 stage 0 |
| 1 | 1 | 0 | 第一轮 stage 1 |
| 2 | 2 | 0 | 第一轮 stage 2 |
| 3 | 0 | 1 | stage 0 第一次 wrap |
| 4 | 1 | 1 | stage 1 第一次 wrap |
| 5 | 2 | 1 | stage 2 第一次 wrap |
| 6 | 0 | 0 | stage 0 第二次 wrap |

phase 绑定的是“同一个 barrier object 的 generation”，所以不是简单的 `k & 1`。只有 `S=1` 时两者才相同。

## 15.5 为什么 empty barrier 要预到达

fresh mbarrier 的 parity 是 0。producer 第一次 acquire 通常等待“旧 phase 0 已经结束”。如果没有初始化转换，它会在一个从未被 consumer 使用过的 stage 上阻塞。

本章手写协议在 prologue 中对每个 `empty[s]` arrive 一次：

```python
for stage in cutlass.range_constexpr(STAGES):
    prims.mbarrier_arrive(empty.subview(stage))
```

parity 从 0 翻到 1，因此第一次 `wait(empty[s], old_phase=0)` 立即成功。这不是伪造 consumer completion，而是显式建立“所有 slots 初始为空”的 pipeline 初态。

高层 pipeline helper 会通过 producer/consumer 初始 `PipelineState.phase` 与 barrier 初始化协议表达同一件事，不应在两套初始化方式上重复翻转。

## 15.6 Producer acquire/commit

软件填充路径是：

```text
wait empty(stage, phase)
  -> producer lanes write SMEM
  -> producer warp publication
  -> elected lane arrive full(stage)
```

关键点是 elected lane 只负责 signal，不代表其他 producer lanes 已写完。commit 前需要 warp/named barrier，把所有 producer writes 排序到 full arrival 之前。

TMA 路径则是：

```text
wait empty(stage, phase)
  -> arrive_expect_tx(full, tile_bytes)
  -> elected issuer launches TMA(load, full barrier)
```

TMA engine 的 `complete_tx::bytes` 取代软件 producer 的“所有 lane 写完”信号。

## 15.7 Consumer wait/release

consumer 侧是：

```text
wait full(stage, phase)
  -> read/compute using stage
  -> all consumer participants finish reading
  -> elected lane arrive empty(stage)
```

release 必须在最后一次 read 之后。若一个 warp 的 lane 0 先读完就立即 release，其他 lanes 仍可能读同一 stage，producer 随后覆盖它。示例在 release 前使用 full-mask warp barrier。

如果 consumer group 是多个 warps、一个 warpgroup 或 cluster CTAs，release count 和 publication scope 都必须扩展，不能继续沿用单 warp 的 elected arrival。

## 15.8 Prologue、steady state 与 tail

### Prologue

建立 barrier 初态并填充最初若干 stages。在 TMA 示例中，producer 对 `min(STAGES,num_tiles)` 个 slots 发出 load，无需等待 consumer release，因为它们从未被使用。

### Steady state

producer 重用较老 stage，consumer 消费已填 stage：

```text
producer: acquire empty[p] -> issue tile k+S
consumer: wait full[c]     -> consume tile k
```

只有这一区域有机会隐藏 latency。tile 数量不超过 stage 数时，程序只有 prologue/drain，没有稳定重叠区。

### Tail/drain

producer 已发完最后一个 async operation，不等于可以退出。它必须确认：

- 所有 transaction completion 已到达；
- consumer 不会再访问任何 stage；
- empty/full barrier storage 不再被远端 arrive；
- TMA store bulk groups 已完成；
- cluster peer lifetime 仍有效。

本章最小程序让 producer/consumer 在 CTA barrier 汇合后才 invalidate。生产 helper 的 `producer_tail(state)` 会逐 stage 等待 empty transition，表达更精确的 producer-side drain。

## 15.9 手写 `PipelineState` 的等价模型

一个 state 至少包含：

```text
count:  已推进多少次
index:  当前 stage
phase:  当前 generation parity
stages: 环形容量（静态）
```

`advance()` 的逻辑是：

```python
count += 1
index += 1
if index == stages:
    index = 0
    phase ^= 1
```

本章 raw primitive 示例直接用 `k % S` 与 `(k // S) & 1`，便于逐轮审计。生产代码使用 `pipeline.make_pipeline_state(PipelineUserType.Producer/Consumer, S)`，避免多个 loop 手写不同推进公式。

## 15.10 Producer 与 consumer 初始 state 不相同

高层 helper 的约定是：

- producer 从 index 0、phase 1 开始，表示 buffer 初始为空；
- consumer 从 index 0、phase 0 开始，等待第一份 full generation。

这与手写 empty pre-signal 是同一个初态的两种编码。调试时应先确认当前 pipeline class 的初始化约定，再解释 state.phase，不能把两套协议拼接。

## 15.11 `Agent` 与 `CooperativeGroup`

`Agent` 描述 participation 的基本单位：

- `Agent.Thread`：任意数量 threads；
- `Agent.Warp`：以 32 threads 为单位；
- `Agent.ThreadBlock`：完整 CTA；
- `Agent.ThreadBlockCluster`：cluster CTAs。

`CooperativeGroup(agent,size)` 表示多少个这类 agents 参与。例如：

```python
producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 128)
```

这常用于 single TMA issuer + 128 compute threads。`size` 会影响 arrival count；它不是注释，也不是 launch block size 的自动别名。

## 15.12 Pipeline helper 的统一接口

常用操作可以直接映射到状态机：

| Helper operation | 手写协议 |
|---|---|
| `producer_try_acquire` | 非阻塞检查 `empty[index]` |
| `producer_acquire` | 等待 `empty[index], phase` |
| `producer_get_barrier` | 获取本 stage full mbarrier，交给 TMA |
| `producer_commit` | 软件/TMA producer 发布 full |
| `consumer_try_wait` | 非阻塞检查 `full[index]` |
| `consumer_wait` | 等待 `full[index], phase` |
| `consumer_release` | 发布 empty |
| `state.advance()` | index wrap + phase toggle |
| `producer_tail` | 等待所有 live slots retire |

`try_*` 返回 token 时，后续 blocking operation 应消费该 token，避免重复检查；不能把 false 当作“可以跳过等待”。

## 15.13 Pipeline class 不是一个通用 barrier 名字

4.7.0 的主要类别包括：

| Class | Producer completion | Consumer/reuse completion | 典型用途 |
|---|---|---|---|
| `PipelineAsync` | software async arrive | empty mbarrier | 通用 producer/consumer |
| `PipelineCpAsync` | `cp.async` group semantics | stage ownership | SM80 copy pipeline |
| `PipelineTmaAsync` | TMA transaction bytes | consumer empty arrive | TMA load mainloop |
| `PipelineTmaStore` | bulk group/fence | store slot retirement | TMA epilogue store |
| `PipelineTmaUmma` | TMA load | UMMA consumer | SM100 mainloop |
| `PipelineAsyncUmma` | async/software producer | UMMA consumer | SM100 mixed producer |
| `PipelineUmmaAsync` | UMMA completion | async consumer | TMEM/epilogue handoff |

选择依据是 completion mechanism 和参与者，不是变量名字叫 load/store。

## 15.14 `PipelineTmaAsync` 的 transaction count

创建时需要：

```python
pipeline.PipelineTmaAsync.create(
    barrier_storage=ptr,
    num_stages=S,
    producer_group=...,
    consumer_group=...,
    tx_count=tile_bytes,
    cta_layout_vmnk=...,
)
```

`tx_count` 必须覆盖一个 stage 的全部 TMA transactions。如果同一 stage 由两个 producer 分别加载 A/B：

- 可以让 barrier arrival count/transaction count表达两个 producers；
- 或分别建立 A/B pipelines；
- 不能只按一份 tile bytes 设置然后等待“任一 copy 完成”。

multicast 还要考虑 receiver completion routing，与第 14 章一致。

## 15.15 Warp specialization 的真正含义

warp specialization 把 CTA threads 分成长期角色：

```text
TMA warp        -> descriptor issue / scheduler
MMA warpgroup   -> Tensor Core compute
epilogue warps  -> accumulator drain / TMA store
```

收益来自：

- 每个角色控制流更单一；
- TMA issuer 不重复执行 compute bookkeeping；
- producer 和 consumer 可以在不同 warp scheduler slot 上并行；
- register budget可以按角色重新分配。

代价包括 named/mbarrier storage、额外 threads、角色负载不均和尾部等待。

## 15.16 不要使用物理 `%warpid` 分配逻辑角色

逻辑 warp 应由：

```python
logical_warp = thread_idx_x // 32
```

得到。物理 warp slot ID 是调度资源标识，在 SM100 上不保证一个 2-warp CTA 恰好得到 slot 0/1。把角色写成 `%warpid == 0` 可能导致 producer warp 根本不存在。

生产代码如果调用 `cute.arch.warp_idx()`，应理解该 wrapper 是否已转换为期望的逻辑/均匀值，并常配合 `make_warp_uniform`。

## 15.17 Register budget 与 `setmaxnreg`

TMA producer通常只持有少量 descriptor/state registers，而 MMA consumer 需要大量 accumulator registers。Blackwell/Hopper warp-specialized kernel 会让 producer relinquish registers、consumer 增加 register budget。

但 register redistribution 受架构、warpgroup 和总 register file 约束：

- 分配总量不能超过 CTA/SM 限制；
- 增加 consumer registers 可能降低 occupancy；
- producer 仍需要足够 state/address registers；
- 编译器实际 allocation 需要 SASS/profiler 证据。

本章最小 copy pipeline 不人为调 budget，避免把 correctness 和 tuning 混在一起。

## 15.18 Stage 数的容量成本

假设每 stage 保存 A/B tile 和额外 metadata：

```text
SMEM_bytes
  = S × (bytes(A_tile) + bytes(B_tile) + padding)
  + barrier/state storage
  + epilogue buffers
```

stage 数从 2 增到 4，主循环 SMEM 基本翻倍。这可能：

- 覆盖更长 TMA latency；
- 同时把每 SM active CTA 从 2 降到 1；
- 让 occupancy/并发下降抵消 overlap；
- 增加 prologue/tail 占比。

因此“越深越快”没有普适性。

## 15.19 计算需要多少 stages

一个粗略模型是：

```text
required_stages ≈ ceil(load_latency / compute_time_per_stage) + safety_margin
```

但实际还受：

- TMA queue/descriptor latency；
- cluster multicast；
- MMA issue rate；
- scheduler/epilogue竞争；
- L2 hit/miss；
- occupancy 和 active clusters；
- problem tail。

模型用于提出候选，不用于替代 benchmark/autotune。

## 15.20 软件 pipeline 示例

[`pipeline_software_fill.py`](../code/15_pipeline/pipeline_software_fill.py) 使用：

- warp 0 producer、warp 1 consumer；
- 128 个 FP32/stage；
- 2、3、4 stage specializations；
- 每 stage full/empty mbarriers；
- producer/consumer release 前 full-mask warp publication；
- CTA tail 和 barrier invalidation。

测试 tile counts 专门覆盖：

- 少于 stages：只有部分 prologue slots；
- 等于 stages：刚好填满；
- stages+1：第一次 reuse/phase change；
- `2*stages+1`：phase 回到 0。

## 15.21 TMA warp-specialized 示例

[`tma_pipeline_warpspec.py`](../code/15_pipeline/tma_pipeline_warpspec.py) 使用：

- warp 0 TMA producer；
- warp 1 SMEM consumer；
- 128×64 FP16 TMA tile；
- 128-byte swizzled SMEM；
- explicit prologue fill；
- steady-state empty acquire；
- load-side transaction completion；
- consumer release；
- CTA tail/invalidate。

它只用 thread stores 写回 GMEM，避免同时引入 TMA store pipeline。真实 GEMM 可把 consumer body替换成 MMA，stage ownership 不变。

运行：

```bash
python discussion/code/15_pipeline/pipeline_software_fill.py
python discussion/code/15_pipeline/tma_pipeline_warpspec.py
```

## 15.22 PTX 证据能证明什么

本章检查：

- mbarrier init/arrive/try-wait/invalidate；
- TMA `complete_tx::bytes`；
- warp publication barrier；
- stage reuse 的 wait/release families。

这些证据证明 lowering 使用了预期 primitive，但不能单独证明 load 与 compute 在时间线上重叠。真实 overlap 需要 profiler timeline、stall reason 和吞吐对比；本章不做 L3 性能结论。

## 15.23 常见死锁与 race

1. 没有把 stages 初始化为 empty。
2. phase 使用 `tile & 1` 而非同一 stage generation。
3. producer commit 前，其他 producer lanes 尚未写完。
4. consumer release 前，其他 consumer lanes 尚未读完。
5. full barrier 有多个 producers，却仍设 arrival count 1。
6. TMA bytes只覆盖 A，不覆盖同 stage 的 B。
7. prologue 发出超过 problem tile count 的 transaction。
8. tail 前 producer 退出，consumer 仍会 arrive remote/local barrier。
9. TMA store 未 bulk-wait 就复用 output SMEM。
10. 128-thread CTA 的每个 warp 都成为 TMA issuer。
11. 用物理 warp slot ID 分配逻辑角色。
12. stage 数变更后只改 SMEM，没改 barrier/state layout。
13. `try_wait` 返回 false 后仍消费 stage。

## 15.24 与后续章节的接口

第 17–24 章会把 consumer body 从“copy out”逐步替换为三代 MMA：

```text
PipelineTmaAsync stage
  -> SM80 ldmatrix / SM90 WGMMA / SM100 UMMA
  -> accumulator ownership
  -> PipelineTmaStore / epilogue
```

pipeline 正确性与 MMA 数学正确性应分开验证。先证明 stage 不被覆盖，再证明 fragment mapping 和 GEMM reference。

## 15.25 检查清单与练习

检查清单：

1. 为每个 stage 画出 ownership 状态图。
2. 列出 producer/consumer group 和 arrival count。
3. 写出 index/phase 推进公式。
4. 标明 prologue 填充多少 slots。
5. 标明 steady state 从哪个 iteration 开始。
6. 证明 commit 前 producer writes 已发布。
7. 证明 release 前 consumer reads 已完成。
8. 对 TMA stage 计算完整 transaction bytes。
9. 对 TMA store 写出 proxy fence、commit、wait。
10. 证明 tail 后没有 in-flight arrive/copy/read。
11. 计算 SMEM/register/occupancy 成本。
12. 将功能、PTX、timeline 和性能证据分开记录。

练习：

1. 为 `S=4` 手算前 13 个 iterations 的 `(stage,phase)`。
2. 把软件 pipeline 改成两个 producer warps，重新设计 full arrival count。
3. 让 consumer 对每个 tile 加 1，再验证结果和 release 位置。
4. 给 TMA pipeline 添加第二个 operand，比较共享/独立 full barrier。
5. 用 `PipelineTmaAsync` 重写 raw mbarrier 版本，逐项对照 state。
6. 加入 `producer_try_acquire`，但保证 false path 最终执行 blocking wait。
7. 用 profiler 比较 2/3 stages，说明结果是否由 latency hiding 或 occupancy 主导。

