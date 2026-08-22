# 第 12 章：warp/CTA/cluster 同步原语

> 范围标签：`[通用/SM90+]`
>
> 前置章节：第 9 章执行层级、第 11 章 Shared Memory
>
> 配套代码：[`12_synchronization/`](../code/12_synchronization/README.md)

## 12.1 本章要回答的问题

第 11 章用两条 CTA barrier 完成了最经典的 SMEM buffer reuse。但真实 kernel 往往不希望所有线程每次都停在同一个位置：

- 一个 warp 负责搬运；
- 若干 warp 负责计算；
- 另一个 warp 负责 epilogue；
- 一个 CTA 等待异步 transaction；
- 一个 cluster 中的 CTA 访问彼此的 DSMEM。

本章要回答：

1. warp、CTA、named subset、mbarrier 和 cluster barrier 的参与者分别是谁？
2. rendezvous、memory ordering 和 async completion 有何区别？
3. member mask 与 named barrier `thread_count` 是怎样的契约？
4. `elect_sync`、vote、shuffle 和 redux 分别传递什么信息？
5. mbarrier 的 arrival count、phase/parity 和 transaction bytes 怎样工作？
6. acquire/release、普通 memory view 与 async proxy 为什么必须配套？
7. cluster barrier、DSMEM 与 `mapa` 怎样建立 remote SMEM 协议？
8. 哪些参与者错误会导致 silent race，哪些会直接死锁？

## 12.2 同步协议的四元组

不要仅凭 API 名字选择 barrier。先写出：

```text
Protocol
  = Participants
  + Completion condition
  + Memory ordering
  + Lifetime / generation
```

### Participants

哪些 lane、warp 或 CTA 必须到达？不参与者是否允许继续运行？

### Completion condition

等待的是所有参与者到达、一个 producer 到达、predicate reduction 完成，还是一批异步字节完成？

### Memory ordering

等待完成后，哪些作用域、哪些 memory proxy 的写入保证可见？

### Lifetime / generation

barrier 是否可以自动复用？下一代用 state token 还是 parity 区分？何时允许 invalidate 或覆盖 storage？

任何一项没有明确，就还没有形成完整同步协议。

## 12.3 Rendezvous、fence 与 completion 不是同义词

这三个概念经常同时出现，但语义不同。

### Rendezvous

参与者在某一点互相等待。例如 CTA `sync_threads`。

### Memory fence/order

约束 barrier 前后的内存操作可见性。Fence 本身通常不保证其他参与者已经执行到某个位置。

### Async completion

确认某个异步 engine 已经完成 transaction。例如 TMA 通过 mbarrier transaction completion 通知 consumer。

因此：

- 单独 fence 不能替代 producer completion；
- 单独 rendezvous 不一定完成另一个 async proxy 的 transaction；
- async completion 也不自动表示所有 CTA threads 都 rendezvous。

## 12.4 同步层级总览

| 原语 | 典型参与范围 | 是否有显式状态对象 | 主要用途 |
|---|---|---:|---|
| `bar_warp_sync(mask)` | 一个 warp 的 mask lanes | 否 | warp 内 rendezvous/可见性 |
| vote/shuffle/redux | 一个 warp 的 mask lanes | 否 | predicate 或 register 数据交换 |
| `sync_threads()` | 整个 CTA | 否 | 全 CTA rendezvous |
| named CTA barrier | 指定 warp 数量 | barrier slot | warp specialization handoff |
| CTA barrier reduction | 指定 CTA threads | barrier slot | rendezvous + predicate reduction |
| mbarrier | 软件定义到达者/transaction | SMEM 64-bit object | split-phase、异步 completion、pipeline |
| cluster barrier | cluster 内所有 CTA | 硬件 cluster state | 发布 DSMEM/cluster state |
| remote mbarrier + `mapa` | cluster scope | peer CTA SMEM object | 跨 CTA completion 与 DSMEM |

## 12.5 Warp member mask 是参与者集合

```python
prims.bar_warp_sync(mask)
```

`mask` 的第 `i` 位表示 lane `i` 是否参加。正确性契约是：

1. mask 中的每个 lane 都必须执行该调用；
2. 所有参与 lane 必须使用相同 mask；
3. mask 外 lane 不需要参加；
4. 参与 lane 的控制流必须能收敛到这次调用。

例如低半 warp：

```python
LOW_MASK = 0x0000FFFF
if lane < 16:
    prims.bar_warp_sync(LOW_MASK)
```

如果 mask 仍为 `FULL_MASK`，但高 16 lanes 绕过调用，低 16 lanes 将等待永远不会到达的参与者。

## 12.6 `bar_warp_sync` 的 memory semantics

warp barrier 不只是等待。对 mask 中的 lanes，它还对 barrier 前后的 shared/global memory operation 提供相应的 acquire-release ordering。

典型模式：

```python
if lane == 0:
    smem[0] = value
prims.bar_warp_sync(cute.arch.FULL_MASK)
observed = smem[0]
```

它只覆盖同一 warp 的 mask participants，不是 CTA barrier。两个不同 warp 即使 lane mask 相同，也不会因此互相等待。

## 12.7 `elect_sync`：选择 single issuer

```python
elected = prims.elect_sync()
if elected:
    single_issuer_operation()
```

在给定 member mask 中恰有一个 participating lane 得到 `True`。关键限制是：

> ISA 不保证 winner 一定是 lane 0。

适合由 elected lane 发出的操作包括：

- `mbarrier_arrive`；
- 一次 TMA descriptor operation；
- 某个 warp 只应执行一次的 commit；
- single-writer marker。

如果算法语义明确要求 lane 0，应写 `if lane == 0`；如果只要求任意唯一 issuer，才使用 election。

不要把只有 elected lane 执行的分支包住本来要求全 mask 参与的 warp collective。

## 12.8 Vote：聚合 predicate

`vote_sync` 处理的是每 lane Boolean，不移动普通数值：

```python
any_true = prims.vote_sync(mask, pred, prims.VoteSync.ANY)
all_true = prims.vote_sync(mask, pred, prims.VoteSync.ALL)
uniform = prims.vote_sync(mask, pred, prims.VoteSync.UNI)
ballot = prims.vote_sync(mask, pred, prims.VoteSync.BALLOT)
```

语义分别是：

- `ANY`：至少一个参与 lane 为真；
- `ALL`：全部参与 lane 为真；
- `UNI`：全部参与 lane 的 predicate 相同；
- `BALLOT`：返回 predicate 为真的 lane bitset。

ballot 常用于 compact、work queue、稀疏分支和 active-lane 计数。它返回的是 lane mask，不是线程总数。

## 12.9 Shuffle：交换 register value

Shuffle 允许 warp lanes 直接读取其他 lane 的 register value：

- `IDX`：指定绝对 source lane；
- `UP`：读取 `lane-delta`；
- `DOWN`：读取 `lane+delta`；
- `BFLY`：读取 `lane XOR delta`。

例如从 lane 7 广播：

```python
broadcast = prims.shfl_sync(
    FULL_MASK,
    lane_value,
    7,
    0x1F,
    prims.Shfl.IDX,
)
```

`mask_and_clamp` 同时编码 segment 和 clamp 边界。做 sub-warp shuffle 时，它与 member mask 是两个不同概念：

- member mask 决定谁参与；
- segmentation/clamp 决定 source lane 如何计算。

## 12.10 Redux：硬件 warp reduction

`redux_sync` 将 participating lanes 的数值规约，并把结果广播给参与 lanes：

```python
warp_sum = prims.redux_sync(
    value,
    prims.ReductionKind.ADD,
    FULL_MASK,
)
```

整数 ADD/MIN/MAX/bitwise 等操作具有较广的 SM80+ 支持。Float reduction 和 `abs`/`nan` modifier 的精确支持范围依赖指令和架构；例如 B200 上可以使用 SM100 的 `redux.sync.max.abs.f32`。

本章只把 redux 当作同步/通信 primitive。如何把 warp result 合并成 CTA reduction、如何构建 Softmax 留到第 16 章。

## 12.11 `sync_threads`：全 CTA barrier

```python
cute.arch.sync_threads()
```

它要求当前 CTA 的全部 threads 到达。这适用于第 11 章的 homogeneous tiled kernel，但对 warp-specialized kernel 可能过宽：producer、consumer 和 epilogue 都会被强制停下。

只要 barrier 在动态分支中，就必须证明整个 CTA 对该分支的选择一致。边界 predicate 应保护内存访问，不应改变 barrier participation。

## 12.12 Named CTA barrier slot

CuTe experimental primitives 暴露：

```python
prims.barrier_cta_arrive(barrier_id, thread_count)
prims.barrier_cta_sync(barrier_id, thread_count=...)
```

每个 CTA 有有限数量的 named slots；当前 wrapper 检查静态 `barrier_id` 在 `0..15`。

`thread_count` 是本次 barrier generation 的总参与 thread 数，不只是 producer 数。静态 count 必须是非零 warp-size multiple。

## 12.13 Arrive/sync split-phase handoff

两个 warp 的 producer/consumer 示例：

```python
if warp == 0:
    produce()
    prims.barrier_cta_arrive(1, thread_count=64)
elif warp == 1:
    prims.barrier_cta_sync(1, thread_count=64)
    consume()
```

参与者表是：

| Role | Threads | Operation |
|---|---:|---|
| producer warp | 32 | arrive，不等待 |
| consumer warp | 32 | arrive 并等待 |
| total | 64 | barrier completion count |

producer 可以在 arrive 后继续执行无依赖工作。consumer 直到 64 个参与者都到达才继续。

如果错误地写 `thread_count=32`，barrier 可能在 consumer 到达前就完成；如果写成 96，却只有 64 threads 参加，则会等待缺失的 32 participants。

## 12.14 Aligned 与 non-aligned CTA barrier

API 提供 unsuffixed 和 `_aligned` 版本。

- non-aligned 版本能表达处于 divergent control flow 中的已知参与子集；
- aligned 版本向编译器/硬件承诺全 CTA converged participation。

`aligned` 不是自动变快的开关，而是更强的前置条件。无法严格证明时，应使用 non-aligned form。

## 12.15 CTA barrier reduction

```python
any_true = prims.barrier_cta_red(
    pred,
    barrier_id=2,
    kind="or",
    thread_count=64,
)
count = prims.barrier_cta_red(
    pred,
    barrier_id=3,
    kind="popc",
    thread_count=64,
)
```

它把 rendezvous 与 predicate reduction 结合：

- `and` 返回全部参与者是否为真；
- `or` 返回是否有参与者为真；
- `popc` 返回真 predicate 的数量。

不要在同一个 active generation/slot 中混用 reducing 和 non-reducing barrier operation。不同协议使用不同 slot，或严格证明上一 generation 已完成再复用。

## 12.16 Named slot 不是 mbarrier object

两者都可以做 split-phase 协议，但抽象不同：

| Named CTA barrier | mbarrier |
|---|---|
| 隐含硬件 slot | 显式 64-bit SMEM object |
| count 通常按 threads | arrival count 可由软件角色定义 |
| CTA scope | CTA 或 cluster scope |
| 适合 warp subset rendezvous | 适合 pipeline、async transaction completion |
| generation 由 slot 协议隐式复用 | state token/parity 显式区分 phase |

mbarrier 不是“更多 named slots”，而是带生命周期和 transaction state 的同步对象。

## 12.17 mbarrier 的物理对象

一个 mbarrier 是位于 Shared Memory 的 64-bit object，至少按 8 bytes 对齐：

```python
mbar = cutlass.Array(
    cutlass.Int64,
    1,
    space=cutlass.AddressSpace.smem,
    alignment=8,
)
```

barrier storage 自身也占用 per-CTA SMEM。多 stage pipeline 通常为每 stage 分配 full/empty barrier，容量应计入第 11 章的 SMEM 公式。

## 12.18 初始化与发布是两个动作

典型初始化协议：

```python
if elected:
    prims.mbarrier_init(mbar, expected_arrivals)
prims.fence_mbarrier_init()
prims.barrier_cta_sync(0)
```

三个动作分别表示：

1. 唯一 issuer 初始化 object；
2. init fence 发布初始化；
3. 相关 CTA threads 在使用前 rendezvous。

如果其他 CTA 还要通过 DSMEM 访问这个 object，仅 CTA barrier 不够；需要 cluster 范围的发布和 rendezvous。

## 12.19 Arrival count

```python
prims.mbarrier_init(mbar, count)
```

`count` 描述当前 phase 需要多少 arrival contribution。每次 software arrive 消耗相应 arrival count；达到零时 phase completion。

它不必等于 CTA threads：

- 一个 elected lane 可以代表整个 producer warp 到达；
- 多个 producer warp 可以各由一个 elected lane arrive；
- TMA completion 可以贡献 transaction completion。

但代表关系必须由此前的 warp/CTA synchronization 保证。例如一个 elected producer lane 在 signal full 前，必须确认同 warp 的所有数据写入已经发布。

## 12.20 State token 与 parity wait

mbarrier 有两类常见等待方式。

### State-token path

```python
state = prims.mbarrier_arrive(mbar)
done = prims.mbarrier_try_wait_timelimit(mbar, state, limit)
```

arrival 返回 state token，wait 判断对应 generation 是否完成。

### Parity path

```python
while not prims.mbarrier_try_wait_parity(
    mbar, phase, time_limit=limit
):
    pass
```

fresh barrier 从 parity 0 开始。等待 parity `p` 的含义是等待当前 phase 离开 `p`，即发生 completion toggle。

try-wait 可以因 time limit 暂时返回 false，因此要放在循环中；time limit 是诊断/重试机制，不是协议正确性的替代品。

## 12.21 双 stage full/empty 状态机

主例为每个 stage 分配两个 barrier：

```text
full[s]  : producer -> consumer
empty[s] : consumer -> producer
```

初始化后先 arrive `empty[s]`，把所有 stage 标记为空。第 `k` 个 tile 使用：

```text
s     = k mod STAGES
phase = floor(k / STAGES) mod 2
```

producer：

```text
wait empty[s], phase
write stage s
publish producer warp writes
arrive full[s]
```

consumer：

```text
wait full[s], phase
read stage s
finish consumer warp reads
arrive empty[s]
```

对 `STAGES=2`：

| tile k | stage | waited phase |
|---:|---:|---:|
| 0 | 0 | 0 |
| 1 | 1 | 0 |
| 2 | 0 | 1 |
| 3 | 1 | 1 |
| 4 | 0 | 0 |

phase 不是 stage index，而是同一个 barrier object 的 generation bit。

## 12.22 Full/empty protocol 与第 11 章两条 barrier

第 11 章：

```text
all CTA produce
  -> CTA barrier
all CTA consume
  -> CTA barrier
```

本章：

```text
producer warp waits empty[s]
  -> writes
  -> arrives full[s]

consumer warp waits full[s]
  -> reads
  -> arrives empty[s]
```

full barrier 对应 producer→consumer 的 RAW protection；empty barrier 对应 consumer→producer 的 WAR protection。改变的是参与者和 split-phase 表达，hazard 本身没有改变。

## 12.23 Transaction bytes

mbarrier 不仅可以等待 software arrivals，还可以跟踪 async transaction bytes。相关操作包括：

- arrive and expect transaction bytes；
- 单独增加 expected transaction bytes；
- async engine 在数据落地时 complete transaction bytes。

barrier completion 需要同时满足：

```text
pending arrivals == 0
and
pending transaction bytes == 0
```

本章 ping-pong 只用 software arrival，目的是先把 stage/phase 状态机讲清楚。第 14 章 TMA 会把 `expect_tx` 与实际 copy byte count 绑定起来。

## 12.24 Acquire/release 与 async proxy

普通线程通过 generic memory view 读写 SMEM；TMA、某些 async copy 和 tensor-core path 可能使用不同的 memory proxy。

正确协议需要同时回答：

1. producer 的写属于哪个 proxy？
2. 哪条 fence 把 generic view 与 async proxy 的观察顺序连接起来？
3. 哪个 completion mechanism 表示数据真正落地？
4. consumer wait 提供哪个 scope 的 acquire ordering？

例如，普通 `sync_threads` 不应被描述成任意 TMA transaction 的 completion wait。反过来，一个 mbarrier transaction 完成也不表示无关 CTA threads 已经 rendezvous。

第 13、14 章会对具体 proxy fence 和 copy operation 给出代码；本章建立判断框架。

## 12.25 mbarrier invalidate 与 lifetime

```python
prims.mbarrier_inval(mbar)
```

只有在所有可能访问该 object 的 participant 都完成后才能 invalidate。对 full/empty ring：

- producer 不能仍在 wait empty；
- consumer 不能仍在 wait full；
- 不能仍有 async engine 将完成事件写入 barrier；
- cluster peer 不能再持有 remote mapped pointer。

主例在两个 warp 退出角色循环后先做 CTA rendezvous，再由 thread 0 invalidate 每个 object。

## 12.26 CTA cluster 与 DSMEM

SM90+ 可以把多个 CTA 作为 cluster launch：

```python
kernel(...).launch(
    grid=grid,
    block=block,
    cluster=(cluster_x, 1, 1),
)
```

每个 CTA 仍有自己的本地 SMEM allocation，但 cluster 中的 CTA 可以通过 distributed shared memory 地址空间访问 peer SMEM。

```python
rank = cute.arch.block_idx_in_cluster()
```

返回当前 CTA 在 cluster 中的 rank，不是整个 grid 的普通 `block_idx`。

## 12.27 `mapa`：映射 corresponding peer address

```python
peer_ptr = prims.mapa(local_ptr, peer_rank)
```

`mapa` 把“当前 CTA 本地 SMEM 中的某个地址”映射到“目标 CTA 对应的 SMEM 地址”。它不是任意 pointer arithmetic：

- source 必须是适合映射的 shared address；
- peer 必须属于同一个 cluster；
- 两个 CTA 的静态 SMEM allocation 结构必须对应；
- peer CTA 必须仍然存活。

映射后的 pointer 可以用于 remote SMEM load/store，也可以指向 remote mbarrier。

## 12.28 Cluster initialization protocol

主例按以下顺序：

```text
each CTA initializes its local mbarrier
  -> fence_mbarrier_init
  -> cluster arrive
  -> cluster wait
  -> first remote mapa/use
```

CTA barrier 只能发布给本 CTA threads，不能保证其他 CTA 已完成 barrier initialization。cluster rendezvous 才能建立第一次 remote access 前的 cluster-wide happens-before。

## 12.29 Remote signal 与 remote read

cluster ring 中 CTA `i`：

1. 把自己的 rank 写入 local SMEM；
2. 映射 predecessor CTA 的 mbarrier；
3. 以 cluster scope arrive predecessor barrier，表示“你的 successor 数据已经就绪”；
4. 等待 successor arrive 自己的 local barrier；
5. 映射 successor 的 data pointer；
6. 读取 successor rank。

方向必须与读取目标匹配。如果通知 successor、却仍读取 successor，那么本地 wait 收到的是 predecessor 的通知，不能为 successor 的数据建立 happens-before；即使某次运行结果正确，协议证明也不成立。

最后再次 cluster rendezvous，保证没有 CTA 提前退出并释放 DSMEM，而 peer 仍在 remote read。

## 12.30 三个可执行示例

### Warp 与 CTA collectives

[`warp_collectives.py`](../code/12_synchronization/warp_collectives.py) 验证：

- full-mask warp shared-memory handoff；
- `elect_sync` 恰好一个 winner，但不假定 winner ID；
- ANY/ALL/BALLOT；
- lane 7 shuffle broadcast；
- integer redux sum；
- 64-thread named CTA arrive/sync；
- CTA OR 和 POPC reductions；
- retained PTX 中对应指令族。

### mbarrier ping-pong

[`mbarrier_pingpong.py`](../code/12_synchronization/mbarrier_pingpong.py) 验证：

- 两个 warps 的 producer/consumer specialization；
- 两个 stages 的 full/empty barrier；
- init publication；
- parity wrap；
- runtime tile count 1、2、3、5、8；
- barrier lifetime 与 invalidate；
- mbarrier PTX instruction families。

### Cluster DSMEM ring

[`cluster_dsmem_ring.py`](../code/12_synchronization/cluster_dsmem_ring.py) 验证：

- cluster launch；
- CTA rank；
- cluster-wide init publication；
- remote mbarrier arrive；
- DSMEM `mapa` read；
- exit 前 cluster rendezvous；
- cluster PTX instruction families。

运行：

```bash
python discussion/code/12_synchronization/warp_collectives.py
python discussion/code/12_synchronization/mbarrier_pingpong.py
python discussion/code/12_synchronization/cluster_dsmem_ring.py --cluster-size 2
```

B200 输出统一记录在 [`validation_log.md`](validation_log.md)。

## 12.31 死锁与 race 模式

### Mask 包含没有执行的 lane

warp barrier、vote、shuffle、redux 都必须满足 member mask participation。

### CTA barrier 放在非统一分支

只有部分 CTA threads 到达全 CTA barrier，其他 threads 不会自动被排除。

### Named barrier count 不匹配

count 太大会永久等待，count 太小会过早释放并形成 race。

### 相同 slot 混合不同协议

尤其不能在同一个 active generation 混用 reducing 和 non-reducing barrier。

### Elected lane 过早 arrive

唯一 issuer 已到达不代表同 warp 的其他 producer 写入都已完成。需要先建立 warp-level publish。

### 漏掉 init fence 或发布 barrier

消费者可能观察到未初始化 mbarrier state。

### Phase 公式错误

把 phase 写成 `tile & 1`，在多 stage ring 中通常不等于同一 barrier object 的 generation。

### 只等待 full，不等待 empty

producer 会覆盖 consumer 尚未读完的 stage。

### 把 time limit 当成功条件

try-wait 返回 false 应继续按协议等待或进入明确错误路径，不能直接消费数据。

### Cluster peer 提前退出

DSMEM pointer 的目标 CTA lifetime 已结束，remote access 不再合法。

### 用 local scope arrive remote barrier

remote mbarrier operation 必须使用匹配的 cluster address/scope semantics。

## 12.32 如何安全分析“坏协议”

不要为了演示 deadlock 而在 GPU 上运行一个无限等待 kernel。更安全的方法是：

1. 列出 participants；
2. 统计每个 barrier generation 的 arrivals；
3. 写出 stage/phase 表；
4. 检查所有 control-flow path 的调用次数与顺序；
5. 使用有 time limit 的 try-wait 做诊断；
6. 把故障变体保留为纸面推演或 compile-only test。

GPU hang 可能影响同一设备上的其他进程，也可能需要重置设备，不应作为教程默认测试。

## 12.33 PTX 证据与功能证据

功能结果正确，只证明当前输入下协议产生了正确输出；PTX 中出现某条 barrier，只证明编译器生成了预期指令族。

高可信验证应组合：

- L0：participants/count/phase 的静态推演；
- L1：reference result；
- L2：不同 runtime tile counts、phase wrap、residue/role cases；
- L4：PTX/SASS instruction evidence；
- sanitizer：在工具支持时检查 race 和非法地址；
- L3：只在明确 benchmark 方法后讨论性能。

没有任何一项能单独证明任意规模、任意调度下绝无同步 bug。

## 12.34 与后续章节的接口

第 13 章会把：

```text
thread-issued cp.async
  -> commit group
  -> wait group / mbarrier completion
```

接到本章的参与者和 completion 模型。

第 14 章会把：

```text
TMA issuer election
  -> expect transaction bytes
  -> async proxy writes SMEM
  -> mbarrier transaction completion
```

接到本章的 transaction bytes 与 proxy ordering。

第 15 章会把 full/empty、stage、phase、producer/consumer warp 组合成完整 pipeline。

第 16 章会把 vote/shuffle/redux 与 SMEM 合并，构建 CTA reduction 和 Softmax。

## 12.35 本章检查清单与练习

设计同步协议时：

1. 写出每条同步原语的 participant set。
2. 写出 completion condition。
3. 写出 memory scope 和 proxy。
4. 标明 aligned/non-aligned 前置条件。
5. named barrier 记录 slot、count 和 operation kind。
6. mbarrier 记录 object、expected arrivals 和 transaction bytes。
7. 对每个 stage 写出 full/empty barrier。
8. 对每轮写出 stage index 与 phase。
9. 证明 elected issuer 之前的 producer work 已发布。
10. 证明 invalidate/CTA exit 前不存在远端或异步使用者。
11. 使用有界诊断，避免运行故意死锁的 kernel。
12. 分开记录功能、边界、PTX 与性能证据。

练习：

1. 把 warp ballot predicate 改为偶数 lane，手算 32-bit mask。
2. 用 shuffle butterfly 实现 warp sum，与 `redux_sync` 结果逐位比较。
3. 把 named CTA handoff 扩展为 1 producer + 2 consumers，重新计算 `thread_count`。
4. 给 CTA barrier reduction 增加 `and` case，并解释返回类型。
5. 把 ping-pong `STAGES` 改为 3，列出前 12 个 tile 的 `(stage,phase)`。
6. 增加两个 producer warps，让每个 warp 一个 elected lane arrive；修改 expected arrival count。
7. 为 mbarrier 状态机画出 fresh、full、empty 和下一 generation 的转换。
8. 把 cluster ring 扩展到 4 CTAs，并验证 rank 映射。
9. 解释为什么 cluster init 不能只用 `sync_threads`。
10. 为第 14 章预先计算一个 `32 x 32 FP16` TMA tile 的 transaction byte count，但暂不发出 TMA copy。
