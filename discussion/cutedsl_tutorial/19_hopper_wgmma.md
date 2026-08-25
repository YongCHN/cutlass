# 第 19 章：Hopper WGMMA、TMA 与 Warpgroup Pipeline

> 范围标签：`[SM90a]`
>
> 前置章节：第 14–15 章 TMA/pipeline、第 17 章 MMA 公共模型、第 18 章 SM80 warp MMA
>
> 配套代码：[`19_hopper_wgmma/`](../code/19_hopper_wgmma/README.md)

## 19.1 本章要回答的问题

Hopper 不只是把 `mma.sync` 换成更大的矩阵指令。本章围绕这条路径展开：

```text
GMEM A/B
  -> TMA descriptor / transaction mbarrier
  -> swizzled SMEM A/B
  -> SMEM descriptor fragments
  -> asynchronous warpgroup WGMMA
  -> FP32 RMEM accumulator
  -> SMEM-staged epilogue
  -> TMA store GMEM C
```

需要回答：

1. warpgroup 的 128-thread participant model 怎样进入 TiledMMA？
2. WGMMA 的 A/B descriptor fragment 与 SM80 RMEM fragment有何不同？
3. `fence → mma_async → commit_group → wait_group` 各保护什么？
4. TMA completion 与 WGMMA completion 为什么是两套协议？
5. producer warp 与 consumer warpgroup 怎样共享 SMEM ring？
6. cluster multicast减少了什么流量，又增加了哪些 lifetime约束？
7. persistent scheduler与 WGMMA 本身是什么关系？
8. FP8 2× accumulator和 fused GELU为什么属于资源管理问题？

## 19.2 架构边界：`SM90a` 不是泛化的 `SM90+`

CuTe DSL 的 `warpgroup.MmaF16BF16Op` 明确要求 `sm_90a`。`a` 表示使用 architecture-specific accelerated features；生成的 cubin 不能假设可在后续 major SM 上直接执行。

本教程 B200 服务器实测：

```text
CUTE_DSL_ARCH=sm_90a 编译：成功
在 B200/SM100 启动 sm_90a image：cudaErrorNoKernelImageForDevice (209)
```

因此本章验证分两层：

- B200 上真实完成 `sm_90a` AOT compilation 与 retained PTX审计（L0/L4）；
- 数值 L1 必须在 H100/H200 上补验。

不能因为 B200 是更新架构就把 compile-only写成 numerical PASS。代码会自动检测 device capability：非 9.x 只编译和检查 PTX，不尝试非法 launch。

## 19.3 从 warp 到 warpgroup

SM80 `mma.sync` 的 participant是一个 32-thread warp。Hopper WGMMA 的 participant是四个连续 warps：

```text
warpgroup = 4 warps = 128 threads
```

所有 threads必须：

- 位于同一 CTA；
- 属于同一个连续 warpgroup；
- converged执行 WGMMA issue/fence/commit/wait sequence；
- 对 descriptor与 accumulator shape达成一致。

一个典型 `m64n64k16` atom：

```text
M = 64
N = 64（N 可在 8..256，步长 8）
K = 16 for FP16/BF16
participants = 128
```

它一条指令就覆盖 SM80 四个 `m16` M strips的规模，但并不意味着只用一条指令就完成大 GEMM；CTA K tile仍需 repeats，CTA M/N也可复制多个 warpgroups。

## 19.4 WGMMA Op 的关键参数

配套代码等价构造：

```python
op = cute.nvgpu.warpgroup.MmaF16BF16Op(
    cutlass.Float16,
    cutlass.Float32,
    (64, 64, 16),
    cute.nvgpu.warpgroup.OperandSource.SMEM,
    cute.nvgpu.OperandMajorMode.K,
    cute.nvgpu.OperandMajorMode.K,
)
tiled_mma = cute.make_tiled_mma(op, (1,1,1))
```

参数除了 dtype/shape，还明确：

- A 来自 SMEM descriptor还是 RMEM；
- A major mode；
- B major mode。

FP16/BF16 的 descriptor route支持 K-major与 MN-major；A 使用 RMEM route时约束更强。FP8 WGMMA K=32，并且 operand major支持范围不同。不要把 FP16配置直接替换 dtype后期待 FP8成立。

## 19.5 `get_slice(0)` 为什么不是只让 thread 0 计算

对单 warpgroup、`atom_layout=(1,1,1)`：

```python
thr_mma = tiled_mma.get_slice(0)
```

这里的 slice index选择的是 TiledMMA 中的 warpgroup atom position，不是 lane id。128 threads都调用同一个 collective atom；各 thread的 accumulator lane mapping由 WGMMA atom TV metadata和实际硬件 lane隐式决定。

若 CTA有两个 WGMMA warpgroups，上游代码会构造 stride=128 的 warpgroup layout：

```text
warpgroup 0 -> slice 0
warpgroup 1 -> slice 128
```

这与第 17 章 SM80 直接用 `get_slice(tid)` 的写法不同，不能机械搬运。

## 19.6 A/B fragment 是 SMEM descriptor

Hopper descriptor route：

```python
t_s_a = thr_mma.partition_A(s_a)
t_s_b = thr_mma.partition_B(s_b)
desc_a = tiled_mma.make_fragment_A(t_s_a)
desc_b = tiled_mma.make_fragment_B(t_s_b)
```

`desc_a/desc_b` 不是把整个 operand加载进 registers。它们编码：

- SMEM base/start address；
- leading/stride offset；
- swizzle mode；
- major mode与 element format；
- partition/repeat coordinate。

WGMMA执行时硬件从 SMEM读取。优点是省掉显式 `ldmatrix → A/B RMEM` 路径并降低 operand register pressure；代价是 descriptor、SMEM lifetime和 async proxy ordering必须正确。

## 19.7 ComposedLayout 的 swizzle为何要移到 pointer

WGMMA fragment verification要求 affine outer layout，swizzle由 iterator/pointer携带。典型分配：

```python
s_a = storage.a.get_tensor(
    composed.outer,
    swizzle=composed.inner,
)
```

这样 Tensor layout仍可被 descriptor分析，而物理地址变换保留在 pointer。若把 ComposedLayout整体传入 `make_fragment_A`，API会提示把 swizzle移到 pointer。

本章使用 `K_SW32`；生产 kernel通常按连续 bit width选择 `K_SW32/K_SW64/K_SW128`，并用 `tile_to_shape`扩展到 CTA tile/stage。

## 19.8 TMA completion：operand何时可读

TMA load依赖 transaction-counted mbarrier：

```text
mbarrier_init(arrival_count=1)
mbarrier_expect_tx(bytes_A + bytes_B)
TMA A -> SMEM, complete_tx(barrier)
TMA B -> SMEM, complete_tx(barrier)
software arrive
mbarrier wait phase
```

只有 arrival count和transaction bytes都完成，consumer才可使用 descriptor读取 SMEM。`wgmma.fence` 不能替代 TMA wait；它解决的是 WGMMA proxy/register ordering，不证明 TMA engine已写完 A/B。

## 19.9 WGMMA 的四步协议

### 19.9.1 `warpgroup.fence()`

把此前 accumulator/A-register writes排序到后续 WGMMA reads之前。即使 A/B来自 descriptor，accumulator初始化仍要被 WGMMA正确观察。

### 19.9.2 `cute.gemm(...)`

发出 `wgmma.mma_async`。它异步更新 accumulator registers，Python调用返回不等于结果可读。

### 19.9.3 `warpgroup.commit_group()`

把此前未提交的 WGMMA operations组成一个 async group。它不等待完成。

### 19.9.4 `warpgroup.wait_group(N)`

等待到至多 N 个更年轻 groups仍 outstanding。消费 accumulator前必须 `wait_group(0)`；pipeline中可用 `wait_group(1)`保留下一组并行度。

最小闭环：

```python
accum.fill(0)
tiled_mma.set(Field.ACCUMULATE, False)
warpgroup.fence()
cute.gemm(tiled_mma, accum, desc_a, desc_b, accum)
warpgroup.commit_group()
warpgroup.wait_group(0)
```

## 19.10 Accumulate field

WGMMA atom带 runtime `Field.ACCUMULATE`：

```text
False: D = A*B，忽略旧 accumulator（首个 K block）
True:  D = A*B + D（后续 K blocks）
```

production mainloop通常首个 issue后立即切换为 True。显式 accumulate state可以避免先用普通 instructions清零大 accumulator，但必须确保首个有效 K block确实以 False执行。

## 19.11 两套 async pipeline不能混为一套

```text
TMA pipeline:
  producer acquire -> TMA issue -> mbarrier complete_tx
  consumer wait -> descriptor safe to read

WGMMA pipeline:
  fence -> mma_async -> commit_group -> wait_group
  accumulator safe to read/reuse
```

SMEM stage reuse还需要第三个条件：所有 WGMMA descriptor reads结束。上游 pipeline在 `wait_group(k_pipe_mmas)` 后 release consumer stage，TMA producer才能覆盖它。

因此一个 stage的状态机是：

```text
EMPTY -> TMA_WRITING -> FULL -> WGMMA_READING -> EMPTY
```

accumulator group的状态机则是：

```text
RMEM_READY -> WGMMA_PENDING -> WGMMA_COMMITTED -> RMEM_READY
```

## 19.12 Warp specialization

Hopper TMA一条指令可搬整个 tile，不需要所有 128 consumer threads参与 issue。典型分工：

```text
warp 0: TMA producer / descriptor prefetch
warps 0..3 or dedicated warpgroup: WGMMA consumer
epilogue warps: RMEM -> SMEM -> TMA store
```

实际生产 kernel可能让一个 warp同时承担 producer辅助角色，也可能独立出多个 warpgroups。角色划分必须匹配：

- participant count；
- barrier arrival count；
- register budget；
- pipeline producer/consumer group；
- early-exit与tail protocol。

## 19.13 Cluster multicast

若 cluster shape为 `(cluster_M, cluster_N)`：

- 沿 N的多个 CTAs需要同一 A tile；
- 沿 M的多个 CTAs需要同一 B tile。

TMA multicast让一个 issuer把 A/B tile路由到多个 CTA SMEM，降低重复 L2/DRAM reads。mask由 CTA layout image计算，不是简单固定常量。

增加的协议：

- cluster-wide mbarrier init publication；
- transaction count考虑 multicast destinations；
- `.shared::cluster` address/routing；
- 所有 peers完成 descriptor reads前保持 SMEM和 CTA lifetime；
- cluster边界 grid padding/launch contract。

multicast不等于共享同一 SMEM地址；每 CTA仍有自己的 SMEM destination，TMA engine负责投递。

## 19.14 Persistent scheduler与 WGMMA解耦

persistent GEMM让有限 CTAs循环领取多个 output tiles，以减少 wave quantization并提高 cache reuse。它改变：

- grid规模；
- tile领取顺序；
- TMA descriptor coordinates；
- pipeline tail与下一 tile重置；
- cluster一致调度。

它不改变单次 WGMMA instruction语义。先证明一个 tile的 TMA/WGMMA protocol，再加入 scheduler，才能定位 deadlock是数据路径还是调度造成。

## 19.15 Epilogue

WGMMA accumulator位于 RMEM，但其 TV Layout通常不适合直接 coalesced GMEM store。上游 epilogue：

```text
wait_group(0)
RMEM accumulator
  -> optional convert/fused op
  -> stmatrix/universal copy to swizzled SMEM
  -> async proxy fence + CTA sync
  -> TMA S2G subtiles
  -> TMA store commit/wait/tail
```

SMEM可复用 mainloop A/B storage，但必须先等所有 warpgroups停止读取 descriptors。cluster>1时还要避免 peer CTA过早退出。

## 19.16 FP8 2× accumulator

FP8 WGMMA K granule更大，输入吞吐高，accumulator register pressure可能成为瓶颈。2× accumulator策略把连续 K groups在两套 RMEM accumulators间交错：

```text
group 0 -> accumulator 0
group 1 -> accumulator 1
group 2 -> accumulator 0
...
```

收益可能包括更多 outstanding WGMMA和依赖链解耦；代价是 accumulator registers翻倍，occupancy可能下降。最后还要逐元素合并两套 accumulator，再进入 GELU/convert epilogue。

## 19.17 Fused GELU 的放置

GELU位于 `wait_group(0)` 之后，因为不能读取 pending accumulator。常见顺序：

```text
acc0 + acc1
 -> FP32 GELU approximation
 -> FP16/FP8 conversion
 -> SMEM epilogue
 -> TMA store
```

融合节省单独 kernel和 GMEM roundtrip，却增加 registers与 transcendental instructions。是否有收益取决于 GEMM tile、epilogue overlap和下游是否真正需要 materialized output。

## 19.18 配套最小代码

[`hopper_wgmma_gemm.py`](../code/19_hopper_wgmma/hopper_wgmma_gemm.py) 固定：

```text
A = (64,16) FP16 K-major
B = (64,16) FP16 K-major
C = (64,64) FP32
one CTA / one warpgroup / one WGMMA
```

它刻意不加入多 stage、cluster或复杂 epilogue，而是 isolating：

1. 两个 TMA descriptors；
2. swizzled SMEM descriptor fragments；
3. transaction barrier；
4. WGMMA四步协议；
5. direct accumulator store。

在 H100/H200 上会执行 PyTorch reference；在 B200上只编译和审计 PTX。

## 19.19 B200 compile/L4结果

```text
PTX TMA load:
  cp.async.bulk.tensor.2d.shared::cta.global...mbarrier...
PTX WGMMA fence:
  wgmma.fence.sync.aligned
PTX WGMMA:
  wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16
PTX commit:
  wgmma.commit_group.sync.aligned
PTX wait:
  wgmma.wait_group.sync.aligned 0
COMPILE-ONLY PASS
```

验证等级：L0/L4。数值结果未在 B200执行，也未记录为 L1。

## 19.20 常见失败模式

- 在 B200上强行启动 `sm_90a` cubin，得到 error 209；
- 只等待 TMA mbarrier，忘记 WGMMA `wait_group`；
- `commit_group` 后立即读 accumulator；
- descriptor pointer没有携带 swizzle或alignment不足；
- producer覆盖 SMEM stage时 WGMMA仍在读取；
- 128 threads未 converged执行 WGMMA control sequence；
- cluster multicast tx count/mask与实际 destinations不一致；
- epilogue复用 A/B SMEM前没有 drain所有 descriptor readers。

## 19.21 本章小结

- WGMMA把 participant扩大到128-thread warpgroup，并支持 SMEM descriptor operands；
- TMA completion与 WGMMA completion是独立状态机；
- 正确主循环必须串联 TMA mbarrier、WGMMA fence/commit/wait和 SMEM stage release；
- cluster multicast、persistent scheduling、FP8 2× accumulator都是在正确单 tile协议之上的资源/调度优化；
- 当前 B200环境完成真实 SM90a编译/PTX验证，但 Hopper数值执行需 H100/H200补验。

下一章进入 Blackwell：accumulator从 RMEM迁移到 TMEM，MMA completion改用 `tcgen05.commit` 到 mbarrier，1CTA/2CTA collective还引入 TMEM allocation permit与跨 CTA ownership。
