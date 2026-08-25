# 第 18 章：Ampere warp MMA 与 TensorOp GEMM

> 范围标签：`[SM80+]`
>
> 前置章节：第 10/13 章 TiledCopy、`cp.async`、`ldmatrix`，第 17 章 MMA 公共模型
>
> 配套代码：[`18_ampere_tensorop/`](../code/18_ampere_tensorop/README.md)

## 18.1 本章数据路径

```text
GMEM A/B
  │  128-bit per-thread cp.async
  ▼
double-buffered SMEM A/B
  │  warp ldmatrix.x4
  ▼
RMEM A/B fragments
  │  mma.sync.m16n8k16
  ▼
FP32 RMEM accumulator
  │  direct teaching epilogue
  ▼
GMEM C
```

本章把第 13 章的两个数据移动原语和第 17 章的 TiledMMA 接成一个完整 K-loop。要回答：

1. `MmaF16BF16Op` 的固定 layout/dtype contract 是什么？
2. 8 个 warp atoms 怎样覆盖 `32×32` output tile？
3. 128-bit `cp.async` TiledCopy 与 16-byte alignment 怎样匹配？
4. double-buffer stage 何时可以覆盖、为什么需要 CTA barrier？
5. `ldmatrix` copy view 怎样连接 SMEM partition 和 MMA fragment？
6. 为什么教学 epilogue 正确但不代表 production coalescing？
7. residue predicate、swizzle、register pipeline 还缺哪些步骤？
8. TensorOp GEMM 与 SIMT SGEMM 应怎样公平比较？

## 18.2 `MmaF16BF16Op` 的硬约束

CuTe DSL 4.7.0 的 SM80 warp op 支持：

| A/B | Accumulator | Instruction shape |
|---|---|---|
| FP16 | FP16 或 FP32 | `(16,8,8)`、`(16,8,16)` |
| BF16 | FP32 | `(16,8,8)`、`(16,8,16)` |

本章固定：

```python
mma_op = cute.nvgpu.warp.MmaF16BF16Op(
    AB_DTYPE,
    cutlass.Float32,
    (16, 8, 16),
)
```

对应 PTX family：

```text
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
```

A layout 固定 row/K-major，B layout 固定 col/K-major；op 不提供任意 transpose switch。CuTe operand B 用 `(N,K)` view，连续 K 对应 PTX 的 col-major operand interpretation。

## 18.3 CTA tiling：8 个 warps覆盖 `32×32`

本章配置：

```text
instruction MNK = (16, 8, 16)
atom layout MNK = ( 2, 4,  1)
TiledMMA footprint = (32, 32, 16)
threads = 2*4*1*32 = 256
```

warp grid 可画成：

```text
             N
       0       8      16      24      32
M  0  warp0   warp2   warp4   warp6
  16  warp1   warp3   warp5   warp7
  32
```

编号由 `Thr Layout VMNK (32,2,4,1):(1,32,64,0)` 推出。每 warp 持有一个 `16×8` C fragment；八个 fragments 无重叠拼成 `32×32`。

同一 A `16×K` strip 被同一 M row 的四个 N-warps复用；同一 B `8×K` strip 被同一 N column 的两个 M-warps复用。这正是第 17 章实测的 A multiplicity=4、B multiplicity=2。

## 18.4 为什么 CTA K tile 选 64

教学 kernel 的 CTA tile 是：

```text
(BM,BN,BK) = (32,32,64)
```

选择 BK=64 有两个直接原因：

1. 每个 K tile 包含 `64/16=4` 个 MMA K-block，能显式展示 repeat mode；
2. A/B 每个 tile 都有 `32*64=2048` 个 b16 values，平均到 256 threads 正好每线程 8 values = 16 bytes，匹配 CuTe `CopyG2SOp` 的 128-bit copy requirement。

这是为了让映射整齐，不是宣称 BK=64 对所有 SM80 shape 最优。production tile 要综合 SMEM、registers、occupancy、latency 与 problem shape autotune。

## 18.5 GMEM → SMEM 的 TiledCopy

copy atom：

```python
cp_atom = cute.make_copy_atom(
    cute.nvgpu.cpasync.CopyG2SOp(
        cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL
    ),
    AB_DTYPE,
    num_bits_per_copy=128,
)
```

thread/value layout：

```python
copy_thr = cute.make_layout((32, 8), stride=(8, 1))
copy_val = cute.make_layout((1, 8))
tiled_copy = cute.make_tiled_copy_tv(cp_atom, copy_thr, copy_val)
```

解释：

```text
thread coord: (row, K-vector), 32*8 = 256 threads
value coord:  (1 row, 8 contiguous b16 values)
coverage:     32 * (8*8) = 32*64 elements
transaction:  8*16 bits = 128 bits
```

因此每线程各向 A/B stage 发一条 16-byte `cp.async`。fake tensor signature 声明 16-byte alignment，SMEM allocation 也使用 `byte_alignment=16`；PTX 实测：

```text
cp.async.cg.shared.global [...], [...], 16, 16;
```

## 18.6 为什么使用 `.cg`

`LoadCacheMode.GLOBAL` 对应只在 L2/global cache 层保证 caching 的 `.cg` 路线，常用于流式读的大 tile，避免无谓占用 L1。是否选择 `.ca`/`.cg` 取决于 reuse、架构和其他 traffic；教程只记录当前 lowering，不把 cache modifier 当普适答案。

`cp.async` 的价值是：

- global load 与 shared store 合成一个 async operation；
- 不需要用普通 registers 暂存搬运值；
- group commit/wait 可以让后续 compute 与未完成 copy 重叠。

但“用了 `cp.async`”并不自动等于发生了有效 overlap，仍要看 stage schedule 和 profiler timeline。

## 18.7 两个 SMEM stages 的物理布局

配套代码分配：

```python
s_a: (32,64,2), stride=(64,1,2048)
s_b: (32,64,2), stride=(64,1,2048)
```

每 operand 每 stage：

```text
32*64*2 bytes = 4096 bytes
```

A/B × 2 stages 合计 16 KiB。stage mode 是最外层的独立连续区域，`read_stage` 在 0/1 间循环。

本章有意使用普通 row-major SMEM，便于手算地址。它满足 correctness 与 `ldmatrix` alignment，但没有使用 tuned Ampere kernel 的 swizzle layout，因此可能有 bank conflicts，不能把本章 benchmark 当性能上限。

## 18.8 Double-buffer prologue

prologue 发出：

```text
group 0: tile 0 -> stage 0
group 1: tile 1 -> stage 1（若存在）
```

每个 tile 的 A/B copies 放在同一 committed group：

```python
cute.copy(... A ...)
cute.copy(... B ...)
cute.arch.cp_async_commit_group()
```

`commit_group` 结束当前 thread 的 async copy group。它不是 CTA barrier，也不表示数据已经能被 `ldmatrix` 读取。

## 18.9 Steady-state stage/phase 表

以三个 K tiles 为例：

| Loop | read stage | wait 后消费 | compute 后覆盖 | outstanding future |
|---:|---:|---|---|---|
| prologue | - | - | tile0→s0, tile1→s1 | tile0/tile1 |
| 0 | 0 | tile0 | tile2→s0 | tile1/tile2 |
| 1 | 1 | tile1 | 无 | tile2 |
| 2 | 0 | tile2 | 无 | 无 |

核心顺序：

```python
cp_async_wait_group(1 or 0)
sync_threads()               # publish current stage to all warps
ldmatrix + mma               # all four K-blocks
sync_threads()               # all readers finished
cp.async tile[k+2] -> consumed stage
commit_group()
```

第一个 CTA barrier 是 producer-completion publication；第二个是 stage-lifetime protection。省略第二个 barrier，快 warp 可能覆盖 stage，而慢 warp 仍在 `ldmatrix` 读取。

## 18.10 `wait_group(1)` 的准确含义

`cp_async_wait_group(1)` 等待到当前 thread 至多还有一个更年轻 committed group 未完成。它不是“等待 group id 1”，group 也没有供用户永久引用的编号。

双缓冲 steady state 中，允许 next stage 保持 outstanding，同时保证 current oldest stage 已完成。最后一个 tile 使用 `wait_group(0)` drain 所有 pending groups。

因为 completion tracking 是 per-thread，而 SMEM 会被全 CTA warps消费，wait 后还必须有 CTA barrier。

## 18.11 SMEM → RMEM：partition、fragment、retile

MMA ownership：

```python
thr_mma = tiled_mma.get_slice(tid)
t_cs_a = thr_mma.partition_A(s_a)
t_cs_b = thr_mma.partition_B(s_b)
t_cg_c = thr_mma.partition_C(g_c)

r_a = tiled_mma.make_fragment_A(t_cs_a[..., 0])
r_b = tiled_mma.make_fragment_B(t_cs_b[..., 0])
r_c = tiled_mma.make_fragment_C(t_cg_c)
```

matrix copy ownership：

```python
ld_atom = cute.make_copy_atom(
    cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), AB_DTYPE
)
copy_s2r_a = cute.make_tiled_copy_A(ld_atom, tiled_mma)
thr_s2r_a = copy_s2r_a.get_slice(tid)
t_cs_a_copy = thr_s2r_a.partition_S(s_a)
t_cr_a_copy = thr_s2r_a.retile(r_a)
```

前一组 views服务 MMA，后一组 views服务 `ldmatrix`。`retile` 没有复制数据，只是把同一 RMEM fragment 表达成 copy atom期望的形状。

## 18.12 K repeat 与四次 MMA

TiledMMA 单次 K footprint 是 16，SMEM stage K=64，所以：

```text
num_k_block = 64/16 = 4
```

每 stage：

```python
for k_block in range(4):
    ldmatrix A[k_block] -> r_a[k_block]
    ldmatrix B[k_block] -> r_b[k_block]
    cute.gemm(tiled_mma, r_c, r_a[k_block], r_b[k_block], r_c)
```

四次 `mma.sync` 累加到同一个 FP32 `r_c`。外层 `k_tile` 继续复用同一 accumulator，直到 problem K 完成。

这展示了两级 K 分解：

```text
problem K
  -> CTA K tiles of 64
       -> instruction K-blocks of 16
```

production kernel 还会在 register level prefetch `k_block+1`，使 `ldmatrix` latency 与当前 MMA 交错。本章顺序 load-then-MMA，保留更清晰的因果关系。

## 18.13 Warp-uniform MMA 与 CTA-uniform stage control

`k_tiles`、`k_tile`、`read_stage` 对整 CTA 一致：

- 每个 warp 的 32 lanes converged 执行 `ldmatrix`/MMA；
- 全 256 threads 到达相同 CTA barriers；
- 没有 lane-local predicate 绕过 collective；
- stage index 更新在所有 threads 上一致。

若把 residue branch 放在 `cute.gemm` 外围，必须保证 branch 是 warp-uniform。更稳妥的做法是让 invalid A/B values 置零而 collective participation 不变。

## 18.14 本章的 shape contract

配套代码要求：

```text
M % 32 == 0
N % 32 == 0
K % 64 == 0
```

这个 contract 使：

- grid 没有 M/N residue CTA；
- 每条 128-bit copy 全部 8 values 有效；
- 每 stage 四个 K-block 完整；
- direct C store 不需要 predicate。

这是有意保留的教学边界，不是 `MmaF16BF16Op` 的硬件限制。第 21 章会从这个基线加入 residue/predication。

## 18.15 Production residue predication

上游 Ampere `tensorop_gemm.py` 的做法值得逐步复用：

1. `local_tile` 取得可能不完整的 A/B tiles；
2. 构造同 shape identity Tensor；
3. 用相同 TiledCopy `partition_S` 得到 per-thread coordinates；
4. M/N predicate 存入 RMEM boolean tensor；
5. K residue 通过 coordinate check；
6. SMEM stage 预先清零，invalid copies 不覆盖零；
7. 所有 threads仍 commit/wait/barrier；
8. C epilogue用 identity Tensor predicate M/N stores。

predicate 应与 copy atom granularity 对齐。128-bit vector 只要跨越边界就不能整体发出；可以降级 scalar tail、padding/alignment contract，或构造每 atom 合法的 mask。

## 18.16 为什么 upstream 把 irregular K tile 放在开头

上游实现会计算负 `residual_k`，把第一个 K tile向前偏移，使 residue 在 mainloop 开始处理。这样：

- 第一个 tile显式做 K checks；
- 之后 steady-state tiles 都是 full K tile；
- 热循环不反复判断“是否最后一个 tile”；
- predicate 可在 residue 后恢复为 full copy。

这是一种控制流整形优化，不改变数学覆盖。前提是 poison/negative coordinates 与 zero fill protocol 被完整证明。

## 18.17 SMEM swizzle 与 bank conflict

本章 row-major stride=64 b16，即相邻 row 相差 128 bytes。它对齐良好，但不同 `ldmatrix` source lanes 可能集中到相同 banks。

上游 tuned kernel 使用：

```text
layout atom + tile_to_shape + bit swizzle
```

构造 64×8 或 8×32 等 matrix-copy-friendly atoms。选择 swizzle 时要同时满足：

- GMEM→SMEM vector copy 的连续维；
- `ldmatrix` row addresses 与 transpose flag；
- bank distribution；
- stage stride 和 base alignment；
- A/B input majorness。

逻辑数值正确无法证明 bank-conflict-free。第 6 章的方法和 Nsight Compute bank metrics要同时使用。

## 18.18 Epilogue：本章 direct store 的边界

本章：

```python
store_atom = cute.make_copy_atom(
    cute.nvgpu.CopyUniversalOp(), cutlass.Float32
)
cute.copy(store_atom, r_c, t_cg_c)
```

优点是 output ownership 与 accumulator mapping 一一对应，便于教学和 correctness 验证。缺点是 MMA C layout 不一定让 global stores 形成理想的 128-bit contiguous transactions。

上游 production epilogue 通常：

```text
FP32 rC
  -> convert/fused op
  -> MMA-owned sC view
  -> CTA barrier
  -> store-coalesced TiledCopy view
  -> predicated vector GMEM store
```

SMEM 在此承担 layout conversion。代价是额外 SMEM traffic、barrier 和容量；收益是 coalesced global store 与融合空间。

## 18.19 Accumulation 精度

本章 FP16/BF16 inputs 均用 FP32 accumulator：

```text
products: input quantization + tensor-core multiply behavior
sum:      FP32 accumulation order
output:   FP32, no cast-back
```

与 PyTorch reference 的差异仍可能来自：

- hardware accumulation tree/order；
- FMA contraction；
- reference backend选择；
- 输入已经量化为 FP16/BF16。

实测最大误差约 `1e-5` 到 `5e-5`，验证阈值使用 `rtol=atol=3e-3`，给不同设备/compiler留余量。阈值不是误差越大越好；应记录实际最大误差并对异常跳变做回归分析。

## 18.20 TensorOp GEMM 与 SIMT SGEMM

上游 Ampere 同时提供 `sgemm.py` 与 `tensorop_gemm.py`：

| 维度 | SIMT SGEMM | TensorOp GEMM |
|---|---|---|
| compute atom | thread-level FP32 FMA | warp `mma.sync` |
| A/B fragment load | ordinary/autovec RMEM load | `ldmatrix`/copy atom |
| main input | 通常 FP32 | FP16/BF16/FP8 等 |
| accumulator | FP32 registers | FP16/FP32 registers，按 op |
| layout constraints | 相对宽松 | 指令 ABI/SMEM layout更强 |
| peak throughput | 较低 | 大幅更高，但需 tile/pipeline喂满 |

公平比较必须固定：

- 相同 M/N/K 与 batch；
- 明确输入精度是否相同；
- 相同 reference 与 tolerance policy；
- 相同 warmup、同步和 L2 policy；
- 报告 TFLOP/s、occupancy、DRAM throughput 和 tensor-core utilization；
- 区分 small-shape latency 与 large-shape throughput。

FP32 SIMT 与 FP16 TensorOp 的结果差异包含输入量化，不能只用一个最大误差数字判定算法优劣。

## 18.21 本章示例与上游 production kernel 的差异

| 能力 | 本章 | 上游 `tensorop_gemm.py` |
|---|---|---|
| CTA tile | 固定 `32×32×64` | FP16/BF16 默认 `128×128×32` |
| pipeline | 2-stage SMEM | 4-stage SMEM + register prefetch |
| G2S | 128-bit `cp.async` | 128-bit `cp.async` |
| SMEM layout | plain row-major | swizzled matrix-copy layout |
| residue | shape contract | M/N/K predicates |
| epilogue | direct FP32 store | SMEM transpose + vector store |
| C dtype/fusion | FP32 only | conversion与 epilogue op |
| rasterization | 无 | grid raster/remap |
| performance claim | 无 | benchmark-oriented |

教程 kernel 的价值是每个 state transition 都能在约 300 行内定位。阅读上游时，可逐行把优化归入这张差异表，而不是把完整实现当黑盒。

## 18.22 B200 验证矩阵

CuTe DSL 4.7.0 / 单张 B200：

| dtype | MNK | CTA tile counts `(M,N,K)` | max error |
|---|---|---|---:|
| FP16 | `(32,32,64)` | `(1,1,1)` | `9.537e-06` |
| FP16 | `(64,96,128)` | `(2,3,2)` | `2.289e-05` |
| FP16 | `(96,64,192)` | `(3,2,3)` | `3.815e-05` |
| BF16 | `(32,32,64)` | `(1,1,1)` | `3.815e-06` |
| BF16 | `(64,96,128)` | `(2,3,2)` | `1.144e-05` |
| BF16 | `(96,64,192)` | `(3,2,3)` | `1.907e-05` |

覆盖：

- 单 CTA / 单 K tile；
- 多 CTA grid；
- 2-stage 刚好填满；
- stage 第一次与第二次 reuse；
- FP16/BF16 两条 MMA dtype path。

PTX 证据：

```text
cp.async.cg.shared.global ..., 16, 16
ldmatrix.sync.aligned.m8n8.x4.shared.b16
mma.sync.aligned.m16n8k16.row.col.f32.{f16|bf16}...f32
bar.sync 0
```

验证等级为 L1/L2/L4。B200 支持 SM80 warp MMA 指令不意味着该 kernel 在 B200 上代表 Ampere 性能；L3 性能比较应在目标 SM80 GPU 和统一 benchmark harness 中进行。

## 18.23 常见失败模式

### 失败 1：copy bits 与 tile coverage不匹配

256 threads × 128 bits 必须刚好覆盖每 operand stage 的 4096 bytes。若 BK 改为 32 而仍用 128-bit atom，会让 thread/value domain 超出 tile。

### 失败 2：只 wait，不做 CTA barrier

每 thread知道自己的 copy完成，不代表其他 threads 的 SMEM writes 都已对所有 consuming warps发布。

### 失败 3：覆盖 stage前缺少 reader barrier

producer开始 `cp.async` 写旧 stage时，另一个 warp可能仍在 `ldmatrix`。

### 失败 4：把 B 当 `(K,N)` partition

`partition_B` 期望 logical `(N,K)`。外部 storage convention 必须在 wrapper中转换。

### 失败 5：`ldmatrix` transpose flag 与 SMEM major不匹配

可能编译通过但 register fragment元素顺序错误，最终 GEMM数值错误。

### 失败 6：partial warp 绕过 MMA

`mma.sync` 是 warp collective；mask values，不要 mask participant lanes。

### 失败 7：看到 Tensor Core PTX 就声称高性能

bank conflicts、poor occupancy、uncoalesced epilogue、small tiles 或 pipeline bubbles都可能让 Tensor Core利用率很低。

## 18.24 从本章走向第 21 章

后续渐进 GEMM 会在保持 reference不变的情况下逐项增加：

1. M/N/K residue 与 identity predicates；
2. swizzled SMEM layouts；
3. three/four-stage `cp.async` pipeline；
4. register prefetch pipeline；
5. SMEM-staged vector epilogue；
6. larger CTA tiles / atom layouts；
7. benchmark 与 tile search。

每步都要回答：减少了什么成本，增加了什么 resource/protocol，PTX或 profiler证据是什么。这样优化链才可归因。

## 18.25 本章小结

- `2×4×1` warp atoms 把 `m16n8k16` 扩展成 `32×32×16` TiledMMA；
- BK=64 同时产生四个 instruction K-repeats，并让 256 threads各发一个 128-bit A/B copy；
- two-stage `cp.async` ring 需要 copy-group wait、consumer publication barrier和 reuse barrier三类时序；
- `make_tiled_copy_A/B + retile` 把 `ldmatrix` TV Layout适配到 MMA RMEM fragments；
- direct FP32 epilogue、plain SMEM和整 tile shape contract保证示例清晰，但不是 production性能设计；
- B200数值与 PTX验证确认完整 `cp.async → ldmatrix → mma.sync` 路径。

下一章进入 Hopper WGMMA：公共 TiledMMA/partition骨架仍在，但 participant scope扩展到 warpgroup，A/B可直接由 SMEM descriptor驱动，copy/MMA同步从 warp synchronous变为显式 fence/commit/wait协议。
