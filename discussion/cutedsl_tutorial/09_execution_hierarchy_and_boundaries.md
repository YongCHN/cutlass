# 第 9 章：执行层级、索引与边界处理

标签：`[通用]`
先修：第 4、7、8 章
状态：正文与示例完成；CuTe DSL 4.7.0 / B200 L1/L2 验证通过。

## 9.1 本章要回答的问题

前几章已经能把 coordinate 映射为地址，也能把 Tensor tile 变成 TensorSSA。但真正启动 kernel 后，还缺一层 ownership：**哪个 thread/warp/CTA 负责哪个坐标，问题尺寸不能整除 tile 时谁负责阻止越界？**

本章建立以下链条：

```text
CUDA participant ID
  -> CTA/tile coordinate
  -> thread-local coordinate
  -> identity Tensor 中的 global coordinate
  -> boundary predicate
  -> safe memory access
```

第 10 章会在这条链上加入 value index，让一个 thread 拥有多个值。本章先保持“一线程一个值”，把执行层级和边界证明单独讲清楚。

## 9.2 CUDA grid、CTA 与 thread

CuTe DSL 直接暴露三维 CUDA special registers：

```python
tx, ty, tz = cute.arch.thread_idx()
bx, by, bz = cute.arch.block_idx()

block_x, block_y, block_z = cute.arch.block_dim()
grid_x, grid_y, grid_z = cute.arch.grid_dim()
```

术语对应：

| CUDA/CuTe 名称 | 常见别名 | 作用域 |
|---|---|---|
| thread | lane participant | 单个执行线程 |
| block | CTA | 一组可共享普通 SMEM、可做 CTA barrier 的 thread |
| grid | kernel grid | 本次 launch 的全部 CTA |
| cluster | CTA cluster | SM90+ 显式组合的若干 CTA |

`block_idx()` 返回的是逻辑 CTA 坐标，不表示 CTA 的实际执行先后。GPU scheduler 可以以任意顺序并发或延后 CTA；正确性不能依赖 `block 0` 一定先于 `block 1` 完成。

同样，device `printf` 的输出顺序也不能用来推断执行顺序。本章的层级示例让每个 thread 写入唯一结果行，再由 CPU reference 验证。

## 9.3 三维坐标的标准 flattening

CUDA 的 x mode 最快变化。CTA 内三维 thread 坐标可展平为：

```text
thread_linear = tx
              + ty * blockDim.x
              + tz * blockDim.x * blockDim.y
```

三维 block 坐标：

```text
block_linear = bx
             + by * gridDim.x
             + bz * gridDim.x * gridDim.y
```

若所有 CTA 的 block shape 相同，全局唯一 participant ID 是：

```text
global_linear = block_linear * threads_per_cta + thread_linear
```

这些公式只建立 participant 编号，不自动表示 Tensor 的 logical index。两者何时相同，取决于调用者选择的 Layout。

例如一个 row-major `(M,N)` Tensor 可用：

```text
row = global_linear // N
col = global_linear % N
```

也可让二维 grid 的 `by` 表示 row tile、`bx` 表示 column tile。第二种方式更自然地保留问题 mode，后续 tile/cluster 调度也更容易扩展。

## 9.4 lane、logical warp 与 warpgroup

### Lane

```python
lane = cute.arch.lane_idx()
```

返回当前 thread 在 warp 内的 lane，范围通常为 `[0,31]`。对标准 CUDA flattening：

```text
lane = thread_linear % 32
```

### Logical warp

```python
warp = cute.arch.warp_idx()
```

CuTe DSL 4.7.0 根据 CTA thread coordinates 计算稳定的 logical warp index：

```text
warp = thread_linear // 32
```

不要把它与 PTX `%warpid` 对应的 physical warp slot 混淆。physical slot 是诊断性调度资源标识，warp 被重新调度后可能改变；ownership、数组索引和同步参与者应使用 stable logical warp index。

### Warpgroup

在需要四个连续 warp 协作的代码中，常写：

```text
warpgroup     = warp // 4
warp_in_group = warp % 4
```

这只是逻辑分组公式。计算出相同 `warpgroup` 不会自动建立同步、共享寄存器或 WGMMA 权限。SM90 WGMMA、第 15 章 warp specialization 和具体 barrier 才定义参与协议。

B200 验证示例采用 `(32,8,1)` CTA，共 256 thread、8 个 warp、2 个逻辑 warpgroup，因此能直接观察 `thread_linear=127 -> 128` 处的 warpgroup 边界。

## 9.5 CTA cluster

SM90+ 可以通过 cluster launch 把多个 CTA 组成 cluster，并使用：

```python
cluster_id = cute.arch.cluster_idx()
cluster_shape = cute.arch.cluster_dim()
```

cluster 引入：

- cluster scope barrier；
- distributed shared memory/remote SMEM；
- TMA multicast mask；
- 2CTA MMA 等协作模式。

普通 `grid/block` launch 没有因此自动变成多 CTA 协作。只有 launch 明确设置 cluster shape，并且 kernel 使用配套同步和 memory scope，cluster coordinate 才构成正确性协议。

本章只建立层级名称；第 12、14、20 章分别讨论 cluster barrier、TMA multicast 和 SM100 2CTA MMA。

## 9.6 1D grid 与 grid-stride loop

一线程一个元素的标准 1D 映射：

```python
idx = bidx * block_dim + tidx
if idx < n:
    dst[idx] = src[idx]
```

grid 数：

```python
num_blocks = cute.ceil_div(n, threads_per_block)
```

若 grid 有意小于全部工作量，可用 grid-stride loop：

```text
idx    = blockIdx.x * blockDim.x + threadIdx.x
stride = gridDim.x * blockDim.x

while idx < n:
    process(idx)
    idx += stride
```

grid-stride 的优点包括限制 resident/launch CTA 数和复用 thread；代价是每个 thread 有循环状态，而且访问次数可能不均匀。

动态 `n` 下，循环变量和 predicate 都属于 GPU 运行期值。第 3 章的 loop-carried type/profile 约束仍然适用。

## 9.7 2D grid 与 tile coordinate

本章 masked vector add 使用 row-major `(M,N)` Tensor：

```text
tile shape = (1,128)
grid.x     = ceil_div(N,128)
grid.y     = M
block.x    = 128
```

映射：

```python
block_x, block_y, _ = cute.arch.block_idx()
tile_coord = (block_y, block_x)
local_coord = (0, tidx)
```

这里故意让 CUDA x dimension 对应连续 column tile，y dimension 对应 row：

```text
global row = block_y
global col = block_x * 128 + tidx
```

相邻 lane 因而访问相邻 row-major FP32 元素。注意，这是我们赋予 grid modes 的业务语义；CUDA 不知道 `x` 是 N、`y` 是 M。

对 GEMM，常见约定又会变成：

```text
blockIdx.x -> N tile
blockIdx.y -> M tile
blockIdx.z -> batch/L
```

每个 kernel 都应在 launch 附近明确记录这种映射。

## 9.8 Residue tile 为什么存在

若 `N=129`、tile N 为 128：

```text
tile 0 covers columns [0,127]
tile 1 logically covers [128,255]
```

第二个 tile 仍有静态 shape `(1,128)`，但只有 column 128 属于真实 problem domain。其余 127 个坐标是 residue tile 中的越界逻辑位置。

保持 tile shape 静态很重要：

- per-thread fragment 大小固定；
- TiledCopy/MMA 类型固定；
- loop 可以展开；
- 同一 kernel 处理 interior 和 boundary tile。

代价是每次 memory access 前必须有合法 predicate，或由更高层 contract 保证该 kernel 只处理整 tile。

## 9.9 Identity Tensor 是 canonical predicate 来源

第 7 章已经构造过 identity Tensor：

```python
identity = cute.make_identity_tensor(problem_shape)
```

关键做法是对 data 和 coordinate Tensor 执行完全相同的 tiling：

```python
a_tile = cute.local_tile(a, tile_shape, tile_coord)
coord_tile = cute.local_tile(identity, tile_shape, tile_coord)
```

同一个 `local_coord`：

```text
a_tile[local_coord]     -> memory value
coord_tile[local_coord] -> original global coordinate
```

边界判断：

```python
global_coord = coord_tile[local_coord]
if cute.elem_less(global_coord, problem_shape):
    ...
```

对二维 coordinate，`elem_less((row,col),(M,N))` 表示各个 mode 都满足边界：

```text
row < M and col < N
```

相比手写 `block_x*tile_n+tidx`，coordinate Tensor 的优势是它能跟随 `zipped_divide`、composition、TiledCopy partition 等复杂 Layout 变换。第 10 章会对 coordinate Tensor 做和数据 Tensor 相同的 `partition_S`。

## 9.10 Predicate 的作用范围

一个 predicate 必须保护所有可能越界的操作：

```python
if pred:
    c[...] = a[...] + b[...]
```

这里 pred 同时保护两个 load 和一个 store。

如果先无条件执行：

```python
value = a[coord]
```

再只给 store 加 predicate，load 已经越界，程序仍然错误。

同理，边界外的 register value 是否为 0、NaN 或未定义，取决于 copy API 的 predicated-load 语义。不能默认“被 mask 的 load 自动得到 0”。若后续会无条件参与 reduction，必须显式建立 neutral value。

## 9.11 Branch、mask 与 divergence

本章一线程一值，使用动态：

```python
if pred:
    load/compute/store
```

尾 warp 中部分 lane 为 true、部分为 false，会发生控制流 divergence。但只有 boundary CTA 受影响，通常是合理的正确性基线。

进入 vectorized TiledCopy 后，一个 thread 拥有多个 predicate value。届时有三种可能：

1. 整个 vector 都合法：发射宽 load/store；
2. 整个 vector 都非法：完全跳过；
3. vector 内部分 lane 合法：scalarize、使用 masked instruction，或拆分 interior/tail kernel。

所以 predicate 与 vectorization 必须共同设计，不能先假设 128-bit copy，再用任意逐元素 mask 保证硬件访问安全。

## 9.12 `assume` 是证明，不是检查

```python
n8 = cute.assume(n, divby=8)
```

这向编译器声明运行期 `n` 一定能被 8 整除。它可以帮助：

- 消除 remainder 路径；
- 证明 tile/divide 合法；
- 推导 alignment/vector width；
- 简化整数运算。

对静态 Python integer，CuTe 可以立即检查：

```python
cute.assume(10, divby=8)  # 构造阶段报错
```

对动态值，它通常不是 kernel 内的 runtime assertion。如果调用者传入 10，却承诺 divby 8，后果可能是错误优化或越界，而不是友好异常。

因此应在 ABI/普通 Python 层先检查：

```python
if n % 8 != 0:
    raise ValueError(...)
```

再把受约束值交给 specialization。或者保留一个通用 masked fallback。

## 9.13 Shape divisibility 与 pointer alignment 是两种契约

以下两句解决不同问题：

```python
n = cute.assume(n, divby=4)
tensor = from_dlpack(x, assumed_align=16)
```

- shape divisible by 4：工作量可按 4 元素分组；
- pointer 16-byte aligned：第一个 vector 起点满足地址对齐。

二者都成立，也不保证任意 slice 后仍然对齐。例如从 FP32 element 1 开始的 view，base pointer 只前进 4 byte，不能再声称 16-byte aligned。

向量化还要求 Layout 的 contiguous mode、每线程起点和访问宽度相容。第 10 章会把这些条件统一到 TV Layout 和 copy atom。

## 9.14 正确性边界与性能友好边界

必须区分：

### 正确性边界

- 每个实际元素恰好被处理；
- 所有 load/store 均在 allocation 内；
- 没有数据竞争；
- 结果与 reference 一致。

### 性能友好边界

- shape 能整除 tile/vector width；
- pointer 和 tile origin 对齐；
- warp 内访问 coalesced；
- predicate 不迫使宽操作 scalarize；
- grid 有足够并行度且不过度浪费 CTA。

本章的 `(1,128)` scalar kernel 同时保证正确和 coalesced，但每 thread 只处理一个 FP32，尚未证明指令层向量化。第 10 章才会验证 128-bit ownership 与生成代码。

## 9.15 动态 shape 与同一 compiled handle

示例把 row-major mode 1 保持为 unit stride：

```python
from_dlpack(tensor).mark_layout_dynamic(leading_dim=1)
```

于是同一个 compiled handle 可以处理：

```text
(1,1)
(3,127)
(3,128)
(5,129)
(7,1003)
```

动态的是 shape/leading dimension leaf；rank 2、dtype FP32、row-major leading mode 和 address-space profile 保持兼容。

每次 launch 的 grid 由动态 `M/N` 重新计算。compiled handle 复用不表示 grid 固定，也不表示所有外部 Tensor layout 都兼容。

## 9.16 可执行示例一：执行层级映射

代码：[execution_hierarchy.py](../code/09_execution_hierarchy/execution_hierarchy.py)

运行：

```bash
cd /volume/njiang/workspace/sandbox/cutedsl_tutorial/code/09_execution_hierarchy
CUDA_VISIBLE_DEVICES=0 /volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python execution_hierarchy.py
```

配置：

```text
grid  = (2,2,1)
block = (32,8,1)
```

每个 thread 写出：

```text
tx, ty, bx, by, lane, warp, warpgroup,
thread_linear, block_linear, global_linear
```

B200 实测边界行：

```text
row 31:  [31,0,0,0,31,0,0,31,0,31]
row 32:  [0,1,0,0,0,1,0,32,0,32]
row 127: [31,3,0,0,31,3,0,127,0,127]
row 128: [0,4,0,0,0,4,1,128,0,128]
row 256: [0,0,1,0,0,0,0,0,1,256]
PASS
```

全部 1024 行与独立 CPU reference 精确一致，L1 通过。

## 9.17 可执行示例二：identity Tensor 尾 tile

代码：[vector_add_masked.py](../code/09_execution_hierarchy/vector_add_masked.py)

运行：

```bash
CUDA_VISIBLE_DEVICES=0 /volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python vector_add_masked.py
```

示例先把输出填为 NaN；任何漏写的合法元素都会让 reference 检查失败。B200 实测：

```text
shape=(1,1), grid=(1,1), tail_columns=1, max_abs_error=0.0e+00
shape=(3,127), grid=(1,3), tail_columns=127, max_abs_error=0.0e+00
shape=(3,128), grid=(1,3), tail_columns=0, max_abs_error=0.0e+00
shape=(5,129), grid=(2,5), tail_columns=1, max_abs_error=0.0e+00
shape=(7,1003), grid=(8,7), tail_columns=107, max_abs_error=0.0e+00
PASS
```

五种 shape 共用一次编译；完整 tile、近乎完整 residue、仅一列 residue 和多 column tile 均与 PyTorch 精确一致，记为 L2 边界验证。完整记录见 [B200 验证日志](validation_log.md)。

## 9.18 常见误区

### 把 `block_idx` 当作执行顺序

它是逻辑坐标。CTA 调度顺序不确定。

### 用 physical warp ID 做 ownership

使用 stable `warp_idx()`；physical slot 只适合诊断。

### 计算了 warpgroup 就认为四个 warp 已同步

分组公式不包含 barrier、fence 或指令参与协议。

### 只保护 store、不保护 load

所有潜在越界 memory operation 都必须在 predicate 内。

### 认为 residue tile 的 shape 会自动缩小

高性能 tile 通常保持静态完整 shape，合法 domain 由 coordinate predicate 描述。

### 把 `assume` 当作 runtime assert

动态 false assumption 违反编译契约；应在调用边界检查或提供 fallback。

### shape 可整除就认为 pointer 对齐

divisibility、base alignment、slice origin 和 Layout contiguous mode 必须分别证明。

## 9.19 本章检查清单与练习

为一个普通 tiled kernel 建立正确性证明：

1. 写出 grid/block 的每个 mode 语义。
2. 写出 CTA/thread flattening 公式。
3. 标出 lane、warp、warpgroup 的参与范围。
4. 写出 CTA tile coordinate 和 thread-local coordinate。
5. 让 data/identity Tensor 经历同一组变换。
6. 列出 predicate 保护的所有 load/store。
7. 分开记录 divisibility 与 alignment contract。
8. 用完整 tile、差 1、余 1 和随机 residue 验证。

练习：

1. 把层级示例的 block 改成 `(16,16,1)`，重新推导 lane/warp 边界。
2. 为三维 grid 写出 batch + row + column tile 映射。
3. 把 masked add 的 tile 改成 `(4,32)`，观察 adjacent lane 对 row-major 地址的访问顺序。
4. 写一个 grid-stride add，用较小 grid 覆盖 1,000,003 个元素。
5. 为 `N % 4 == 0` 建立 Python fast-path 检查，再安全调用带 `assume(n,divby=4)` 的 specialization。
6. 故意只 predicate store，用 compute-sanitizer 观察越界 load；不要在未隔离的生产进程中运行错误版本。

下一章将在同一个 CTA tile 内引入 value index，用 Thread-Value Layout 精确描述每个 thread 拥有的多个元素，并通过 TiledCopy 完成 GMEM ↔ RMEM 的向量化复制。
