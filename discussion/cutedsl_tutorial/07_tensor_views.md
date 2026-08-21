# 第 7 章：Tensor、切片与坐标 Tensor

标签：`[通用]`
先修：第 4–6 章
状态：正文与示例完成；CuTe DSL 4.7.0 / B200 L1 验证通过。

## 7.1 本章要回答的问题

Layout 只计算 offset，不拥有数据。**Tensor 如何把存储和 Layout 组合起来？**

CuTe 的定义可以浓缩为：

```text
Tensor = Engine o Layout
T(c)   = Engine(Layout(c))
```

其中 Engine 通常是 Pointer/Iterator，也可以是坐标 counting iterator。Layout 决定“给定逻辑坐标，访问哪个 offset”；Engine 决定“这个 offset 如何被解释或解引用”。

这条分解是后续所有 tiled kernel 的主线：

```text
同一块存储 + 不同 Layout = 不同逻辑视图
同一 Layout + 不同 Engine = 相同映射、不同数据来源
```

## 7.2 Memory-backed Tensor 的组成

最直接的构造：

```python
from cutlass.cute.runtime import make_ptr

ptr = make_ptr(
    cutlass.Float32,
    address,
    cute.AddressSpace.gmem,
    assumed_align=16,
)
layout = cute.make_layout((M, N), stride=(ld, 1))
tensor = cute.make_tensor(ptr, layout)
```

访问：

```text
tensor((i,j))
  = load ptr[layout((i,j))]
  = load ptr[i*ld+j]
```

Tensor 不负责分配或拥有指针指向的生命周期。它是一个 typed view：

- element type 来自 Pointer/Iterator；
- shape、stride/profile 来自 Layout；
- address space 和 alignment 信息来自 Engine；
- slicing/tiling 同时变换 iterator 和 Layout。

两个 Tensor 可以别名同一地址。CuTe 不会因为逻辑 view 不同而自动消除写冲突。

## 7.3 Engine 不一定是普通指针

`make_tensor` 可接收：

- GMEM/SMEM/RMEM 等地址空间中的 Pointer；
- 某些硬件 descriptor/iterator；
- 整数或整数 tuple 形成的 coordinate counting iterator。

这也是 identity Tensor 不需要内存的原因。它的“元素”由坐标迭代器计算，而不是从某块 allocation load。

前一章的 ComposedLayout 说明 Layout 侧也不一定只是 affine stride；但 memory op 是否支持该组合，仍由具体 Tensor type 与 lowering 决定。

## 7.4 地址空间不是装饰

常见空间：

| 空间 | 典型 Engine/对象 | 主要特征 |
|---|---|---|
| GMEM | global Pointer/Tensor | 跨 CTA 可见，高延迟，关注 transaction/coalescing |
| SMEM | shared Pointer/Tensor | CTA/cluster 协作，关注同步、bank、容量 |
| RMEM | register-backed Tensor | thread-local，静态容量，常用作 fragment storage |
| TMEM | Blackwell tensor memory view | SM100 Tensor Core 专用，访问与 ownership 有专门规则 |

同一个 shape/layout 不能消除地址空间语义。例如把一个 SMEM pointer 错标为 GMEM 并不是“性能差一点”，而是生成错误的地址空间操作。

RMEM Tensor 与第 8 章的 TensorSSA 也不同：RMEM Tensor 是可 load/store 的寄存器 backing storage；TensorSSA 是一个不可变 SSA value。

## 7.5 三条常用构造路径

### `make_ptr + make_tensor`

调用者已有裸地址，自己声明 dtype、address space、alignment 和 Layout：

```python
from cutlass.cute.runtime import make_ptr

ptr = make_ptr(cutlass.Float32, address, cute.AddressSpace.gmem)
mA = cute.make_tensor(ptr, cute.make_layout((M,N), stride=(ld,1)))
```

优点是 ABI 明确，也能绕开某些外部框架 shape-1 mode 的 stride 规范化。代价是调用者必须正确维护生命周期、dtype、shape、stride 和 stream contract。

### DLPack `from_dlpack`

```python
from cutlass.cute.runtime import from_dlpack
mA = from_dlpack(torch_tensor, assumed_align=16)
```

它从 PyTorch/JAX/NumPy 等 DLPack 对象导入 pointer、dtype、shape 和 stride。教程示例主要使用这条路径。

`assumed_align` 是由调用者提供给编译器的承诺，不是运行期帮你重新对齐。承诺高于真实地址或派生 view 能保证的对齐会导致错误代码生成。

### Fake Tensor / symbolic descriptor

编译时不想分配真实数据，可使用 PyTorch FakeTensor 或 runtime 的 fake tensor helpers 建立 signature，再让匹配的真实对象执行。它适合 AOT/框架集成，但不改变 signature compatibility 规则。

## 7.6 完整索引与 partial evaluation

完整坐标返回一个元素：

```python
value = tensor[(i, j)]
```

整数固定对应 mode，`None` 保留对应 mode：

```python
row    = tensor[(i, None)]
column = tensor[(None, j)]
```

返回的是新 Tensor view：

- iterator 被推进到被固定坐标的起点；
- Layout 只保留 `None` 对应的 mode；
- 原 storage 没有复制。

对 hierarchical Layout，slice coordinate 必须与 profile congruent。不要把 Python/NumPy 的切片直觉无条件套到 CuTe 的 nested mode 上；先打印结果 shape/layout。

## 7.7 `local_tile`：同一操作同时变换指针和 Layout

概念定义：

```text
local_tile(input, tiler, coord)
  = zipped_divide(input, tiler)[(None, coord)]
```

对 shape `(4,6)`、tiler `(2,3)`：

```text
zipped_divide shape = ((2,3),(2,2))
                       Tile   Rest
```

`coord=(block_x,block_y)` 从 Rest 中选一个 tile，返回 shape `(2,3)` 的 view。

示例：

```python
src_tile = cute.local_tile(src, (2,3), (block_x, block_y))
```

与手工写：

```text
base pointer + block_x * tile_stride_x + block_y * tile_stride_y
```

相比，`local_tile` 保留了 Layout profile，后续仍可继续 partition 或 compose。

若使用 `proj`，可从一个更高 rank 的 tiler 中投影出与当前 Tensor 相关的 mode；GEMM 中用同一个 `(M,N,K)` tiler 分别切 A(M,K)、B(N,K)、C(M,N) 时经常采用这一模式。

## 7.8 `domain_offset`

`domain_offset(coord, tensor)` 让 Tensor 的 Engine 前进到 `layout(coord)` 对应的位置，同时保留原 Layout：

```python
shifted = cute.domain_offset((3,5), tensor)
```

可理解为：

```text
shifted.iterator = tensor.iterator + tensor.layout((3,5))
shifted.layout   = tensor.layout
```

它与 slice 的用途不同：slice 通常固定/删除 mode；domain offset 保持同一 domain profile，只改变 origin。构造滑动窗口或处理 tile residue 时很有用。

调用者仍要保证新 origin 后的所有访问在 allocation 内。这个操作只建立 view，不自动加边界 predicate。

## 7.9 Identity/coordinate Tensor：让坐标经历同样的变换

构造：

```python
identity = cute.make_identity_tensor(src.shape)
```

它满足：

```text
identity((i,j)) = (i,j)
```

关键技巧是让 data Tensor 和 coordinate Tensor 经历完全相同的操作：

```python
src_tile   = cute.local_tile(src,      tile_shape, tile_coord)
coord_tile = cute.local_tile(identity, tile_shape, tile_coord)
```

于是同一个 local coordinate `lc`：

```text
src_tile[lc]   -> 对应的数据
coord_tile[lc] -> 数据的原始全局坐标
```

本章示例用全局坐标构造可人工检查的结果：

```python
dst_tile[lc] = src_tile[lc] + row * 10 + col
```

第 9 章会把这个模式改成：

```python
predicate = cute.elem_less(coord_tile[lc], problem_shape)
```

从而为不完整尾 tile 建立安全边界。

coordinate Tensor 携带坐标，不是“预先分配一份坐标矩阵”；其 Engine 是 counting iterator。

## 7.10 Recast：改变元素解释，不改变底层字节

`recast_ptr`/`recast_tensor` 用另一个 dtype 重新解释同一块 storage。典型用途：

- 以 128-bit 宽度组织 vectorized copy；
- 在整数与浮点位表示间做 bit-level 操作；
- 处理 packed/sub-byte 数据。

若从 FP32 recast 为更宽或更窄的 element type，Layout 也必须按 bit width 合法变换。检查：

1. 总 bit 数是否可整除；
2. contiguous mode 是否允许合并/拆分；
3. base 和派生 pointer 是否满足新类型 alignment；
4. slice 后的起点是否仍然对齐；
5. aliasing 的读写顺序是否合法。

Recast 不是数值转换。它不把浮点 1.0 计算成整数 1，而是重新解释 bit pattern。

## 7.11 Static、dynamic 与 compact Layout signature

`from_dlpack(tensor)` 默认把可见 shape/stride 值带入编译 signature。若这些 leaf 是 static，shape 变化会要求不同 specialization。

```python
dynamic = from_dlpack(tensor).mark_layout_dynamic(leading_dim=0)
```

对一维 contiguous Tensor：

- rank=1 和 dtype 固定；
- stride-1 leading mode 保持可证明；
- extent 成为运行期值；
- 同一 compiled handle 可接受多个长度。

本章实测用一次编译运行：

```text
n=17
n=257
n=1003
```

`mark_layout_dynamic` 适用于 flat Layout；`leading_dim` 指定哪个 mode 保持 unit stride，并会检查输入是否真的满足。多维 compact shape 还可用 `mark_compact_shape_dynamic` 表达 shape 与派生 compact stride 的关系，避免把所有 stride 都降为互不相关的动态整数。

动态化不是越多越好：

- static extent 有利于 unroll、vectorization 和更强的整除证明；
- dynamic signature 减少 specialization 数量，却可能保留更多运行期算术；
- rank、dtype、address space、profile 等结构通常仍是类型的一部分。

需要根据部署 shape 分布做取舍，而不是一律动态或一律静态。

## 7.12 DLPack 与框架边界的注意事项

### 生命周期

编译/执行期间，底层 framework Tensor 必须存活。CuTe view 不接管 allocation 所有权。

### Stream

不同框架的 DLPack stream 约定可能不同。示例在同一 PyTorch CUDA context/stream 中运行并在检查前同步；集成异步流水线时应显式传递/协调 stream，而不是靠全局 synchronize 掩盖依赖。

### Shape-1 mode

部分 DLPack 实现会规范化 size-1 mode 的 stride。若内核依靠该 stride 传播 alignment 或布局信息，可改用裸 pointer wrapper 加显式 Layout，或者在 ABI 处单独传入 shape/stride。

### 非连续 view

`from_dlpack` 能携带 stride，不代表任意内核都支持任意非连续 Layout。内核的 vectorization、copy atom 或 MMA path 可能要求 compact/特定 stride。

## 7.13 可执行示例与实测结果

### Tensor view 与坐标 Tensor

代码：[tensor_views.py](../code/07_tensor/tensor_views.py)

运行：

```bash
cd /volume/njiang/workspace/sandbox/cutedsl_tutorial/code/07_tensor
CUDA_VISIBLE_DEVICES=0 /volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python tensor_views.py
```

输入 `(4,6)` 被四个 CTA 分为 `(2,3)` tile。每个 tile 用相同 local coordinate 访问 data 与 identity Tensor。B200 实测输出：

```text
dst=
tensor([[ 0.,  2.,  4.,  6.,  8., 10.],
        [16., 18., 20., 22., 24., 26.],
        [32., 34., 36., 38., 40., 42.],
        [48., 50., 52., 54., 56., 58.]])
PASS
```

### 动态 Layout

代码：[dynamic_layout.py](../code/07_tensor/dynamic_layout.py)

运行：

```bash
CUDA_VISIBLE_DEVICES=0 /volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python dynamic_layout.py
```

一次 `cute.compile` 后复用同一个 handle：

```text
n=17, first=-6.0, last=26.0
n=257, first=-6.0, last=506.0
n=1003, first=-6.0, last=1998.0
PASS
```

三种结果都与 PyTorch `src * 2` 精确一致。这是 L1 数值验证；没有覆盖非连续动态 stride。完整记录见 [B200 验证日志](validation_log.md)。

## 7.14 常见误区

### 把 Tensor 当成 allocation owner

Tensor 是 view。复制 Tensor 对象不复制数据，也不延长外部 allocation 的所有权语义。

### 以为 slice/local_tile 会搬数据

这些操作主要变换 Engine origin 与 Layout；真正的数据搬运由 load/store/copy 指令发生。

### 把 identity Tensor 当成 GMEM 坐标数组

它由 coordinate iterator 计算坐标，不需要单独的数据 allocation。

### 高报 `assumed_align`

它是编译器契约。错误承诺可能让生成的宽访存对未对齐地址执行。

### 认为 dynamic Layout 可改变 rank/profile

动态的是 leaf value。rank、嵌套结构、dtype 等仍须与 compiled signature 兼容。

### 从 DLPack 导入成功就认为任意 stride 都高效

语义可表示、内核可 lower、访存高效是三件事。

## 7.15 本章检查清单与练习

分析一个 Tensor 时：

1. 分开写 Engine 与 Layout。
2. 标明 dtype、address space、alignment 和生命周期 owner。
3. 枚举三组 logical coordinate 到 pointer offset。
4. 对每次 slice/tile 写出新 shape/profile 和 origin。
5. 若需要边界，给 data view 配一个 congruent coordinate view。
6. 标出 signature 中 static/dynamic 的 shape/stride leaf。
7. 检查框架 stream 与非连续 stride contract。

练习：

1. 把 `tensor_views.py` 的 tile 改成 `(1,3)`，写出 grid 和 coordinate 输出。
2. 用 `domain_offset` 建立从 `(1,2)` 开始的 `(4,6)` 同 profile view，并列出安全访问区域。
3. 为 padded `(3,4):(8,1)` 构造裸 pointer Tensor，与 PyTorch `as_strided` reference 对照。
4. 给动态示例增加二维 row-major shape，明确哪个 extent/stride 动态。
5. 构造一个 shape-1 mode，比较 DLPack 导入和显式 `make_ptr + make_tensor` 的 Layout 打印。

下一章会把 memory-backed Tensor 一次性 load 成 thread-local TensorSSA，进入寄存器级 elementwise、broadcast、slice 与 reduction 数据流。
