# 第 10 章：TiledCopy 与 Thread-Value Layout

标签：`[通用/SM80+]`
先修：第 5、7–9 章
状态：正文与示例完成；CuTe DSL 4.7.0 / B200 L0/L1/L2/L4 验证通过。

## 10.1 本章要回答的问题

第 9 章让一个 thread 负责一个值。高带宽 kernel 通常需要每个 thread 一次拥有多个值，并让整个 CTA 恰好覆盖一个 tile。核心问题变成：**谁复制哪些值，这个 ownership 如何从 Layout 中推导，而不是散落在手写除法和取模中？**

CuTe 的 canonical chain 是：

```text
CopyOp
  -> CopyAtom
  -> TiledCopy
  -> ThrCopy
  -> partition_S / partition_D
  -> per-thread Tensor views
  -> RMEM fragments
  -> cute.copy
```

Thread-Value Layout，简称 TV Layout，把 ownership 表达成一个函数：

```text
(thread index, value index) -> logical coordinate within a tile
```

一旦这个函数正确，同一 ownership 可以同时应用于 data Tensor、destination Tensor 和 identity Tensor。

## 10.2 为什么手写 `thread * V + value` 不够

一维连续数组可以写：

```text
index = thread_id * values_per_thread + value_id
```

但真实 kernel 很快会出现：

- 二维/三维 tile；
- hierarchical value modes；
- thread 在 M/N 上的二维排列；
- source 与 destination 的不同 Layout；
- SMEM swizzle；
- ldmatrix/MMA 固定 ownership；
- 一个 copy atom 一次处理多个元素；
- tile repeat/stage modes。

继续手写 `%` 和 `//` 会把硬件参与者、逻辑坐标和物理 offset 混在一起。TV Layout 则把它们拆成：

```text
(T,V) --TV Layout--> tile logical coordinate
                     --Tensor Layout--> memory offset
```

这正是第 5 章 composition 的实际用途。

## 10.3 Thread Layout 与 Value Layout 的方向

`make_layout_tv` 接受两个 compact Layout：

```python
thr_layout = cute.make_layout((2,3), stride=(3,1))
val_layout = cute.make_layout((2,2), stride=(2,1))
tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
```

必须注意输入与输出方向。

### Thread Layout

```text
tile thread-coordinate -> thread ID
```

`thr_layout=(2,3):(3,1)` 表示一个 `(2,3)` thread grid，mode 1 快速变化：

```text
row 0: T0 T1 T2
row 1: T3 T4 T5
```

它不是 `thread ID -> coordinate`。可视化通常画的是 inverse mapping，因此容易把方向看反。

### Value Layout

```text
per-thread value-coordinate -> value ID
```

`val_layout=(2,2):(2,1)` 表示每线程四个 value：

```text
V0 V1
V2 V3
```

### TV Layout

CuTe 先用 raked product 组合 thread/value replication，再通过 right inverse 得到：

```text
(thread ID, value ID) -> tile logical index/coordinate
```

这是 kernel partition 真正消费的方向。

## 10.4 `make_layout_tv` 的输出

小例子得到：

```text
tiler_mn = (4,6)
tv_layout = ((3,2),(2,2)):((8,2),(4,1))
```

domain profile：

```text
((thread submodes), (value submodes))
```

外部仍可用扁平 `(tid,vid)` coordinate 调用：

```python
logical_index = tv_layout((tid, vid))
coordinate = cute.idx2crd(logical_index, tiler_mn)
```

示例 ownership：

```text
(T0,V0) -> (0,0)
(T0,V1) -> (0,1)
(T0,V2) -> (1,0)
(T0,V3) -> (1,1)

(T1,V0) -> (0,2)
...

(T5,V3) -> (3,5)
```

六个 thread × 四个 value 恰好覆盖 24 个 coordinate。

`tiler_mn` 是 TV Layout 覆盖的 logical tile shape，不是 thread block shape：

```text
CTA threads = size(thr_layout) = 6
values/thread = size(val_layout) = 4
tile values = size(tiler_mn) = 24
```

## 10.5 Coverage 证明

一个正确的 TV ownership 至少要证明：

### 范围合法

```text
0 <= tv_layout(tid,vid) < size(tiler_mn)
```

### Injective

不同 `(tid,vid)` 不映射到同一个 tile coordinate。

### Size 相等

```text
thread_count * value_count = tile_size
```

在有限集合上，范围合法、injective 和 size 相等共同证明 exact coverage：无洞、无重叠。

如果设计本来包含 broadcast/reduction ownership，injective 不一定是目标；但普通 copy 必须明确谁负责重复位置，以及并发写是否安全。

## 10.6 CopyOp：一条复制操作的能力

CopyOp 表达最底层复制类别，例如：

```python
cute.nvgpu.CopyUniversalOp()
```

Universal copy 类似普通赋值，不附带特殊 memory order/cache 属性。后续还会遇到：

- GMEM → RMEM 专用 copy；
- RMEM → GMEM；
- SMEM ↔ RMEM；
- `cp.async`；
- `ldmatrix/stmatrix`；
- TMA；
- TMEM load/store。

CopyOp 本身还没有绑定 dtype 对应的 value layout，也没有扩展到整个 CTA tile。

## 10.7 CopyAtom：最小不可再分的 copy

构造：

```python
copy_atom = cute.make_copy_atom(
    cute.nvgpu.CopyUniversalOp(),
    cutlass.Float32,
    num_bits_per_copy=128,
)
```

CopyAtom 绑定：

- CopyOp；
- internal element type；
- atom source/destination TV Layout；
- 一次 atom execution 的 bit width。

FP32 的 128-bit atom 对应四个 32-bit element：

```text
ATOM_V = 128 / 32 = 4 values
```

`num_bits_per_copy` 省略或为 0 时，Universal copy 允许编译器在可证明安全的范围内自动选择宽度。这不是“无条件 128-bit”。最终宽度仍受 Layout、alignment、predicate 和后端 lowering 影响。

CopyAtom 是 instruction-level contract。错误地声明 128 bit，但 source/destination 起点没有 16-byte 对齐，属于调用者违反契约。

## 10.8 TiledCopy：把 Atom 铺到 CTA tile

已知 thread/value layouts 时：

```python
tiled_copy = cute.make_tiled_copy_tv(
    copy_atom,
    thr_layout,
    val_layout,
)
```

它内部：

1. 调用 `make_layout_tv` 推导 TV Layout；
2. 将 atom value mode 与 TV ownership 对齐；
3. 建立覆盖整个 `tiler_mn` 的 TiledCopy 类型。

重要属性：

```python
tiled_copy.tiler_mn
tiled_copy.layout_tv_tiled
tiled_copy.layout_src_tv_tiled
tiled_copy.layout_dst_tv_tiled
tiled_copy.size
```

source/destination tiled layouts 可能不同。Universal copy 的两边通常相似，但 `ldmatrix`、transpose copy、TMA 等不能假设 `partition_S == partition_D`。

## 10.9 `ThrCopy`：选择当前 thread

kernel 中：

```python
tidx, _, _ = cute.arch.thread_idx()
thr_copy = tiled_copy.get_slice(tidx)
```

`ThrCopy` 不是执行 copy 的命令；它是 TiledCopy 在当前 logical thread 上的 ownership slice。

随后分别 partition：

```python
thread_src = thr_copy.partition_S(block_src)
thread_dst = thr_copy.partition_D(block_dst)
```

得到：

```text
value/repeat coordinates -> source/destination address
```

如果错误地把 destination 也一律 `partition_S`，Universal copy 可能偶然工作，但在 source/destination atom layouts 不同的操作上会失败。教程代码坚持按角色使用 `partition_S` 和 `partition_D`。

## 10.10 为什么 partition shape 看起来比 value count 复杂

小例子中每线程有四个 value，但实测：

```text
partition_shape=((1,(2,2)),1,1)
```

原因是 partition 结果不仅保留用户的 value profile，还保留 CopyAtom value mode 与可能的 repeat/rest modes。概念上通常标为：

```text
(AtomV, RestV..., RestTile...)
```

`cute.size(thread_src)==4`，但 rank/depth 不应被粗暴压成 `(4,)`。后续 copy algorithm 根据 mode 0 是否能被 atom 消费，递归遍历其余 modes。

调试时同时打印：

```python
thread_src.shape
thread_src.layout
cute.size(thread_src)
```

不要只看总元素数猜 mode 语义。

## 10.11 CTA tiling 与 TiledCopy partition 是两层操作

Host JIT 先把完整 Tensor 切成 CTA tiles：

```python
g_a = cute.zipped_divide(a, tiler_mn)
# ((TileM,TileN),(RestM,RestN))
```

kernel 用 block index 选择一个 tile：

```python
block_coord = ((None,None), bidx)
block_a = g_a[block_coord]
# (TileM,TileN)
```

再用 TiledCopy 把 CTA tile 分给 thread：

```python
thread_a = tiled_load.get_slice(tidx).partition_S(block_a)
```

两层 ownership 不应混淆：

```text
grid/CTA ownership: which tile belongs to this CTA?
TV ownership:       which values inside that tile belong to this thread?
```

## 10.12 从 GMEM view 到 RMEM fragment

partition 后的 `thread_a` 仍是指向 GMEM 的 Tensor view。建立 congruent RMEM storage：

```python
fragment_a = cute.make_fragment_like(thread_a)
```

复制：

```python
cute.copy(load_atom, thread_a, fragment_a)
```

计算进入第 8 章 TensorSSA 模型：

```python
result = fragment_a.load() + fragment_b.load()
fragment_c.store(result)
```

写回：

```python
cute.copy(store_atom, fragment_c, thread_c)
```

完整数据路径：

```text
GMEM Tensor
  -> CTA tile
  -> ThrCopy source partition
  -> CopyAtom load
  -> RMEM Tensor
  -> TensorSSA
  -> RMEM Tensor
  -> CopyAtom store
  -> ThrCopy destination partition
  -> GMEM Tensor
```

## 10.13 `cute.copy` 为什么接收 Atom 而不是 TiledCopy

TiledCopy/ThrCopy 已经完成 ownership partition。此时 source/destination Tensor 的 mode 0 与 CopyAtom 对齐：

```python
cute.copy(copy_atom, thread_src, thread_dst)
```

copy algorithm：

- 用 atom 消费最内层 V mode；
- 对 repeat/rest modes 生成或展开循环；
- 检查 source/destination 相应 mode size；
- 可接收与 partition profile 相容的 predicate；
- 对需要单线程 election 的 atom 自动处理 election。

所以：

```text
TiledCopy answers ownership
CopyAtom answers one copy action
cute.copy walks the partitioned tensors
```

## 10.14 一个可手算的 TV Layout

向量化示例采用：

```text
threads = 128
values/thread = 4 FP32
tiler = (1,512)
tv_layout = (128,4):(4,1)
```

函数：

```text
logical_index(tid,vid) = tid*4 + vid
```

因此：

```text
T0 owns 0,1,2,3
T1 owns 4,5,6,7
...
T127 owns 508,509,510,511
```

每线程四个相邻 FP32 正好 16 byte；相邻 thread 的 vector 起点也相邻。一个 warp 覆盖连续：

```text
32 threads * 16 bytes = 512 bytes
```

这同时建立 per-thread vectorization 和 warp-level coalescing 的必要 Layout 条件。

## 10.15 Vector width 的完整证明

源码写 `num_bits_per_copy=128` 还不够。至少检查：

1. dtype：FP32 × 4 = 128 bit；
2. value ownership：每 thread 的四个 value 在 memory 中连续；
3. base pointer：至少 16-byte aligned；
4. row leading dimension：每个 row origin 保持 16-byte aligned；
5. CTA tile origin：`block_n * 512 * 4 bytes` 对齐；
6. source/destination 都满足相同宽度；
7. 没有部分有效的 atom 访问；
8. 最终 PTX/SASS 确实生成宽操作。

完整路径使用 shape `(4,1024)`：

- `1024 % 512 == 0`，无 residue CTA；
- row stride 为 4096 byte；
- 每个 CTA N tile 起点相隔 2048 byte；
- PyTorch allocation base 至少满足传入的 16-byte contract。

## 10.16 PTX 中 128-bit 的不同拼写

不要把“FP32 四元素”硬编码为只有一种 PTX 文本。128 bit 可被表示成：

```text
ld.global.v4.b32
```

也可表示成：

```text
ld.global.v2.b64
```

两者总 payload 都是 128 bit。B200 / CuTe DSL 4.7.0 实际生成：

```text
ld.global.v2.b64 {%rd8, %rd9}, [%rd5];
ld.global.v2.b64 {%rd10, %rd11}, [%rd6];
st.global.v2.b64 [%rd7], {%rd13, %rd12};
```

因此验证脚本接受 `v4.b32` 或 `v2.b64`，但仍要求两个输入 load 和一个输出 store 都出现 128-bit 证据。

PTX 证明了编译器 IR 到 PTX 的指令形态，记为 L4；它没有证明 memory throughput 已接近峰值。后者需要第 29 章的 benchmark 与 profiler 方法。

## 10.17 Identity Tensor 也要做同一个 partition

通用 residue 路径：

```python
identity = cute.make_identity_tensor(problem_shape)
g_coord = cute.zipped_divide(identity, tiler_mn)
```

kernel 中：

```python
block_coord_tensor = g_coord[block_coord]
thread_coord = thread_store.partition_D(block_coord_tensor)
```

data destination 和 coordinate Tensor 使用同一个 destination ThrCopy：

```text
thread_c[i]     -> destination address
thread_coord[i] -> address对应的 global coordinate
```

生成逐 value predicate：

```python
for i in cutlass.range_constexpr(cute.size(predicate)):
    predicate[i] = cute.elem_less(thread_coord[i], problem_shape)
```

这种写法即使 TV Layout 改成二维 raked ownership，也不需要重写手工 global index 公式。

## 10.18 Predicated copy 的 neutral value

被 predicate 屏蔽的 load lane 不应被假设为自动得到 0。示例先初始化：

```python
fragment_a.fill(0.0)
fragment_b.fill(0.0)
```

再执行：

```python
cute.copy(load_atom, thread_a, fragment_a, pred=predicate)
```

于是 invalid lane 保持 neutral value 0，进入 TensorSSA 加法仍有定义。最终 store 也使用相同 predicate，不会写出 problem domain。

对 max reduction，neutral value 可能是负无穷；对乘法可能是 1。neutral value 由后续数学决定，不总是 0。

## 10.19 Predicate 与 128-bit atom 的冲突

若四元素 atom 中只有一个 lane 合法，硬件不能简单发射一次越界 128-bit load，再说“只使用第一项”。安全策略包括：

1. 让 copy lowering 根据逐 value predicate 选择安全宽度；
2. interior kernel 使用显式宽 atom，tail kernel 使用 scalar/masked copy；
3. allocation 做合法 padding，并证明宽 load 不跨 allocation；
4. 使用目标架构支持的真正 masked vector operation。

本章采用前两种思想：

- `aligned_elementwise`：显式 128-bit，只接受 full tile；
- `masked_elementwise`：相同 TV ownership，但 atom 宽度交给编译器在 predicate 下安全选择。

因此只对 aligned path 声明 L4 128-bit 证据，不声称 residue path 仍使用同样的指令宽度。

## 10.20 `autovec_copy`

当 source/destination 已经是静态 per-thread Tensor，且不需要显式 TiledCopy ownership时，可以使用：

```python
cute.autovec_copy(src, dst)
```

CuTe DSL 4.7.0 的实现会综合：

- source/destination 最大共同 Layout；
- Layout max alignment；
- pointer max alignment；
- dtype bit width；
- 当前实现的最大 copy bit cap；

选择安全 vector width；失败时退回 basic copy。

它解决的是“给定两个已经 partition 好的 Tensor，选多宽的 copy”，而 `TiledCopy` 主要解决“CTA/thread/value ownership 怎么 partition”。两者不是互斥关系：复杂 kernel 常先通过 TiledCopy/MMA partition 得到 per-thread view，再在合适的局部路径使用 auto-vectorization。

## 10.21 Cache 与专用 CopyOp

Universal copy 不表达特殊 cache/memory-order 语义。如果需要：

- load cache mode；
- eviction priority；
- volatile/invariant；
- acquire/release；

应选择对应 GMEM↔RMEM 或 SMEM↔RMEM CopyOp，并传递合法属性。

性能 hint 不能破坏正确性。例如把会被其他 participant 更新的数据声明 invariant，可能让编译器缓存旧值。同步与 memory order 会在第 12 章系统展开。

## 10.22 可执行示例一：ownership 可视化

代码：[tiled_copy_visual.py](../code/10_tiled_copy/tiled_copy_visual.py)

运行：

```bash
cd /volume/njiang/workspace/sandbox/cutedsl_tutorial/code/10_tiled_copy
/volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python tiled_copy_visual.py
```

示例验证：

- `make_layout_tv` 的 tiler/TV Layout；
- `make_tiled_copy_tv` 的 tiler；
- 六个 `ThrCopy.partition_S`；
- 每个 partition coordinate 与 TV Layout 映射一致；
- 24 个 `(thread,value)` 映射 injective 且完整覆盖 `(4,6)`。

B200 环境关键输出：

```text
tiler_mn=(4, 6)
tv_layout=((3,2),(2,2)):((8,2),(4,1))
thread 0: partition_shape=((1,(2,2)),1,1)
  (0,0) -> 0 -> (0,0)
  (0,3) -> 5 -> (1,1)
...
  (5,3) -> 23 -> (3,5)
PASS
```

这是 L0 ownership/coverage 验证。

## 10.23 可执行示例二：完整与 residue elementwise

代码：[vectorized_elementwise.py](../code/10_tiled_copy/vectorized_elementwise.py)

运行：

```bash
CUDA_VISIBLE_DEVICES=0 /volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python vectorized_elementwise.py
```

### Aligned full-tile path

```text
[aligned] tiler_mn=(1,512)
[aligned] tv_layout=(128,4):(4,1)
aligned shape=(4,1024), max_abs_error=0.0e+00,
ptx_vector_loads=2, ptx_vector_stores=1
```

数值 L1 与 PTX L4 均通过。

### Dynamic residue path

同一个 compiled handle 验证：

```text
masked shape=(1,1), tail_values=1, max_abs_error=0.0e+00
masked shape=(3,511), tail_values=511, max_abs_error=0.0e+00
masked shape=(3,512), tail_values=0, max_abs_error=0.0e+00
masked shape=(5,513), tail_values=1, max_abs_error=0.0e+00
masked shape=(7,1003), tail_values=491, max_abs_error=0.0e+00
PASS
```

输出预填 NaN，所有合法值与 PyTorch 精确一致，记为 L2 边界验证。完整命令和 PTX 行见 [B200 验证日志](validation_log.md)。

## 10.24 常见误区

### 把 Thread Layout 当成 `tid -> coordinate`

输入 Thread Layout 的方向是 tile thread-coordinate 到 thread ID；TV Layout 才是 `(tid,vid)` 到 logical coordinate。

### 只比较 `size(thread_partition)`

必须保留 AtomV/RestV/repeat profile。相同总 size 不表示 copy algorithm 看到相同结构。

### source 和 destination 都调用 `partition_S`

按角色使用 `partition_S/partition_D`，不要依赖 Universal copy 的偶然对称性。

### 设置 128-bit 就认为一定安全

还要证明 pointer、row/tile origin、contiguous Layout 和 full atom validity。

### 有 predicate 就认为越界 vector load 安全

predicate 粒度必须与 atom/lowering 匹配。部分有效 vector 需要安全的 tail 策略。

### PTX 有宽 load 就声称带宽最优

指令形态只是 L4 结构证据。occupancy、请求合并、cache、问题规模和时钟仍影响实测性能。

### 把 RMEM fragment 当成 TensorSSA

fragment 是 storage；`fragment.load()` 才产生 TensorSSA。

## 10.25 第一阶段闭环

到本章为止，读者已经能完整解释一个 vectorized elementwise kernel：

```text
Python/JIT signature
  -> dynamic problem Layout
  -> CTA tile via zipped_divide
  -> TV ownership via TiledCopy
  -> per-thread GMEM views
  -> predicated CopyAtom
  -> RMEM fragments
  -> TensorSSA computation
  -> predicated vector/scalar-safe store
  -> PyTorch reference + PTX evidence
```

这完成了教程第一交付阶段“语言模型、Layout、Tensor 和 TiledCopy”。接下来的第 11 章不再改变 ownership 基础，而是把其中一段数据路径扩展为：

```text
GMEM -> SMEM -> RMEM -> SMEM/GMEM
```

并加入 CTA 内协作、shared-memory allocation、bank conflict 和 buffer reuse。

## 10.26 本章检查清单与练习

设计 TiledCopy 时：

1. 分别写出 Thread Layout 和 Value Layout 的方向。
2. 打印 `tiler_mn` 与 TV Layout。
3. 手算至少两个 thread 的全部 value coordinates。
4. 证明 coverage、无重叠或明确重复语义。
5. 区分 CTA tiling 与 thread partition。
6. 对 source/destination 使用正确 partition API。
7. 标注 AtomV、RestV 和 repeat modes。
8. 证明 dtype、vector bits 和 alignment。
9. 为 identity Tensor 使用相同 destination ownership。
10. 单独设计 full-tile 和 partially valid atom 的策略。
11. 用数值 reference 验证正确性，用 PTX/SASS 验证指令形态。

练习：

1. 把小例子的 Thread Layout 改为 `(3,2):(2,1)`，画出新的 ownership。
2. 把 Value Layout 改成每线程 `(1,4)`，比较 tiler 和 partition profile。
3. 将 FP32 aligned path 改为 FP16；保持 128 bit 时每线程应拥有多少 value？
4. 故意把 row stride 改为不能保持 16-byte alignment 的值，解释为何 full path contract 失效。
5. 为 residue path 实现 host dispatch：整 tile 使用 aligned specialization，最后一个 tile 使用 masked specialization。
6. 用 `autovec_copy` 替换局部 Universal CopyAtom，比较生成 PTX。
7. 添加 benchmark，但必须记录 warmup、iteration、同步、总字节数和 B200 环境；不要把首次编译时间计入 kernel 时间。

下一章将进入 shared memory，使用同样的 TV ownership 实现一个经典 tiled transpose/搬运 kernel，并第一次系统处理 CTA barrier 与 SMEM buffer reuse。
