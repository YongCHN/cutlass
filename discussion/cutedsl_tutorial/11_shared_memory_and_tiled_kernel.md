# 第 11 章：Shared Memory 分配与经典 tiled kernel

> 范围标签：`[通用]`
>
> 前置章节：第 6 章 Swizzle、第 7 章 Tensor、第 9 章边界处理、第 10 章 TiledCopy
>
> 配套代码：[`11_shared_memory/tiled_transpose.py`](../code/11_shared_memory/tiled_transpose.py)

## 11.1 本章要回答的问题

第 10 章已经能够回答“一个 CTA 中的每个线程拥有哪些 GMEM 元素”。本章在这条数据路径中插入一块由整个 CTA 协作使用的 Shared Memory：

```text
GMEM -> SMEM -> RMEM -> GMEM
```

我们要进一步回答：

1. 什么情况下引入 SMEM，而不是直接在 GMEM 和 RMEM 之间移动数据？
2. `SmemAllocator` 分配的是怎样的对象，allocation 的大小何时确定？
3. `@cute.struct`、raw bytes、array 和 Tensor 分配分别适合什么场景？
4. `size(layout)` 与真正占用容量的 `cosize(layout)` 为什么不能混淆？
5. producer 写完 SMEM 后，consumer 为什么必须同步才能读取？
6. 同一块 buffer 被下一轮复用前，为什么通常还需要第二个 barrier？
7. residue tile 中部分线程没有合法 GMEM 元素时，怎样保证同步不会死锁？
8. padding 为什么能改变 bank 映射？它又为什么不能自动构成性能结论？

本章选择 tiled transpose 作为主例。Transpose 不引入 MMA，也不把注意力提前带到 GEMM 数值计算上，却能同时暴露 coalescing、跨线程交换、SMEM bank、边界谓词和 buffer reuse。

## 11.2 SMEM 解决的不是“容量更大”

Shared Memory 的主要价值是 CTA 内线程协作。常见用途包括：

- 一个线程从 GMEM 读取，另一个线程消费；
- 改变数据排列，使下一阶段获得合适的连续访问；
- 合并重复的 GMEM 读取；
- 为 MMA、规约或 epilogue 构造期望的 tile；
- 作为异步 copy 和计算之间的 staging buffer。

因此，引入 SMEM 的核心问题不是“能否把数据放进去”，而是：

```text
谁写？
谁读？
何时可见？
何时允许覆盖？
```

如果每个元素始终由同一个线程读取、计算和写回，像第 10 章的 elementwise add，那么 SMEM 往往只会增加指令和同步开销。

## 11.3 从 private ownership 到 cooperative ownership

第 10 章的路径是：

```text
GMEM tile
  -> TiledCopy partition
  -> per-thread RMEM fragment
  -> TensorSSA
  -> per-thread GMEM destination
```

其中 source 和 destination ownership 可以完全属于同一个线程。Transpose 不同：

- loader thread `(tx, ty)` 写入 `smem[ty+j, tx]`；
- storer thread `(tx, ty)` 读取 `smem[tx, ty+j]`；
- 同一个逻辑元素的 writer 和 reader 通常不是同一个线程。

SMEM 在这里相当于一次 CTA 范围的 ownership exchange。Layout 描述“元素放在哪里”，barrier 描述“ownership 何时可以从 producer 转给 consumer”。

## 11.4 `SmemAllocator` 的基本模型

CuTe DSL 4.7 的典型入口是：

```python
allocator = cutlass.utils.SmemAllocator()
```

allocator 支持四类常用分配：

| API | 返回对象 | 典型用途 |
|---|---|---|
| `allocate(bytes)` | byte pointer | 手工管理的原始区域 |
| `allocate(dtype)` | typed pointer | 标量状态、计数器 |
| `allocate_array(dtype, n)` | typed pointer | 一维数组、barrier 数组 |
| `allocate_tensor(dtype, layout)` | `cute.Tensor` | 按 Layout 访问的 tile |
| `allocate(StructType)` | struct instance | 把多个 buffer、barrier 和状态组织为共享 storage |

allocator 是 bump allocation 模型：每次分配都在前一个区域之后选择满足 alignment 的起始位置。它不会像通用 heap 一样提供运行期 `free`。

这意味着 buffer lifetime 与 alias/reuse 通常由 kernel 协议管理，而不是由 allocator 自动分析。

## 11.5 静态 Layout 与 launch-time dynamic SMEM

“dynamic shared memory”很容易引起误解。这里必须区分两个维度。

### Allocation 结构是静态的

`allocate_tensor` 当前要求 static Layout：

```python
assert cute.is_static(layout)
```

元素类型、Layout、`cosize` 和 alignment 在 specialization 中必须能够确定。不能把任意运行时 `M`、`N` 直接拿来构造一块动态大小的 SMEM Tensor。

### Launch 字节数可以自动推导

当所有 allocation 都通过 `SmemAllocator` 表达时，kernel launch 可以省略 `smem=`：

```python
kernel(...).launch(grid=grid, block=block)
```

编译流程会推导这个 specialization 需要的动态 SMEM 字节数并配置 launch。设备端的：

```python
cute.arch.dynamic_smem_size()
```

返回实际配置给当前 CTA 的 launch-time dynamic SMEM 大小。

所以正确说法是：

> SMEM storage 的类型和 Layout 是静态 specialization；launch 配置中的字节数可以由编译器自动推导。

## 11.6 显式 `smem=` 是契约，不是扩容操作

也可以在 launch 时显式传递 `smem=bytes`。但这不是“需要多少就自动分多少”的运行时 allocator：

- 小于实际 allocation 需求时，kernel 的 storage contract 不成立；
- 大于实际需求时，多出的 launch capacity 不会自动生成新的 Tensor；
- 较大的 per-CTA SMEM 配置还可能减少同时驻留的 CTA 数量。

教程的同步 baseline 优先使用自动推导，并在结果中读取 `dynamic_smem_size()` 验证实际字节数。

## 11.7 `@cute.struct`：组织 storage，而不是改变 lifetime

当 kernel 同时需要多个 tile、barrier 和标量状态时，可以定义：

```python
@cute.struct
class SharedStorage:
    data: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, 1024],
        16,
    ]
    barrier: cute.struct.Align[cutlass.Int64, 8]
```

然后：

```python
storage = allocator.allocate(SharedStorage)
```

`MemRange` 表达固定长度区域，`Align[T, N]` 提高字段 alignment。struct 会计算字段 offset、padding、整体 size 与整体 alignment。

但 struct 只负责物理组织：

- 不会让 producer/consumer 自动同步；
- 不会自动判断两个字段能否 alias；
- 不会自动缩短某个 stage 的 lifetime；
- 不会替代 TiledCopy 的 ownership。

## 11.8 Alignment 的三层含义

讨论 SMEM alignment 时，至少要分清：

1. allocation base alignment；
2. Layout 中某个逻辑坐标对应地址的 alignment；
3. 某条 copy 指令对访问宽度和地址的 alignment 要求。

例如：

```python
smem = allocator.allocate_tensor(
    cutlass.Float32,
    layout,
    byte_alignment=16,
)
```

只保证 Tensor 起始地址至少 16-byte aligned。它不保证每一行开头都是 16-byte aligned，也不保证任意切片仍满足 128-bit copy 的对齐要求。

第 10 章的结论仍然适用：base pointer alignment、row stride、tile origin 和 per-thread address 都必须共同证明。

## 11.9 容量看 `cosize`，不是只看 `size`

对紧凑 Layout：

```text
shape  = (32,32)
stride = (32,1)
size   = 1024 elements
cosize = 1024 elements
bytes  = 4096
```

加入一列 padding：

```text
shape  = (32,33)
stride = (33,1)
size   = 1056 elements
cosize = 1056 elements
bytes  = 4224
```

更一般地，allocation 使用的是：

```text
bytes = cosize(layout) * element_bits / 8
```

对带洞、嵌套或 composed Layout，只比较逻辑元素数量可能低估真实地址跨度。

## 11.10 主例的 thread 与 value ownership

主例使用：

```text
tile        = 32 x 32
block       = (32, 8, 1)
threads     = 256
values/thread/tile = 4
```

block 内线性线程号仍以 `x` 维为最快变化方向。因此一个 warp 固定 `ty`，`tx` 从 0 到 31。

每个线程处理：

```python
for j in (0, 8, 16, 24):
    local_row = ty + j
    local_col = tx
```

定义域大小为：

```text
256 threads * 4 values = 1024 values
```

恰好覆盖一个 `32 x 32` tile。这个手写 mapping 与第 10 章的 TV ownership 是同一个问题，只是本章把映射直接写成坐标，便于把注意力放在 SMEM 和同步上。

## 11.11 GMEM → SMEM：row-wise producer phase

加载阶段写入：

```python
smem[local_row, tx] = src[global_row, global_col]
```

同一 warp 中 `tx` 连续，因此 source GMEM 地址与 SMEM destination 地址都沿连续列变化。这是 transpose 的 coalesced producer phase。

residue 坐标写入加法意义上的中性值 0：

```python
if cute.elem_less((global_row, global_col), problem_shape):
    smem[local_row, tx] = src[global_row, global_col]
else:
    smem[local_row, tx] = 0.0
```

这里 neutral fill 有两个作用：

- 不读取越界 GMEM；
- 不让未初始化 SMEM 参与后续跨线程读取。

## 11.12 第一条 barrier：publish loaded tile

所有 producer 写入后调用：

```python
cute.arch.sync_threads()
```

它提供两个方面的语义：

1. CTA 中所有线程到达 rendezvous；
2. barrier 之前的 CTA 可见内存操作，对 barrier 之后的参与线程可见。

仅有 control-flow convergence 但没有 memory ordering 不够；仅有 fence 但没有等待所有 producer 完成也不够。`sync_threads` 在经典 synchronous tiled kernel 中同时承担两者。

## 11.13 SMEM → RMEM → GMEM：transposed consumer phase

同步后，线程读取：

```python
value = smem[tx, ty + j]
dst[output_row, output_col] = value
```

注意 SMEM 坐标从 load 时的：

```text
(ty+j, tx)
```

交换为：

```text
(tx, ty+j)
```

这一步经过线程私有标量 `value`，因此数据路径是：

```text
GMEM source
  -> cooperative SMEM tile
  -> per-thread RMEM scalar
  -> transposed GMEM destination
```

## 11.14 第二条 barrier：protect buffer reuse

如果一个 CTA 只处理一个 tile，kernel 结束本身会终止所有 SMEM 使用，最后一条 barrier 可能不需要。

本章主例故意让每个 CTA 连续处理两个 row tiles：

```python
for tile_round in range(2):
    produce_to_smem()
    sync_threads()
    consume_from_smem()
    sync_threads()
```

第二条 barrier 防止下一轮 producer 覆盖当前 SMEM，而某些 consumer 仍在读取旧 tile。

两个 barrier 保护的是两个不同的 hazard：

| Barrier | 防止的 hazard |
|---|---|
| producer → consumer | read-after-write：consumer 过早读取 |
| consumer → next producer | write-after-read：producer 过早覆盖 |

这正是后续 full/empty mbarrier 和多级 pipeline 的同步 baseline。

## 11.15 为什么 residue 线程不能提前 return

错误写法：

```python
if not valid:
    return
sync_threads()
```

如果只有部分 CTA 线程提前退出，其余线程将在 CTA barrier 永久等待。

正确结构是让所有线程执行相同数量、相同顺序的 CTA barrier，只对内存操作加 predicate：

```python
if load_valid:
    load()
else:
    neutral_fill()

sync_threads()

if store_valid:
    store()

sync_threads()
```

甚至当第二个 tile 整体越界时，本例仍让全部线程完成空 producer phase、两次同步和空 consumer phase。这样 barrier participation 与 shape 无关。

## 11.16 Bank conflict 的最小模型

对连续 FP32 word，可先用经典 32-bank 模型分析：

```text
bank = word_address mod 32
```

这是一种地址映射分析模型。最终 bank 宽度、广播/多播规则和具体指令行为仍应结合目标架构验证。

### Compact stride 32

consumer warp 访问：

```text
smem[tx, fixed_column]
word_offset = tx * 32 + fixed_column
bank        = fixed_column
```

32 个 lane 落到同一个 bank 的不同地址，形成经典列访问 conflict。

### Padded stride 33

```text
word_offset = tx * 33 + fixed_column
bank        = (tx + fixed_column) mod 32
```

32 个 lane 被打散到 32 个 bank。

padding 没有改变逻辑 tile，只改变物理 row stride。这与第 6 章 Swizzle 的目标一致：改变地址映射而不改变逻辑坐标语义。

## 11.17 Padding、Swizzle 与性能结论

从 bank 映射可以严格推出“两个版本的 bank ownership 不同”。但不能只凭这个推出端到端性能提升，因为：

- 编译器可能改变 load/store 指令形态；
- transpose 还受 GMEM、launch、occupancy 和指令吞吐影响；
- padded 版本增加了 SMEM 容量；
- 小问题可能由 launch latency 主导。

因此本章区分：

- L0：Layout/bank 映射证明；
- L1/L2：数值与 residue 正确性；
- L4：PTX 中确有 shared load/store 和 CTA barrier；
- L3：只有按第 29 章统一 benchmark 方法采样后才建立性能结论。

## 11.18 一个 specialization 的 SMEM 容量

主例用 `PAD` 作为 compile-time specialization：

```python
smem_layout = cute.make_ordered_layout(
    (32, 32 + PAD),
    order=(1, 0),
)
```

因此：

```text
PAD=0 -> 32 * 32 * 4 = 4096 bytes
PAD=1 -> 32 * 33 * 4 = 4224 bytes
```

同一个 `PAD` specialization 则复用一个动态 shape compiled handle，覆盖：

- 单元素；
- 小于 tile；
- 恰好一个完整 tile；
- 奇数 tile 数；
- 多 CTA 和第二轮整体越界。

这是第 2、9、10 章“静态策略 + 动态问题”的延续。

## 11.19 SMEM 容量与 occupancy

假设每个 CTA 需要：

```text
smem_per_cta = storage_bytes
```

仅从 SMEM 约束看，每个 SM 最多可驻留的 CTA 数上界近似为：

```text
floor(smem_capacity_per_sm / smem_per_cta)
```

实际 occupancy 还同时受以下约束：

- threads/CTA；
- warps/CTA；
- registers/thread；
- architecture CTA limit；
- cluster placement；
- carveout 和 launch attributes。

因此“减少 128 bytes SMEM”不一定增加 resident CTA；只有跨过某个离散资源门槛时才会改变 occupancy。

## 11.20 Stage 数的容量公式

后续流水线把单 buffer 扩展成多 stage：

```text
total_smem
  = stages * bytes_per_stage
  + barrier_storage
  + epilogue_storage
  + alignment_padding
```

增加 stage 可能提高 latency hiding，也可能减少 occupancy。第 15 章会把这个容量公式与 producer/consumer phase 一起分析；本章只建立单 buffer reuse 的同步基础。

## 11.21 手工 alias 为什么危险

为了省容量，可以让两个逻辑 buffer 复用同一 raw allocation，但必须证明 lifetime 不重叠：

```text
last use of A
  -> synchronization/fence
  -> first overwrite as B
```

常见错误包括：

- 只证明同一线程完成 A，没证明其他线程完成 A；
- 地址对齐只满足 A，不满足 B；
- 两个异步 proxy 对同一 storage 的完成顺序未定义；
- compiler-visible type/layout 与手工 recast 后实际访问不一致。

教程在基础章节优先显式分配。只有 capacity 成为真实瓶颈时，再引入带 lifetime 证明的 alias。

## 11.22 可执行示例结构

[`tiled_transpose.py`](../code/11_shared_memory/tiled_transpose.py) 包含：

1. `SmemAllocator.allocate_tensor` 分配 static SMEM Tensor；
2. `PAD=0/1` 两个 specialization；
3. 每 CTA 两轮 tile processing 和真实 buffer reuse；
4. load/store 双侧 residue predicate；
5. invalid load 的 neutral fill；
6. 自动 SMEM size 推导与设备端回读；
7. 一个 compiled handle 复用多组 dynamic shapes；
8. PyTorch exact transpose reference；
9. retained PTX 中 shared load/store 和 barrier 检查。

运行：

```bash
python discussion/code/11_shared_memory/tiled_transpose.py
```

B200 实机输出与具体 CuTe DSL 版本统一记录在 [`validation_log.md`](validation_log.md)。

## 11.23 与后续章节的边界

本章中的 GMEM → SMEM copy 是同步执行的：线程发出 load，得到数据，再写 SMEM。

后续章节分别增加：

- 第 12 章：更细粒度的参与者和 barrier 状态机；
- 第 13 章：`cp.async` 与 `ldmatrix/stmatrix`；
- 第 14 章：TMA descriptor、partition 与 transaction completion；
- 第 15 章：多 stage、phase、warp specialization；
- 第 17 章以后：把同一个 SMEM ownership 模型接到 MMA。

不要在本章把 `sync_threads` 描述成异步 copy completion，也不要把 padded transpose 描述成 TMA pipeline。

## 11.24 常见误区

### 把 dynamic SMEM 理解为 dynamic Layout

launch 字节数可推导，不代表 `allocate_tensor` 接受任意运行时 Layout。

### 用 `size(layout)` 计算 allocation

带 padding、洞或 composed mapping 时，应基于 `cosize` 和 element width。

### 只同步 producer，不保护 reuse

第一条 barrier 防止过早读，第二条 barrier 防止过早覆盖。

### residue thread 直接 return

只要后面存在 CTA barrier，参与者集合就必须保持一致。

### 只保护 store

越界 GMEM load 和未初始化 SMEM read 都属于 correctness bug。

### allocation 对齐就认为所有行都对齐

row stride 和 tile/slice origin 仍会改变实际地址。

### padding 后直接宣称性能提升

bank 映射是机制证据；kernel 时间和硬件计数器才是性能证据。

### 把 `sync_threads` 当作跨 CTA barrier

它的参与范围仅是当前 CTA。普通 CTA 之间没有这种隐式同步关系。

## 11.25 本章检查清单与练习

设计一个 synchronous SMEM tiled kernel 时：

1. 写出 producer 和 consumer 的 thread/value ownership。
2. 计算每个 allocation 的 dtype、Layout、alignment 与 `cosize`。
3. 标出每一次 ownership transfer。
4. 为每条 barrier 写明参与者集合。
5. 区分 read-after-write 与 write-after-read barrier。
6. 保证 residue 路径执行相同的 barrier 序列。
7. 对 invalid load 写入明确 neutral value。
8. 分别证明 GMEM、SMEM 和 GMEM store 的边界。
9. 计算 per-CTA SMEM bytes 与可能的 occupancy 门槛。
10. 用数值 reference 验证结果，用 PTX/SASS 验证指令形态。

练习：

1. 把每 CTA 两轮改为一轮，解释第二条 barrier 是否仍必要。
2. 把 `BLOCK_Y` 从 8 攺为 4 或 16，重新证明 coverage。
3. 手算 compact/padded 版本中 warp 0 的全部 bank 编号。
4. 用第 6 章的 Swizzle 替换 padding，并比较逻辑 Layout 与物理地址。
5. 把手写 ownership 改写成 TiledCopy partition，验证坐标集合完全一致。
6. 增加 FP16 specialization，重新计算 SMEM bytes 和 bank mapping。
7. 故意删掉第二条 barrier，只做纸面 hazard trace；不要把存在数据竞争的程序作为正确示例运行。
