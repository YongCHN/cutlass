# 第 4 章：Layout 是坐标到索引的函数

标签：`[通用]`
先修：第 2、3 章
状态：正文与示例完成；CuTe DSL 4.7.0 / B200 L0/L1 验证通过。

## 4.1 本章要回答的问题

CuTe 输出中的：

```text
(3,4):(4,1)
```

不是一个 Tensor，也不是“12 个元素的二维数组”。它是一个函数的紧凑表示：

```text
Layout = Shape : Stride
         domain   mapping rule
```

对上面的 Layout，二维坐标 `(i,j)` 被映射为：

```text
L(i,j) = i * 4 + j * 1
```

本章要建立的核心心智模型是：**Layout 只负责映射，不拥有数据。** 到第 7 章把 Layout 与 Pointer/Iterator 组合后，映射结果才会真正成为一次内存访问的 offset。

## 4.2 三种容易混淆的量

后续讨论必须严格区分：

| 名称 | 示例 | 含义 |
|---|---|---|
| logical coordinate | `(1,2)` | domain 中的多维位置 |
| logical linear index | `5` | 按 Shape 的默认逻辑次序枚举 domain |
| physical index/offset | `6` 或 `7` | Layout 应用 Stride 后的函数值 |

以 `Shape=(3,4)` 为例，CuTe 展开 logical linear index 时采用最左 mode 最快变化的 colexicographic 次序：

```text
0 → (0,0)
1 → (1,0)
2 → (2,0)
3 → (0,1)
4 → (1,1)
5 → (2,1)
```

因此 `idx2crd(5, (3,4)) == (2,1)`。同一个 logical index 5 进入不同 Layout：

```text
(3,4):(1,3)  → 5
(3,4):(4,1)  → 9
```

前者和默认逻辑枚举一致；后者先得到坐标 `(2,1)`，再计算 `2*4+1=9`。

不要把 logical linear index 直接当成物理地址，也不要默认 CuTe 的整数枚举顺序就是某个框架的 contiguous 顺序。

## 4.3 Shape、Stride 与 Coord 都是 IntTuple

CuTe 的 `IntTuple` 是递归定义：

```text
IntTuple := integer | tuple[IntTuple, ...]
```

所以这些都是合法形状：

```text
8
(3,4)
((2,3),4)
(2,(2,2))
```

Shape 和 Stride 必须 congruent，也就是具有相同的树形轮廓。合法示例：

```python
shape  = ((2, 3), 4)
stride = ((1, 2), 6)
layout = cute.make_layout(shape, stride=stride)
```

坐标也遵循同一层级：

```text
coord = ((i0,i1),j)
L(coord) = i0*1 + i1*2 + j*6
```

`((1,2),3)` 因而映射到 `1 + 4 + 18 = 23`。

层级不是纯粹的括号装饰。后续 tile、thread/value partition、Copy Atom 和 MMA fragment 都使用它保留“这一组 mode 属于同一逻辑维度”的信息。

## 4.4 rank、depth、size 与 cosize

这四个量回答不同问题：

### `rank`

最外层有多少个 mode。标量整数按 rank 1 处理：

```text
rank(8)           = 1
rank((3,4))       = 2
rank(((2,3),4))   = 2
rank((2,3))       = 2   # 取上例 mode 0 后
```

### `depth`

Shape 的最大嵌套层数：

```text
depth(8)          = 0
depth((3,4))      = 1
depth(((2,3),4))  = 2
```

所以 `rank` 不会把所有 leaf 简单计数。`((2,3),4)` 有三个整数 leaf，但顶层 rank 仍是 2。

### `size`

domain 的元素数，即所有 shape leaf 的乘积：

```text
size(((2,3),4)) = 2 * 3 * 4 = 24
```

### `cosize`

让 Layout 产生的所有非负 offset 都可作为下标时，所需地址 span 的上界；对常规非负 strided Layout，可理解为最大 offset 加 1：

```text
cosize(L) = L(size(L)-1) + 1
```

`size` 和 `cosize` 完全可能不同：

| Layout | size | cosize | 原因 |
|---|---:|---:|---|
| `(3,4):(4,1)` | 12 | 12 | compact row-major |
| `(3,4):(8,1)` | 12 | 20 | row 间有 padding holes |
| `(3,4):(0,1)` | 12 | 4 | 三个 row 映射到同一组 offset |

`cosize` 不是“实际访问了多少个互异元素”。带 padding 时它包含洞；带别名时多个坐标可能指向同一 offset。它表达的是 codomain span，而不是集合去重计数。

## 4.5 `make_layout`

最直接的构造函数：

```python
layout = cute.make_layout(shape, stride=stride)
```

如果省略 stride，CuTe 生成 compact left-most stride：

```python
cute.make_layout((3,4))
# (3,4):(1,3)
```

“left-most”意味着 mode 0 最快变化。若约定 mode 0 是 row、mode 1 是 column，它通常称为 column-major；但 Layout 自己并不知道“row”或“column”，这些名字来自调用者赋予 mode 的语义。

显式 row-major：

```python
cute.make_layout((3,4), stride=(4,1))
```

Stride 是 keyword-only 参数。坚持写出 `stride=` 能避免把 `(shape,stride)` 误写成一个嵌套 shape。

## 4.6 手算四种二维映射

以下统一把第一个 mode 解释为 row，第二个解释为 column。

### Compact left-major：`(3,4):(1,3)`

```text
      col 0  1  2  3
row 0      0  3  6  9
row 1      1  4  7 10
row 2      2  5  8 11
```

### Row-major：`(3,4):(4,1)`

```text
      col 0  1  2  3
row 0      0  1  2  3
row 1      4  5  6  7
row 2      8  9 10 11
```

### Padded row-major：`(3,4):(8,1)`

```text
      col 0  1  2  3
row 0      0  1  2  3
row 1      8  9 10 11
row 2     16 17 18 19
```

offset 4–7 和 12–15 是 padding hole。12 个逻辑坐标需要覆盖到 offset 19，所以 `cosize=20`。

### Broadcast row：`(3,4):(0,1)`

```text
      col 0  1  2  3
row 0      0  1  2  3
row 1      0  1  2  3
row 2      0  1  2  3
```

zero stride 表示该 mode 不影响输出 offset。读取时它是广播视图；并行写入时多个 logical coordinate 会别名到同一位置，必须额外证明不存在数据竞争。

## 4.7 `make_ordered_layout`

手算 stride 容易在高 rank 时出错。`make_ordered_layout` 通过从最快到最慢的 mode 顺序生成 compact stride：

```python
left = cute.make_ordered_layout((3,4), order=(0,1))
row  = cute.make_ordered_layout((3,4), order=(1,0))
```

结果分别等价于：

```text
(3,4):(1,3)
(3,4):(4,1)
```

`order` 描述的是 majorness/变化顺序，不是 shape 的重排，也不会交换逻辑坐标含义。换句话说，`(i,j)` 仍是 `(i,j)`，只是它被映射到不同 offset。

## 4.8 `make_identity_layout` 不是 contiguous Layout

Identity Layout 将坐标映射回坐标：

```python
identity = cute.make_identity_layout((3,4))
identity((1,2)) == (1,2)
```

它的 stride 以 scaled basis 形式表示，例如 `1@0`、`1@1`。输出属于坐标空间，而不是单个内存 offset。

Identity Layout 的主要价值是构造 coordinate Tensor：数据 tile 和 identity tile 使用相同的切片/partition 操作，一个产生数据访问，另一个保留原始全局坐标，用于边界谓词。第 7、9 章会完整实现这一模式。

因此不要用 `make_identity_layout` 替代 `(shape):(compact stride)`。

## 4.9 Layout 的两种调用方式

### 直接应用坐标

```python
offset = layout((row, column))
```

### 显式坐标转换

```python
offset = cute.crd2idx((row, column), layout)
coord  = cute.idx2crd(logical_index, layout.shape)
```

`layout(coord)` 和 `crd2idx(coord, layout)` 对普通 Layout 表达同一映射。`idx2crd` 只根据 Shape 展开 logical index，并不根据某个 Layout 的 stride 做“物理地址反解”。

物理 offset 到坐标可能根本没有唯一逆：

- broadcast Layout 是多对一；
- padded Layout 的洞没有对应坐标；
- 任意 stride 还可能产生其他 collision。

真正的 inverse 属于第 5 章 Layout 代数，并且需要满足相应前置条件。

## 4.10 静态 Layout 与动态 Layout

第 2 章的 static/dynamic 规则直接延伸到 Shape 和 Stride leaf。

静态：

```python
layout = cute.make_layout((3,4), stride=(8,1))
```

- shape、stride、size、cosize 和映射可在生成阶段求值；
- Python `print(layout)` 能看到完整值；
- 编译器可以据此消除索引计算或展开结构。

动态：

```python
layout = cute.make_layout(
    (rows, columns),
    stride=(leading_dimension, cutlass.Int32(1)),
)
```

- rank/depth/tree profile 仍然固定；
- leaf 值在 GPU 运行时进入；
- `size`、`cosize` 和 offset 可能成为 staged value；
- 同一 compiled handle 可复用不同的 rows/columns/leading dimension。

动态不等于“运行时任意改变 Layout 类型”。它允许固定 value tree 中的整数 leaf 变化，不允许调用时把 rank 2 变成 rank 3，或把 `(rows,cols)` 换成 `((a,b),cols)`。

来自 PyTorch/DLPack 的动态 Tensor Layout 以及 shape compatibility 会在第 7 章结合 Tensor ABI 单独讨论。

## 4.11 Layout 不等于性能

一个 Layout 可以：

- 数学映射正确；
- `size/cosize` 计算正确；
- 甚至是双射；
- 但仍产生低效的 global transaction 或严重 SMEM bank conflict。

本章只判断函数语义。性能还取决于：

- 哪些 thread 同时访问哪些 coordinate；
- 元素 dtype 和 vector width；
- alignment；
- memory space；
- transaction/bank 粒度；
- 是否满足 TMA、ldmatrix、MMA 的布局约束。

第 6 章会把 Swizzle 和 bank conflict 加入模型，第 10 章再把 thread/value Layout 与 TiledCopy 组合起来。

## 4.12 可执行示例

代码：[layout_basics.py](../code/04_layout/layout_basics.py)

第一部分在编译阶段构造并断言：

- compact left-major；
- row-major；
- padded row-major；
- zero-stride broadcast；
- ordered Layout；
- Identity Layout；
- hierarchical Layout；
- `rank/depth/size/cosize/crd2idx/idx2crd`。

第二部分启动一个 32-thread kernel，在 GPU 运行时用动态 rows、columns 和 leading dimension 构造 Layout，再把每个坐标的 offset 写回 Tensor。一个 compiled handle 连续验证：

```text
(3,4):(8,1)
(2,5):(7,1)
```

运行：

```bash
cd /volume/njiang/workspace/sandbox/cutedsl_tutorial/code/04_layout
CUDA_VISIBLE_DEVICES=0 /volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python layout_basics.py
```

B200 / CuTe DSL 4.7.0 实测的静态映射表和两组动态结果均与 reference 一致，程序输出 `PASS`。完整命令与结果见 [B200 验证日志](validation_log.md)。

## 4.13 常见误区

### 把 Layout 当作数据容器

Layout 没有元素存储。`layout((i,j))` 返回映射值，不返回矩阵元素。

### 把 `size` 当成分配长度

带 padding 时应关注 `cosize`；否则合法坐标可能生成超过 `size-1` 的 offset。

### 把 `cosize` 当成互异地址数量

zero stride/collision 会让多个坐标别名；`cosize` 只是 codomain span。

### 默认右边 mode 最快

`make_layout(shape)` 默认是 compact left-most。需要 conventional row-major 时显式 stride 或 `make_ordered_layout`。

### 把 Identity Layout 当作 stride-1 storage

Identity Layout 返回 coordinate，不是线性 offset。

### 动态 shape 可以改变层级

动态 leaf 可以变，rank/depth/profile 仍是 JIT 类型的一部分。

## 4.14 本章检查清单与练习

面对任意 Layout，先完成：

1. 写出完整 `Shape:Stride`。
2. 标注每个 mode 的业务语义。
3. 手算至少三个 coordinate 的映射。
4. 分别计算 rank、depth、size、cosize。
5. 判断是否有 padding、broadcast 或 collision。
6. 标出 static/dynamic leaf。
7. 不做线程映射分析前，不下性能结论。

练习：

1. 枚举 `(4,3):(1,8)` 的二维映射表，并计算 size/cosize。
2. 为 Shape `(2,(2,3))` 分别生成 left-major 和自定义 hierarchical stride。
3. 构造一个 `size=12`、`cosize=6` 且存在 collision 的非 zero-stride Layout。
4. 把动态示例改成 column-major leading dimension，并写出 PyTorch reference。
5. 用仓库 `print_latex.py` 生成一个二维 Layout 的 LaTeX，核对它与手算表格一致。

下一章将在函数视角之上引入 composition、divide、product、complement 和 inverse，解释 tile 与 partition 为什么都可以统一成 Layout 代数。
