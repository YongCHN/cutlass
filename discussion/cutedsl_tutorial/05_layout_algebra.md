# 第 5 章：Layout 代数

标签：`[通用]`
先修：第 4 章
状态：正文与示例完成；CuTe DSL 4.7.0 / B200 L0 验证通过。

## 5.1 本章要回答的问题

第 4 章把 Layout 定义为坐标到索引的函数。本章再向前一步：**tile、partition 和 thread/value 映射为什么都能写成 Layout 变换？**

答案不是“CuTe 提供了很多辅助函数”，而是这些操作都在变换同一个对象：

```text
domain coordinate --Layout--> codomain index
```

有的操作保持函数不变、只规范化表示；有的操作复合两个函数；有的操作把 domain 分解成 tile 内坐标和 tile 坐标；还有的操作为一个已有映射补齐其余地址。读任何 Layout 代数表达式时，都应同时追踪：

1. 新 domain 的树形结构；
2. 每个 mode 的语义；
3. 新函数如何映射到原 codomain；
4. 操作需要哪些可整除、无碰撞或覆盖条件。

## 5.2 先区分“函数”和“表示”

两个 Layout 的打印形式不同，仍可能表示相同函数。例如一个 hierarchical Layout 可以被 `coalesce` 压平；只要对 domain 中所有逻辑位置得到相同 offset，它们在映射意义上就是等价的。

反过来，两个 Layout 的 shape 相同，并不表示映射相同：

```text
(2,3):(1,2)
(2,3):(3,1)
```

因此示例不会只断言字符串，而是检查性质：

```python
for i in cutlass.range_constexpr(cute.size(original)):
    assert normalized(i) == original(i)
```

这种 property-driven 检查方式比绑定某个版本的规范化打印结果更稳健。

## 5.3 `coalesce`：保持映射的规范化

`coalesce(layout)` 尝试：

- 删除 size-1 mode；
- 合并在逻辑枚举次序上连续、在 codomain 中也连续的相邻 mode；
- 降低不必要的嵌套深度。

以 left-major 的两个相邻 mode 为例：

```text
(M,N):(S, M*S)  ->  (M*N):S
```

因为：

```text
i*S + j*(M*S) = (i + j*M)*S
```

若 stride 关系不满足，合并会改变函数，`coalesce` 就必须保留边界。它不是无条件 `flatten`，也不是让所有 Layout 变 contiguous。

实验中的输入：

```text
((2,(3,4)),(3,2),1):((4,(8,24)),(2,6),12)
```

被规范化为：

```text
(24,6):(4,2)
```

两者 size 相同，并且对所有 logical index 的函数值相同。

需要保留特定 mode 边界时，可以按 mode 指定 profile，而不是先把语义结构全部压掉。后续 TiledCopy/MMA 代码中，过早 coalesce 常会让 thread/value 或 M/N/K 边界变得难以辨认。

## 5.4 `composition`：真正的函数复合

对 Layout `A` 与 `B`：

```text
C = composition(A, B)
C(x) = A(B(x))
```

代码中的例子：

```python
outer = cute.make_layout((6, 2), stride=(8, 2))
inner = cute.make_layout((4, 3), stride=(3, 1))
composed = cute.composition(outer, inner)
```

验证的不是输出长什么样，而是：

```python
assert composed(i) == outer(inner(i))
```

顺序不能颠倒。`composition(outer, inner)` 先把输入交给 `inner`，再把结果交给 `outer`。

常见用途包括：

- 用一个 tile Layout 重新参数化原 Tensor/Layout；
- 把 thread/value coordinate 映射到 tile coordinate；
- 将局部排列附着在原地址映射之前；
- 用 inverse 把一个表示空间拉回另一个表示空间。

组合前必须检查 `inner` 的 codomain 是否是 `outer` 可接受的 coordinate/index 范围。函数可以成功构造，不代表它一定覆盖整个 outer domain，也不代表没有 collision。

## 5.5 Divide：把 domain 拆成 tile 与 rest

设目标 Layout 为：

```text
target = (8,6):(6,1)
tiler  = (2,3)
```

语义上要把每个原坐标写成：

```text
original coordinate = coordinate within tile + coordinate of tile
```

这两组坐标分别简称 Tile 与 Rest。四个 divide 变体的主要区别是**如何保存这组 mode 的层级**，而不是覆盖不同的数据。

### `logical_divide`

```text
((2,4),(3,2)):((6,12),(1,3))
```

每个原 mode 被替换为 `(TileMode, RestMode)`：

```text
((TileM, RestM), (TileN, RestN))
```

适合逐个追踪原 mode 的来源。

### `zipped_divide`

```text
((2,3),(4,2)):((6,1),(12,3))
```

所有 tile mode 和所有 rest mode 分别 zip 在一起：

```text
((TileM, TileN), (RestM, RestN))
```

这正是许多 block tile 代码最自然的形式：第一组是 CTA 内 tile，第二组是 grid 中的 tile 坐标。

### `tiled_divide`

```text
((2,3),4,2):((6,1),12,3)
```

tile modes 保持为一组，rest modes 展开到外层：

```text
((TileM, TileN), RestM, RestN)
```

### `flat_divide`

```text
(2,3,4,2):(6,1,12,3)
```

所有 mode 都在同一层：

```text
(TileM, TileN, RestM, RestN)
```

四种结果的 domain size 都是 48，且都重新参数化同一个目标映射。选择依据是下游 API 希望看到哪种 profile；不要只因为 flat 形式更短就丢弃有用层级。

若 shape 不能被 tiler 整除，divide 的结果会涉及向上取整或 residue 语义，合法坐标与实际数据边界不再相同。第 9 章会用 identity Tensor 为尾块建立 predicate。

## 5.6 Product：复制 atom 覆盖更大 domain

Divide 是拆分；Product 可以理解为把一个已有 block/atom 与一个 replication Layout 组合，铺满更大 domain。

实验使用：

```text
block       = (2,2):(2,1)
replication = (3,2):(1,3)
```

得到的总 size 是：

```text
size(product) = size(block) * size(replication) = 4 * 6 = 24
```

对应四种结构：

| API | 结果 shape | 阅读方式 |
|---|---|---|
| `logical_product` | `((2,2),(3,2))` | atom 与 replication 保留层级 |
| `zipped_product` | `((2,2),(3,2))` | 按 zip profile 分组；本例恰与 logical 相同 |
| `tiled_product` | `((2,2),3,2)` | atom 为一组，复制 mode 展开 |
| `flat_product` | `(2,2,3,2)` | 全部展开 |

“size 乘起来”只是必要检查，不足以证明覆盖正确。还应检查：

- atom 的 `cosize` 与 replication stride 是否配合；
- 不同副本是否意外别名；
- 最终 codomain 是否有洞；
- 下游是否依赖 atom 的层级边界。

## 5.7 `complement`：补齐未覆盖地址

`complement(layout, target_size)` 构造一个与已有 Layout 互补的映射，使 product 可以覆盖目标 codomain。

最简单的例子：

```text
layout              = 4:1
complement(layout,24)= 6:4
```

原 Layout 覆盖一个 4 元素连续块；complement 以 stride 4 选择六个块起点：

```text
0, 4, 8, 12, 16, 20
```

两者组合后可覆盖 `[0,24)`。

这不是集合论中任意形式的“补集”，而是为 Layout product 构造可组合的坐标映射。目标 size、原映射的 injectivity 以及 stride 结构都会影响结果；不能把它当作任意稀疏 Layout 的自动修复器。

## 5.8 Left inverse 与 right inverse

函数复合中：

```text
right inverse R: L(R(x)) = x
left  inverse Q: Q(L(x)) = x
```

对 domain/codomain 上的双射 permutation，两者一致。实验中的：

```text
L = (2,3):(3,1)
inverse = (3,2):(2,1)
```

满足两边复合关系。

但一般函数不一定有双侧逆：

- broadcast/collision Layout 不是 injective，不能唯一恢复输入；
- padded Layout 的 codomain 有洞，不是 surjective 到整个 span；
- inverse 只对其定义/覆盖范围有意义。

所以 inverse 不是“把 stride 倒过来”。使用前先写出希望成立的复合等式，再在有效范围上断言它。

## 5.9 结构操作：`select`、`group_modes`、`flatten`、`slice_`

以：

```text
base = (2,3,4):(12,4,1)
```

为例。

### `select`

```python
cute.select(base, mode=[0, 2])
# (2,4):(12,1)
```

选择 mode 0 和 2，并保持指定顺序。它不是运行时数据 gather，而是静态选择 Layout mode。

### `group_modes`

```python
cute.group_modes(base, 0, 2)
# ((2,3),4):((12,4),1)
```

把半开区间 `[0,2)` 组合成一个 hierarchical mode。函数没有改变，profile 改变了。

### `flatten`

```python
cute.flatten(grouped)
# (2,3,4):(12,4,1)
```

移除层级括号，但不承诺像 `coalesce` 那样合并相邻 mode。`flatten` 改树形表示，`coalesce` 还会依据 stride 关系规范化。

### `slice_`

```python
cute.slice_(base, (1, None, None))
# (3,4):(4,1)
```

整数固定一个 mode，`None` 保留一个 mode。对 Layout，结果描述余下 domain 的映射；对 Tensor，还会同时推进 iterator，详见第 7 章。

## 5.10 `tile_to_shape`

`tile_to_shape(atom, target_shape, order)` 重复 atom，直到构造出目标 shape：

```python
atom = cute.make_layout((2,2), stride=(2,1))
cute.tile_to_shape(atom, (4,6), order=(1,0))
# ((2,2),(2,3)):((2,12),(1,4))
```

阅读为：

- 第一组 `(2,2)` 是 atom 内坐标；
- 第二组 `(2,3)` 是复制次数；
- `order` 决定扩展各 mode 的顺序。

`target_shape` 必须与 atom 的覆盖方式相容。即便最终 size 正确，也仍应检查输出 mode 的语义和映射表，而不是把它当成普通 reshape。

## 5.11 为每个 mode 做语义标注

复杂代码中，建议紧挨 Layout 记录 profile：

```python
gA = cute.zipped_divide(mA, cta_tiler)
# gA: ((TileM, TileK), (RestM, RestK))
```

比变量名更可靠的检查方式是把 shape 一起打印：

```python
print(f"gA={gA}")
```

对于更深层结构，使用逐层命名：

```text
((ThrM,ThrN),(ValM,ValN),(RestM,RestN))
```

后续每执行一次 `select/group/flatten/coalesce`，都同步更新注释。很多所谓“Layout bug”其实是代码仍按旧 profile 解读新结果。

## 5.12 可执行示例与实测结果

代码：[layout_algebra_lab.py](../code/05_layout_algebra/layout_algebra_lab.py)

运行：

```bash
cd /volume/njiang/workspace/sandbox/cutedsl_tutorial/code/05_layout_algebra
/volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python layout_algebra_lab.py
```

示例在生成/编译阶段覆盖：

- `coalesce` 的映射保持；
- `composition` 的函数复合；
- 四种 divide 与 product 的 size/profile；
- complement 的块起点；
- left/right inverse 的复合等式；
- `select/group_modes/flatten/slice_`；
- `tile_to_shape`。

B200 / CuTe DSL 4.7.0 实测以 `PASS` 结束，关键 Layout 输出已记录在 [验证日志](validation_log.md)。这是 L0 结构/代数验证；本章没有发射数据处理 kernel，也没有性能结论。

## 5.13 常见误区

### 只看结果 size

相同 size 可以对应不同 profile、不同 stride、collision 或 padding。至少同时检查 shape、stride 和若干映射值。

### 把 `coalesce` 当作 `flatten`

`flatten` 去层级；`coalesce` 还会在合法时合并 mode 并删除 size-1 mode。

### 忘记 composition 的执行顺序

`composition(A,B)(x) == A(B(x))`，先执行右边。

### 把四种 divide/product 当成四种数据算法

它们主要表达同一逻辑变换的不同 mode 分组。选错 profile 会让下游索引难写，但不应凭名称猜数据方向。

### 对任意 Layout 求逆

collision、broadcast 和 padding 会破坏双侧逆条件。先写出并验证所需的左逆或右逆等式。

### 变换后仍沿用旧 mode 注释

Layout 代数最危险的错误常不是 offset 公式，而是把 `(Tile,Rest)` 误读成 `(Rest,Tile)`。

## 5.14 本章检查清单与练习

面对一个 Layout 变换：

1. 写出输入和输出的完整 `Shape:Stride`。
2. 为每层 mode 标注语义。
3. 写出要保持的函数等式。
4. 检查 size、cosize、injectivity 与覆盖范围。
5. 用小 shape 枚举映射，而不是只比较字符串。
6. 明确变换发生在编译期还是包含动态 leaf。

练习：

1. 手算 `(8,12)` 被 `(2,3)` divide 后四种 profile。
2. 构造一个无法完全 coalesce 的三 mode Layout，解释是哪一对 stride 破坏条件。
3. 对 `(2,3):(3,1)` 枚举 Layout 与 inverse 的双向映射表。
4. 用 `logical_product` 把 `(2,2)` atom 扩展到 24 个位置，并检查是否无洞无碰撞。
5. 把实验中的 `base` 先 `group_modes` 再 `slice_`，与先 `slice_` 再 `group_modes` 比较 profile。

下一章会在普通 Layout 外再叠加 bit-level 地址变换，解释 Swizzle、ComposedLayout 与 shared-memory bank conflict 的关系。
