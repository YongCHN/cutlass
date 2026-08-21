# 第 8 章：TensorSSA 与寄存器级数据流

标签：`[通用]`
先修：第 2、7 章
状态：正文与示例完成；CuTe DSL 4.7.0 / B200 L1 验证通过。

## 8.1 本章要回答的问题

第 7 章的 Tensor 是 Engine 与 Layout 组成的 view。**为什么寄存器 fragment 不能一律当成普通 memory-backed Tensor？**

因为编译器中的寄存器计算更自然地表示为 SSA value：每个值只定义一次，运算产生新值，而不是在一个可寻址数组上反复 load/store。CuTe 用 `TensorSSA` 把一个静态 shape 的 thread-local 向量值保留为 Tensor 形状：

```text
memory Tensor --load()--> TensorSSA --compute--> TensorSSA --store()--> memory Tensor
```

这让编译器同时看到：

- 一组寄存器值；
- 它们的 dtype；
- CuTe shape/profile；
- elementwise、broadcast、slice、reduction 的数据流。

## 8.2 三个容易混淆的对象

| 对象 | 语义 | 是否可寻址/修改 | 常见操作 |
|---|---|---|---|
| GMEM/SMEM Tensor | 外部或共享 storage 的 view | 是 | 索引、copy、load/store |
| RMEM Tensor | thread-local register backing storage | 是 | `fragment[i]=...`、`load/store` |
| TensorSSA | immutable thread-local value | 否 | 算术、broadcast、slice、reduce |

RMEM Tensor 虽然物理上也对应寄存器，却仍采用 memory-like 接口：

```python
fragment[0] = 1.0
value = fragment.load()
```

`value` 才是 TensorSSA。二者的区别不是“在哪里存”，而是**编程语义是 storage 还是 value**。

## 8.3 `load()` 是 storage 到 value 的边界

对静态 shape memory Tensor：

```python
a_value = a.load()
```

产生一个 TensorSSA，逻辑 shape 与 `a.shape` 一致。其底层通常是扁平 MLIR vector，但 CuTe 继续保存 nested shape，因此后续算术仍能按 mode 推导。

一次完整 `load()` 要把该 Tensor view 的所有元素放进一个 thread 的寄存器数据流。它适合已经 partition 到每线程的小 fragment，不应对大型 CTA/global Tensor 直接使用，否则寄存器数和 IR 大小会迅速膨胀。

动态 shape 不能成为 TensorSSA shape：SSA vector lane 数必须在编译时确定。通常先用 tiling/partition 得到静态 per-thread fragment，再 `load()`。

## 8.4 `store()` 是 value 到 storage 的边界

```python
dst.store(result)
```

要求：

- TensorSSA shape 与 destination Tensor shape congruent；
- dtype 可按规则安全转换；
- destination address space 支持 store；
- 调用者已满足同步和边界条件。

`store()` 不是给 TensorSSA 增加可变性。它只是把一个完整 value 写到另一个 storage view。

标量索引写：

```python
fragment[i] = value
```

修改的是 memory/RMEM Tensor backing storage，不是原 TensorSSA。

## 8.5 `make_rmem_tensor`

显式创建 register-backed storage：

```python
row_bias = cute.make_rmem_tensor((2,1), cutlass.Float32)
row_bias[0] = 10.0
row_bias[1] = 20.0
row_value = row_bias.load()
```

参数可以是 shape 或 Layout：

```python
fragment = cute.make_rmem_tensor(layout, dtype)
```

使用显式 Layout 时，可让 RMEM fragment 与 copy/MMA 的 thread-value profile 对齐。分配量必须是静态的；不要用运行期 shape 决定每线程寄存器数组大小。

## 8.6 `make_fragment_like`

已有一个 Tensor view，希望建立 shape/layout congruent 的 RMEM storage：

```python
fragment = cute.make_fragment_like(per_thread_tensor)
```

还可显式指定 dtype：

```python
acc_fragment = cute.make_fragment_like(per_thread_tensor, cutlass.Float32)
```

常见路径：

```text
GMEM/SMEM per-thread view
  -> make_fragment_like
  -> cute.copy / load
  -> TensorSSA compute
  -> fragment store
  -> cute.copy to destination
```

本章示例用：

```python
fragment.store(result)
round_trip = fragment.load()
```

明确展示 value-storage-value 边界。真实 kernel 中，如果没有 copy/MMA API 需要 memory-like fragment，编译器有时可以直接让 TensorSSA 沿数据流传递；是否插入 RMEM backing storage应由接口需求决定。

## 8.7 Elementwise 运算与 dtype promotion

TensorSSA 支持常见逐元素操作：

```python
c = a + b
d = a * 2.0
mask = a > b
e = cute.math.sqrt(a)
```

每次表达式都产生新 TensorSSA。Python 变量重新绑定：

```python
x = x + 1.0
```

看起来像更新 `x`，实际含义是新 SSA value 替代 Python 名称的旧绑定，不会回写最初 load 的 Tensor。

dtype 按 CuTe DSL 的数值规则 promotion/cast。不要依赖 Python literal 的偶然类型，特别是在低精度和 MMA accumulator 路径中；关键边界显式写：

```python
scale = cutlass.Float32(2.0)
```

写回较窄 dtype 时，要明确 rounding/saturation 支持，而不是把 bit-width 相同误当作数值转换必然安全。

## 8.8 Broadcasting 的 shape 规则

TensorSSA 使用与 NumPy 类似的 size-1 mode 广播，并从 CuTe mode 顺序推导目标 shape。

示例：

```text
src         (2,3)
row_bias    (2,1)
column_bias (1,3)
result      (2,3)
```

代码：

```python
result = src_value * 2.0 + row_bias.load() + column_bias.load()
```

等价于：

```text
result[i,j] = src[i,j]*2 + row_bias[i,0] + column_bias[0,j]
```

广播不会给源 TensorSSA 增加 storage，只会在生成的 vector value 中复制/重排相应 lane。

需要警惕 CuTe 的 left-most/colexicographic 传统与外部框架打印顺序。shape 兼容并不保证你对 mode 语义的命名正确；对小矩阵始终用非对称数值验证。

## 8.9 TensorSSA slicing

和 Tensor view 类似，`None` 保留 mode，整数固定 mode：

```python
column_1 = value[(None, 1)]
```

若 `value.shape == (2,3)`，结果 shape 是 `(2,)`，包含：

```text
value[0,1], value[1,1]
```

完整整数坐标会返回 scalar，而保留至少一个 mode 的 slice 返回 TensorSSA。切片产生新 value，不与原 value 建立可写别名。

对 nested shape，slice spec 要匹配 profile。若下游 destination 的 shape 不 congruent，`store()` 会在编译阶段拒绝，而不会像某些数组库那样隐式 reshape。

## 8.10 Reduction 与 `reduction_profile`

接口：

```python
reduced = value.reduce(
    cute.ReductionOp.ADD,
    init_value,
    reduction_profile=profile,
)
```

`reduction_profile` 使用：

- `None`：保留该 mode；
- `1`：沿该 mode 规约；
- 标量 `0` 等形式可表达规约全部或相应 profile（以具体 shape/API 为准）。

对 `(2,3)`：

```python
row_sums = value.reduce(
    cute.ReductionOp.ADD,
    0.0,
    reduction_profile=(None, 1),
)
```

保留 mode 0，规约 mode 1，结果 shape `(2,)`。

相反：

```text
(1,None)
```

规约 mode 0、保留 mode 1，得到 shape `(3,)`。

`init_value` 参与每个 reduction group，必须是对应操作的正确 identity 或有意设置的初值。求和通常用 0，乘积用 1，max/min 要根据 dtype 选择合法极值。

本章是单线程 TensorSSA 内规约。跨 lane、warp、CTA 的规约需要 shuffle/shared memory/barrier 等通信，将在第 12、16 章讨论。

## 8.11 “mutation-looking” syntax 的正确解释

需要区分三种写法：

```python
fragment[i] = x        # 修改 RMEM Tensor storage
dst.store(value)       # 把 SSA value 写入 destination storage
value = value + 1.0    # 新 SSA value，Python 名称重新绑定
```

TensorSSA 本身是 immutable。不要期望：

```python
value[i] = x
```

像数组一样就地更新。需要修改局部 lane 时，应使用能返回新 TensorSSA 的 elementwise/select/where/slice 组合，或先写入 RMEM Tensor 再 load 新 value。

SSA 思维能避免一个常见误解：源码中同名变量出现多次，不代表硬件中一定存在多份寄存器数组；寄存器分配器会基于 live range 复用物理寄存器。反过来，一个看似短的 TensorSSA 表达式也可能因过多同时存活的 value 导致高 register pressure。

## 8.12 MMA accumulator fragment 的前置模型

Tensor Core 章节会出现：

- per-thread A/B operand fragment；
- accumulator fragment；
- MMA 指令更新后的新 accumulator value；
- TMEM/RMEM 间的专用 load/store。

本章模型直接延伸到那里：

```text
partitioned Tensor view
  -> fragment storage / load
  -> TensorSSA operand
  -> MMA produces accumulator value
  -> epilogue elementwise/broadcast/reduction
  -> store
```

但 accumulator shape 不是用户随意选择的二维数组，而由 MMA atom、thread ownership 和 instruction shape 决定。只有掌握第 10 章 TV Layout 与后续 MMA partition，才能正确解释每个 TensorSSA lane 属于哪一个矩阵坐标。

## 8.13 可执行示例与实测结果

代码：[tensorssa_ops.py](../code/08_tensorssa/tensorssa_ops.py)

运行：

```bash
cd /volume/njiang/workspace/sandbox/cutedsl_tutorial/code/08_tensorssa
CUDA_VISIBLE_DEVICES=0 /volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python tensorssa_ops.py
```

一个 GPU thread 完成：

1. `(2,3)` GMEM Tensor load；
2. `(2,1)` 与 `(1,3)` RMEM bias 构造；
3. TensorSSA elementwise 与广播；
4. `make_fragment_like` 的 store/load 往返；
5. 逐行 ADD reduction；
6. 第 1 列 slice；
7. 三个 GMEM 结果写回。

B200 / CuTe DSL 4.7.0 实测：

```text
dst=
tensor([[12.5000, 15.0000, 17.5000],
        [28.5000, 31.0000, 33.5000]])
row_sums=[45.0, 93.0]
selected_column=[15.0, 31.0]
PASS
```

结果与 PyTorch reference 精确一致，L1 通过。该示例没有跨线程通信，也不作性能声明。完整记录见 [B200 验证日志](validation_log.md)。

## 8.14 常见误区

### 把 RMEM Tensor 与 TensorSSA 当成同一种对象

前者是 storage 语义，后者是 immutable value 语义；`load/store` 是两者的显式边界。

### 对大型 Tensor 直接 `.load()`

一个 thread 会承担整个静态 TensorSSA。先 partition 到合理的 per-thread fragment。

### 认为 `x = x + 1` 会回写源 Tensor

它只创建并绑定新 SSA value。需要持久化必须 `store()`。

### 忘记广播后的 mode 语义

shape 数值兼容不等于 row/column 命名正确。使用非对称数据和 reference 检查。

### 混淆单线程 reduction 与并行 reduction

TensorSSA `reduce` 只规约当前 thread 持有的 lanes，不自动跨 thread 通信。

### 让动态 shape 进入 TensorSSA

TensorSSA 的向量 lane 数必须静态。动态 problem shape 应在 tiling/boundary 层处理。

## 8.15 本章检查清单与练习

设计寄存器数据流时：

1. 标出每个对象是 memory Tensor、RMEM Tensor 还是 TensorSSA。
2. 只在明确边界调用完整 `load/store`。
3. 为每个 TensorSSA 写出 static shape 和 dtype。
4. 手算 broadcast target shape。
5. 为 reduction profile 标出保留/规约 mode。
6. 跟踪 live value，避免无意延长 accumulator 生命周期。
7. 分开验证单线程数值与跨线程通信。

练习：

1. 给示例增加 column sums，并写出 `reduction_profile`。
2. 把第 1 列 slice 改成第 0 行，核对结果 shape。
3. 用 `cute.where` 为负值实现 ReLU，证明源 TensorSSA 未改变。
4. 将输出 dtype 改为 FP16，明确转换位置和 reference tolerance。
5. 把 `(2,3)` 扩大到 `(8,8)`，比较生成 IR、register usage 与运行结果；解释为何这不代表合理的生产 per-thread tile。

下一章将回到多线程执行层级，把 thread/block coordinate、identity Tensor predicate 和不完整尾 tile 组合成一个边界安全的普通 kernel。
