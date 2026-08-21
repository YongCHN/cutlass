# 第 2 章：类型系统、静态值与动态值

标签：`[通用]`
先修：第 0、1 章
状态：正文和三个示例完成；已在 B200 / CuTe DSL 4.7.0 上通过 L1 验证。

## 2.1 本章要回答的问题

CuTe DSL 使用 Python 语法，却不承诺完整的 Python 运行时语义。一个变量一旦参与生成 GPU IR，它的标量类型、所属阶段以及 value tree 结构就成为编译契约。理解这三件事，才能解释为什么某些普通 Python 写法会在 DSL 中被拒绝。

本章建立三个正交维度：

```text
值是什么类型？       Int32 / Float32 / Boolean / Tensor / ...
值在哪个阶段存在？   compile-time Python / runtime staged value
值怎样组成结构？     scalar leaf / tuple / dataclass / dynamic value tree
```

## 2.2 Python 值不是自动等于编译期常量

JIT 参数默认按动态参数处理。调用点传入 Python `5`，若形参标注为 `cutlass.Int32`，它会成为运行时 `i32` 参数；只有显式标注 `cutlass.Constexpr` 才属于编译期：

```python
@cute.jit
def f(x: cutlass.Int32, tile_size: cutlass.Constexpr):
    ...
```

- `x` 进入生成函数 ABI，同一 compiled handle 可以接收不同的 `x`。
- `tile_size` 不进入运行时 ABI；它参与 specialization。
- `tile_size` 变化通常需要另一份 specialization。

因此，“调用者传的是 Python int”和“它在 DSL 中是静态值”不是同一句话。阶段由 JIT 参数协议和注解决定。

## 2.3 DSL 标量类型

常用类型包括：

| 类别 | 代表类型 | 用途 |
|---|---|---|
| 布尔 | `cutlass.Boolean` | runtime predicate、mask |
| 有符号整数 | `Int8/16/32/64` | 索引、计数、整数计算 |
| 无符号整数 | `Uint8/16/32/64` | 位模式、无符号数据 |
| 浮点 | `Float16/BFloat16/Float32/Float64` | 数据和累加 |
| 低精度 | FP8、FP6、FP4 等 | 后续低精度章节 |
| 抽象基类 | `cutlass.Numeric` | `type[Numeric]` 形式的 dtype 元配置 |

`cutlass.Numeric` 通常用于描述“某一种数值 dtype 的类型”，例如 `element_type: type[cutlass.Numeric]`；它不是把所有运行时数值都装进同一个动态类型的 Python `Number`。

建议把动态常量也写出目标类型：

```python
value = x * cutlass.Int32(3) + cutlass.Int32(1)
scale = cutlass.Float32(0.5)
```

这样既能控制 IR 位宽，也能避免 Python literal 参与混合运算时产生依赖上下文的推断。

## 2.4 显式转换、提升与溢出

示例使用：

```python
integer_value = x * cutlass.Int32(3) + cutlass.Int32(1)
floating_value = cutlass.Float32(integer_value) * scale
```

核心原则是：跨整数/浮点域时显式转换，不依赖隐式提升猜测。还要区分：

1. **数值转换**：如 `Float32(i32_value)`，改变数值表示。
2. **位重解释**：保持比特、换解释方式，需要专门 API，不能用普通 cast 冒充。
3. **指针/向量 recast**：改变访问粒度，还额外受容量、对齐和向量宽度约束。

整数运算遵循固定宽度语义。shape、stride、linear index 可能超过范围时，应在 host 侧证明边界并选择合适位宽。特别注意：当前 CuTe Layout 代数的 shape/stride 主要是 32 位语义；不能因为普通标量支持 `Int64` 就假定 Layout 的所有内部计算也自动升级为 64 位。

## 2.5 `Constexpr` 与 `const_expr`

二者职责不同：

- `cutlass.Constexpr` 是函数参数的阶段/类型提示。
- `cutlass.const_expr(expr)` 告诉前端：这个条件必须在代码生成阶段求值。

```python
@cute.kernel
def epilogue(x: cutlass.Float32,
             do_square: cutlass.Constexpr):
    if cutlass.const_expr(do_square):
        x = x * x
```

`do_square=False` 的 specialization 中不会留下平方分支；这不是 GPU 在运行时选择 false，而是该代码根本没有进入这一版生成 IR。

反过来，把动态 predicate 强塞给 `const_expr` 是阶段错误：

```python
# 错误：x 运行时才知道
if cutlass.const_expr(x > 0):
    ...
```

4.7.0 会报告 `PHASE_REQUIRES_CONSTANT`，而不是悄悄把它变成动态分支。

## 2.6 静态结构可以装动态叶子

Python tuple、list、dict 和 frozen dataclass 可以在编译期间定义固定的树形结构，同时其叶子可以是 Tensor 或 DSL scalar：

```text
KernelParams                       固定结构
├── width: Python int              静态叶子
├── src: cute.Tensor               动态叶子集合
├── dst: cute.Tensor               动态叶子集合
├── scale: cutlass.Float32         动态叶子
└── bias: cutlass.Float32          动态叶子
```

编译器会 flatten 动态叶子为 MLIR/C ABI 参数，并在 JIT 函数或 kernel 内重建原来的容器。关键不变式是：树的形状、字段顺序和每个动态叶子的类型必须稳定。

允许的典型操作：

- 编译期遍历一个固定 tuple；
- frozen dataclass 中携带 Tensor 和动态标量；
- 构造一个同结构的新 tuple/dataclass 值。

不允许的典型操作：

- 根据 runtime 值 append/pop 改变 list 长度；
- 使用 runtime `Int32` 索引普通 Python list；
- 在动态循环中把某个字段从 `Int32` 换成 `Float32`；
- 让不同动态分支返回不同长度或不同字段的树。

普通 Python list 是元编程容器，不是 GPU random-access memory。运行时索引应该落到 Tensor、Pointer、Vector 或编译器明确支持的结构上。

## 2.7 四类结构不要混淆

### Frozen dataclass / NamedTuple

适合只读参数包。字段通过 pytree/value tree 展开，字段自身不可原地替换。要“修改”它，应构造一个新值。

### `@cute.native_struct`

适合 kernel 内可变的 by-value 聚合状态，例如循环中的 accumulator。字段更新会生成 LLVM struct value 更新。它和 Python dataclass 不是同一表示。

### `@cute.struct`

描述一段内存的字段布局，常配合 `SmemAllocator`：

```python
@cute.struct
class SharedRecord:
    values: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, 4], 16
    ]
    checksum: cutlass.Float32
```

- `MemRange[T, N]` 是连续 N 个 T 的内存范围。
- `Align[T, bytes]` 提高字段对齐约束。
- 标量字段访问得到 memory-backed wrapper；读取值要使用类似 `field.ptr.load()` 的操作。
- 它主要服务于 SMEM 等显式内存布局，不是普通的 by-value 参数包。

CuTe DSL 4.7.0 的 `@cute.struct` 在类创建时需要具体字段类型。本教程实测发现，对该模块启用 `from __future__ import annotations` 会把注解延迟成字符串并导致类型识别失败，因此结构定义示例没有启用 postponed annotations。

### 自定义 DynamicExpression/JitArgument

高级用户可以实现 `__extract_mlir_values__`、`__new_from_mlir_values__` 等协议，自行定义 value tree 的 flatten/reconstruct。它适合框架对象或复杂动态状态，但协议与编译上下文高度相关。本教程基础代码优先使用内建容器，协议实现留到框架集成和调试章节。

## 2.8 动态控制流的 join 约束

动态 `if` 的两条路径最终会在 IR 中汇合。汇合点上的变量必须具有相同类型和结构：

```python
value = cutlass.Int32(1)
if predicate:
    value = cutlass.Float32(2.0)  # 错误
out[0] = value
```

4.7.0 会报告 `TYPE_UNSTABLE_JOIN`。原因不是编译器“不够聪明”，而是汇合后的一个 SSA value 不可能同时既是 `i32` 又是 `f32`。

修复方法是进入分支前先决定统一类型：

```python
value = cutlass.Float32(1.0)
if predicate:
    value = cutlass.Float32(2.0)
```

同样规则也适用于循环携带值：一轮迭代输出的 value tree 必须能作为下一轮完全相同签名的输入。

## 2.9 其他当前限制

- 不支持依赖运行时数据改变类型的 dependent type。
- `global` 不受支持。
- 不能捕获当前 JIT 上下文之外的 `nonlocal` 动态状态。
- 动态 Python 容器不能改变结构。
- DSL 对象依赖 MLIR context，不应跨编译上下文用 `lru_cache` 保存并复用。
- 类型错误应尽量在带注解的函数边界暴露，而不是推迟到深层 lowering。

## 2.10 正向示例：标量与 specialization

代码：[types_and_constexpr.py](../code/02_types/types_and_constexpr.py)

这个示例验证：

- `Int32` 算术；
- `Int32 → Float32` 显式转换；
- runtime predicate；
- `Constexpr` 控制的两个 specialization；
- static 参数从 compiled handle 的 runtime ABI 中消失。

实测：

```text
square_result=False: i32=[16, 1], f32=[8.0]
square_result=True: i32=[16, 1], f32=[64.0]
PASS
```

## 2.11 结构示例

代码：[struct_and_dataclass.py](../code/02_types/struct_and_dataclass.py)

它用 frozen dataclass 把 Tensor、动态 scale/bias 和固定 width 传给 kernel，再用 `@cute.struct` 定义带 16-byte 对齐数组和 checksum 的共享内存布局。

```text
result=[1.25, 2.75, 4.25, 5.75, 14.0]
PASS
```

本例只有一个 thread，刻意不讨论线程间共享和 barrier；这些属于第 11、12 章。

## 2.12 可执行负向测试

代码：[expected_compile_errors.py](../code/02_types/expected_compile_errors.py)

脚本要求下面三次编译全部失败，否则测试自身失败：

| 错误 | 4.7.0 诊断类别 |
|---|---|
| 动态值传给 `const_expr` | `PHASE_REQUIRES_CONSTANT` |
| 动态值索引 Python list | `PHASE_DYNAMIC_INDEX` |
| 动态分支改变变量类型 | `TYPE_UNSTABLE_JOIN` |

错误消息文本可能随版本变化，所以测试只把“必须失败”作为稳定契约，把诊断类别作为当前版本的观察结果。

完整命令与环境见 [B200 验证日志](validation_log.md)。

## 2.13 检查清单与练习

写任何新 kernel 前先问：

1. 每个参数是 static 还是 dynamic？
2. 每个动态 scalar 的确切位宽是什么？
3. 跨类型运算是否显式转换？
4. Python 容器的结构会不会被 runtime 值改变？
5. 动态分支/循环前后的 value tree 是否同构？
6. 这里需要 dataclass、native struct，还是 memory struct？

练习：

1. 给正向示例增加 `Float16 → Float32` 累加，再比较舍入误差。
2. 把 frozen dataclass 字段改为可变赋值，观察 Python 或 DSL 在哪个阶段拒绝。
3. 给负向测试加入“动态 list append”和“分支改变 tuple 长度”。
4. 用 `@cute.native_struct` 实现包含 `total/count` 的动态 accumulator。

下一章把阶段和 value tree 规则应用到 `if/for/while`，观察它们如何变成或不变成 GPU IR。
