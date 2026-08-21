# 第 3 章：控制流与元编程

标签：`[通用]`
先修：第 1、2 章
状态：正文和控制流/IR 示例完成；已在 B200 / CuTe DSL 4.7.0 上通过 L1 验证。

## 3.1 本章要回答的问题

看到一段 Python `for` 或 `if`，不能只凭语法判断它执行在哪里。必须同时看：

1. 控制结构使用哪个 API；
2. bound/predicate 是静态还是动态值；
3. 前端选择编译期执行还是生成结构化 IR；
4. 后端是否进一步 unroll、fold 或消除这段 IR。

前端 staging 和后端优化是两件事。即使后端最终展开某个短循环，也不能倒推它在 Python 编译期就是 `range_constexpr`。

## 3.2 三种 `for` 范围

CuTe DSL 4.7.0 的控制流规则是：

| 写法 | 前端行为 | 主要用途 |
|---|---|---|
| `range(...)` | 生成动态/结构化 IR 循环 | 通用运行时循环 |
| `cutlass.range(...)` | 生成 IR，可携带 unroll/pipeline 提示 | 需要编译提示的循环 |
| `cutlass.range_constexpr(...)` | Python 编译期展开 | 很小的固定重复结构 |

这里的 Python built-in `range` 在 DSL 函数 AST 中由前端接管，不能按普通解释器里的 eager loop 来理解。

### `range_constexpr`

```python
for i in cutlass.range_constexpr(4):
    fragment[i] = ...
```

循环体在生成阶段执行四次，`i` 是 Python/Constexpr 值。它适合：

- 固定 rank/mode 遍历；
- 小型 fragment lane 展开；
- 按静态 stage 数生成代码；
- 对固定 tuple/list 做元编程。

代价是代码体积随次数增长。不要用它展开很大的 problem dimension。

4.7.0 的阶段检查很严格：bound 必须直接表现为代码读取阶段可知的 Python 整数。实测中，即使 frozen dataclass 内含固定 `width`，把 `params.width` 直接传给 `range_constexpr` 仍被 `ARG_NON_CONSTANT` 拒绝。稳妥做法是使用直接的 Constexpr 参数、literal 或当前前端能明确证明的静态表达式。

### `range`

```python
acc = cutlass.Int32(0)
for i in range(bound):
    acc = acc + i
```

当 `bound` 是 runtime `Int32` 时，生成真正的循环和 loop-carried `acc`。同一 compiled handle 可以用不同 bound 启动，而不重新 specialization。

### `cutlass.range`

```python
for i in cutlass.range(bound, unroll=2):
    acc = acc + i
```

它和动态 `range` 同属 IR 循环，但允许向后端表达 `unroll` 和 `prefetch_stages` 等意图。提示不是无条件性能保证：最终是否展开、寄存器是否增加、流水化是否合法，仍取决于 target 和循环体。

## 3.3 静态与动态 `if`

静态分支：

```python
if cutlass.const_expr(add_relu):
    value = max(value, 0)
```

只生成选中的分支，适合 dtype、tile、可选 epilogue 和硬件路径选择。static flag 是 specialization key 的一部分。

动态分支：

```python
if index < num_elements:
    dst[index] = src[index]
```

predicate 运行时才知道，前端保留两条路径和汇合点。尾部 mask、数据依赖选择等通常属于这一类。

两者的选择不是“哪种语法更快”，而是信息何时可得。把频繁变化的数据条件做成 Constexpr 会制造 specialization 爆炸；把固定算法开关留成 runtime branch 又会保留不必要的指令和分歧。

## 3.4 动态 `while`

普通 `while predicate` 会生成 IR：

```python
countdown = bound
while countdown > cutlass.Int32(0):
    countdown = countdown - cutlass.Int32(1)
```

循环退出需要可证明会发生，否则 kernel 可能长期运行。GPU 高性能 kernel 通常更偏好边界清晰的 `for`，但 `while` 对 scheduler、work queue、变长迭代仍然重要。

`while cutlass.const_expr(condition)` 则在 Python 生成阶段循环。此时必须保证条件由静态值驱动，并且生成阶段本身能够终止。

## 3.5 Loop-carried value tree

动态循环会把循环前仍需更新的值变成 IR `iter_args`：

```text
iteration k input tree
       ↓
    loop body
       ↓
iteration k+1 input tree
```

输入和输出必须同构：

- 动态叶子数量相同；
- 对应叶子的 MLIR 类型相同；
- tuple/list/dict/dataclass 的嵌套结构相同；
- body 内新建且未作为 loop result 携出的局部值，不能在循环外凭空使用。

这就是第 2 章 value tree 规则在控制流中的具体形式。复杂 accumulator 应使用同结构的新值，或使用 DSL 明确支持的 mutable native struct。

## 3.6 动态控制流当前不支持的写法

CuTe DSL 4.7.0 的动态控制流 body 不支持普通 Python 中常见的一些早退行为：

- `break`；
- `continue`；
- `return` 直接离开动态 region；
- `raise`；
- 作为占位语句的 `pass`。

常见重写方法：

- 把 `break` 改成循环条件中的 active predicate；
- 把 `continue` 改成 `if active:` 包围有效工作；
- 把早退改成 host 侧检查或 kernel 内 predicate；
- 把错误条件写入 status tensor，再由 host 检查。

不要把异常处理当成 device control flow。JIT 编译失败与 GPU runtime status 是不同的错误通道。

## 3.7 元编程的正确粒度

编译期元编程最适合生成结构，而不是搬运 problem data：

```python
if cutlass.const_expr(dtype is cutlass.Float16):
    tiled_mma = make_fp16_mma(...)
else:
    tiled_mma = make_bf16_mma(...)

for stage in cutlass.range_constexpr(num_stages):
    build_stage_view(stage)
```

典型静态项：

- dtype；
- tile/cluster shape；
- pipeline stage 数；
- Copy/MMA Atom；
- 可选 fusion；
- architecture-specific 策略。

典型动态项：

- 数据指针；
- problem shape；
- tail predicate；
- alpha/beta 等频繁变化标量；
- scheduler 当前 tile id。

判断标准是：该值变化时，是否真的值得生成另一份机器代码？

## 3.8 Unroll 与软件流水化提示

`cutlass.range(bound, unroll=N)` 向编译器表达展开意图。展开可能：

- 减少 loop overhead；
- 暴露更多 instruction-level parallelism；
- 同时增加寄存器压力和代码体积。

`prefetch_stages=N` 用于让编译器构造 prologue/main-loop 形式的软件流水化。仓库文档将这项能力标为 experimental，并限定在 SM90+。它不能替代正确的 buffer stage、barrier 和 producer/consumer 协议；这些内容在第 13–15 章与 TMA pipeline 一起验证。

基础章节只验证 `unroll=2` 的语义正确性，不做性能结论。

## 3.9 JIT 函数的返回边界

DSL helper 可以在同一个编译上下文中返回用于继续生成代码的静态对象或受支持表达式；但不能把它等同于“任意 Python 函数返回任意 GPU runtime 对象”。尤其是顶层 compiled host ABI，不应依赖普通 Python 式的动态复合返回。

教程的可执行 kernel 遵循稳定模式：

- 数据结果写入输出 Tensor；
- 编译期 helper 返回 Layout/Atom/配置对象供调用者继续构造；
- runtime status 写入显式 buffer，或通过 CUDA error/synchronization 报告。

## 3.10 AST rewrite 与 tracing 如何协作

控制流能保留，是因为默认 frontend 不只是运行一次 Python trace：

```text
源码 AST
  ↓ 预处理 if/for/while/function boundary
带 callback 的 Python 函数
  ↓ proxy 参数执行 + 运算 tracing
结构化 MLIR（scf.if / scf.for / scf.while）
  ↓ lowering / optimization
PTX → CUBIN → GPU
```

`@cute.jit(preprocess=False)` 是 tracing-only 模式，只适合确认没有动态控制流的 straight-line code。对 kernel 教程默认保留 `preprocess=True`，以免未执行路径从 trace 中消失。

## 3.11 可执行示例

代码：[control_flow.py](../code/03_control_flow/control_flow.py)

同一个 kernel 同时包含：

- `range_constexpr(4)` 的编译期展开；
- `range(bound)` 的动态循环；
- runtime 奇偶分支；
- `Constexpr` 控制的静态 `+100` 分支；
- 动态 `while`；
- `cutlass.range(..., unroll=2)`。

`add_hundred=True` 的 compiled handle 连续接收动态 bound 5 和 6；`False` 建立另一份 specialization。B200 实测：

```text
add_hundred=True, bound=5: result=[6, 130, 0, 20]
add_hundred=True, bound=6: result=[6, 125, 0, 30]
specialization add_hundred=True: mlir_chars=4721, scf.for=2, scf.while=1, scf.if=2
add_hundred=False, bound=5: result=[6, 30, 0, 20]
specialization add_hundred=False: mlir_chars=4641, scf.for=2, scf.while=1, scf.if=2
PASS
```

原始 IR 中存在两个 `scf.for`、一个 `scf.while` 和两个 `scf.if`。`range_constexpr(4)` 本身不产生第三个 `scf.for`，这为“编译期展开”提供了直接证据。

示例显式设置：

```text
CUTE_DSL_KEEP=ir-debug
CUTE_DSL_DUMP_DIR=<示例目录>/generated_ir
```

这是因为 4.7.0 默认不保留 `compiled.__mlir__`。落盘目录属于生成物，不应提交到 Git。

完整运行命令见 [B200 验证日志](validation_log.md)。

## 3.12 如何选择控制流

| 问题 | 选择 |
|---|---|
| 次数很小且决定生成结构 | `range_constexpr` |
| 次数运行时才知道 | `range` |
| 动态循环需要 unroll/pipeline hint | `cutlass.range` |
| 算法开关、dtype 路径固定 | `if const_expr(...)` |
| tail/data predicate 运行时变化 | 普通动态 `if` |
| 需要提前退出 | 重写为 active predicate/循环条件 |

## 3.13 练习

1. 把 `range_constexpr(4)` 改为 `range(4)`，重新统计原始 IR 的 `scf.for`。
2. 删除 `const_expr(add_hundred)`，观察 static flag 是否仍能被前端按预期处理，并解释差异。
3. 把动态循环中的 `Int32` accumulator 改成某条路径返回 `Float32`，核对错误位置。
4. 用 active predicate 重写一个“找到第一个满足条件元素后停止累加”的循环。
5. 比较 `unroll=1/2/4` 的 PTX、寄存器用量和运行时间；在完成规范 benchmark 前不要宣称哪一个更快。

下一章正式进入 CuTe 的核心对象：把 Layout 视为从逻辑坐标到物理索引的函数。
