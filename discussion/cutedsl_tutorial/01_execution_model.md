# 第 1 章：最小程序与 CuTe DSL 执行模型

标签：`[通用]`
先修：第 0 章
状态：正文和两个示例完成；已在 B200 / CuTe DSL 4.7.0 上通过 L1 验证。

## 1.1 本章要解决的困惑

CuTe DSL 代码看起来像 Python，但同一函数里可能同时出现 Python `print`、动态 GPU 值、静态配置、kernel launch 和普通 PyTorch 测试。如果不先区分执行阶段，后面几乎一定会误解控制流、Layout 和 pipeline。

我们先建立三层调用模型：

```text
普通 Python main()
  ├─ 创建 tensor、reference、测试参数
  └─ 调用 @cute.jit host function
         ├─ 编译期构造 Layout/Atom/launch 配置
         └─ launch @cute.kernel
                 └─ GPU runtime：thread/block 指令和数据访问
```

## 1.2 `@cute.kernel`

`@cute.kernel` 定义 GPU kernel。它负责：

- 读取 thread/block/cluster 坐标；
- 操作传入的 Tensor、Pointer 和动态标量；
- 使用 copy、MMA、barrier 等 device operation；
- 由 host function 指定 grid、block、cluster、SMEM 和 stream 后启动。

kernel 本身不决定输入 tensor 怎样创建，也不负责 reference 验证。

最小例子中的 kernel：

```python
@cute.kernel
def hello_world_kernel(meta_value: cutlass.Constexpr,
                       runtime_value: cutlass.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    if tidx == 0:
        cute.printf(
            "GPU says: tidx={}, meta_value={}, runtime_value={}",
            tidx, meta_value, runtime_value,
        )
```

这里 `tidx` 和 `runtime_value` 在 GPU 执行时才有具体值；`meta_value` 在 specialization 时已经确定，会被嵌入生成代码。

## 1.3 `@cute.jit`

`@cute.jit` 函数是可编译、可组合的 DSL 函数。顶层 host JIT 常用于：

- 根据 dtype/shape 构造 Layout；
- 创建 CopyAtom/TiledCopy/TiledMMA；
- 计算 grid、cluster、dynamic SMEM；
- 调用一个或多个 kernel launch；
- 把静态配置传给 kernel。

```python
@cute.jit
def launch_hello_world(meta_value: cutlass.Constexpr,
                       runtime_value: cutlass.Int32):
    hello_world_kernel(meta_value, runtime_value).launch(
        grid=(1, 1, 1),
        block=(1, 1, 1),
    )
```

把 launch 单独放在 `@cute.jit` 中非常重要：复杂教程会在这里完成大量编译期构造，而 kernel 只接收已经准备好的对象。

## 1.4 普通 Python 层

普通 Python 负责运行环境和测试：

```python
def main():
    cutlass.cuda.initialize_cuda_context()
    launch_hello_world(7, 11)
    cutlass.cuda.stream_sync(cutlass.cuda.default_stream())
    print("PASS")
```

它可以自由使用 argparse、PyTorch、文件系统和测试框架。不要把这些普通 Python 行为误认为 device code。

## 1.5 静态参数与动态参数

CuTe DSL 默认把 JIT 参数视为动态参数；显式标注 `cutlass.Constexpr` 后，它变成静态参数。

| 属性 | 静态参数 | 动态参数 |
|---|---|---|
| 典型标注 | `cutlass.Constexpr` | `cutlass.Int32`、Tensor 等 |
| 值何时已知 | specialization/编译期 | 调用/运行期 |
| 是否进入生成函数 ABI | 否 | 是 |
| 改变值的效果 | 通常产生新 specialization | 可复用同一 specialization |
| 适合内容 | tile shape、dtype、算法开关 | 数据指针、长度、alpha/beta |

静态参数类似 C++ template parameter，但由 Python 值提供。它让编译器删除无用分支、展开循环并选择不同硬件 op；代价是静态值变化可能触发重新编译。

## 1.6 代码为什么“执行两次”

更准确地说，同一段源码参与两个阶段：

### 编译阶段

1. 前端处理 Python AST，把受支持的 `if/for/while` 变成结构化 IR。
2. 函数使用 proxy 参数运行，重载操作记录计算。
3. 静态 Python 对象直接参与元编程。
4. 产生并 lower IR，最终得到 GPU code。

### GPU 运行阶段

1. driver 按 launch configuration 启动 kernel。
2. thread/block index 获得实际值。
3. dynamic 参数和 tensor 数据被读取。
4. `cute.printf` 等 device operation 真正执行。

因此，源码中的普通 Python `print` 会看到动态 proxy，而不是 GPU 中某个 thread 的实际值。

## 1.7 `print` 与 `cute.printf`

示例 [compile_time_vs_runtime.py](../code/01_execution_model/compile_time_vs_runtime.py) 同时使用两者。

```python
print(f"[compile:kernel] dynamic_value={dynamic_value}")
```

这是 Python 编译阶段输出。动态值通常显示为未知值或 proxy 表示。

```python
cute.printf(
    "[runtime:gpu] dynamic_value={}",
    dynamic_value,
)
```

这会生成运行时代码，看到的才是本次调用传入的实际值。

实际调试原则：

- 想检查 Layout、类型、静态 shape：用 Python `print`。
- 想检查 thread index、动态标量或 device 数据：用 `cute.printf`。
- 大规模 kernel 中限制打印 thread/block，否则输出量会迅速失控。

## 1.8 JIT specialization 与缓存

`compile_time_vs_runtime.py` 先显式编译，再复用 compiled handle：

```python
compiled_static_7 = cute.compile(launch_phase_demo, 7, 11)
compiled_static_7(11)
compiled_static_7(13)

compiled_static_8 = cute.compile(launch_phase_demo, 8, 13)
compiled_static_8(13)
```

预期语义：

1. 第一次 `cute.compile` 为 `static_value=7` 建立 specialization。
2. static 参数不进入运行时 ABI，所以 `compiled_static_7` 调用时只传 `dynamic_value`。
3. 两次动态调用复用同一个 compiled handle，不重新 tracing。
4. `static_value` 改为 8 时，再显式编译另一份 specialization。

进程间或磁盘缓存是否命中受版本和缓存设置影响；但生成函数签名中 static 参数被消去、dynamic 参数保留这一语义不变。教程在需要精确控制编译与测量边界时，统一保留 compiled handle，而不是把装饰后的函数直接调用多次。

实测时，`static_value=7` 的 tracing 只在建立 compiled handle 时出现；随后以动态值 11 和 13 调用时只出现 GPU 运行期输出。改为 `static_value=8` 后出现第二次 tracing：

```text
compile specialization: static_value=7
[compile:jit] static_value=7
[compile:jit] dynamic_value=?
[compile:kernel] static_value=7
[compile:kernel] dynamic_value=?
first runtime call: dynamic_value=11
[runtime:gpu] static_value=7, dynamic_value=11, sum=18
second runtime call: reuse specialization, dynamic_value=13
[runtime:gpu] static_value=7, dynamic_value=13, sum=20
compile another specialization: static_value=8
[compile:jit] static_value=8
[compile:jit] dynamic_value=?
[compile:kernel] static_value=8
[compile:kernel] dynamic_value=?
third runtime call: dynamic_value=13
[runtime:gpu] static_value=8, dynamic_value=13, sum=21
PASS
```

这段输出直接验证了“静态值决定 specialization、动态值复用运行时 ABI”的模型。完整运行信息见 [B200 验证日志](validation_log.md)。

## 1.9 launch configuration

最小例子使用：

```python
.launch(grid=(1, 1, 1), block=(1, 1, 1))
```

后面会逐步加入：

- `cluster=(...)`：thread-block cluster；
- `smem=...`：dynamic shared memory；
- `stream=...`：框架或 CUDA stream；
- programmatic launch/event 属性。

grid 和 block 描述执行资源，Layout 描述 thread/value 与逻辑数据之间的映射。两者有关联，但不是同一个东西。

## 1.10 最小程序逐层追踪

完整代码：[hello_world.py](../code/01_execution_model/hello_world.py)

运行：

```bash
cd /volume/njiang/workspace/sandbox/cutedsl_tutorial/code/01_execution_model
/volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python hello_world.py
```

调用链：

```text
main
  → initialize_cuda_context
  → launch_hello_world(7, 11)
      → specialize meta_value=7
      → runtime ABI carries runtime_value
      → hello_world_kernel(...).launch(1 block, 1 thread)
          → GPU thread 0 executes cute.printf
  → stream_sync
  → Python prints PASS
```

`stream_sync` 使 device 输出和 kernel failure 在程序结束前可见。后续使用 PyTorch tensor 时，也必须明确处理 stream 和同步边界。

2026-08-21 实测输出：

```text
GPU says: tidx=0, meta_value=7, runtime_value=11
PASS
```

## 1.11 常见错误

### 把 Python 容器当成运行时可变对象

Python tuple/list 很适合编译期配置，但不能在 device dynamic branch 中任意改变结构。

### 用 Python `print` 期待 tensor 数值

编译阶段只有 proxy；要打印 device 值必须生成 device print，或把结果写回 tensor 后在 Python 读取。

### 静态参数过多

把频繁变化的长度、alpha 等误标为 Constexpr 会造成 specialization 爆炸和重复编译。

### 忘记同步

kernel launch 通常异步。测试输出、计时和错误报告都需要正确同步，不能用普通 Python 时间直接包围异步 launch 后就下结论。

### 源码与 wheel 不匹配

基础例子可能碰巧运行，高级 TMA/pipeline 代码却会在很后面才失败。第 0 章的版本检查不能省略。

## 1.12 深挖问题回顾

1. **解决什么问题？** 用 Python 做高效 GPU kernel 的元编程与结构表达。
2. **输入输出是什么？** 静态配置和动态 ABI 参数进入 host JIT，再进入 kernel。
3. **编译期/运行期如何分？** Constexpr/Python 对象属于编译期；thread index、Tensor 数据和 dynamic scalar 属于运行期。
4. **哪些 thread 参与？** 由 launch 的 grid/block 和 kernel 分支共同决定。
5. **同步在哪里？** 最小例只在 host 末尾 stream sync；后续会加入 device barrier/pipeline。
6. **如何验证？** 观察 GPU print、进程返回码和明确的 `PASS`；数值 kernel 还必须与 reference 比较。

## 1.13 练习

1. 把 block 改成 4 threads，只让偶数 thread 打印。
2. 保持 static value 不变，连续传入三个 dynamic value，观察输出。
3. 改变 static value，比较首次调用延迟。
4. 删除 `stream_sync`，观察 device print 的顺序为何不再是可靠契约。
5. 故意把字符串或 Float32 传给 `runtime_value`，阅读类型错误。

下一章将系统整理 DSL 标量类型、Constexpr、struct/dataclass 和 dynamic value tree。
