# CuTe DSL 教程：B200 验证日志

本文只记录实际执行过的命令和结果。教程正文中的“已验证”状态应能回溯到这里；没有记录的示例不视为已通过 GPU 验证。

## 验证环境

验证日期：2026-08-21

| 项目 | 值 |
|---|---|
| 主机工作目录 | `/volume/njiang/workspace/sandbox` |
| GPU | NVIDIA B200（SM100，compute capability 10.0） |
| Driver | 590.48.01 |
| Python | 3.12.3 |
| CuTe DSL | 4.7.0 |
| PyTorch | 2.9.1+cu129 |
| PyTorch CUDA | 12.9 |
| 教程解释器 | `/volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python` |

CuTe DSL 4.7.0 通过 Mac 下载 Linux x86_64 wheels、复制到远端、再以 `--no-index` 离线安装。系统原有的 4.4.2 环境未被修改。

## L0：环境探针

代码：[env_probe.py](../code/00_environment/env_probe.py)

```bash
cd /volume/njiang/workspace/sandbox/cutedsl_tutorial/code/00_environment
/volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python env_probe.py
```

关键输出：

```text
python=3.12.3
platform=Linux x86_64
cutlass=4.7.0
cutlass_path=/volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/lib/python3.12/site-packages/nvidia_cutlass_dsl/dsl_packages/cutlass/__init__.py
torch=2.9.1+cu129
torch_cuda=12.9
cuda_available=True
gpu=NVIDIA B200
compute_capability=10.0
runnable_tracks=language,layout,tensor,tiled-copy,sm80-warp-mma,sm90-tma-wgmma,sm100-tcgen05-tmem
PASS
```

结论：版本、导入路径、CUDA 可用性和目标架构均符合教程基线，L0 通过。

## L1：最小 kernel

代码：[hello_world.py](../code/01_execution_model/hello_world.py)

```bash
cd /volume/njiang/workspace/sandbox/cutedsl_tutorial/code/01_execution_model
CUDA_VISIBLE_DEVICES=0 /volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python hello_world.py
```

输出：

```text
GPU says: tidx=0, meta_value=7, runtime_value=11
PASS
```

结论：`@cute.jit` host function、`@cute.kernel` launch、动态参数传递和 device `printf` 均正常，L1 通过。

## L1：编译期与运行期

代码：[compile_time_vs_runtime.py](../code/01_execution_model/compile_time_vs_runtime.py)

```bash
cd /volume/njiang/workspace/sandbox/cutedsl_tutorial/code/01_execution_model
CUDA_VISIBLE_DEVICES=0 /volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python compile_time_vs_runtime.py
```

关键结果：

- `static_value=7` 建立一次 compiled handle。
- 动态值 11 和 13 复用该 handle，GPU 分别得到 18 和 20。
- `static_value=8` 建立另一份 specialization，GPU 得到 21。
- 程序输出 `PASS`，退出码为 0。

结论：静态参数 specialization 与动态运行时参数的边界符合正文模型，L1 通过。

## L2 边界样例：标量 vector add

代码：[vector_add.py](../code/01_vector_add/vector_add.py)

```bash
cd /volume/njiang/workspace/sandbox/cutedsl_tutorial/code/01_vector_add
CUDA_VISIBLE_DEVICES=0 /volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python vector_add.py --num-elements 1000003
```

输出：

```text
Compiling CuTe DSL vector-add kernel...
Launching with 1000003 elements...
GPU: NVIDIA B200
max_abs_error: 0.000e+00
first five results: [-0.05659037083387375, 0.5634089708328247, -0.4320492446422577, -0.2658909559249878, -0.050401270389556885]
PASS
```

`1,000,003` 不是 256 的整数倍，因此该次运行覆盖了最后一个 block 的越界保护。结果与 PyTorch reference 完全一致。

这里标记为“L2 边界样例”，表示非整 block shape 已覆盖；尚未声称完成多 dtype、空 tensor、超大 shape 或性能验证。

## L1：标量类型与 Constexpr specialization

代码：[types_and_constexpr.py](../code/02_types/types_and_constexpr.py)

```bash
cd /volume/njiang/workspace/sandbox/cutedsl_tutorial/code/02_types
CUDA_VISIBLE_DEVICES=0 /volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python types_and_constexpr.py
```

输出：

```text
square_result=False: i32=[16, 1], f32=[8.0]
square_result=True: i32=[16, 1], f32=[64.0]
PASS
```

结论：`Int32` 算术、`Int32 → Float32` 转换、动态 predicate 和两个静态 specialization 的数值结果均正确，L1 通过。

## L1：Frozen dataclass 与 `cute.struct`

代码：[struct_and_dataclass.py](../code/02_types/struct_and_dataclass.py)

```bash
cd /volume/njiang/workspace/sandbox/cutedsl_tutorial/code/02_types
CUDA_VISIBLE_DEVICES=0 /volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python struct_and_dataclass.py
```

输出：

```text
result=[1.25, 2.75, 4.25, 5.75, 14.0]
PASS
```

结论：frozen dataclass 的静态结构/动态叶子传递、带 alignment 的 `MemRange`、SMEM struct 标量读写均正确，L1 通过。

开发过程中确认两个 4.7.0 约束：

- `@cute.struct` 需要类创建时的具体注解，不能让 postponed annotations 把字段类型变成字符串。
- `cute.struct` 标量字段是 memory-backed wrapper；读取数值使用其 pointer load，而不是直接当作 DSL scalar 写入 Tensor。

## L0 负向测试：阶段与类型错误

代码：[expected_compile_errors.py](../code/02_types/expected_compile_errors.py)

```bash
cd /volume/njiang/workspace/sandbox/cutedsl_tutorial/code/02_types
CUDA_VISIBLE_DEVICES=0 /volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python expected_compile_errors.py
```

关键输出：

```text
dynamic value passed to const_expr: expected failure: DSLUserCodeError: error[PHASE_REQUIRES_CONSTANT]
dynamic Python-list index: expected failure: DSLUserCodeError: error[PHASE_DYNAMIC_INDEX]
dynamic branch changes value type: expected failure: DSLUserCodeError: error[TYPE_UNSTABLE_JOIN]
PASS (all failures were expected)
```

结论：三项非法程序均在编译期被拒绝，负向契约 L0 通过。测试不绑定完整诊断文本，以免小版本改进错误措辞时造成脆弱测试。

## L1/L4：控制流与原始 MLIR

代码：[control_flow.py](../code/03_control_flow/control_flow.py)

```bash
cd /volume/njiang/workspace/sandbox/cutedsl_tutorial/code/03_control_flow
CUDA_VISIBLE_DEVICES=0 /volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python control_flow.py
```

输出：

```text
add_hundred=True, bound=5: result=[6, 130, 0, 20]
add_hundred=True, bound=6: result=[6, 125, 0, 30]
specialization add_hundred=True: mlir_chars=4721, scf.for=2, scf.while=1, scf.if=2
add_hundred=False, bound=5: result=[6, 30, 0, 20]
specialization add_hundred=False: mlir_chars=4641, scf.for=2, scf.while=1, scf.if=2
PASS
```

结论：

- 同一个 compiled handle 正确处理动态 bound 5 和 6，数值 reference L1 通过。
- static `add_hundred` 产生 True/False 两份 specialization。
- `CUTE_DSL_KEEP=ir-debug` 保存的原始 IR 含两个 `scf.for`、一个 `scf.while` 和两个 `scf.if`。
- `range_constexpr(4)` 没有产生额外的 `scf.for`，为编译期展开提供 L4 结构证据。

## L0/L1：Layout 基础与动态 Layout

代码：[layout_basics.py](../code/04_layout/layout_basics.py)

```bash
cd /volume/njiang/workspace/sandbox/cutedsl_tutorial/code/04_layout
CUDA_VISIBLE_DEVICES=0 /volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python layout_basics.py
```

关键输出：

```text
left_major=(3,4):(1,3)
row_major=(3,4):(4,1)
padded=(3,4):(8,1)
broadcast_rows=(3,4):(0,1)
hierarchical: rank=2, depth=2, size=24, cosize=24
dynamic case shape=(3, 4), stride=(8, 1): [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19]
dynamic case shape=(2, 5), stride=(7, 1): [0, 1, 2, 3, 4, 7, 8, 9, 10, 11]
PASS
```

结论：静态 Layout 构造、层级属性、映射表与 identity coordinate 的 L0 断言通过；同一 compiled handle 在 GPU 上正确处理两组动态 shape/leading dimension，L1 通过。

## L0：Layout 代数

代码：[layout_algebra_lab.py](../code/05_layout_algebra/layout_algebra_lab.py)

```bash
cd /volume/njiang/workspace/sandbox/cutedsl_tutorial/code/05_layout_algebra
/volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python layout_algebra_lab.py
```

关键输出：

```text
coalesce: ((2,(3,4)),(3,2),1):((4,(8,24)),(2,6),12) -> (24,6):(4,2)
composition: (6,2):(8,2) o (4,3):(3,1) -> ((2,2),3):((24,2),8)
logical_divide=((2,4),(3,2)):((6,12),(1,3))
zipped_divide=((2,3),(4,2)):((6,1),(12,3))
tiled_divide=((2,3),4,2):((6,1),12,3)
flat_divide=(2,3,4,2):(6,1,12,3)
complement(4:1, 24)=6:4
right_inverse=(3,2):(2,1)
left_inverse=(3,2):(2,1)
tile_to_shape((2,2):(2,1), (4,6))=((2,2),(2,3)):((2,12),(1,4))
PASS
```

结论：`coalesce` 映射保持、composition 复合等式、divide/product size、complement 覆盖、inverse 左右复合和结构变换断言全部通过。该示例只做生成/编译阶段结构验证，记为 L0。

## L0/L1：Swizzle 与 ComposedLayout

代码：[swizzle_visualizer.py](../code/06_swizzle/swizzle_visualizer.py)、[smem_bank_probe.py](../code/06_swizzle/smem_bank_probe.py)

```bash
cd /volume/njiang/workspace/sandbox/cutedsl_tutorial/code/06_swizzle
/volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python swizzle_visualizer.py
CUDA_VISIBLE_DEVICES=0 /volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python smem_bank_probe.py
```

可视化关键输出：

```text
outer=32:32
swizzle=S<5,0,5>
composed=S<5,0,5> o 0 o 32:32
0: 0 -> 0 -> 0
1: 32 -> 33 -> 1
31: 992 -> 1023 -> 31
PASS
```

SMEM 往返输出：

```text
dst=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0]
PASS
```

结论：ComposedLayout 映射、bank 静态模型和 XOR involution 的 L0 断言通过；32 个 thread 使用 canonical `s128b` 成对变换物理 SMEM pointer 后精确写回 lane id，L1 通过。

开发中确认：低层逐 lane `apply_swizzle(...).store()` 不应与按 CuTe Tensor/128-bit vector remap 的 `load_swizzled()` 混搭。修正后的探针让 physical-pointer 写读保持同一抽象层。本节不声称测得 bank-conflict 性能。

## L1：Tensor tile 与 coordinate Tensor

代码：[tensor_views.py](../code/07_tensor/tensor_views.py)

```bash
cd /volume/njiang/workspace/sandbox/cutedsl_tutorial/code/07_tensor
CUDA_VISIBLE_DEVICES=0 /volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python tensor_views.py
```

输出：

```text
dst=
tensor([[ 0.,  2.,  4.,  6.,  8., 10.],
        [16., 18., 20., 22., 24., 26.],
        [32., 34., 36., 38., 40., 42.],
        [48., 50., 52., 54., 56., 58.]])
PASS
```

结论：四个 CTA 对 data Tensor 和 identity Tensor 应用相同 `(2,3)` `local_tile`，由 coordinate Tensor 恢复的全局 row/column 与 PyTorch reference 精确一致，L1 通过。

## L1：动态 Tensor Layout 的 compiled-handle 复用

代码：[dynamic_layout.py](../code/07_tensor/dynamic_layout.py)

```bash
cd /volume/njiang/workspace/sandbox/cutedsl_tutorial/code/07_tensor
CUDA_VISIBLE_DEVICES=0 /volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python dynamic_layout.py
```

输出：

```text
n=17, first=-6.0, last=26.0
n=257, first=-6.0, last=506.0
n=1003, first=-6.0, last=1998.0
PASS
```

结论：用长度 17 的动态一维 Layout 建立一次 compiled handle 后，同一 handle 正确执行长度 17、257、1003，并与 `src * 2` 精确一致。该 L1 验证覆盖动态 extent 和非整 block 尾部，不覆盖非连续动态 stride。

## L1：TensorSSA 寄存器数据流

代码：[tensorssa_ops.py](../code/08_tensorssa/tensorssa_ops.py)

```bash
cd /volume/njiang/workspace/sandbox/cutedsl_tutorial/code/08_tensorssa
CUDA_VISIBLE_DEVICES=0 /volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python tensorssa_ops.py
```

输出：

```text
dst=
tensor([[12.5000, 15.0000, 17.5000],
        [28.5000, 31.0000, 33.5000]])
row_sums=[45.0, 93.0]
selected_column=[15.0, 31.0]
PASS
```

结论：memory Tensor load、RMEM fragment store/load、TensorSSA elementwise/broadcast、`(None,1)` 规约 profile 和 `(None,1)` slice 均与 PyTorch reference 精确一致，L1 通过。该示例为单线程静态 shape，不覆盖跨线程规约。

## 后续记录约定

每个新示例至少记录：

1. 教程章节和代码路径；
2. 完整运行命令；
3. GPU、CuTe DSL 版本和关键参数；
4. reference、误差阈值与边界 shape；
5. 验证等级 L0–L4；
6. 失败时保留错误摘要和修复原因。
