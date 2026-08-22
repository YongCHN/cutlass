# CuTe DSL 教程：B200 验证日志

本文只记录实际执行过的命令和结果。教程正文中的“已验证”状态应能回溯到这里；没有记录的示例不视为已通过 GPU 验证。

## 验证环境

验证日期：2026-08-21 至 2026-08-22

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

## 第 9 章：执行层级、索引与边界处理

验证代码：

- [`09_execution_hierarchy/execution_hierarchy.py`](../code/09_execution_hierarchy/execution_hierarchy.py)
- [`09_execution_hierarchy/vector_add_masked.py`](../code/09_execution_hierarchy/vector_add_masked.py)

### L1：执行层级与线性索引

执行命令：

```bash
cd /volume/njiang/workspace/sandbox/cutlass
.venv-cutedsl-4.7/bin/python \
  discussion/code/09_execution_hierarchy/execution_hierarchy.py
```

关键输出：

```text
columns=tx,ty,bx,by,lane,warp,warpgroup,thread_linear,block_linear,global_linear
row 0: [0,0,0,0,0,0,0,0,0,0]
row 31: [31,0,0,0,31,0,0,31,0,31]
row 32: [0,1,0,0,0,1,0,32,0,32]
row 127: [31,3,0,0,31,3,0,127,0,127]
row 128: [0,4,0,0,0,4,1,128,0,128]
row 255: [31,7,0,0,31,7,1,255,0,255]
row 256: [0,0,1,0,0,0,0,0,1,256]
row 1023: [31,7,1,1,31,7,1,255,3,1023]
PASS
```

结论：

- 二维 block `(32, 8, 1)` 和二维 grid `(2, 2, 1)` 的 1024 条记录均与独立的 CPU 参考实现逐项一致；
- `lane_idx`、`warp_idx`、逻辑 warpgroup 编号、block 内线性线程号和全局线性线程号的边界点均符合预期；
- 本例中的 warpgroup 是为了讲解而按连续 4 个 warp 分组得到的逻辑编号，不把它冒充为某条硬件 warpgroup 指令的执行语义。

### L2：动态形状、残块和坐标谓词

执行命令：

```bash
cd /volume/njiang/workspace/sandbox/cutlass
.venv-cutedsl-4.7/bin/python \
  discussion/code/09_execution_hierarchy/vector_add_masked.py
```

关键输出：

```text
shape=(1,1), grid=(1,1), tail_columns=1, max_abs_error=0.0e+00
shape=(3,127), grid=(1,3), tail_columns=127, max_abs_error=0.0e+00
shape=(3,128), grid=(1,3), tail_columns=0, max_abs_error=0.0e+00
shape=(5,129), grid=(2,5), tail_columns=1, max_abs_error=0.0e+00
shape=(7,1003), grid=(8,7), tail_columns=107, max_abs_error=0.0e+00
PASS
```

结论：

- 同一个动态形状编译句柄覆盖整块、单元素、差一个元素和多 CTA 残块等情况；
- 数据 Tensor 与 identity Tensor 使用相同的 `local_tile`，再通过 `cute.elem_less` 生成坐标谓词；
- 谓词同时保护输入读取和输出写回；输出缓冲区预填 `NaN`，因此验证也能发现合法位置漏写；
- 五组形状均与 PyTorch 参考结果精确一致。

---

## 第 10 章：TiledCopy 与线程—值布局

验证代码：

- [`10_tiled_copy/tiled_copy_visual.py`](../code/10_tiled_copy/tiled_copy_visual.py)
- [`10_tiled_copy/vectorized_elementwise.py`](../code/10_tiled_copy/vectorized_elementwise.py)

### L0：手算 TV Layout 与覆盖性证明

执行命令：

```bash
cd /volume/njiang/workspace/sandbox/cutlass
.venv-cutedsl-4.7/bin/python \
  discussion/code/10_tiled_copy/tiled_copy_visual.py
```

关键输出：

```text
thr_layout=(2,3):(3,1)
val_layout=(2,2):(2,1)
tiler_mn=(4, 6)
tv_layout=((3,2),(2,2)):((8,2),(4,1))
tiled_copy_tiler=(4:1, 6:1)
thread 0: partition_shape=((1,(2,2)),1,1)
  (0,0)->0->(0,0)
  (0,3)->5->(1,1)
thread 5: partition_shape=((1,(2,2)),1,1)
  (5,0)->18->(2,4)
  (5,3)->23->(3,5)
PASS
```

结论：

- 6 个线程、每线程 4 个值恰好覆盖 `4 x 6` 的逻辑 tile；
- 程序逐项验证了 `thread,value -> logical index -> coordinate` 与 `TiledCopy` 分区结果的一致性；
- 映射满足单射，且定义域、值域大小同为 24，因此完成了“无重叠、无遗漏”的有限覆盖证明。

### L1/L2/L4：显式 128-bit 路径、残块路径与 PTX 证据

执行命令：

```bash
cd /volume/njiang/workspace/sandbox/cutlass
.venv-cutedsl-4.7/bin/python \
  discussion/code/10_tiled_copy/vectorized_elementwise.py
```

关键输出：

```text
[aligned] tiler_mn=(1, 512)
[aligned] tv_layout=(128,4):(4,1)
aligned shape=(4, 1024), max_abs_error=0.0e+00, ptx_vector_loads=2, ptx_vector_stores=1
PTX: ld.global.v2.b64 {%rd8,%rd9}, [%rd5];
PTX: ld.global.v2.b64 {%rd10,%rd11}, [%rd6];
PTX: st.global.v2.b64 [%rd7], {%rd13,%rd12};
masked shape=(1,1), cta_tiles=1, tail_values=1, max_abs_error=0.0e+00
masked shape=(3,511), cta_tiles=3, tail_values=511, max_abs_error=0.0e+00
masked shape=(3,512), cta_tiles=3, tail_values=0, max_abs_error=0.0e+00
masked shape=(5,513), cta_tiles=10, tail_values=1, max_abs_error=0.0e+00
masked shape=(7,1003), cta_tiles=14, tail_values=491, max_abs_error=0.0e+00
PASS
```

结论：

- 两条路径共用 `(128, 4):(4, 1)` 的 TV Layout：128 个线程各拥有连续 4 个 FP32 元素，一个 CTA 处理 512 个元素；
- 对满足完整 tile 和对齐前提的路径，两个输入读取和一个输出写回均使用显式 128-bit Copy Atom；数值结果精确一致；
- 保留下来的 PTX 中出现两条 `ld.global.v2.b64` 和一条 `st.global.v2.b64`。这里 `v2.b64` 与常见的 `v4.b32` 都表示 128-bit 传输，不能只按一种文本拼写判断向量化；
- 通用残块路径使用逐 value 谓词，越界输入 fragment 先填充加法单位元 0，再只对合法位置读写；五组动态形状均精确通过；
- 显式 128-bit 结论只属于满足完整 tile 与对齐契约的路径，不外推到逐 value 谓词的通用残块路径。

---

## 第 11 章：Shared Memory 分配与经典 tiled kernel

验证代码：[`11_shared_memory/tiled_transpose.py`](../code/11_shared_memory/tiled_transpose.py)

### 首次编译问题：动态 shape 不能进入 Python `assert`

初版在 `@cute.jit` 中写了：

```python
assert src.shape[0] == dst.shape[1]
```

编译器正确报告 `PHASE_REQUIRES_CONSTANT`：动态 Tensor extent 是 staged runtime value，不能作为 Python `assert` 的条件。修复方式不是增加 `assume`，而是由 host test harness 构造严格匹配的 transposed destination；kernel 只消费这项 runtime contract。

这与第 2、3 章的 staging 规则一致，也说明 `assume`、compile-time assert 和 runtime shape validation 不能混为一谈。

### L0/L1/L2/L4：SMEM 容量、residue、buffer reuse 与 PTX

执行命令：

```bash
cd /volume/njiang/workspace/sandbox/cutlass
/volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python \
  discussion/code/11_shared_memory/tiled_transpose.py
```

关键输出：

```text
pad=0, shape=(1,1), smem_bytes=4096, grid=(1,1), max_abs_error=0.0e+00
pad=0, shape=(31,33), smem_bytes=4096, grid=(2,1), max_abs_error=0.0e+00
pad=0, shape=(32,32), smem_bytes=4096, grid=(1,1), max_abs_error=0.0e+00
pad=0, shape=(65,67), smem_bytes=4096, grid=(3,2), max_abs_error=0.0e+00
pad=0, shape=(97,129), smem_bytes=4096, grid=(5,2), max_abs_error=0.0e+00
pad=1, shape=(1,1), smem_bytes=4224, grid=(1,1), max_abs_error=0.0e+00
pad=1, shape=(31,33), smem_bytes=4224, grid=(2,1), max_abs_error=0.0e+00
pad=1, shape=(32,32), smem_bytes=4224, grid=(1,1), max_abs_error=0.0e+00
pad=1, shape=(65,67), smem_bytes=4224, grid=(3,2), max_abs_error=0.0e+00
pad=1, shape=(97,129), smem_bytes=4224, grid=(5,2), max_abs_error=0.0e+00
PTX shared_loads=16, shared_stores=32, barriers=8
PTX: ld.shared.b32  %r45, [%r16];
PTX: st.shared.b32  [%r10], %r38;
PTX: bar.sync  0;
PTX: bar.sync  0;
PASS
```

结论：

- compact `(32,32):(32,1)` 和 padded `(32,33):(33,1)` specialization 的自动 launch SMEM 分别为 4096 和 4224 bytes，与 `cosize * sizeof(FP32)` 精确一致；
- 两个 specialization 都复用一个 dynamic-shape compiled handle，覆盖单元素、双向 residue、完整 tile、多 CTA 和每 CTA 第二轮整体越界；
- 输出预填 `NaN`，所有结果与 `src.transpose(0,1)` 精确一致；
- 每个 CTA 复用同一 SMEM buffer 处理两个 row tiles，producer→consumer 和 consumer→next-producer 两条 CTA barrier 均存在；
- PTX 确认 shared load/store 与 `bar.sync`；
- compact/padded bank ownership 完成 L0 映射证明，但本章未进行 L3 benchmark，不把映射差异表述成实测性能提升。

---

## 第 12 章：warp/CTA/cluster 同步原语

验证代码：

- [`12_synchronization/warp_collectives.py`](../code/12_synchronization/warp_collectives.py)
- [`12_synchronization/mbarrier_pingpong.py`](../code/12_synchronization/mbarrier_pingpong.py)
- [`12_synchronization/cluster_dsmem_ring.py`](../code/12_synchronization/cluster_dsmem_ring.py)

### L1/L4：warp collectives 与 named CTA barrier

执行命令：

```bash
cd /volume/njiang/workspace/sandbox/cutlass
/volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python \
  discussion/code/12_synchronization/warp_collectives.py
```

关键输出：

```text
warp PASS: elected_lane=0, ballot=0xff, broadcast=70, redux_sum=496
cta PASS: handoff=0xc0de, any_lane_zero=1, even_count=32
PTX elect: elect.sync  %r7|%p4, -1;
PTX vote: vote.sync.any.pred  %p1, %p7, -1;
PTX shuffle: shfl.sync.idx.b32  %r1, %r6, 0, 31, -1;
PTX redux: redux.sync.add.s32  %r5, %r2, %r10;
PTX warp_barrier: bar.warp.sync  -1;
PTX cta_arrive: barrier.arrive  1, 64;
PTX cta_sync: barrier.sync  1, 64;
PTX cta_red: barrier.red.or.pred  %p2, %r11, %r10, %p1;
PASS
```

结论：

- `elect_sync` 恰好产生一个 winner；本次运行选择 lane 0，但验证只检查“唯一且 marker 匹配”，不把 lane 0 当作 ISA 保证；
- predicate `lane < 8` 的 ANY/ALL/BALLOT 分别得到 1、0、`0xff`；lane 7 的数值 70 成功广播，`0..31` redux sum 为 496；
- 64-thread named CTA barrier 完成 producer/consumer handoff，CTA OR 与 POPC 得到 1 和 32；
- PTX 中八类预期指令族全部存在；当前 B200 toolchain 对 named CTA barrier 使用 `barrier.arrive/sync/red` 拼写，因此检查器按语义 family 同时接受新旧合法拼写。

### L1/L2/L4：双 stage mbarrier ping-pong

执行命令：

```bash
cd /volume/njiang/workspace/sandbox/cutlass
/volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python \
  discussion/code/12_synchronization/mbarrier_pingpong.py
```

关键输出：

```text
num_tiles=1, stages=2, wraps=0, dst0=0.0, max_abs_error=0.0e+00
num_tiles=2, stages=2, wraps=1, dst0=32.0, max_abs_error=0.0e+00
num_tiles=3, stages=2, wraps=1, dst0=96.0, max_abs_error=0.0e+00
num_tiles=5, stages=2, wraps=2, dst0=320.0, max_abs_error=0.0e+00
num_tiles=8, stages=2, wraps=4, dst0=896.0, max_abs_error=0.0e+00
PTX init: mbarrier.init.shared.b64  [%r2], %r18;
PTX init_fence: fence.mbarrier_init.release.cluster;
PTX arrive: mbarrier.arrive.shared.b64  %rd3, [%r5];
PTX wait_parity: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 ...;
PTX invalidate: mbarrier.inval.shared.b64  [%r2];
PASS
```

结论：

- 一个 producer warp、一个 consumer warp、两个 stages 和每 stage 一对 full/empty barrier 形成闭合协议；
- runtime tile count 跨过 0、1、2、4 次 stage ring wrap，覆盖两种 parity；
- producer signal full 和 consumer release empty 前都有显式 warp rendezvous；退出前进行 CTA rendezvous，再 invalidate barrier storage；
- 五组结果与按 tile 规约的 PyTorch reference 精确一致，mbarrier lifecycle 指令族全部在 PTX 中确认。

### L1/L2/L4：CTA cluster、DSMEM 与 `mapa`

在把最初的 2-CTA ring 扩展到 4 CTAs 时，协议审计发现“通知 successor、等待 predecessor、再读取 successor”只在 cluster size 为 2（predecessor 与 successor 相同）时天然闭合；对更大 ring，数值偶然正确并不能证明 successor 写入已发布。最终实现改为“写本地值后通知 predecessor、等待 successor 通知本地 barrier、再读取 successor”，随后重新运行 2/4-CTA 两组测试。

执行命令：

```bash
cd /volume/njiang/workspace/sandbox/cutlass
/volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python \
  discussion/code/12_synchronization/cluster_dsmem_ring.py --cluster-size 2
/volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python \
  discussion/code/12_synchronization/cluster_dsmem_ring.py --cluster-size 4
```

关键输出：

```text
cluster_size=2, dst=[1, 0], PASS
cluster_size=4, dst=[1, 2, 3, 0], PASS
PTX mapa: mapa.shared::cluster.u32  %r13, %r2, %r5;
PTX cluster_arrive: barrier.cluster.arrive.relaxed;
PTX cluster_wait: barrier.cluster.wait;
PTX remote_arrive: mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%rd2];
```

结论：

- 2-CTA 和 4-CTA cluster 都正确读取 successor rank；
- local mbarrier initialization 在第一次 remote access 前经过 init fence 与 cluster rendezvous；
- 每个 CTA 在写完本地值后通知 predecessor，并等待 successor 通知自己的 local barrier，因此 wait 与随后读取 successor 的目标严格匹配；remote arrive 使用 cluster scope，remote data load 使用 `mapa.shared::cluster`；
- 所有 CTA 在 invalidate/exit 前再次 cluster rendezvous，保护 peer DSMEM lifetime；
- PTX 检查按 `mbarrier.arrive` 与 `shared::cluster` 同行出现判断 remote arrive，允许 order/scope qualifier 插入其中；三个第 12 章程序使用彼此隔离的 artifact 子目录，避免交叉命中其他程序的指令证据。

---

## 第 13 章：`cp.async` 与 `ldmatrix/stmatrix`

验证代码：

- [`13_async_copy_and_matrix/cp_async_roundtrip.py`](../code/13_async_copy_and_matrix/cp_async_roundtrip.py)
- [`13_async_copy_and_matrix/ldmatrix_roundtrip.py`](../code/13_async_copy_and_matrix/ldmatrix_roundtrip.py)

### L1/L2/L4：per-thread `cp.async`、copy group 与 CTA publication

执行命令：

```bash
cd /volume/njiang/workspace/sandbox/cutlass
CUDA_VISIBLE_DEVICES=0 \
  /volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python \
  discussion/code/13_async_copy_and_matrix/cp_async_roundtrip.py
```

输出：

```text
n=512, tile=512, max_abs_error=0.0e+00
n=2048, tile=512, max_abs_error=0.0e+00
n=8192, tile=512, max_abs_error=0.0e+00
PTX copy: cp.async.cg.shared.global [%r8], [%rd7], 16;
PTX commit: cp.async.commit_group;
PTX wait: cp.async.wait_group  0;
PTX cta_barrier: barrier.sync  0;
PASS
```

结论：

- 一个动态 compiled handle 覆盖 1、4、16 个 CTA tiles；
- 每个 thread 发出一条 16-byte `cp.async.cg`，copy-group commit/wait 均得到 PTX 证据；
- lane `t` 消费 lane `(t+1) mod 128` 的 SMEM vector，再写回该 vector 的 global slot，因而 CTA barrier 承担真实的 cross-thread publication，而非装饰性同步；
- 三组输出与输入逐位一致；示例明确只接受完整 512-element tiles，不声称覆盖 residue。

### L1/L2/L4：b16 matrix fragment x1/x2/x4

执行命令：

```bash
cd /volume/njiang/workspace/sandbox/cutlass
CUDA_VISIBLE_DEVICES=0 \
  /volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python \
  discussion/code/13_async_copy_and_matrix/ldmatrix_roundtrip.py
```

关键输出：

```text
dtype=fp16, x1, shape=(8, 8), words_per_lane=1: PASS
dtype=fp16, x2, shape=(16, 8), words_per_lane=2: PASS
dtype=fp16, x4, shape=(32, 8), words_per_lane=4: PASS
dtype=bf16, x1, shape=(8, 8), words_per_lane=1: PASS
dtype=bf16, x2, shape=(16, 8), words_per_lane=2: PASS
dtype=bf16, x4, shape=(32, 8), words_per_lane=4: PASS
PTX ldmatrix: ldmatrix.sync.aligned.m8n8.x1.shared.b16 ...
PTX ldmatrix: ldmatrix.sync.aligned.m8n8.x2.shared.b16 ...
PTX ldmatrix: ldmatrix.sync.aligned.m8n8.x4.shared.b16 ...
PTX stmatrix: stmatrix.sync.aligned.m8n8.x1.shared.b16 ...
PTX stmatrix: stmatrix.sync.aligned.m8n8.x2.shared.b16 ...
PTX stmatrix: stmatrix.sync.aligned.m8n8.x4.shared.b16 ...
PASS
```

结论：

- FP16/BF16 与 x1/x2/x4 的六组数值 roundtrip 全部 bit-exact；
- 每个 8×8 b16 fragment 对应每 lane 一个 32-bit carrier word，xN 返回 N words/lane；
- retained PTX 独立确认六种 instruction specialization；
- 示例整体要求 SM90+，因为 `stmatrix` 不是 SM80 指令；`ldmatrix` 的较早架构可用性在正文中单独说明。

---

## 第 14 章：TMA descriptor、partition、proxy 与 multicast

验证代码：

- [`14_tma/tma_copy_v0.py`](../code/14_tma/tma_copy_v0.py)
- [`14_tma/tma_transpose_v1.py`](../code/14_tma/tma_transpose_v1.py)
- [`14_tma/tma_multicast.py`](../code/14_tma/tma_multicast.py)

### L1/L2/L4：`TmaInfo`、`group_modes` 与 `tma_partition`

执行命令：

```bash
cd /volume/njiang/workspace/sandbox/cutlass
CUDA_VISIBLE_DEVICES=0 \
  /volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python \
  discussion/code/14_tma/tma_copy_v0.py
```

输出：

```text
shape=(128, 128), tiles=(1,1): PASS
shape=(256, 128), tiles=(2,1): PASS
shape=(256, 384), tiles=(2,3): PASS
PTX tma_load: cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes...
PTX tma_store: cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group...
PTX expect_tx: mbarrier.expect_tx.relaxed.cta.shared.b64 ...
PTX store_commit: cp.async.bulk.commit_group;
PTX store_wait: cp.async.bulk.wait_group  0;
PASS
```

结论：

- `128×128` FP16 SMEM tile 对应 32768 transaction bytes；
- `local_tile → group_modes → tma_partition → (None,bidx,bidy)` 覆盖 1×1、2×1 和 2×3 tile grids；
- 所有输出与输入逐位一致；
- retained PTX 分别命中 G2S TMA load、S2G TMA store、transaction expectation 和 store bulk-group completion，避免仅凭共同 token 把 load 误认成 store。

### L1/L2/L4：双 SMEM buffer TMA transpose 与 proxy fence

执行命令：

```bash
cd /volume/njiang/workspace/sandbox/cutlass
CUDA_VISIBLE_DEVICES=0 \
  /volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python \
  discussion/code/14_tma/tma_transpose_v1.py
```

输出：

```text
shape=(128, 128) -> (128, 128): PASS
shape=(256, 128) -> (128, 256): PASS
shape=(256, 384) -> (384, 256): PASS
PTX tma_load: cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes...
PTX proxy_fence: fence.proxy.async.shared::cta;
PTX tma_store: cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group...
PTX store_wait: cp.async.bulk.wait_group  0;
PASS
```

开发中第一次运行发生 `unspecified launch failure`。协议审计确认 `elect_one` 是 warp scope：128-thread CTA 的初版让四个 warps 都初始化同一 barrier 并发出 TMA。修复后只有 warp 0 执行 barrier init/load/store issuer path，四个 warps共同完成 SMEM transpose；store commit/wait 保持 warp-uniform。

最终结论：

- 单 tile 方阵与两组非方形多 tile shape 均与 `src.T` 逐位一致；
- tile 内 `(row,col)→(col,row)` 和 global grid `(tile_m,tile_n)→(tile_n,tile_m)` 均已覆盖；
- thread generic-proxy writes 与 TMA async-proxy read 之间的 fence 得到 PTX 证据；
- 修复过程说明“唯一 elected lane”必须同时限定在哪个 warp 中产生。

### L1/L4：two-CTA TMA multicast

执行命令：

```bash
cd /volume/njiang/workspace/sandbox/cutlass
CUDA_VISIBLE_DEVICES=0 \
  /volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python \
  discussion/code/14_tma/tma_multicast.py
```

输出：

```text
cluster=2, tile=(128,64), both CTA copies exact
PTX multicast: cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2 ...
PTX expect_tx: mbarrier.arrive.expect_tx.shared.b64 ...
PTX cluster_arrive: barrier.cluster.arrive.relaxed;
PTX cluster_wait: barrier.cluster.wait;
PASS
```

结论：

- CTA rank 0 只发出一次 TMA multicast，mask `0b11` 把同一 `128×64` FP16 tile 投递给两个 CTAs；
- leader transaction count 为单份 descriptor bytes 的两倍；
- leader 等待两份 completion 后，通过 cluster barrier 把 ready state 发布给 non-leader；
- 两个 CTA 的输出与同一输入 tile 逐位一致，PTX 确认 multicast、`cta_group::2` 与 cluster barrier；
- 本例固定为 SM100 two-CTA routing，不把结论外推到任意 cluster size。

---

## 第 15 章：多级流水线与 warp specialization

验证代码：

- [`15_pipeline/pipeline_software_fill.py`](../code/15_pipeline/pipeline_software_fill.py)
- [`15_pipeline/tma_pipeline_warpspec.py`](../code/15_pipeline/tma_pipeline_warpspec.py)

### L1/L2/L4：2/3/4-stage software-fill pipeline

执行命令：

```bash
cd /volume/njiang/workspace/sandbox/cutlass
CUDA_VISIBLE_DEVICES=0 \
  /volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python \
  discussion/code/15_pipeline/pipeline_software_fill.py
```

关键输出：

```text
stages=2, tiles=1, trace=[0:(s0,p0)]: PASS
stages=2, tiles=5, trace=[0:(s0,p0) 1:(s1,p0) 2:(s0,p1) 3:(s1,p1) 4:(s0,p0)]: PASS
stages=3, tiles=7, trace=[0:(s0,p0) 1:(s1,p0) 2:(s2,p0) 3:(s0,p1) 4:(s1,p1) 5:(s2,p1) 6:(s0,p0)]: PASS
stages=4, tiles=9, trace=[0:(s0,p0) 1:(s1,p0) 2:(s2,p0) 3:(s3,p0) 4:(s0,p1) 5:(s1,p1) 6:(s2,p1) 7:(s3,p1) 8:(s0,p0)]: PASS
PTX init: mbarrier.init.shared.b64 ...
PTX arrive: mbarrier.arrive.shared.b64 ...
PTX wait: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 ...
PTX warp_publish: bar.warp.sync -1;
PTX invalidate: mbarrier.inval.shared.b64 ...
PASS
```

结论：

- 2、3、4-stage 三个 specialization 共验证 14 组 runtime tile count；
- case 覆盖部分 fill、刚好填满、第一次 stage reuse 和 phase 回到 0；
- producer/consumer 分别使用 warp 0/1，每 stage 都有 full/empty mbarrier；
- commit/release 前均有 full-mask warp publication，退出前 CTA drain 后才 invalidate；
- 全部输出与输入逐位一致，五类 mbarrier/warp 指令族得到 PTX 证据。

### L1/L2/L4：warp-specialized TMA pipeline

执行命令：

```bash
cd /volume/njiang/workspace/sandbox/cutlass
CUDA_VISIBLE_DEVICES=0 \
  /volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python \
  discussion/code/15_pipeline/tma_pipeline_warpspec.py
```

输出：

```text
stages=2, tiles=1, shape=(128, 64), wraps=0: PASS
stages=2, tiles=2, shape=(128, 128), wraps=1: PASS
stages=2, tiles=3, shape=(128, 192), wraps=1: PASS
stages=2, tiles=5, shape=(128, 320), wraps=2: PASS
stages=3, tiles=1, shape=(128, 64), wraps=0: PASS
stages=3, tiles=3, shape=(128, 192), wraps=1: PASS
stages=3, tiles=4, shape=(128, 256), wraps=1: PASS
stages=3, tiles=7, shape=(128, 448), wraps=2: PASS
PTX tma_load: cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes ...
PTX expect_tx: mbarrier.arrive.expect_tx.shared.b64 ...
PTX wait: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 ...
PTX release: mbarrier.arrive.shared.b64 ...
PTX invalidate: mbarrier.inval.shared.b64 ...
PASS
```

结论：

- warp 0 执行 TMA producer prologue/steady loop，warp 1 读取 swizzled SMEM 并 release；
- 2/3-stage 共八组 shape 覆盖 tile count 小于、等于和大于 stage depth，以及两次 wrap；
- TMA transaction completion、producer acquire、consumer release 和 tail lifetime 形成闭环；
- 所有 FP16 输出 bit-exact，retained PTX 确认 TMA、expect-tx 和 mbarrier state transitions；
- 本章不根据指令存在声称真实 latency overlap，L3 timeline/性能留待统一 benchmark。

---

## 第 16 章：分层规约与 Online Softmax

验证代码：

- [`16_reduction_and_softmax/reduction_ladder.py`](../code/16_reduction_and_softmax/reduction_ladder.py)
- [`16_reduction_and_softmax/online_softmax.py`](../code/16_reduction_and_softmax/online_softmax.py)

### L1/L2/L4：thread-vector → warp → CTA reduction ladder

执行命令：

```bash
cd /volume/njiang/workspace/sandbox/cutlass
CUDA_VISIBLE_DEVICES=0 \
  /volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python \
  discussion/code/16_reduction_and_softmax/reduction_ladder.py
```

输出：

```text
rows=1, width=128, items/thread=1: PASS
rows=7, width=128, items/thread=1: PASS
rows=33, width=128, items/thread=1: PASS
rows=1, width=512, items/thread=4: PASS
rows=7, width=512, items/thread=4: PASS
rows=33, width=512, items/thread=4: PASS
rows=1, width=1024, items/thread=8: PASS
rows=7, width=1024, items/thread=8: PASS
rows=33, width=1024, items/thread=8: PASS
PTX vector_load: ld.global.v8.b32 ...
PTX shuffle: shfl.sync.bfly.b32 ...
PTX shared_store: st.shared.b32 ...
PTX cta_barrier: barrier.sync 0;
PASS
```

结论：

- 每线程先规约 1/4/8 个连续 FP32，warp butterfly 后由四个 warp leaders 写 SMEM partials，warp 0 完成 CTA merge；
- sum/max 同时验证，使用小整数值域确保 FP32 sum 与 reference bit-exact；
- 三个 row width 和三个 row count 共九组均通过；
- PTX 确认 x8 vector load、shuffle、shared store 和 CTA publication。

### L1/L2/L4：ragged-mask online Softmax

执行命令：

```bash
cd /volume/njiang/workspace/sandbox/cutlass
CUDA_VISIBLE_DEVICES=0 \
  /volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python \
  discussion/code/16_reduction_and_softmax/online_softmax.py
```

输出：

```text
shape=(1,1), lengths=[1], max_error=0.000e+00: PASS
shape=(5,31), lengths=[1, 8, 16, 24, 31], max_error=5.960e-08: PASS
shape=(7,128), lengths=[1, 22, 43, 64, 86, 107, 128], max_error=1.192e-07: PASS
shape=(9,257), lengths=[1, 33, 65, 97, 129, 161, 193, 225, 257], max_error=1.192e-07: PASS
shape=(4,1000), lengths=[1, 334, 667, 1000], max_error=2.980e-08: PASS
PTX shuffle: shfl.sync.bfly.b32 ...
PTX exp: ex2.approx.ftz.f32 ...
PTX shared_store: st.shared.b32 ...
PTX cta_barrier: barrier.sync 0;
PASS
```

结论：

- thread-local online scan、warp `(m,l)` merge、SMEM cross-warp merge 和 second-pass normalize 形成完整两遍算法；
- runtime lengths 覆盖 sub-warp、warp residue、CTA width、多轮动态 loop 和 1000-column row；
- invalid positions作为 identity operands，仍参与所有 collectives，输出固定为 0；
- probability、row sum 和内部 `(row_max,row_exp_sum)` stats 均与 PyTorch reference 对照；
- fast exponential lower 为 `ex2.approx.ftz.f32`，五组最大逐元素误差不超过 `1.192e-7`，在记录的 tolerance 内通过。

---

## 后续记录约定

每个新示例至少记录：

1. 教程章节和代码路径；
2. 完整运行命令；
3. GPU、CuTe DSL 版本和关键参数；
4. reference、误差阈值与边界 shape；
5. 验证等级 L0–L4；
6. 失败时保留错误摘要和修复原因。
