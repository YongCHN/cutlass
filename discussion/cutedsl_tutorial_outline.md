# CuTe DSL 教程总纲与知识点清单

> 基线：CUTLASS `4.7.0`，提交 `7107b055`。大纲来自对 `examples/python/CuTeDSL` 全部示例材料的审阅。后续正文与可执行代码均放在 `discussion` 下。

## 1. 教程定位

### 1.1 目标读者

- 会写 Python，理解基本 CUDA 术语（thread、warp、block、shared memory）。
- 想从普通 CUDA kernel 进入 CuTe 的 Layout/Tensor 抽象。
- 需要开发、修改或读懂 Ampere/Hopper/Blackwell 上的高性能 kernel。
- 不要求熟悉 CuTe C++ 模板，但有 CUTLASS/CUDA 经验会更容易进入架构篇。

### 1.2 完成教程后的能力

读者应能：

1. 正确区分 Python 元编程、DSL 编译期和 GPU 运行期。
2. 手算并用程序验证 Layout、Tensor、tile、slice、partition 的结果。
3. 写出带边界处理、向量化、SMEM staging 和正确同步的 kernel。
4. 根据目标架构构造 TiledCopy/TiledMMA，并说清每条数据路径。
5. 从 naive GEMM 逐步演化到 TMA、warp-specialized、persistent GEMM。
6. 对 kernel 做 reference 校验、benchmark、autotune、IR/PTX 检查和性能归因。
7. 把 kernel 接入 PyTorch/JAX/TVM FFI，或导出为 AOT/C++ 可调用产物。
8. 能沿统一框架阅读 FMHA、MoE、block-scaled GEMM 和分布式示例。

### 1.3 不做什么

- 不把 API reference 原样翻译成中文。
- 不承诺一个 GEMM 配置对所有 shape、dtype 和 GPU 都最优。
- 不把 experimental Primitives/Task Scheduling 描述为稳定接口。
- 不用复杂生产 kernel 代替最小可运行例子。
- 不混淆 CUTLASS Python/CuTe DSL 与旧版 CUTLASS Python 高层封装。

## 2. 教程的可复现约定

### 2.1 章节标签

每章和每个代码例子使用以下标签：

- `[通用]`：架构无关概念或所有 CuTe DSL 支持架构通用。
- `[SM80+]`：Ampere/Ada warp-level MMA 或 `cp.async` 路线。
- `[SM90+]`：Hopper TMA/WGMMA/cluster 路线。
- `[SM100+]`：Blackwell tcgen05/TMEM/UMMA 路线。
- `[SM103]`、`[SM120]`：仅对应架构特性。
- `[Experimental]`：Primitives、Task Scheduling、`cute_ext` 等接口。

### 2.2 每个可执行例子的固定结构

后续代码统一放在 `discussion/code/<chapter>/`，每个例子至少包含：

1. 文件头：目标架构、最低 CUDA/driver、dtype、shape/alignment 限制。
2. `@cute.kernel`：只放 device 逻辑。
3. `@cute.jit` host：构造参数、配置 launch、传入 stream。
4. `run()`：生成输入、编译/缓存、执行。
5. `reference()` 与 `verify()`：默认执行正确性检查。
6. 可选 `benchmark()`：warmup、重复次数、同步点、GB/s 或 TFLOP/s。
7. CLI：`--help` 可查看参数；默认 shape 要能快速完成。
8. 预期输出：成功标志、最大误差、必要时打印 Layout/IR。

### 2.3 验证层级

- L0：语法/导入、静态 Layout 断言，不要求 GPU。
- L1：小 shape 数值正确性。
- L2：随机 shape、边界和多 dtype 参数化测试。
- L3：benchmark 与性能回归，记录 GPU、频率、CUDA、提交号。
- L4：IR/PTX/SASS 或 profiler 证据，验证优化确实生成并生效。

当前工作区是 Darwin/arm64，而仓库文档声明 CuTe DSL 支持 Linux x86_64/aarch64；因此本阶段只制定运行规范。后续 GPU 例子的“已执行”状态必须在受支持的 Linux + NVIDIA GPU 环境中确认，不能以静态阅读替代。

## 3. 正文大纲

## 第一篇：建立正确的语言心智模型

### 第 0 章：版本、环境与阅读地图 `[通用]`

**要回答的问题**：这套教程对应什么版本？我的机器能跑哪条路线？

**深入知识点**：

- CUTLASS 4.x、CUTLASS Python、CuTe DSL、CuTe C++ 的关系。
- wheel 与仓库源码必须匹配的原因；提交号和 API changelog。
- Linux、Python、CUDA、driver、SM 架构的兼容矩阵。
- SM80、SM90、SM100/103、SM120 四条硬件路线。
- 稳定 API、experimental Primitives、Task Scheduling 的边界。
- 如何运行仓库例子，如何保留 IR/PTX/CUBIN，如何读报错。

**计划代码**：`00_env_probe.py`，打印 package/version、GPU compute capability，并给出可运行章节列表。

**主要取材**：`README.md`、`media/docs/pythonDSL/quick_start.rst`、`overview.rst`、`limitations.rst`。

### 第 1 章：最小程序与两层调用模型 `[通用]`

**要回答的问题**：`@cute.kernel`、`@cute.jit`、Python 调用者分别负责什么？

**深入知识点**：

- device kernel、host JIT function、普通 Python 三层边界。
- `kernel(...).launch(grid, block, cluster, smem, stream)` 的职责。
- JIT specialization、首次编译与后续调用。
- AST rewrite + tracing 的混合编译流程。
- Python `print`、`cute.printf`、compile-time proxy 的差异。
- 编译产物从 DSL IR 到 PTX/CUBIN 的路径。

**计划代码**：`01_hello_world.py`、`01_compile_time_vs_runtime_print.py`。

**主例**：`cute/notebooks/hello_world.ipynb`、`experimental/primitives/tutorial/01_hello_world.py`。

### 第 2 章：类型系统、静态值与动态值 `[通用]`

**要回答的问题**：为什么看起来是 Python，变量却不能随意改变类型和结构？

**深入知识点**：

- `cutlass.Int32/Float32/Boolean/Numeric` 与 Python 标量的映射。
- `cutlass.Constexpr`、`cutlass.const_expr` 与编译期分支。
- 类型提升、显式转换、整数宽度和溢出意识。
- Python tuple/list/dataclass 作为静态结构时的规则。
- `@cute.struct`、`MemRange`、alignment 和不可变 dataclass。
- DynamicExpression/value tree、循环前后结构不变式。
- 不支持的 dependent type、动态 list 索引、早退、`global/nonlocal`。

**计划代码**：`02_types_and_constexpr.py`、`02_struct_and_dataclass.py`、`02_expected_compile_errors.py`。

**主例**：`data_types.ipynb`、`dataclass_immutable.py`、`08_mlir_value_tree_debug.py`。

### 第 3 章：控制流与元编程 `[通用]`

**要回答的问题**：一个 `for`/`if` 是在 Python 中展开，还是留到 GPU 运行？

**深入知识点**：

- Python `range`、`cutlass.range`、`cutlass.range_constexpr`。
- 静态与动态 `if/for/while` 的 lowering。
- loop-carried value 的类型/结构一致性。
- `unroll`、`prefetch_stages` 与软件流水化提示。
- 编译期生成特化 kernel，避免运行时分支。
- JIT 函数返回值与调用约束。

**计划代码**：`03_control_flow.py`，同一函数分别用静态/dynamic shape 编译并比较 IR。

**主要取材**：`dsl_control_flow.rst`、`dsl_code_generation.rst`、`dsl_dynamic_layout.rst`。

## 第二篇：CuTe 的核心——Layout 与 Tensor

### 第 4 章：Layout 是坐标到索引的函数 `[通用]`

**要回答的问题**：`(shape):(stride)` 到底描述了什么？

**深入知识点**：

- coordinate、logical index、physical offset。
- Shape/Stride 的层级 tuple，rank、depth、size、cosize。
- 静态整数与动态整数进入 Layout 后的区别。
- row-major、column-major、带 padding、broadcast/zero-stride Layout。
- `make_layout`、`make_ordered_layout`、`make_identity_layout`。
- 用表格和小矩阵手算 Layout 映射。

**计划代码**：`04_layout_basics.py`，枚举坐标并验证 offset。

**主例**：`cute_layout_algebra.ipynb`、`print_latex.py`。

### 第 5 章：Layout 代数 `[通用]`

**要回答的问题**：为什么 tile、partition、thread-value 映射都能写成 Layout 变换？

**深入知识点**：

- `coalesce` 的前后条件与按 mode 合并。
- `composition` 的函数复合含义。
- `logical/zipped/tiled/flat_divide`：用 tile 拆分 domain。
- `logical/zipped/tiled/flat_product`：复制 tile 覆盖更大 domain。
- complement、left/right inverse 的用途和合法条件。
- `tile_to_shape`、`slice_`、`select`、`group_modes`、`flatten`。
- 结果 shape 中每个 mode 的语义标注方法。

**计划代码**：`05_layout_algebra_lab.py`，每个操作都打印映射表并断言双射/覆盖关系。

**主例**：`cute_layout_algebra.ipynb`；对照 `test/python/pycute` 和 CuTe C++ Layout 文档。

### 第 6 章：Swizzle、ComposedLayout 与 bank conflict `[通用/SM80+]`

**要回答的问题**：逻辑 Layout 和地址变换如何叠加？

**深入知识点**：

- `ComposedLayout` 的 inner transform、offset、outer layout。
- bit-level swizzle 参数和可视化。
- SMEM bank、transaction width、alignment 的关系。
- gather/scatter 视图与非平凡 iterator。
- TMA/ldmatrix 对 SMEM Layout 的额外约束。
- “逻辑形状正确”不等于“访存高效”。

**计划代码**：`06_swizzle_visualizer.py`、`06_smem_bank_probe.py`。

**主例**：`composed_layout.ipynb`、`cp_async_shared_global_swizzled.py`、`tma_swizzle_modes.py`。

### 第 7 章：Tensor、切片与坐标 Tensor `[通用]`

**要回答的问题**：Tensor 如何把存储和 Layout 组合起来？

**深入知识点**：

- `Pointer/Iterator + Layout = Tensor`。
- GMEM/SMEM/RMEM/TMEM address space 与元素类型。
- `make_ptr`、`make_tensor`、DLPack `from_dlpack`。
- full/partial evaluation、`None` 切片、`local_tile`、`domain_offset`。
- identity/coordinate Tensor 如何携带坐标而非数据。
- `recast_ptr/recast_tensor`、alignment 与向量宽度。
- static/dynamic/compact Layout 对 JIT signature 的影响。

**计划代码**：`07_tensor_views.py`、`07_dynamic_layout.py`。

**主例**：`tensor.ipynb`、`call_bypass_dlpack.py`、`torch_fake_tensor.py`。

### 第 8 章：TensorSSA 与寄存器级数据流 `[通用]`

**要回答的问题**：为什么寄存器 fragment 不是普通 memory-backed Tensor？

**深入知识点**：

- TensorSSA 的值语义与 memory Tensor 的区别。
- `make_rmem_tensor`、`make_fragment_like`、load/store 边界。
- elementwise、broadcast、reduction、slice 的 shape 规则。
- mutation-looking syntax 与 SSA 更新。
- MMA accumulator fragment 的前置知识。

**计划代码**：`08_tensorssa_ops.py`。

**主例**：`tensorssa.ipynb`、`basic_types.py`。

## 第三篇：从普通 kernel 到高效数据移动

### 第 9 章：执行层级、索引与边界处理 `[通用]`

**要回答的问题**：Layout 如何映射到 thread/warp/block/cluster？

**深入知识点**：

- `thread_idx/block_idx/block_dim/grid_dim`。
- lane、warp、warpgroup、CTA、cluster 的参与者范围。
- 1D/2D grid-stride loop 和 tile coordinate。
- identity Tensor + predicate、residue tile、mask。
- `assume`、shape divisibility、alignment contract。
- 正确性边界与性能友好边界的区别。

**计划代码**：`09_vector_add_scalar.py`、`09_vector_add_masked.py`。

**主例**：`elementwise_add.ipynb`、Ampere `elementwise_add.py`。

### 第 10 章：TiledCopy 与 Thread-Value Layout `[通用/SM80+]`

**要回答的问题**：谁复制哪些值，如何从 Layout 中看出来？

**深入知识点**：

- CopyOp → CopyAtom → TiledCopy → ThrCopy。
- thread layout、value layout、TV Layout。
- `partition_S/partition_D` 的 shape 与 ownership。
- vectorized load/store、copy bits、alignment。
- `make_tiled_copy_tv`、`autovec_copy`、`cute.copy`。
- predicate 与 vectorization 共存时的尾部处理。

**计划代码**：`10_tiled_copy_visual.py`、`10_vectorized_elementwise.py`。

**主例**：`elementwise_add.ipynb`、`07_vectorized_array.py`、Ampere elementwise 示例。

### 第 11 章：Shared Memory 分配与经典 tiled kernel `[通用]`

**要回答的问题**：何时需要 SMEM，如何正确分配和复用？

**深入知识点**：

- `SmemAllocator`、`@cute.struct` storage、byte alignment。
- 静态与动态 shared-memory size。
- GMEM → SMEM → RMEM → GMEM 的基本 tiled 路径。
- `sync_threads`、可见性与 buffer reuse。
- bank conflict、占用率和 stage 数的容量约束。

**计划代码**：`11_tiled_transpose.py` 或 `11_tiled_gemm_smem.py`。

**主例**：`smem_allocator.py`、`dynamic_smem_size.py`、`03_gemm_tiled_smem.py`。

### 第 12 章：warp/CTA/cluster 同步原语 `[通用/SM90+]`

**要回答的问题**：barrier 的作用域、到达计数和内存可见性如何配套？

**深入知识点**：

- warp sync、named barrier、CTA barrier reduction。
- `elect_one/elect_sync`、vote、shuffle、redux。
- mbarrier init/arrive/wait、phase bit、transaction bytes。
- acquire/release、fence、async proxy 与普通 memory view。
- cluster barrier、DSMEM、`mapa` 与 remote SMEM。
- divergence 和参与者数不匹配导致的死锁模式。

**计划代码**：`12_warp_collectives.py`、`12_mbarrier_pingpong.py`。

**主例**：`mbarrier.py`、`warp_named_barrier.py`、`mapa.py`、vote/shuffle/redux 示例。

### 第 13 章：`cp.async` 与 `ldmatrix/stmatrix` `[SM80+/SM90+]`

**要回答的问题**：Ampere 风格 tensor-core kernel 怎样把 A/B 搬进寄存器？

**深入知识点**：

- per-thread `cp.async` 与 bulk async copy。
- copy group commit/wait 与 mbarrier completion。
- `ldmatrix/stmatrix` 的 lane-to-fragment 映射。
- SMEM swizzle 与 matrix load 的对齐要求。
- 软件填充 pipeline 与硬件异步 copy 的差别。

**计划代码**：`13_cp_async_roundtrip.py`、`13_ldmatrix_roundtrip.py`。

**主例**：`cp_async_shared_global.py`、`cp_async_bulk.py`、`ldmatrix_stmatrix.py`。

### 第 14 章：TMA 从 descriptor 到 partition `[SM90+]`

**要回答的问题**：为什么 TMA 不是“换一个 copy op”这么简单？

**深入知识点**：

- TensorMap/TMA descriptor：global layout、box/tile、swizzle、interleave。
- `make_tiled_tma_atom` 与 `TmaInfo`。
- `group_modes` 为什么要把 atom domain 放到 mode 0。
- `tma_partition` 输入/输出 shape 的逐 mode 推导。
- G2S、S2G、multicast、cluster mask。
- mbarrier transaction bytes、TMA store fence/wait。
- tensormap replace、online descriptor、prefetch descriptor。

**计划代码**：`14_tma_copy_v0.py`、`14_tma_transpose_v1.py`、`14_tma_multicast.py`。

**主例**：Blackwell `tutorial_tma/tma_v0.py`、`v1.py`、`v2.py`；原语 TMA 目录作为对照。

### 第 15 章：多级流水线与 warp specialization `[SM90+/SM100+]`

**要回答的问题**：如何证明 producer/consumer 不覆盖仍在使用的数据？

**深入知识点**：

- prologue、steady state、epilogue/tail。
- stage index、phase、`PipelineState` 的循环推进。
- CooperativeGroup、Agent、producer/consumer participant。
- acquire/commit、try_wait/wait、release、producer tail。
- TMA async、TMA store、TMA-UMMA、UMMA-async pipeline 的差异。
- warp role、register budget 与 latency hiding。
- stage 数的 SMEM 容量、延迟覆盖和 occupancy 权衡。

**计划代码**：`15_pipeline_software_fill.py`、`15_tma_pipeline_warpspec.py`。

**主例**：`async_pipeline.ipynb`、`pipeline_software_fill.py`、TMA pipeline 原语、`tma_v2.py`。

### 第 16 章：规约与 Softmax `[通用/SM90+]`

**要回答的问题**：同一个 reduction 为什么要按 thread、warp、CTA、cluster 分层？

**深入知识点**：

- identity、associativity、数值类型与累加精度。
- thread-vector、shuffle tree、SMEM block reduction。
- cluster SMEM/TMEM reduction。
- max/sum 双规约与 online softmax 更新公式。
- 数值稳定性、mask、exp 近似和误差容忍。
- bandwidth-bound kernel 的性能测量。

**计划代码**：`16_reduction_ladder.py`、`16_online_softmax.py`。

**主例**：五级 reduction 示例、`06_softmax.py`、CTA norm/RMSNorm。

## 第四篇：Tensor Core 与 GEMM

### 第 17 章：MMA Atom、TiledMMA 与 partition 公共模型 `[通用]`

**要回答的问题**：硬件指令、空间复制和每线程 fragment 如何连起来？

**深入知识点**：

- MmaOp → MmaAtom → TiledMma → ThrMma。
- instruction shape、atom layout、permutation/tile footprint。
- `partition_A/B/C` 的 thread/value/repeat modes。
- `make_fragment_A/B/C` 与 descriptor/寄存器/TMEM 的差异。
- `cute.gemm` 的 accumulator 更新语义。
- 从 Layout 证明 tile coverage、无重叠和 fragment 对齐。

**计划代码**：`17_mma_layout_inspector.py`，不追求性能，专门打印 partition shape 与 ownership。

**主要取材**：三份 MMA programming guide、`05_minimal_tensor_mma.py`。

### 第 18 章：Ampere warp MMA `[SM80+]`

**数据路径**：GMEM → `cp.async` → SMEM → `ldmatrix` → RMEM → `mma.sync` → RMEM/SMEM/GMEM。

**深入知识点**：

- `MmaF16BF16Op`、warp atom 与多 warp tiling。
- A/B register fragment 和 C accumulator fragment。
- SMEM Layout 与 ldmatrix tiled copy。
- K-loop 双缓冲、predicate、epilogue coalescing。
- SIMT SGEMM 与 tensorop GEMM 的性能/精度对照。

**计划代码**：`18_ampere_tensorop_gemm.py`。

**主例**：Ampere `sgemm.py`、`tensorop_gemm.py`。

### 第 19 章：Hopper WGMMA `[SM90+]`

**数据路径**：GMEM → TMA → SMEM descriptor → WGMMA；accumulator 位于 RMEM。

**深入知识点**：

- warpgroup 参与模型与 WGMMA op。
- A/B descriptor fragment 和 fence/commit/wait 序列。
- TMA producer 与 WGMMA consumer 的 warp specialization。
- cluster multicast、persistent scheduler、TMA store epilogue。
- FP8 2x accumulator 与 fused GELU。

**计划代码**：`19_hopper_wgmma_gemm.py`。

**主例**：Hopper dense GEMM、persistent GEMM、FP8 示例。

### 第 20 章：Blackwell tcgen05、TMEM 与 UMMA `[SM100+]`

**数据路径**：GMEM → TMA → SMEM/TMEM → tcgen05 UMMA → TMEM → RMEM/SMEM → TMA store。

**深入知识点**：

- TMEM 的用途、列分配、holding buffer、dealloc/relinquish 协议。
- `tcgen05.mma`、1CTA 与 2CTA CtaGroup。
- SMEM descriptor、S2T copy、TMEM load/store。
- UMMA completion barrier 与跨 CTA accumulator ownership。
- CTA cluster、multicast mask、leader CTA。
- tcgen05 pipeline lifecycle 和常见 fence。

**计划代码**：`20_tcgen05_1cta.py`、`20_tcgen05_2cta.py`、`20_tmem_roundtrip.py`。

**主例**：tcgen05 原语目录、`05_minimal_tensor_mma.py`、Blackwell GEMM 0/1。

### 第 21 章：从 naive GEMM 到可用 GEMM `[通用 + 架构分支]`

**要回答的问题**：每个优化究竟减少了哪种成本？

**深入知识点**：

- GEMM 的 M/N/K 层次分解与算术强度。
- naive global-memory kernel 的访存冗余。
- SMEM tiling、register blocking、tensor-core atom。
- host 侧 Layout 约定：A(M,K)、B(N,K)/B(K,N)、C(M,N)。
- residue/predication 与“只支持整 tile”基线的取舍。
- reference、误差阈值、TFLOP/s 计算。

**计划代码**：`21_gemm_0_naive.py`、`21_gemm_1_smem.py`、对应架构 tensor-core 版。

**主例**：原语教程 02/03/05、`tour_to_sol_gemm.ipynb`。

### 第 22 章：Blackwell GEMM 逐版演化 `[SM100+]`

**讲解方式**：以 `fp16_gemm_0` 为可执行基线，逐个 diff 引入：

1. 2CTA MMA + TMA multicast；
2. TMA/MMA/epilogue warp specialization；
3. static persistent scheduler；
4. CLC dynamic scheduler；
5. preferred/fallback dynamic cluster；
6. initial/rolling TMA prefetch；
7. dequant + GEMM Programmatic Dependent Launch。

**每一步都深入分析**：新增资源、warp role、barrier/pipeline、grid 公式、适用 shape、潜在收益、额外约束和失败模式。

**计划代码**：`22_blackwell_gemm_step_0.py` 至 `step_7.py`，共享测试驱动，避免复制 host boilerplate。

**主例**：`tutorial_gemm/fp16_gemm_0.py` 至 `fp16_gemm_6.py`。

### 第 23 章：Epilogue、融合与自定义操作 `[SM90+/SM100+]`

**要回答的问题**：累加器如何安全、高吞吐地落回 GMEM，并顺便完成融合？

**深入知识点**：

- RMEM/TMEM → SMEM → GMEM 的 tiled epilogue。
- epilogue subtile、多 stage TMA store、register pressure。
- alpha/beta、bias、activation、residual、amax。
- `dsl_user_op` 与 inline PTX 的安全边界。
- EFC/EVT 运算树、broadcast 输入与数据移动策略。

**计划代码**：`23_gemm_bias_relu.py`、`23_custom_epilogue.py`。

**主例**：dense alpha-beta、EFC 四个入口示例、`inline_ptx.py`。

### 第 24 章：低精度与 block scaling `[SM100/SM103/SM120]`

**要回答的问题**：FP4/FP8 的难点为何不仅是换 dtype？

**深入知识点**：

- FP8、NVFP4、MXFP4/6/8 的数据与累加类型。
- packed narrow type、recast、量化/反量化。
- SFA/SFB vector size、逻辑 shape 与物理存储 Layout。
- scale-factor GMEM/SMEM/TMEM mapping。
- block-scaled MMA operand bundle 与 S2T copy。
- amax、误差模型、emulated reference。
- SM100、SM103、SM120 指令与布局差异。

**计划代码**：`24_nvfp4_scale_layout.py`、`24_nvfp4_gemm_1cta.py`、`24_nvfp4_gemm_2cta.py`。

**主例**：NVFP4 tutorial 0/1、blockscaled GEMM 与 SM103/SM120 示例。

## 第五篇：调度与综合应用

### 第 25 章：Persistent、Grouped、Blockwise 与 MoE 调度 `[SM90+/SM100+]`

**深入知识点**：

- static/dynamic persistent tile scheduler。
- raster/swizzle、wave quantization、负载不均衡。
- batched、pointer-array、grouped、contiguous-grouped 的数据模型。
- 动态 problem shape 和在线 TMA descriptor 更新。
- MoE expert/token range、workspace、descriptor init kernel。
- blockwise accumulator 更新、masked/contiguous grouped 变体。

**计划代码**：`25_persistent_scheduler_sim.py`、`25_grouped_gemm_small.py`。

**主例**：Hopper/Blackwell grouped GEMM、MoE scheduler/utils、blockwise 三例。

### 第 26 章：FMHA、MLA、RMSNorm 与 SSD 的解剖方法 `[SM80+/SM90+/SM100+]`

**统一解剖维度**：

1. 数学分块与中间张量；
2. 每个 warp/warpgroup 的角色；
3. TMA/cp.async/MMA 数据路径；
4. pipeline dependency graph；
5. online reduction 和数值稳定性；
6. mask、变长、mixed-input/FP8；
7. scheduler 与 epilogue。

**计划代码**：先提供 `26_attention_math_reference.py` 和小型 fused attention，再引用生产 kernel；不在一章中重写 4000 行实现。

**主例**：Ampere FlashAttention/HSTU、Hopper FMHA、Blackwell FMHA forward/backward、mixed-input FMHA、MLA、Mamba2 SSD、RMSNorm。

### 第 27 章：多 GPU 与通信计算融合 `[SM100+，多 GPU]`

**深入知识点**：

- NVSHMEM 初始化、symmetric tensor、peer/multicast tensor 与手工释放。
- one-shot/two-shot all-reduce、reduce-scatter、all-gather GEMM。
- `multimem.ld_reduce/st/red` 与 NVLS。
- flag、Lamport、memory ordering 与跨 GPU 可见性。
- GEMM epilogue 与 collective overlap。
- 多进程启动、P2P/multicast capability 检查和验证。

**计划代码**：`27_all_reduce_simple.py`、`27_all_reduce_tma.py`；要求独立的多 GPU 测试说明。

**主例**：`cute/blackwell/kernel/distributed` 全目录。

## 第六篇：编译、调试、性能与生态集成

### 第 28 章：JIT signature、动态 Layout 与缓存 `[通用]`

**深入知识点**：

- 参数 tracing：static vs dynamic、Tensor/Pointer/tuple/dataclass。
- static Layout specialization 与 dynamic/compact Layout reuse。
- `cute.compile`、显式/隐式编译、cache key。
- fake tensor、symbolic int、无 GPU 输入的编译流程。
- `lru_cache` 与 MLIR context 敏感对象的风险。
- 分离编译、编译延迟和 launch 延迟。

**计划代码**：`28_jit_cache_probe.py`、`28_fake_tensor_compile.py`。

**主例**：dynamic-layout 文档、Ampere/Blackwell `compile_bmm`、fake tensor 示例。

### 第 29 章：正确性、调试与性能分析 `[通用]`

**深入知识点**：

- CPU/PyTorch reference、dtype-aware tolerance、随机与边界测试。
- Python print、`cute.printf`、`cute.print_tensor`、LaTeX Layout。
- line info、IR/PTX/CUBIN 保存、编译诊断、错误最小化。
- compute-sanitizer、Nsight Systems/Compute 的使用边界。
- warmup、同步、L2 flush、CUDA Graph 对 benchmark 的影响。
- occupancy、register spill、SMEM/TMEM、memory throughput、tensor-core utilization。
- autotune search space、cache 和“只对当前 shape 最优”的风险。
- IKET in-kernel event tracing。

**计划代码**：`29_debuggable_kernel.py`、`29_benchmark_harness.py`、`29_autotune_copy.py`。

**主例**：print/benchmark/autotune notebooks、`error_reporting.py`、IKET GEMM。

### 第 30 章：PyTorch、JAX 与 TVM FFI `[通用]`

**深入知识点**：

- DLPack conversion、stream、device、layout/alignment contract。
- 直接传 framework tensor、绕过 DLPack、aliasing。
- JAX `cutlass_call`、TensorSpec、JIT、custom partitioning/sharding、export。
- TVM FFI JIT/AOT、Torch/JAX 调用、fake tensor、错误传播。
- framework graph/shape polymorphism 与 CuTe specialization 的边界。

**计划代码**：`30_torch_vector_add.py`、`30_jax_vector_add.py`、`30_tvm_ffi_add_one.py`。

**主例**：`dsl_tutorials/jax`、`tvm_ffi`、`call_bypass_dlpack.py`。

### 第 31 章：AOT、C/C++ 导出与自定义 JIT 参数 `[通用]`

**深入知识点**：

- JIT 与 AOT 的部署权衡。
- Low-level CuTe ABI、symbol、dynamic shape slot、stream 参数。
- shared library 动态加载与静态链接。
- Python runtime `load_module`。
- 自定义对象的 `__extract_mlir_values__` / `__new_from_mlir_values__` 协议。
- C++ tensor wrapper 与 CMake 扩展。

**计划代码**：复用并精简 `export_to_c`、`load_in_python`、C++ loader 和 FFI tensor 示例。

**主例**：`dsl_tutorials/export`、`ffi/jit_argument.py`、TVM AOT bundle。

### 第 32 章：高级 launch 与执行编排 `[SM90+]`

**深入知识点**：

- stream 与 event 的所有权和 ABI。
- CUDA Graph capture/replay。
- launch completion event 与 programmatic event。
- Programmatic Dependent Launch 的 wait/launch_dependents 协议。
- cooperative launch、resident grid 限制和 grid-wide barrier。
- dynamic/preferred/fallback cluster launch 参数。

**计划代码**：`32_cuda_graph.py`、`32_pdl_pair.py`、`32_cooperative_barrier.py`。

**主例**：CUDA Graph notebook、launch events、PDL、cooperative launch、FP16 GEMM 4/6。

### 第 33 章：Task Scheduling `[Experimental, SM100+]`

**深入知识点**：

- 为什么在 warp-specialized kernel 上引入静态 schedule 验证。
- MemoryResource、producer/consumer work、TaskLocalVariable。
- Task、schedule、resource dependency graph、domain loop。
- PipelineConfig、allocator、`try_*` 与阻塞调用。
- persistent work queue、dynamic domain。
- cluster GEMM、NVFP4、Split-K、PipelineGroup merge/fork。
- FrontendNext 限制：现阶段只验证新写的 Primitives kernel。

**计划代码**：沿仓库 tutorial 01–07 逐章精简，保留 raw kernel 对照和 schedule graph。

**主例**：`experimental/task_scheduling/blackwell/tutorial` 全目录。

## 4. 附录规划

### 附录 A：命名约定与读代码速查

- `g/s/r/t` 前缀：global/shared/register/tensor memory。
- `mA/tAgA/tAsA/thr_mma/tCrC` 等组合命名的逐段含义。
- shape mode、tile mode、stage mode、MMA mode 的标注模板。
- 常见 host/kernel 类的阅读顺序。

### 附录 B：Layout 与容量计算手册

- row/column-major stride。
- tile count、grid、cluster grid。
- SMEM bytes、mbarrier bytes、stage 上限。
- TMEM column 规划。
- TMA transaction bytes、multicast traffic。
- GEMM FLOPs、arithmetic intensity、TFLOP/s。

### 附录 C：同步正确性检查表

- barrier 初始化者、参与者、到达计数。
- transaction bytes 和 phase。
- producer/consumer state 是否等量推进。
- tail 是否回收全部 stage。
- fence 是否覆盖正确的 memory proxy/scope。
- 分支中的所有参与者是否保持一致。

### 附录 D：架构/API 对照表

| 架构 | MMA 参与者 | A/B 主要来源 | accumulator | 典型搬运 |
|---|---|---|---|---|
| SM80/89 | warp | RMEM | RMEM | cp.async + ldmatrix |
| SM90 | warpgroup | SMEM descriptor / RMEM A | RMEM | TMA + WGMMA |
| SM100/103 | CTA/2CTA | SMEM/TMEM | TMEM | TMA + tcgen05 |
| SM120 | warp | RMEM | RMEM | TMA/cp.async + ldmatrix |

### 附录 E：示例索引与兼容性矩阵

- 把 207 个 Python 示例按主题、架构、稳定性、最低依赖和正文入口建立索引。
- 标记“教学主例 / 对照例 / 生产例”。
- 记录仓库升级后失效的 API、路径和运行参数。

### 附录 F：术语表

Layout、mode、atom、TV Layout、fragment、descriptor、TMA、TMEM、WGMMA、UMMA、mbarrier、stage、phase、warp specialization、persistent scheduler、CLC、EFC、NVLS 等。

## 5. 深入剖析时必须回答的通用问题

后续每个知识点都按下面的模板讲，不只说明“API 怎么写”：

1. 它解决什么硬件或编译问题？
2. 输入、输出的类型与 Layout 是什么？
3. 每个 shape mode 的语义是什么？
4. 哪些量在编译期，哪些量在运行期？
5. 哪些 thread/warp/CTA 参与，谁拥有哪些值？
6. 数据经过哪些 memory spaces？
7. 同步的 scope、计数、phase 和 fence 是什么？
8. 对齐、整除、dtype、架构限制是什么？
9. 如何构造最小正确例和反例？
10. 如何验证生成代码和数值正确性？
11. 它的性能收益来自延迟、带宽、并行度还是 occupancy？
12. 生产示例中它与哪些其他机制组合？

## 6. 推荐编写顺序

为尽早形成一条可执行学习路径，正文按四个里程碑交付：

1. **基础闭环**：第 0–10 章。读者能写、运行、验证 vectorized elementwise kernel，并真正理解 Layout/Tensor。
2. **数据移动闭环**：第 11–16 章。读者能写 SMEM/TMA pipeline 和分层 reduction。
3. **GEMM 闭环**：第 17–24 章。读者能按自己的 GPU 选择 SM80/90/100 路线，并完成渐进 GEMM。
4. **生产闭环**：第 25–33 章。调度、应用、调试、部署、框架集成和实验性 TS。

示例审阅结论与取材说明见 [cutedsl_example_survey.md](cutedsl_example_survey.md)。
