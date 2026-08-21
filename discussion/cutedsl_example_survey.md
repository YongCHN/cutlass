# CuTe DSL 示例审阅与教程取材记录

> 审阅基线：CUTLASS `4.7.0`，提交 `7107b055`（2026-08-20）。本文只记录教程设计阶段的取材结论，不替代后续章节正文。

## 1. 审阅范围

本轮完整盘点了 `examples/python/CuTeDSL` 下的教学与示例材料，并按文件结构、入口函数、装饰器、核心 API、内存路径、同步方式、目标架构和验证方式逐项阅读。范围包括：

| 类型 | 数量 | 主要用途 |
|---|---:|---|
| Python | 207 | Kernel、host JIT、测试、benchmark、框架集成 |
| Notebook | 13 | 基础概念、Layout/Tensor、elementwise、pipeline、GEMM、JAX |
| Markdown | 13 | DSL 功能、TMA、GEMM、分布式、Task Scheduling 教学说明 |
| C++ | 4 | FFI、AOT 导出后的 C++ 调用 |
| Shell | 3 | 动态加载、静态链接、C++ bundle |
| CMake | 1 | 自定义 JIT 参数的 C++ 扩展 |
| requirements | 1 | TVM FFI 示例依赖 |
| 合计 | 242 | 不含图片、缓存与生成物 |

同时交叉核对了 `media/docs/pythonDSL` 下的 Quick Start、语言模型、控制流、动态 Layout、JIT 参数、限制、调试、框架集成和三代 MMA 编程指南。`operators/.../providers/cutedsl` 属于产品集成实现而非教学示例，本轮不计入上表；后续讲解 EFC、调度器和算子封装时可把它作为补充源码。

## 2. 示例全景

| 示例群 | Python 数量 | 读到的核心主题 | 在教程中的定位 |
|---|---:|---|---|
| `cute/notebooks` | 12 个 notebook | Hello World、类型、打印、Tensor、TensorSSA、Layout 代数、ComposedLayout、elementwise、异步流水线、autotune、CUDA Graph、SOL GEMM | 入门主线 |
| `dsl_tutorials` | 28 | JIT 调用、DLPack 绕行、struct/dataclass、动态 SMEM、allocator、inline PTX、launch 属性、JAX、TVM FFI、AOT/C 导出、fake tensor、IKET | 工程化与互操作 |
| `cute/ampere` | 7 | elementwise、autotune、SIMT/warp MMA GEMM、FlashAttention/HSTU | SM80 架构支线 |
| `cute/hopper` | 7 | TMA、WGMMA、warp specialization、persistent/grouped GEMM、FMHA、CTA norm | SM90 架构支线 |
| `cute/blackwell` | 68 | TMA/tcgen05/TMEM、1CTA/2CTA、persistent/动态调度、FP4/FP8、FMHA/MLA/SSD、MoE、EFC、分布式 | 高性能主案例与进阶 |
| `cute/blackwell_geforce` | 4 | SM120 warp MMA、block-scaled GEMM、cooperative/ping-pong | SM120 专题 |
| `cute_ext` | 6 | experimental decorator、扩展版 dense/block-scaled GEMM、指针数组 | 实验 API 对照 |
| `experimental/primitives` | 63 | 基本类型、barrier、shuffle/vote/redux、cp.async、TMA、tcgen05、规约，以及 01–10 渐进教程 | 底层原语实验室 |
| `experimental/task_scheduling` | 17 | resource/task/schedule、静态验证、动态 domain、cluster GEMM、NVFP4、Split-K、pipeline merge/fork | 独立实验篇 |
| `helpers` / `utils` | 6 | FMHA 调度与 mask、稀疏压缩/仿真 | 应用支撑与源码索引 |

## 3. 仓库中已有的四条渐进路线

### 3.1 概念路线：Notebook

推荐顺序是：

1. `hello_world.ipynb`：`@cute.kernel`、`@cute.jit`、编译产物。
2. `data_types.ipynb`：DSL 标量类型、转换与运算符。
3. `print.ipynb`：Python `print` 与 GPU `cute.printf` 的阶段差异。
4. `tensor.ipynb`：Pointer + Layout = Tensor、DLPack、切片、坐标 Tensor。
5. `tensorssa.ipynb`：寄存器 Tensor、算术、broadcast、reduction。
6. `cute_layout_algebra.ipynb` 与 `composed_layout.ipynb`：coalesce、composition、divide/product、swizzle/gather。
7. `elementwise_add.ipynb`：naive → vectorized → TV Layout → 高阶算子。
8. `async_pipeline.ipynb`：同步通信 → 单级异步 → 多级循环缓冲。
9. `benchmark_autotune.ipynb`、`cuda_graphs.ipynb`：性能工程与图捕获。
10. `tour_to_sol_gemm.ipynb`：从完整 GEMM 骨架走向软件流水化。

JAX notebook 是一条独立互操作路线，覆盖 vector add、SAXPY、ReLU、GEMM、sharding 和 export。

### 3.2 原语路线：01–10

`experimental/primitives/tutorial` 给出了仓库里最连续的底层教学阶梯：

1. 最小 kernel/host。
2. 每个线程计算一个输出的 naive GEMM。
3. 经典 SMEM tiled GEMM。
4. TMA + mbarrier 单级加载。
5. 最小 tcgen05 MMA 与 warp specialization。
6. 八种 Softmax 实现，展示串行、warp、block 与 online 算法。
7. 向量化 load/store、切片、对齐、mask 和 broadcast。
8. MLIR value tree 与动态循环中复合对象的结构不变式。
9. 4D Tensor 经 TMA 映射到 2D tile。
10. FP8 tcgen05 MMA。

同目录其余示例形成原语字典：

- 同步：warp/CTA named barrier、mbarrier 生命周期、elect、vote、shuffle、redux。
- 数据移动：`cp.async`、bulk copy、prefetch、SMEM swizzle、`ldmatrix/stmatrix`。
- TMA：load/store、multicast、swizzle、tensormap replace、single-warp/warp-specialized/ping-pong/count-2 pipeline。
- TMEM/tcgen05：1CTA、2CTA、cluster multicast、block scale、A-from-TMEM、S2T、TMEM load/store。
- 规约：thread → warp → block SMEM → cluster SMEM/TMEM。

这些 API 在当前版本仍标记为 experimental，教程会单独加稳定性标签，不与稳定 CuTe API 混写。

### 3.3 Blackwell TMA 路线：V0–V2

| 示例 | 增量 |
|---|---|
| `tma_v0.py` | `group_modes`、`tma_partition`、单个 mbarrier、G2S/S2G |
| `tma_v1.py` | load/transpose/store 三类 warp、SMEM swizzle、生产者/消费者 |
| `tma_v2.py` | 多 stage 循环缓冲、phase、overlap 与 pipeline tail |

这组示例适合作为 TMA 正文，而不是直接从完整 GEMM 中反推 TMA 语义。

### 3.4 Blackwell GEMM 路线

| 示例 | 相比前一版新增的知识 |
|---|---|
| `fp16_gemm_0.py` | 1CTA tcgen05、TMA、SMEM/TMEM、非 warp-specialized 多级主循环 |
| `fp16_gemm_1.py` | 2CTA MMA、cluster、TMA multicast、更深 AB stages |
| `fp16_gemm_2.py` | TMA/MMA/epilogue warp specialization、TMA store epilogue |
| `fp16_gemm_3.py` | static persistent tile scheduler |
| `fp16_gemm_3_1.py` | CLC dynamic persistent scheduler |
| `fp16_gemm_4.py` | dynamic/preferred/fallback cluster |
| `fp16_gemm_5.py` | 初始 + rolling TMA prefetch |
| `fp16_gemm_6.py` | dequant + GEMM 的 Programmatic Dependent Launch |
| `nvfp4_gemm_0.py` | 1CTA NVFP4 block-scaled GEMM 与 scale-factor Layout |
| `nvfp4_gemm_1.py` | 2CTA block-scaled MMA 与 multicast |

这条路线非常适合“逐版 diff”式讲解。教程不会把每版完整代码重复抄写，而会保留一个可执行基线，再用小步提交展示每个优化的必要条件、收益来源和同步不变式。

## 4. 生产示例揭示出的专题边界

### 4.1 架构差异不能被抹平

- SM80 warp MMA：A/B/C 都进入 RMEM；常见路径是 GMEM → `cp.async` → SMEM → `ldmatrix` → RMEM → `mma.sync`。
- SM90 WGMMA：warpgroup 发射；A/B 通常由 SMEM descriptor 直接供给 WGMMA，累加器在 RMEM；TMA 和 warp specialization 成为主结构。
- SM100 tcgen05：引入 TMEM、UMMA、TMEM 分配/归还协议，以及 1CTA/2CTA 指令和 cluster multicast。
- SM120 block-scaled warp MMA 又回到 warp-synchronous 模型，但增加 SFA/SFB 寄存器片段和新的数据类型/指令约束。

因此 MMA 不应写成一章“大一统 API 罗列”，而应先讲 Atom/Tiled Operation 公共模型，再分架构讲数据路径和同步协议。

### 4.2 高性能 kernel 有稳定的解剖框架

Ampere、Hopper、Blackwell 的生产示例虽然细节不同，但反复出现同一骨架：

1. host 侧检查 dtype、对齐、problem shape、tile/cluster 合法性；
2. 构造 global Tensor、TiledMMA、SMEM Layout、TMA/Copy Atom；
3. 计算 shared-memory stages、TMEM columns、grid 和 scheduler 参数；
4. kernel 内划分 warp/warpgroup 角色并初始化 barrier/pipeline；
5. tile/partition A、B、C/D；
6. prologue → steady-state mainloop → tail；
7. accumulator 经 RMEM/TMEM → SMEM → GMEM 的 epilogue；
8. host 侧编译、运行、reference 校验和 benchmark。

后续每个大型案例都应沿这八个问题解剖，避免按源文件行号平铺。

### 4.3 应用示例是组合题，不适合过早出现

- FMHA/MLA/SSD：多个 MMA、online softmax、跨 warp 流水线、mask、量化转换与调度器组合。
- Grouped GEMM/MoE：动态 problem、pointer array、在线 TMA descriptor、持久化调度。
- Blockwise/blocked-scaled：额外 accumulator 更新路径、scale-factor memory layout、S2T copy。
- EFC：把 epilogue 运算树与 GEMM 数据移动解耦。
- Distributed：NVSHMEM symmetric allocation、P2P/multicast address、Lamport/flag 协议、`multimem` 指令和通信计算融合。

这些示例将在核心篇完成后作为“综合案例”，并明确先修章节。

## 5. 教程设计结论

1. **先语言、再 CuTe、后架构。** `@jit`/`@kernel`、静态与动态值、Layout/Tensor 是后续一切的共同语法。
2. **Layout 必须单独形成一组练习。** 大多数难点不是 API 名，而是某次 tile/partition 后每个 mode 的含义和 thread-value ownership。
3. **内存搬运要按路径讲。** Universal copy、`cp.async`、TMA、`ldmatrix/stmatrix`、TMEM copy 分别解决不同 memory-space 与参与者问题。
4. **流水线要先讲正确性，再讲 overlap。** stage、phase、producer/consumer state、tail 和 barrier 到达计数是核心不变式。
5. **GEMM 用渐进 diff。** 以简单、可校验版本为基线，依次加入 tensor core、TMA、warp specialization、persistent scheduler 和 prefetch。
6. **每个例子必须有支持矩阵。** 至少标注稳定性、最低 SM、输入约束、预期输出、reference 和 benchmark 命令。
7. **正式 API 与 experimental 分轨。** Primitives 和 Task Scheduling 单列，不让读者误以为它们与主 CuTe API 有相同兼容性承诺。
8. **固定源码基线。** 当前仓库同时存在版本敏感信息，例如 Quick Start 仍写 4.4，而根 README 是 4.7；`limitations.rst` 与 Blackwell preferred-cluster 示例也存在表述差异。教程必须记录提交号，并在升级时运行兼容性检查。

## 6. 后续章节的源码索引原则

每章只选三类源码：

- **主例**：本章真正逐行讲解并在 `discussion/code` 提供精简可执行版。
- **对照例**：用于解释另一种实现或架构差异，只引用关键段落。
- **生产例**：展示同一概念在复杂 kernel 中如何组合，不要求初学者一次读懂。

完整章节安排和每章的主例计划见 [cutedsl_tutorial_outline.md](cutedsl_tutorial_outline.md)。
