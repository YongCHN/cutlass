# CuTe DSL 教程

本目录是 [教程总纲](../cutedsl_tutorial_outline.md) 的正文实现。教程按依赖关系分为四个交付阶段；章节只有在正文、配套代码和相应级别验证都完成后才标记为“完成”。

## 阅读方式

- 第一次阅读请按章节顺序推进。Layout、Tensor、TiledCopy 是后续 TMA/MMA 的共同语言，不建议跳过。
- 只关心某一架构时，也要先完成第一阶段，再进入 SM80、SM90 或 SM100 分支。
- 所有可执行代码位于 `discussion/code`。每章会给出本地路径、远端路径、运行命令和实际验证结果。
- `[Experimental]` 内容使用独立标记，其兼容性承诺与稳定 CuTe API 不同。

## 交付状态

### 阶段一：语言模型、Layout、Tensor 和 TiledCopy

| 章 | 内容 | 正文 | 代码 | B200 验证 |
|---:|---|---|---|---|
| 0 | [版本、环境与阅读地图](00_environment_and_roadmap.md) | 完成 | `env_probe.py` | L0 通过（4.7.0） |
| 1 | [最小程序与执行模型](01_execution_model.md) | 完成 | 2 个示例 | L1 通过（4.7.0） |
| 2 | [类型系统、静态值与动态值](02_types_and_values.md) | 完成 | 3 个示例 | L0/L1 通过（4.7.0） |
| 3 | [控制流与元编程](03_control_flow_and_metaprogramming.md) | 完成 | 1 个示例 | L1/L4 通过（4.7.0） |
| 4 | [Layout 是坐标到索引的函数](04_layout_basics.md) | 完成 | 1 个示例 | L0/L1 通过（4.7.0） |
| 5 | [Layout 代数](05_layout_algebra.md) | 完成 | 1 个示例 | L0 通过（4.7.0） |
| 6 | [Swizzle 与 ComposedLayout](06_swizzle_and_composed_layout.md) | 完成 | 2 个示例 | L0/L1 通过（4.7.0） |
| 7 | [Tensor、切片与坐标 Tensor](07_tensor_views.md) | 完成 | 2 个示例 | L1 通过（4.7.0） |
| 8 | [TensorSSA](08_tensorssa.md) | 完成 | 1 个示例 | L1 通过（4.7.0） |
| 9 | 执行层级、索引与边界 | 待编写 | 标量 vector add 已验证 | L2 边界样例通过（4.7.0） |
| 10 | TiledCopy 与 TV Layout | 待编写 | 待编写 | — |

### 阶段二：SMEM、同步、TMA、流水线与规约

对应第 11–16 章，尚未开始正文。

### 阶段三：三代 Tensor Core 与渐进 GEMM

对应第 17–24 章，尚未开始正文。

### 阶段四：生产主题与工程化

对应第 25–33 章，尚未开始正文。

## 版本基线

- 教程源码：CUTLASS 4.7.0，提交 `7107b055`。
- GPU 验证环境：`/volume/njiang/workspace/sandbox/.venv-cutedsl-4.7`。
- 系统自带 CuTe DSL 4.4.2 保持不变；教程验证必须显式使用 4.7.0 虚拟环境。

每次实机运行的命令、输出和验证范围统一记录在
[B200 验证日志](validation_log.md) 中。
