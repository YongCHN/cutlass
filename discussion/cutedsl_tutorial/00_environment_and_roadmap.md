# 第 0 章：版本、环境与阅读地图

标签：`[通用]`
状态：正文与环境探针完成；已在 B200 / CuTe DSL 4.7.0 上通过 L0 验证。

## 0.1 本章目标

学完本章后，你应该能够：

1. 区分 CUTLASS、CUTLASS Python、CuTe 和 CuTe DSL。
2. 说明为什么教程源码版本必须与安装的 wheel 对齐。
3. 根据 GPU compute capability 选择 SM80、SM90 或 SM100 路线。
4. 在不修改系统 Python 的情况下验证教程环境。
5. 理解本文对“代码已验证”的严格含义。

## 0.2 我们究竟在学习什么

CUTLASS 是 NVIDIA 面向高性能线性代数和相关 kernel 的组件集合。CuTe 是其中用于描述层级化 Layout、Tensor、Copy 和 MMA 的核心抽象。CuTe DSL 则把这套模型放进 Python 语法中，让我们用 Python 做元编程，同时仍然显式控制 GPU 的线程层级、内存层级和硬件指令。

可以把本教程使用的软件栈理解为：

```text
普通 Python：测试、数据生成、配置、框架互操作
        ↓
CuTe DSL：@cute.jit、@cute.kernel、Layout/Tensor/Copy/MMA
        ↓
CUTLASS Python 编译器：AST rewrite、tracing、MLIR lowering
        ↓
CUDA 工具链：PTX、CUBIN、driver launch
        ↓
NVIDIA GPU：warp、CTA、cluster、Tensor Core、TMA、TMEM
```

这里的“Python kernel”不是让 Python 解释器在 GPU 上执行。Python 在编译阶段帮助构造程序；真正运行在 GPU 上的是编译后的机器代码。

## 0.3 CUTLASS C++、旧 CUTLASS Python 与 CuTe DSL

容易混淆的三个名字：

- **CUTLASS C++**：以 C++ template 为主要配置和组合机制。
- **旧版 CUTLASS Python**：偏向从 Python 选择和生成已有 CUTLASS 操作，不等同于逐条编写 GPU kernel。
- **CuTe DSL**：低层 kernel authoring DSL，与 CuTe C++ 的 Layout/Tensor/Atom 模型保持对应。

本教程只讲第三种，同时在必要处引用 CuTe C++ 文档帮助理解数学模型。

## 0.4 为什么必须固定版本

CuTe DSL 仍在快速演进。装饰器、动态 Layout、pipeline、TMA/tcgen05 helper 和 experimental frontend 都可能在相邻版本间改变。若源码来自一个提交、wheel 却来自另一个版本，会出现三类问题：

1. 导入路径或函数签名不存在。
2. 示例能够编译，但 Layout/launch 语义已经改变。
3. production 示例依赖新 compiler lowering，旧 wheel 无法识别。

本教程固定：

```text
CUTLASS repository: 4.7.0
Git commit:          7107b055
CuTe DSL wheel:      4.7.0
```

GPU 服务器原先预装 4.4.2。我们没有覆盖它，而是在工作目录中建立了隔离环境：

```text
/volume/njiang/workspace/sandbox/.venv-cutedsl-4.7
```

以后教程命令统一使用：

```bash
/volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python
```

不要依赖 `python3` 恰好指向哪个环境。

## 0.5 当前验证机器

| 项目 | 值 |
|---|---|
| GPU | 8 × NVIDIA B200 |
| Compute capability | 10.0（SM100） |
| 单卡显存 | 约 183 GB |
| Driver | 590.48.01 |
| 系统 `nvidia-smi` CUDA 字段 | 13.1 |
| Python | 3.12.3 |
| PyTorch | 2.9.1+cu129 |
| 教程 CuTe DSL | 4.7.0（隔离环境） |
| 工作目录 | `/volume/njiang/workspace/sandbox` |

注意：`nvidia-smi` 的 CUDA Version 表示驱动可支持的最高 CUDA driver API 版本，不等价于某个 Python 包实际绑定的 CUDA toolkit 版本。判断编译环境时要同时查看 CuTe DSL wheel、PyTorch CUDA、driver 和工具链。

## 0.6 离线环境规则

GPU 服务器不能访问外网。因此依赖安装遵循：

```text
Mac 下载 Linux x86_64 wheel
        ↓
核对包名、版本、Python ABI、manylinux 标签
        ↓
scp 到 /volume/njiang/workspace/sandbox/offline_packages
        ↓
远端 pip --no-index --find-links ...
        ↓
在工作目录中的虚拟环境验证
```

禁止在远端盲目执行联网 `pip install`。这既会失败，也会让复现步骤变得不完整。

4.7.0 实际离线包包括：

- `nvidia-cutlass-dsl`
- `nvidia-cutlass-dsl-libs-base`
- `nvidia-cutlass-dsl-libs-core`
- `nvidia-cutlass-dsl-libs-cu12`
- `nvidia-cutlass-dsl-libs-cu13`
- `nvidia-cuda-nvdisasm`

元包在 `[cu13]` 模式下仍声明 `libs-cu12`，所以离线准备时必须以 wheel metadata 的完整依赖链为准，不能只凭 extra 名称猜测。

## 0.7 架构路线

| 路线 | 最低能力 | 主要内容 |
|---|---:|---|
| 通用核心 | CuTe DSL 支持架构 | 语言模型、Layout、Tensor、Copy |
| SM80 | 8.x | `mma.sync`、`cp.async`、`ldmatrix`、warp MMA |
| SM90 | 9.0 | TMA、WGMMA、warpgroup、cluster |
| SM100/103 | 10.0/10.3 | tcgen05、TMEM、1CTA/2CTA UMMA |
| SM120 | 12.0 | Blackwell GeForce warp MMA、block scaling |

B200 可以运行通用核心、SM80 风格的通用 kernel、SM90+ TMA 概念以及 SM100 专属示例；但不能把“新架构能运行”自动理解为“旧架构指令路径原样执行”。架构专题仍要使用对应 op 和 target 验证。

## 0.8 稳定 API 与 experimental API

本教程将两类接口分轨：

- 稳定主线：`cutlass.cute` 的 Layout、Tensor、Copy/MMA 以及正式 pipeline helper。
- `[Experimental]`：Primitives、Task Scheduling、`cute_ext`、FrontendNext。

实验接口很有教学价值，尤其适合直接观察 barrier、TMA、tcgen05 指令，但其函数名和 lowering 可能更快变化。正文会明确标记，不把实验代码悄悄混入基础模板。

## 0.9 环境探针

代码：[env_probe.py](../code/00_environment/env_probe.py)

运行：

```bash
cd /volume/njiang/workspace/sandbox/cutedsl_tutorial/code/00_environment
/volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python env_probe.py
```

探针检查：Python/平台、CuTe DSL 版本与实际导入路径、PyTorch CUDA、GPU 名称、compute capability，并输出可学习路线。它不编译 kernel，也不修改环境。

2026-08-21 实测摘要：

```text
python=3.12.3
cutlass=4.7.0
torch=2.9.1+cu129
torch_cuda=12.9
cuda_available=True
gpu=NVIDIA B200
compute_capability=10.0
runnable_tracks=language,layout,tensor,tiled-copy,sm80-warp-mma,sm90-tma-wgmma,sm100-tcgen05-tmem
PASS
```

完整命令与导入路径见 [B200 验证日志](validation_log.md)。

## 0.10 “已验证”是什么意思

教程采用五级验证：

- **L0**：静态语法、导入或 Layout 断言。
- **L1**：小规模 GPU 数值正确性。
- **L2**：边界 shape、多 dtype、随机输入。
- **L3**：规范 benchmark 和性能回归。
- **L4**：检查 IR/PTX/SASS 或 profiler 证据。

章节写了代码但没有在支持环境执行，不能标“已验证”。仅仅返回 `PASS` 也不够：数值 kernel 必须有独立 reference；性能结论必须记录硬件和测量方法。

## 0.11 本章检查清单

- [x] `cutlass.__version__` 与教程基线一致。
- [x] `cutlass.__file__` 指向教程虚拟环境，不是系统 4.4.2。
- [x] `torch.cuda.is_available()` 为真。
- [x] compute capability 与目标章节标签匹配。
- [x] GPU 例子在 `/volume/njiang/workspace/sandbox` 下执行。
- [x] 新依赖全部通过 Mac 下载、传输和离线安装。

下一章将运行第一个真正的 CuTe DSL kernel，并观察编译期和 GPU 运行期的区别。
