# CuTe DSL 教程开发环境

本文记录后续教程编写、示例执行和性能验证必须遵守的环境约束。

## 本地开发机

- 系统：macOS arm64。
- 仓库目录：`/Users/yong/workarea/develop/github/yongchn/cutlass`。
- 用途：阅读和编辑代码、下载源码/安装包/第三方依赖、准备离线传输文件。
- 限制：不能在本机执行需要 NVIDIA GPU 的 CuTe DSL kernel。

## GPU 开发机

- SSH：`ssh root@116.178.239.130 -p 32045`。
- 工作目录：`/volume/njiang/workspace/sandbox`。
- GPU：8 × NVIDIA B200（SM100），每张约 183 GB 显存。
- 驱动：590.48.01。
- `nvidia-smi` 显示的 CUDA 版本：13.1。
- MIG：未启用。
- 教程隔离环境：`/volume/njiang/workspace/sandbox/.venv-cutedsl-4.7`。
- 教程 CuTe DSL：4.7.0；系统 Python 中原有的 4.4.2 保持不变。

## 网络与依赖规则

GPU 开发机不能访问外网。后续工作必须遵循以下流程：

1. 不在 GPU 开发机上直接执行需要外网的下载或安装命令。
2. 所需源码、wheel、压缩包和其他第三方依赖先下载到本地 Mac。
3. 在本地核对文件版本及其与仓库提交、Python、CUDA 和目标架构的兼容性。
4. 将文件复制到 GPU 开发机的 `/volume/njiang/workspace/sandbox` 或其子目录。
5. 在 GPU 开发机上使用本地文件或离线索引完成安装。
6. 记录实际安装版本、传输位置和验证命令，确保教程示例可复现。

教程示例必须显式使用隔离环境中的解释器：

```text
/volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python
```

除非用户以后明确更新这些信息，否则本文所述环境和规则视为后续工作的固定前提。
