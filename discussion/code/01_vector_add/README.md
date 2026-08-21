# CuTe DSL Vector Add

这是教程开发环境中的第一个已执行示例，计算：

```text
c[i] = a[i] + b[i]
```

## Kernel 结构

- 每个 CUDA thread 处理一个 FP32 元素。
- block 固定使用 256 个 thread。
- grid 使用 `ceil_div(num_elements, 256)`。
- kernel 中使用 `linear_idx < num_elements` 保护尾部，因而输入长度不必是 256 的整数倍。
- `@cute.kernel` 定义 GPU 代码，`@cute.jit` host function 负责计算 launch 配置。
- 普通 Python `run()` 负责构造 PyTorch tensor、调用 `cute.compile` 和数值校验。

## 本地与远端位置

- 本地源码：`discussion/code/01_vector_add/vector_add.py`
- GPU 机源码：`/volume/njiang/workspace/sandbox/cutedsl_tutorial/code/01_vector_add/vector_add.py`

## 已验证环境

- GPU：NVIDIA B200（SM100）
- CuTe DSL：4.7.0（隔离环境；另曾在系统 4.4.2 上通过）
- PyTorch：2.9.1+cu129
- Driver：590.48.01

## 运行命令

```bash
cd /volume/njiang/workspace/sandbox/cutedsl_tutorial/code/01_vector_add
CUDA_VISIBLE_DEVICES=0 /volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python vector_add.py --num-elements 1000003
```

选择 `1,000,003` 是为了验证非整 block 的尾部处理。

## 实际验证结果

```text
Compiling CuTe DSL vector-add kernel...
Launching with 1000003 elements...
GPU: NVIDIA B200
max_abs_error: 0.000e+00
first five results: [-0.05659037083387375, 0.5634089708328247, -0.4320492446422577, -0.2658909559249878, -0.050401270389556885]
PASS
```

这里暂不加入向量化和 TiledCopy；它们会作为后续章节相对于这个标量基线的增量优化。
