# 第 7 章代码

本目录包含两个互补示例：

- `tensor_views.py`：对 memory-backed Tensor 与 identity Tensor 做相同的 `local_tile`，由坐标 Tensor 恢复全局坐标。
- `dynamic_layout.py`：把一维 Tensor 的 extent 标记为动态，证明同一个 compiled handle 可处理多个长度。

在 B200 验证机上运行：

```bash
cd /volume/njiang/workspace/sandbox/cutedsl_tutorial/code/07_tensor
CUDA_VISIBLE_DEVICES=0 /volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python tensor_views.py
CUDA_VISIBLE_DEVICES=0 /volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python dynamic_layout.py
```

验证范围只覆盖 FP32、二维整 tile 切分和一维 contiguous 动态 Layout；边界 tile、非连续动态 stride 与向量化留到第 9、10 章。
