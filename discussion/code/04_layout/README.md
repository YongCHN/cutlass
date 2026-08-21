# 第 4 章代码：Layout 基础

代码 `layout_basics.py` 包含两部分：

- 编译期构造并断言 compact、row-major、padding、broadcast、hierarchical 和 identity Layout；
- GPU kernel 在运行时构造动态 Layout，并把每个二维坐标对应的 offset 写回 Tensor 验证。

运行：

```bash
cd /volume/njiang/workspace/sandbox/cutedsl_tutorial/code/04_layout
CUDA_VISIBLE_DEVICES=0 /volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python layout_basics.py
```

同一个动态 compiled handle 会分别处理 `(3, 4):(8, 1)` 和 `(2, 5):(7, 1)`。
