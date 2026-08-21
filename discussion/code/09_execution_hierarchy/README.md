# 第 9 章代码：执行层级、索引与边界

- `execution_hierarchy.py`：用一个二维 grid 和二维 CTA 验证 thread、lane、warp、warpgroup、CTA 与全局线性编号。
- `vector_add_masked.py`：对数据 Tensor 和 identity Tensor 应用相同 `(1,128)` tile，以全局坐标生成尾 tile predicate；同一个动态 Layout 编译句柄运行多组 shape。

在 B200 验证机上运行：

```bash
cd /volume/njiang/workspace/sandbox/cutedsl_tutorial/code/09_execution_hierarchy
CUDA_VISIBLE_DEVICES=0 /volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python execution_hierarchy.py
CUDA_VISIBLE_DEVICES=0 /volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python vector_add_masked.py
```

第二个示例有意保持每线程一个标量。第 10 章再用 Thread-Value Layout 将每线程 ownership 扩展为多个连续值，并讨论 predicate 与向量化的相互约束。
