# 第 16 章代码：分层规约与 Online Softmax

- `reduction_ladder.py`：连续 thread vector、warp shuffle tree、SMEM cross-warp CTA reduction；
- `online_softmax.py`：带 runtime ragged mask 的 `(max, exp-sum)` online pair reduction。

在仓库根目录运行：

```bash
python discussion/code/16_reduction_and_softmax/reduction_ladder.py
python discussion/code/16_reduction_and_softmax/online_softmax.py
```

两个示例都执行 reference 校验并检查 retained PTX。性能数据不在本章宣称，统一留到第 29 章的方法学中测量。
