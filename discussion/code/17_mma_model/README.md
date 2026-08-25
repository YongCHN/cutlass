# 第 17 章代码：MMA 公共模型与 ownership inspector

- `mma_layout_inspector.py`：分别构造单 warp 和 `2×4×1` warp replication 的
  `m16n8k16` TiledMMA，打印静态 TV Layout/partition shape，导出每线程的 A/B/C
  坐标，并执行真实 MMA。

在仓库根目录运行：

```bash
python discussion/code/17_mma_model/mma_layout_inspector.py
```

程序同时验证 FP16/BF16 数值结果、partition coverage/multiplicity，以及 retained
PTX 中的 `ldmatrix` 和 `mma.sync`。本例是 ownership inspector，不用于性能声明。
