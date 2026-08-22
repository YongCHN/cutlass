# 第 15 章代码：多级流水线与 warp specialization

- `pipeline_software_fill.py`：线程搬运的 full/empty mbarrier stage ring；
- `tma_pipeline_warpspec.py`：TMA producer warp、SMEM consumer warp，以及显式 prologue/steady/tail。

在仓库根目录运行：

```bash
python discussion/code/15_pipeline/pipeline_software_fill.py
python discussion/code/15_pipeline/tma_pipeline_warpspec.py
```

两个示例都验证多个 stage depth 和跨 phase wrap 的 runtime tile count，并检查保留 PTX 中的 pipeline 指令族。

