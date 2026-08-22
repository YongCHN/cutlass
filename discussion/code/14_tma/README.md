# 第 14 章代码：TMA descriptor、partition 与 multicast

- `tma_copy_v0.py`：`TmaInfo`、`group_modes`、`tma_partition` 和单 stage G2S/S2G；
- `tma_transpose_v1.py`：TMA load、线程级 SMEM 转置、proxy fence、TMA store；
- `tma_multicast.py`：两 CTA cluster 的一次读取、多 CTA 投递协议。

在仓库根目录运行：

```bash
python discussion/code/14_tma/tma_copy_v0.py
python discussion/code/14_tma/tma_transpose_v1.py
python discussion/code/14_tma/tma_multicast.py
```

三个程序都要求 SM90+；本教程在 SM100 B200 上验证。
