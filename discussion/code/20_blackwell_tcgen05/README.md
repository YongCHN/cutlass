# 第 20 章代码：Blackwell tcgen05、TMEM 与 UMMA

- `tmem_roundtrip.py`：32-column TMEM allocation、store/load wait、dealloc 与
  allocation-permit relinquish；
- `tcgen05_1cta_gemm.py`：独立固定形状 `128×128×64` CTA_1 FP16 GEMM；
- `tcgen05_2cta_gemm.py`：调用仓库 canonical `2cta_mma_basic.py`，固定三组
  cluster-grid 验证并审计 CTA_2 PTX。保留 canonical kernel 可避免教程复制一份容易
  与 experimental API 漂移的 500 行 cluster protocol。

在完整 CUTLASS checkout 根目录运行：

```bash
python discussion/code/20_blackwell_tcgen05/tmem_roundtrip.py
python discussion/code/20_blackwell_tcgen05/tcgen05_1cta_gemm.py
python discussion/code/20_blackwell_tcgen05/tcgen05_2cta_gemm.py
```

三个程序都要求 datacenter Blackwell（SM100/SM103），执行数值 reference 并检查
retained PTX；consumer Blackwell/Thor 不能运行 TMEM/tcgen05 MMA 路线。
