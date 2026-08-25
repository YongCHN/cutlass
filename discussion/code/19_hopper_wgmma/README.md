# 第 19 章代码：Hopper WGMMA

- `hopper_wgmma_gemm.py`：最小 `TMA → SMEM descriptor → WGMMA → FP32 RMEM`
  GEMM，固定 `(64,64,16)`。

在仓库根目录运行：

```bash
python discussion/code/19_hopper_wgmma/hopper_wgmma_gemm.py
```

程序始终按 `sm_90a` 编译并检查 retained PTX。H100/H200 上继续执行数值验证；
B200 不能启动 SM90a cubin，因此只报告真实 compile/L4 PASS，不能把它记为 L1。
