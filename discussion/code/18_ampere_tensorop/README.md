# 第 18 章代码：Ampere warp Tensor Core GEMM

- `ampere_tensorop_gemm.py`：固定 `32×32×64` CTA tile、8 个 warp MMA atoms、
  2-stage `cp.async` SMEM ring、`ldmatrix` S2R 和 FP32 accumulator/direct store。

在仓库根目录运行：

```bash
python discussion/code/18_ampere_tensorop/ampere_tensorop_gemm.py
```

默认验证 FP16/BF16 和三组 shape。为保持 pipeline/ownership 可见，本教学基线要求
M/N 是 32 的正整数倍、K 是 64 的正整数倍；residue predication 在第 21 章加入。程序检查 PyTorch
reference 与 retained PTX，不做性能声明。
