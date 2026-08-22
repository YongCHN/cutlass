# 第 13 章代码：`cp.async` 与 matrix fragment

- `cp_async_roundtrip.py`：每线程 16-byte `cp.async.cg`、group commit/wait 和 CTA publication；
- `ldmatrix_roundtrip.py`：一个 warp 执行 b16 `ldmatrix/stmatrix` 的 x1/x2/x4 roundtrip。

在仓库根目录运行：

```bash
python discussion/code/13_async_copy_and_matrix/cp_async_roundtrip.py
python discussion/code/13_async_copy_and_matrix/ldmatrix_roundtrip.py
```

两个程序都会做 bit-exact 数值校验，并在保留的 PTX 中检查目标指令。
