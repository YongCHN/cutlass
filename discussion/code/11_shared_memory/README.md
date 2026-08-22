# 第 11 章代码：Shared Memory 与 tiled transpose

`tiled_transpose.py` 包含 compact (`32 x 32`) 和 padded (`32 x 33`) 两种 SMEM Layout。每个 CTA 连续处理两个 tile，从而真实触发 buffer reuse 所需的第二个 CTA barrier；同一编译句柄覆盖完整 tile 和 residue tile。

在仓库根目录运行：

```bash
python discussion/code/11_shared_memory/tiled_transpose.py
```

程序会验证数值转置、自动推导的 SMEM 字节数，并从保留的 PTX 中检查 shared load/store 和 CTA barrier。padding 对 bank conflict 的影响属于访问映射结论；只有配套计时或性能计数器才能进一步形成性能结论。
