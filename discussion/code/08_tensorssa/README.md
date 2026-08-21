# 第 8 章代码

`tensorssa_ops.py` 用一个线程完整展示：

- memory-backed Tensor 的 `load()` 如何产生 TensorSSA；
- `(2,3) + (2,1) + (1,3)` 的广播；
- `make_fragment_like` 创建的 RMEM fragment 与 TensorSSA 之间的 `store/load`；
- `reduction_profile=(None, 1)` 的逐行规约；
- `(None, 1)` 对 TensorSSA 的切片。

在 B200 验证机上运行：

```bash
cd /volume/njiang/workspace/sandbox/cutedsl_tutorial/code/08_tensorssa
CUDA_VISIBLE_DEVICES=0 /volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python tensorssa_ops.py
```

该示例只验证单线程、静态 shape 的寄存器级值语义。跨线程规约与 MMA accumulator fragment 分别留到第 16 章和 Tensor Core 篇。
