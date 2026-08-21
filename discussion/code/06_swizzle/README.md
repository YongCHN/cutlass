# 第 6 章代码：Swizzle 与 ComposedLayout

运行：

```bash
cd /volume/njiang/workspace/sandbox/cutedsl_tutorial/code/06_swizzle
/volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python swizzle_visualizer.py
CUDA_VISIBLE_DEVICES=0 /volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python smem_bank_probe.py
```

visualizer 证明函数映射与 XOR involution；GPU probe 则让 32 个 thread 成对使用同一物理指针 swizzle 写入和读回共享内存。它验证地址变换的正确性，但不把静态 bank 模型冒充成 profiler 性能数据。
