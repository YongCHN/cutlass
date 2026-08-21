# 第 10 章代码：TiledCopy 与 Thread-Value Layout

- `tiled_copy_visual.py`：用 6 个 thread × 每线程 4 个 value 覆盖 `(4,6)` tile，逐项核对 TV Layout 与 `ThrCopy.partition_S` ownership。
- `vectorized_elementwise.py`：实现两个共享 TV Layout 的路径：完整 tile 使用显式 128-bit Copy Atom，并从 PTX 验证 `ld/st.global.v2.b64`（后端也可能表示为 `v4.b32`）；通用路径使用 identity Tensor 和逐 value predicate 处理 residue tile。

在 B200 验证机上运行：

```bash
cd /volume/njiang/workspace/sandbox/cutedsl_tutorial/code/10_tiled_copy
/volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python tiled_copy_visual.py
CUDA_VISIBLE_DEVICES=0 /volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python vectorized_elementwise.py
```

`generated_artifacts/` 保存本次编译的 PTX，但由 `.gitignore` 排除。示例只把 PTX 指令形态标记为 L4 证据；没有 benchmark，因此不声称达到 B200 峰值带宽。
