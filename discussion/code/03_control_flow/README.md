# 第 3 章代码：控制流与元编程

运行：

```bash
cd /volume/njiang/workspace/sandbox/cutedsl_tutorial/code/03_control_flow
CUDA_VISIBLE_DEVICES=0 /volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python control_flow.py
```

脚本会在同目录的 `generated_ir/` 中保存原始 MLIR，并打印 `scf.for`、`scf.while`、`scf.if` 的数量。`generated_ir/` 是运行生成物，不纳入版本控制。
