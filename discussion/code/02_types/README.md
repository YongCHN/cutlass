# 第 2 章代码：类型与值

运行环境：NVIDIA B200、CuTe DSL 4.7.0，解释器：

```text
/volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python
```

远端目录：

```text
/volume/njiang/workspace/sandbox/cutedsl_tutorial/code/02_types
```

按顺序运行：

```bash
CUDA_VISIBLE_DEVICES=0 /volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python types_and_constexpr.py
CUDA_VISIBLE_DEVICES=0 /volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python struct_and_dataclass.py
CUDA_VISIBLE_DEVICES=0 /volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python expected_compile_errors.py
```

第三个脚本是负向测试：三次编译都失败才输出 `PASS`。
