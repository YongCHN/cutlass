# 第 6 章：Swizzle、ComposedLayout 与 bank conflict

标签：`[通用/SM80+]`
先修：第 4、5 章
状态：正文与示例完成；CuTe DSL 4.7.0 / B200 L0/L1 验证通过。

## 6.1 本章要回答的问题

普通 Layout 用整数乘加把坐标映射成 offset，但高性能 shared-memory Layout 往往还要对地址 bit 做局部排列。本章回答：**逻辑 Layout 和非仿射地址变换怎样叠加，为什么这能改变 bank 映射而不改变逻辑数据？**

需要同时保留两种视角：

```text
逻辑视角：coordinate -> logical offset
物理视角：logical offset -> transformed physical offset
```

Swizzle 改的是第二段。只要写入和读取使用同一契约，上层仍可按原 logical coordinate 访问数据。

## 6.2 `ComposedLayout` 的三部分

CuTe 的 ComposedLayout 可写成：

```text
ComposedLayout = inner o offset o outer
```

对坐标 `c`：

```text
C(c) = inner(offset + outer(c))
```

三部分各自承担：

| 部分 | 作用 |
|---|---|
| `outer` | 从逻辑坐标生成基础 offset，通常是普通 affine Layout |
| `offset` | 在进入地址变换前平移基础 offset |
| `inner` | 对 offset 做最终变换；本章使用 XOR Swizzle |

构造：

```python
outer = cute.make_layout(32, stride=32)
swizzle = cute.make_swizzle(5, 0, 5)
composed = cute.make_composed_layout(swizzle, 0, outer)
```

输出：

```text
S<5,0,5> o 0 o 32:32
```

ComposedLayout 不是把 Swizzle “烘焙”成普通 stride。XOR 是 bit-level 非仿射变换，无法用单一整数 stride 完整表达；因此 inner 与 outer 必须在类型中分别保留。

当前 CuTe DSL 的 `make_composed_layout` 要求 outer 是 affine `Layout`。若需要继续组合，应先明确哪一层是地址变换、哪一层是坐标重参数化，不能随意把任意 composed object 再塞进 outer 位置。

## 6.3 `Swizzle<B,M,S>` 的 bit 含义

对正 shift 的常见形式 `Swizzle<B,M,S>`：

- `B`：参与 XOR 的 bit 数；
- `M`：保持不动的最低 bit 数，也是目标 bit 区域的起点；
- `S`：source bit 区域相对 target bit 区域的位移。

直观表示：

```text
target bits [M, M+B) ^= source bits [M+S, M+S+B)
```

例如：

```text
S<5,0,5>
```

把 bit `[5,10)` XOR 到 bit `[0,5)`。对实验中的 offset：

```text
logical = lane * 32
physical = (lane * 32) XOR lane
```

所以：

```text
lane 0: 0   -> 0
lane 1: 32  -> 33
lane 2: 64  -> 66
...
lane 31:992 -> 1023
```

当 source/target bit 区域不重叠时，同一 XOR 变换是 involution：

```text
swizzle(swizzle(x)) = x
```

这解释了为何同一个描述符既能编码逻辑到物理，也能恢复对应映射；它不表示任意两次不同层级的 API 调用都可混用，地址单位和向量粒度仍必须一致。

## 6.4 从 offset 到 shared-memory bank

对常见的 32-bit word 访问，可用简化模型：

```text
bank = element_offset mod 32
```

若一个 warp 的 lane `i` 都访问：

```text
logical_offset = i * 32
```

则：

```text
bank = (i * 32) mod 32 = 0
```

所有 lane 落到同一个 bank。应用 `S<5,0,5>` 后：

```text
physical_offset = (i * 32) XOR i
bank = physical_offset mod 32 = i
```

静态模型里，32 个 lane 分散到 32 个 bank。

但这个模型只是理解工具。真实 bank 行为还受以下因素影响：

- dtype 宽度；
- 每条指令的 vector width；
- warp 请求被拆成多少个 memory transaction；
- 相同地址广播；
- alignment 与跨 transaction 边界；
- 具体 load/store/TMA/ldmatrix 指令的地址解释。

因此“offset 表中 bank 互异”不是 profiler 结论。要声称性能改善，必须在目标架构上固定指令形态并测量 bank-conflict counter、吞吐或延迟。

## 6.5 地址单位：element offset 与 byte address

这是本章最重要的工程边界之一。

### Layout/visualizer 路径

示例 `swizzle_visualizer.py` 把值解释为 FP32 element offset，因此：

```text
bank = offset % 32
```

它特意选择 `S<5,0,5>`，让表格可以手算。

### Pointer 路径

`Pointer.apply_swizzle` 操作底层 SMEM 地址。标准描述符：

```python
cutlass.Swizzle.from_name("s32b")
cutlass.Swizzle.from_name("s64b")
cutlass.Swizzle.from_name("s128b")
```

分别编码 CUDA/TMA 约定的 32B、64B、128B swizzle；例如 `s128b` 对应 `S<3,4,3>`。其中最低 4 个 byte-address bit 保持不动，体现 16-byte 基础粒度。

不要把 visualizer 中按 element 编号定义的 `S<5,0,5>` 机械搬到 byte pointer 上。若 dtype 为 FP32，同一个数值增量在两层相差 4 倍；一旦忽略单位，bit 区域和 bank 推导都会错位。

## 6.6 低层物理指针与高层向量 API 不要混搭

CuTe DSL 4.7.0 提供两组相关接口：

### 低层物理地址

```python
physical_ptr = logical_ptr.apply_swizzle(swizzle)
physical_ptr.store(value)
value = logical_ptr.apply_swizzle(swizzle).load()
```

适合必须拿到物理地址的操作，例如某些 `cp.async` destination，以及本章逐 lane 的标量探针。

### 高层 swizzled load/store

```python
vec = logical_ptr.load_swizzled(swizzle, count=N)
logical_ptr.store_swizzled(vec, swizzle)
```

它构造带 swizzle 信息的 CuTe Tensor，并围绕 128-bit vector remap 地址。文档允许标量，但高效用法是成对使用相同宽度的 vector load/store。

开发探针时曾把：

```text
apply_swizzle(...).store(scalar)
```

与：

```text
load_swizzled(...)
```

混在一起。程序能编译运行，却产生 lane 内可重复的置换，而不是原序列。最终示例刻意让物理写和物理读成对，从而验证同一地址契约。

可迁移规则是：

1. 先决定使用 physical-pointer 还是 typed/vector Tensor 层；
2. 读写两侧保持同一层级、同一 swizzle、同一 logical base；
3. 宽向量路径保持相同 `count` 和合法 alignment；
4. 不把“同一个 Swizzle 对象”误当成“任意 API 组合都等价”。

## 6.7 Alignment、period 与 vector width

Swizzle 不会取消对齐要求。反而，TMA/ldmatrix 和宽 load/store 往往同时要求：

- allocation base 满足 16B/32B/128B 等对齐；
- tile shape 与 swizzle period 相容；
- 每线程起点不破坏 vector alignment；
- 下一行/下一 tile 的 leading dimension 满足编码约束。

示例分配：

```python
smem = cutlass.Array(
    cutlass.Float32,
    1024,
    space=cutlass.AddressSpace.smem,
    alignment=128,
)
```

这里的 `alignment=128` 是 byte alignment。它保证 allocation base 的契约，但不能证明每个派生 pointer 都仍有 128-byte alignment；pointer 加上 element offset 后，可证明的 alignment 会下降。

## 6.8 Gather/scatter 与非平凡 iterator

ComposedLayout 的思想不限于 shared-memory bank swizzle：outer 负责坐标映射，inner 可以代表额外地址变换。这使它在抽象上接近 gather/scatter view 或带排列的 iterator。

但 CuTe Tensor 是否支持某种 composed inner，取决于具体 lowering：

- 普通 affine outer 最容易被分析；
- 自定义 inner function 可能不能用于 memory-backed Tensor；
- vectorized load/store 对 composed layout 可能有额外限制；
- 某些硬件指令只接受有限的预定义 swizzle 模式。

所以“Layout 代数上可表达”不等于“目标 memory op 可 lower”。应把构造合法性、正确性和指令可用性分成三层验证。

## 6.9 TMA 与 `ldmatrix` 的附加约束

后续章节会分别展开，这里先建立边界：

- TMA tensor map 有明确的 swizzle enum、box extent、global/SMEM stride 和 alignment 规则；
- TMA 写入 SMEM 的物理布局必须与 consumer 使用的 swizzle 完全一致；
- `ldmatrix/stmatrix` 按 warp lane 和矩阵片段解释地址，不接受任意“数学上无冲突”的 Layout；
- MMA operand Layout 还受数据类型、instruction shape 和 major mode 约束。

因此不要从本章的 `S<5,0,5>` 教学表直接推导 TMA descriptor。生产代码优先从对应 copy/MMA atom 需要的 Layout 出发，再验证 bank 行为。

## 6.10 两个可执行示例

### 静态映射可视化

代码：[swizzle_visualizer.py](../code/06_swizzle/swizzle_visualizer.py)

运行：

```bash
cd /volume/njiang/workspace/sandbox/cutedsl_tutorial/code/06_swizzle
/volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python swizzle_visualizer.py
```

它验证：

- `ComposedLayout(c) == swizzle(outer(c))`；
- 原 offset 全落 bank 0；
- swizzled offset 分散到 bank 0–31；
- concrete XOR 映射满足 involution。

B200 环境输出从：

```text
0: 0 -> 0 -> 0
1: 32 -> 33 -> 1
```

一直到：

```text
31: 992 -> 1023 -> 31
PASS
```

这是 L0 映射验证。

### B200 SMEM 往返

代码：[smem_bank_probe.py](../code/06_swizzle/smem_bank_probe.py)

运行：

```bash
CUDA_VISIBLE_DEVICES=0 /volume/njiang/workspace/sandbox/.venv-cutedsl-4.7/bin/python smem_bank_probe.py
```

32 个 thread 使用 canonical `s128b` 描述符，将 lane id 写入 swizzled physical pointer；CTA 同步后，再由相同 logical pointer 和相同变换读回。实测：

```text
dst=[0.0, 1.0, ..., 30.0, 31.0]
PASS
```

这是 L1 数值正确性验证，不是性能 benchmark。完整记录见 [B200 验证日志](validation_log.md)。

## 6.11 常见误区

### 把 Swizzle 当成 shape 变换

Swizzle 改地址 bit，不交换 logical mode，也不改变 domain size。

### 混淆 element offset 与 byte address

先写清单位，再标 bit 区域。Pointer preset 不应从 element 表格按名称猜出。

### 只在写入端应用 Swizzle

consumer 必须使用匹配的逻辑到物理映射，否则读到的是排列后的物理数据。

### 混用 physical-pointer 和 typed vector API

即使描述符相同，二者的 lowering 粒度也可能不同。保持读写 API 成对。

### 看到 32 个不同 bank 就声称“无冲突且更快”

真实冲突按指令和 transaction 判断。静态表只能生成假设，必须用 profiler/benchmark 验证性能。

### 认为任何 Swizzle 都可交给 TMA/MMA

硬件 descriptor 只支持规定模式，还要同时满足 shape、stride 与 alignment。

## 6.12 本章检查清单与练习

分析一个 swizzled SMEM Layout 时：

1. 写出 outer Layout、offset 和 inner transform。
2. 标注 offset 是 element 还是 byte 单位。
3. 展开 `B/M/S` 对应的 source/target bit。
4. 对一个 warp 枚举 logical offset、physical offset 和 bank。
5. 写明读写两端使用的 API 层级和 vector width。
6. 检查 allocation 与派生 pointer alignment。
7. 把静态映射结论与实测性能结论分开。

练习：

1. 为 `S<2,1,3>` 标出参与 XOR 的 bit 区域，并计算 16 个输入。
2. 把 visualizer 的 logical stride 改成 16，观察 bank 分布是否仍为双射。
3. 为 FP16 重写 bank 计算，解释为什么 element offset 不能直接 `% 32`。
4. 把 SMEM probe 扩展为每线程一个 128-bit vector，并成对使用 `store_swizzled/load_swizzled`。
5. 用 Nsight Compute 对 swizzled/unswizzled 的固定指令版本测 bank-conflict counter；不要只比较 wall-clock 单次运行。

下一章会把 Pointer/Iterator 与 Layout 组合成真正的数据视图 Tensor，并复用 identity Tensor 同步追踪 tile 的全局坐标。
