# 第 16 章：分层规约与 Online Softmax

> 范围标签：`[通用/SM90+/SM100+][Experimental]`
>
> 前置章节：第 8 章 TensorSSA、第 10 章 ownership、第 11–12 章 SMEM/同步
>
> 配套代码：[`16_reduction_and_softmax/`](../code/16_reduction_and_softmax/README.md)

## 16.1 本章要回答的问题

同一个 `sum/max` 为什么要按 thread、warp、CTA、cluster 分层？Softmax 又为什么不只是“先 max 再 sum”？本章回答：

1. reduction 的 identity、associativity、dtype 分别约束什么？
2. thread vector、warp shuffle、CTA SMEM、cluster DSMEM/TMEM 各解决哪一级通信？
3. 为什么 FP32 sum 的 reduction tree 可能与 PyTorch 顺序不同？
4. online Softmax 的 `(max,sum)` state 怎样合并？
5. mask lane 如何保留 collective participation？
6. fast exp 与数值稳定性怎样设 reference tolerance？
7. reduction/Softmax 的 bandwidth 性能应该怎样测？

## 16.2 Reduction 是代数与 ownership 的组合

一个 reduction 需要：

```text
(value set, combine operator, identity, ownership partition, communication tree)
```

例如 sum：

```text
combine(a,b) = a+b
identity = 0
```

max：

```text
combine(a,b) = max(a,b)
identity = -infinity
```

identity 用于没有合法输入的 lane；associativity 允许把顺序重组为并行树；ownership 决定每个 participant 先局部处理哪些元素。

## 16.3 数学结合律不等于浮点逐位结合

实数加法满足结合律，但 IEEE 浮点会舍入：

```text
(a+b)+c != a+(b+c)
```

因此并行 sum 应区分：

- 数学 reference 是否相同；
- accumulation dtype 是否相同；
- reduction tree 是否相同；
- 允许的误差阈值；
- 是否要求 deterministic/bitwise reproducible。

本章 reduction 示例使用小整数的 FP32 表示，所有中间和可精确表示，从而对 sum 做 bit-exact 验证。Softmax 使用 transcendental approximation，采用显式 tolerance。

## 16.4 Accumulation dtype

FP16/BF16 输入通常不应直接用输入 dtype 累加长向量。常见策略：

| Input | Accumulator | 原因 |
|---|---|---|
| FP16/BF16 | FP32 | 减少舍入和 overflow/underflow |
| FP8/FP4 | FP32 或受控 FP16 | scale + 极窄 dynamic range |
| INT8 | INT32 | 保留乘加精度 |
| FP32 | FP32/FP64 | 按精度与吞吐需求选择 |

“输出 dtype 是 FP16”不代表 reduction state 也应是 FP16。accumulator cast-back 应在语义边界明确发生。

## 16.5 第一级：thread-local vector fold

让每线程先处理连续 values：

```python
values = ptr.load(count=ITEMS_PER_THREAD)
local_sum = values.reduce("add")
local_max = values.reduce("max")
```

优点：

- 连续 vector load；
- 大部分 combine 不需要线程通信；
- warp reduction 的输入从 `32×items` 压缩成每 lane 一个 scalar；
- 增加 ILP。

代价是每线程 registers 随 `ITEMS_PER_THREAD` 增加。过大的 vector 可能产生 spill 或降低 occupancy。

## 16.6 第二级：warp shuffle tree

butterfly all-reduce：

```python
for offset in (16, 8, 4, 2, 1):
    value = combine(value, shuffle_bfly(value, offset))
```

五步后 full warp 的每个 lane 都得到相同结果。若只有 lane 0 需要最终值，也可以用 down tree，但必须理解哪些 lanes 的中间值仍有效。

shuffle 只移动 registers，不经过 SMEM，适合 warp 内 communication。member mask、active lanes 和 divergence 仍须满足第 12 章契约。

## 16.7 `redux.sync` 与 shuffle tree

某些 integer add/min/max/bitwise operation 可由 `redux.sync` 一条硬件 collective 完成。float support 随架构和 operation 变化，不能假定任意 dtype/op 都有等价指令。

选择原则：

- ISA 原生支持且语义匹配时使用 redux；
- 需要自定义 state（如 Softmax pair）时使用 shuffle；
- 需要固定可解释 tree 时使用显式 shuffle；
- 用 PTX/SASS 确认实际 lowering。

## 16.8 第三级：CTA cross-warp reduction

warp shuffle 不能跨 warp。经典 CTA ladder 是：

```text
each warp reduces locally
  -> lane 0 stores one partial to SMEM
  -> CTA barrier
  -> warp 0 loads W partials, other lanes use identity
  -> warp 0 reduces again
  -> lane 0 publishes CTA result
```

对 128-thread CTA，SMEM 只需 4 个 partials/operator，而不是 128 个 thread values。第一级压缩越充分，cross-warp communication 越小。

## 16.9 为什么需要 CTA barrier

warp leaders 属于不同 warps。warp 0 读取 partial 前必须确认：

- 四个 warp leader 都已 store；
- shared writes 对 warp 0 可见；
- 没有 warp 因分支绕过 rendezvous。

warp barrier 或 shuffle 都不能完成 cross-warp publication。若用 atomic/REDS 聚合，仍需初始化、最终读取和 generation reuse 的顺序协议。

## 16.10 SMEM atomic/REDS 与显式 combine

cross-warp 可以选择：

1. 每 warp leader 对一个 SMEM scalar做 atomic/REDS；
2. 每 warp leader 写独立 slot，warp 0 显式 combine。

前者 storage 少、代码短，但 operation/dtype 支持有限，且 contention/ordering 需分析。后者通用、tree 可见，更适合教程和复合 state。

本章代码对 sum/max 都用独立 slots，使两种 operator 共享同一清晰协议。

## 16.11 第四级：cluster reduction

当一个输出 tile 由多个 CTAs 分担（例如 split-K GEMM），CTA partial 还需要 cluster/global 合并。SM90+ 路线可用：

- 每 CTA 把 partial 写入 DSMEM-visible SMEM；
- cluster barrier发布所有 partial；
- root CTA 用 `mapa` 读取 peer SMEM；
- root 合并并写结果；
- 所有 CTAs 在 peer read 完成前保持 lifetime。

也可以使用 remote atomic/REDS，arrival/memory scope 要匹配 cluster。第 12 章的 remote barrier 与 DSMEM lifetime 直接适用。

## 16.12 SM100 TMEM reduction

Blackwell MMA accumulator 常驻 TMEM，但 TMEM 是 CTA-private。cluster reduction 的典型路径不是“CTA 0 直接读取 peer TMEM”，而是：

```text
peer TMEM accumulator
  -> tcgen05 load/drain
  -> peer SMEM tile
  -> cluster publish
  -> root mapa(peer SMEM)
  -> elementwise cluster combine
```

必须同时管理 TMEM allocation/deallocation、async TMEM load/store wait、SMEM proxy fence和 cluster lifetime。该路线在第 23/30 章结合 GEMM/distributed reduction 展开，本章先建立层级模型。

## 16.13 Reduction ladder 示例

[`reduction_ladder.py`](../code/16_reduction_and_softmax/reduction_ladder.py) 对每行执行：

```text
contiguous vector load
  -> local Vector.reduce(sum,max)
  -> warp butterfly(sum,max)
  -> four SMEM partial pairs
  -> CTA barrier
  -> warp 0 butterfly
  -> two row outputs
```

specializations：

| items/thread | row width | 目的 |
|---:|---:|---|
| 1 | 128 | scalar baseline |
| 4 | 512 | 16-byte thread vector |
| 8 | 1024 | 32-byte logical vector / 更多 ILP |

每个 specialization 测 1、7、33 rows；sum/max 全部 bit-exact，并检查 vector load、shuffle、SMEM store 和 CTA barrier PTX。

## 16.14 Softmax 的稳定三遍形式

对一行 `x_i`：

```text
m = max_i x_i
l = sum_i exp(x_i - m)
y_i = exp(x_i - m) / l
```

减去最大值保证 exponent ≤ 0，避免正向 overflow。它通常需要：

1. max pass；
2. exp+sum pass；
3. normalize pass。

如果 exp 临时值写回 GMEM/SMEM，可少一次重新计算但增加 storage traffic；选择取决于 row width、register/SMEM 容量和 bandwidth。

## 16.15 Online Softmax 单元素更新

online state 是：

```text
(m, l)
m = 当前最大值
l = sum exp(x_i - m)
```

加入新元素 `x`：

```text
m_new = max(m, x)
l_new = l * exp(m - m_new) + exp(x - m_new)
```

若 `x <= m`，旧 scale 为 1，只增加 `exp(x-m)`；若 `x > m`，所有旧 exponent 都必须乘 `exp(m-x)` 重新缩放到新基准。

这样一遍 scan 同时得到 row max 与 normalized exp-sum，随后第二遍产生输出。

## 16.16 两个 online states 怎样合并

对于两个互不重叠子集的 states `(m_a,l_a)`、`(m_b,l_b)`：

```text
m = max(m_a, m_b)
l = l_a * exp(m_a - m)
  + l_b * exp(m_b - m)
```

这就是可用于 parallel reduction 的 combine。它在精确实数意义上可结合，因此可以：

- thread 内顺序 scan；
- warp 内 shuffle tree；
- CTA 内 SMEM partial merge；
- cluster 内进一步 merge。

有限精度下不同 tree 仍可能有微小差异。

## 16.17 Online state 的 identity

理想 identity 是：

```text
m = -infinity
l = 0
```

但直接计算两个 identity 的 `m_a - m` 会出现 `-inf - -inf = NaN`。本章使用最小有限 FP32：

```text
m = -FLT_MAX
l = 0
```

identity 与 identity 合并时 scale 是 `exp(0)`，仍乘 0；与有限值合并时 identity contribution 保持 0，避免 NaN。

另一种实现是显式判断 `l==0`，但会给 combine tree 增加分支。

## 16.18 Ragged/causal mask 的参与者规则

本章每行有 runtime `valid_cols`：

- `col < valid_cols` 才更新 online state；
- invalid lanes 保留 identity pair；
- 所有 lanes 仍执行 shuffle；
- 所有 warps 仍到达 CTA barrier；
- invalid output 写 0。

mask 改变的是 reduction operand，不是 collective participant set。这与第 9/12 章的 residue/barrier 原则一致。

## 16.19 全 mask 行要单独定义

如果 `valid_cols=0`，数学 Softmax 没有自然概率分布：

- max 是 `-∞`；
- sum 是 0；
- normalize 会除 0。

不同框架可能要求输出全 0、NaN、保留 mask sentinel 或直接拒绝。本章 host contract 要求每行至少一个有效元素，测试 lengths 从 1 开始。生产 API 必须把全 mask 行语义写清，而不是依赖偶然行为。

## 16.20 `fastmath=True` 与 `ex2.approx`

CuTe fast exp 可能 lower 为近似 base-2 exponential，例如 PTX `ex2.approx.ftz.f32`，并通过常数完成自然指数换底。它通常比精确数学库快，但带来：

- approximation error；
- flush-to-zero 行为；
- 极小概率项差异；
- reduction tree 放大/缩小误差。

因此 Softmax 不能沿用 bit-exact reference。阈值应基于 dtype、输入范围、row width 和下游需求制定，并记录最大误差而非只打印 PASS。

## 16.21 Online Softmax 示例

[`online_softmax.py`](../code/16_reduction_and_softmax/online_softmax.py) 使用：

- 128-thread CTA/row；
- thread-strided online scan；
- warp `(m,l)` butterfly；
- 四个 SMEM pair partials；
- warp 0 CTA pair merge；
- runtime ragged lengths；
- second pass normalize；
- stats `(row_max,row_exp_sum)` 输出。

测试 shape：

```text
(1,1), (5,31), (7,128), (9,257), (4,1000)
```

覆盖 sub-warp、warp residue、完整 CTA width、多轮 loop 和 1000-column row。B200 最大 probability error 约 `1.2e-7`。

## 16.22 Softmax 与 attention 的接口

FMHA 中 score tile 通常不完整落到 GMEM：

```text
QK^T fragment
  -> scale + mask
  -> online max/sum update
  -> rescale old accumulator
  -> accumulate P*V
```

当新 score block 提高 running max 时，不仅 `l` 要 rescale，已有 output accumulator `O` 也要乘：

```text
alpha = exp(m_old - m_new)
O_new = alpha * O_old + exp(S_block - m_new) * V_block
```

本章 pair reduction是第 26 章 FMHA online state 的数学基础。

## 16.23 Norm/RMSNorm 的相同层级

LayerNorm/RMSNorm 也使用相同 ladder：

- thread local sum/sumsq；
- warp reduction；
- CTA/cluster merge；
- 计算 mean/variance 或 RMS；
- normalize + affine transform。

差别在 operator/state：Welford state 通常是 `(count,mean,M2)`，也需要定义可结合 merge，而不是分别 sum/sumsq 后假定所有数值范围都稳定。

## 16.24 性能：先算最低流量

out-of-place Softmax 至少读取输入并写输出：

```text
minimum_bytes = rows * cols * (sizeof(input) + sizeof(output))
```

如果三遍都重新读 GMEM或写 exp 临时值，实际 bytes 更高。benchmark 必须说明：

- 是否从 cache 热数据开始；
- exp 临时值在哪里；
- mask/lengths traffic 是否计入；
- input/output dtype；
- warmup、重复、同步；
- GB/s 是按最低算法流量还是实际估算流量。

本章只做 L1/L2/L4，不报告 GB/s；统一 benchmark 方法留到第 29 章。

## 16.25 选择 row-to-CTA 映射

常见方案：

| Row width | 映射候选 |
|---|---|
| 很短（≤warp capacity） | one warp/row，多个 rows/CTA |
| 中等 | one CTA/row |
| 很长 | 多 CTA/row + second-stage/cluster reduction |
| attention tile | warpgroup/CTA tile，与 MMA Layout 对齐 |

本章始终 one CTA/row，重点是层级协议，不声称对长度 1 或 31 性能最优。

## 16.26 常见错误

1. max 的 invalid identity 写成 0，导致全负输入错误。
2. sum 用 FP16/BF16 长向量累加。
3. warp shuffle mask 包含未执行 collective 的 lane。
4. lane 0 读取 cross-warp SMEM partial 前没有 CTA barrier。
5. identity lanes 未初始化就参加第二级 warp reduction。
6. 把 floating sum 与不同 tree 的 reference 要求 bit-exact。
7. Softmax 直接计算 `exp(x)`，没有减 max。
8. running max 更新时只加新 exponent，没有 rescale旧 sum。
9. 合并 online pairs 时直接 `l_a+l_b`，忽略不同 max 基准。
10. 用 `(-inf,0)` 却没有处理 identity-identity 的 NaN。
11. masked lanes 跳过 warp/CTA collective。
12. 全 mask 行语义未定义。
13. fast exp 使用零 tolerance。
14. 只验证每行和为 1，不与逐元素 reference 比较。

## 16.27 与后续章节的接口

第 17–24 章会把 reduction consumer 接到 MMA accumulator fragment/TMEM：

- GEMM split-K partial reduction；
- epilogue amax/scaling；
- SM100 TMEM drain；
- cluster tile merge。

第 26 章 FMHA 会复用 online `(m,l)`，并增加 output accumulator rescale。第 30 章分布式 kernel 会把同样的 associative-state 思维扩展到 GPU 间 collective。

## 16.28 检查清单与练习

检查清单：

1. 写出 operator、identity 和 state dtype。
2. 说明数学结合律与浮点 reproducibility。
3. 标注每线程 ownership 和 vector width。
4. 标注 warp tree、active mask 和结果所在 lanes。
5. 标注 cross-warp SMEM slots 与 CTA barrier。
6. cluster reduction证明 DSMEM/TMEM handoff 与 lifetime。
7. Softmax 写出 online update 和 pair merge。
8. mask 用 identity operand，不改变 collective participation。
9. 单独定义全 mask 行。
10. 根据 fastmath/dtype/width制定 tolerance。
11. 同时验证逐元素结果、row sum 和内部 stats。
12. 性能报告写清实际/最低 memory traffic。

练习：

1. 给 reduction ladder 增加 min，并写出 identity。
2. 把 warp butterfly 改成 down tree，说明哪些 lanes 的结果有效。
3. 用 SMEM atomic add 替换 cross-warp slots，比较协议与 PTX。
4. 给 Softmax 增加 `valid_cols=0` 的明确全零语义。
5. 将 input 改为 FP16、accumulator 保持 FP32，制定合理 tolerance。
6. 把 one CTA/row 改成 four warps/four rows，分析短行利用率。
7. 将 online state 扩展为 `(m,l,O)`，实现一个不含 MMA 的分块 attention toy。

