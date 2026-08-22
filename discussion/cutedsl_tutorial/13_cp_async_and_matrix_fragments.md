# 第 13 章：`cp.async` 与 `ldmatrix/stmatrix`

> 范围标签：`[SM80+/SM90+][Experimental]`
>
> 前置章节：第 6 章 Swizzle、第 10 章 TiledCopy、第 11 章 SMEM、第 12 章同步
>
> 配套代码：[`13_async_copy_and_matrix/`](../code/13_async_copy_and_matrix/README.md)

## 13.1 本章要回答的问题

Ampere 风格 Tensor Core kernel 的 A/B 数据路径通常不是“每个线程从 GMEM 读几个标量，然后直接执行 MMA”，而是：

```text
GMEM
  -- per-thread cp.async --> SMEM tile
  -- warp ldmatrix -------> register fragment
  -- mma -----------------> accumulator fragment
```

本章集中回答：

1. `cp.async` 的异步对象、copy group 和可见性分别是什么？
2. `commit_group`、`wait_group(N)` 与 CTA barrier 为什么缺一不可？
3. SM80 per-thread copy、SM90 descriptor-less bulk copy、TMA 有何区别？
4. `ldmatrix.x1/x2/x4` 怎样把 SMEM 的 8×8 b16 矩阵分发到 32 个 lanes？
5. `stmatrix` 为什么不能被笼统标成 SM80 指令？
6. Layout、swizzle、对齐和 row-start address 怎样共同决定 fragment？
7. “用了异步指令”为什么不等于“形成了流水线”？

## 13.2 先画清数据路径

第 10 章的 TiledCopy 描述“哪个 thread 拥有哪些 values”。本章增加两个硬件边界：

| 边界 | 发送者 | 目标 | 主要抽象 |
|---|---|---|---|
| GMEM → SMEM | SM80+ 每线程 | SMEM byte region | `cp.async` + copy group |
| SMEM → RMEM | 一个 warp 协作 | packed register words | `ldmatrix` |
| RMEM → SMEM | 一个 warp 协作 | matrix rows | `stmatrix`（SM90+） |

`cp.async` 解决“不要先把 GMEM 数据落进普通寄存器再写 SMEM”；`ldmatrix` 解决“按 MMA 需要的 lane/register 形式读取 SMEM”。两者不是同一种 copy，也不共享同一种完成协议。

## 13.3 Per-thread `cp.async`

SM80 non-bulk `cp.async` 由每个线程独立给出：

- 一个 SMEM destination address；
- 一个 GMEM source address；
- 4、8 或 16-byte copy size（具体 cache form 有额外限制）；
- cache operator。

本章示例采用：

```python
prims.cp_async_shared_global(
    smem.data_ptr() + thread_base,
    src.iterator.raw_ptr() + tile_base + thread_base,
    16,
    "cg",
)
```

每线程搬运 4 个 FP32，即 16 bytes。`cg` 形式绕过 L1，只在 16-byte copy 上合法；需要 4/8-byte copy 时应根据 API/ISA 约束选择 `ca`，不能只修改常量。

### 地址契约

16-byte 指令不自动证明地址满足 16-byte 对齐。示例的证明是：

```text
base pointer: assumed aligned
thread_base = tid * 4 FP32 = tid * 16 bytes
tile_base   = block * 512 FP32 = block * 2048 bytes
```

因此每个 source/destination 都保持 16-byte alignment。

## 13.4 Copy group 是每线程异步队列

典型序列是：

```python
issue_cp_async()
prims.cp_async_commit_group()
prims.cp_async_wait_group(0)
```

含义不是“整个 CTA 创建一个全局 group object”。更准确的模型是：

1. 当前线程发出若干 pending `cp.async`；
2. `commit_group` 把这些操作提交为下一组；
3. `wait_group(N)` 等到最多还有 `N` 个较新的 committed groups 未完成；
4. `wait_group(0)` 等待当前线程此前提交的所有 group 完成。

因此 group depth 可以表达 overlap：producer 在使用较老 stage 时允许若干较新 groups 保持 pending。单 stage roundtrip 使用 `0`，是最保守的完成点。

## 13.5 `wait_group(0)` 不能替代 CTA publication

本章代码在 wait 后仍执行 CTA barrier：

```python
prims.cp_async_wait_group(0)
prims.barrier_cta_sync(0)
```

原因是两个协议回答不同问题：

- wait：调用线程发出的 async copy 是否完成；
- CTA barrier：所有 producer 是否到达，并把 SMEM tile 发布给全部 CTA consumers。

示例让 lane `t` 读取 lane `(t+1) mod BLOCK` 写入的 16-byte region，再把它写回该 region 对应的 global slot。输出仍等于输入，但这使 barrier 真正承载跨线程依赖，编译器不能把它当作同线程私有 roundtrip 删除。

## 13.6 尾块不是免费得到的

[`cp_async_roundtrip.py`](../code/13_async_copy_and_matrix/cp_async_roundtrip.py) 明确要求：

```text
N % (128 threads × 4 elements/thread) == 0
```

生产 kernel 常见策略包括：

1. 单独的完整 tile 与 residue kernel；
2. 对 source-size 使用合法字节数，让硬件对剩余目标字节 zero-fill；
3. predicate 后改走同步标量/向量 copy；
4. 对输入做 padding，并把 padding 写进 API contract。

不要让越界 lane 跳过 CTA barrier。predicate 应保护 copy operand，而不是改变同步参与者集合。

## 13.7 SM90 descriptor-less bulk async

SM90 还有不使用 TensorMap descriptor 的 bulk copy，例如 GMEM↔SMEM 的一段连续 bytes。它与 SM80 `cp.async` 的核心区别是：

| 维度 | SM80 per-thread | SM90 bulk |
|---|---|---|
| issuer | 每线程各自提供地址 | 通常 elected 单线程发出大块 copy |
| 描述对象 | raw source/destination pointer | raw pointer + bulk byte count |
| G2S completion | copy group | 常接 mbarrier transaction bytes |
| S2G completion | group commit/wait | bulk group commit/wait |
| 多维坐标 | 软件计算地址 | 仍是线性 byte region |

它适合连续大块传输，但仍不是 TMA。第 14 章的 TMA descriptor 会额外编码多维 shape/stride、box、swizzle、OOB 行为等信息。

## 13.8 `ldmatrix` 的最小数据单位

对本章的 b16 路径，一份 matrix fragment 是：

```text
8 rows × 8 columns × 16 bits = 128 bytes
```

warp 有 32 lanes，因此每个 lane 获得：

```text
128 bytes / 32 lanes = 4 bytes = one 32-bit register word
```

这一个 word 打包两个 b16 元素。于是：

| 指令 | 逻辑输入 | 每 lane 返回 |
|---|---|---:|
| `ldmatrix.x1.m8n8.b16` | 1 个 8×8 | 1 个 i32 carrier |
| `ldmatrix.x2.m8n8.b16` | 2 个 8×8 | 2 个 i32 carriers |
| `ldmatrix.x4.m8n8.b16` | 4 个 8×8 | 4 个 i32 carriers |

这里的 `xN` 是 instruction fragment count，不是任意用户 Layout 的“复制 N 次”语义。

## 13.9 谁提供 row address

所有 32 lanes 都必须执行 warp matrix instruction，但并非每条地址 operand 都被同等使用。对 b16 m8n8 row form：

- x1 需要 8 个 row starts；
- x2 需要 16 个 row starts；
- x4 需要 32 个 row starts。

示例把第 `lane` 个地址统一写成：

```python
row_ptr = smem_in.data_ptr() + lane * 8
registers = prims.ldmatrix(row_ptr, NUM_MATRICES, prims.MMALayout.ROW)
```

在 x1/x2 中，超出所需 row-start 数量的 lanes 仍参与指令，只是其地址 operand 不构成额外矩阵行。统一公式减少了人为分支，也保持 warp convergence。

## 13.10 `MMALayout.ROW/COL` 不是普通 Tensor transpose

`MMALayout` 选择 matrix instruction 的 row/non-transposed 或 column/transposed form，改变 lane/register 内的 fragment 映射。它不会：

- 改写 GMEM Tensor 的 shape；
- 自动创建转置后的普通二维 Tensor；
- 自动修复 SMEM pitch 或 swizzle；
- 把 packed b8 carrier 转换成 b16 数值。

要确认输出逻辑坐标，必须同时分析 source row starts、instruction variant、register order 和 store-side mapping。

## 13.11 `stmatrix` 的架构边界

`ldmatrix` 是 Ampere Tensor Core 数据路径的常见组成部分；`stmatrix` 则是 SM90+ matrix store 能力。本章 roundtrip 同时使用二者，所以可执行示例最低要求 SM90。

对 SM80 kernel，应把路线写成：

```text
cp.async (SM80+) -> ldmatrix (SM75+/SM80 path) -> mma -> other store path
```

不能因为章节讨论 Ampere 风格加载，就把 `stmatrix` 也误标成 SM80。

## 13.12 Layout、Swizzle 与地址是同一份契约

`ldmatrix` 接收 shared address，不接收 CuTe Layout 对象。硬件只会解释最终地址和指令 variant。因此：

```text
logical coordinate
  -- SMEM Layout / swizzle --> physical byte address
  -- ldmatrix -------------> lane/register fragment
```

如果 producer 按 swizzled layout 写入，却按 compact row-major 公式生成 row pointer，代码可能：

- 读到错误元素；
- 仍然保持地址合法，因而没有异常；
- 在对称输入下偶然通过；
- 只有进入 MMA 后才表现为数值错误。

可靠做法是从 SMEM Tensor/Layout 推导 slice 或 address，而不是在远离 Layout 定义处手写另一个物理 pitch。

## 13.13 对齐要求要分层证明

至少记录三层：

1. SMEM allocation base alignment；
2. 每个 matrix row start 的对齐；
3. swizzle 后地址是否保持 instruction 需要的对齐。

本章静态 SMEM 数组使用 128-byte base alignment，每行 8×b16=16 bytes，因而所有 row starts 都是 16-byte aligned。这个证明不能直接外推到带 padding、interleave 或 sub-byte dtype 的 layout。

## 13.14 从异步 copy 到真正 pipeline

下面的程序虽然发出了异步指令，却没有隐藏延迟：

```text
issue -> commit -> immediately wait -> consume
```

要形成 pipeline，至少要有：

- 两个或更多独立 stages；
- producer 在当前 stage 消费期间填充未来 stage；
- 对每个 stage 的 full/empty 生命周期；
- prologue、steady state 和 tail；
- 资源预算允许 overlap 后仍保持 occupancy。

本章只建立数据移动 primitive。第 15 章才把 group depth 或 mbarrier phase 组织成可复用 pipeline。

## 13.15 两个可执行示例

### `cp_async_roundtrip.py`

验证内容：

- 同一 compiled handle 运行 512、2048、8192 个 FP32；
- 每线程恰好一条 16-byte `cp.async.cg`；
- cross-thread SMEM consumption；
- 精确数值 reference；
- PTX 中 copy、commit、wait 和 CTA barrier。

### `ldmatrix_roundtrip.py`

验证内容：

- FP16 和 BF16；
- x1、x2、x4；
- 普通 row-major Tensor → SMEM → packed registers → SMEM → Tensor；
- bit-exact roundtrip；
- PTX 中六种 matrix instruction specialization。

运行：

```bash
python discussion/code/13_async_copy_and_matrix/cp_async_roundtrip.py
python discussion/code/13_async_copy_and_matrix/ldmatrix_roundtrip.py
```

B200 的完整输出见 [`validation_log.md`](validation_log.md)。

## 13.16 常见错误

1. 发出 `cp.async` 后只做 CTA barrier，不做 copy-group wait。
2. `wait_group(0)` 后跨线程读 SMEM，却没有 CTA/named barrier publication。
3. 用 `cg` 发出 4/8-byte copy。
4. source/destination 没有满足实际 vector alignment。
5. residue lane 直接跳过全 CTA barrier。
6. 把 x4 理解成任意 4×4 或四倍向量化。
7. 只有部分 lanes 执行 `ldmatrix/stmatrix`。
8. 把 `MMALayout.COL` 当成高层 Tensor transpose。
9. swizzled producer 与 compact row-pointer 公式混用。
10. 在 SM80 目标上使用 `stmatrix`。
11. 看到异步指令就宣称实现了 latency-hiding pipeline。

## 13.17 与后续章节的接口

第 14 章将把线性、线程发出的 copy 提升为 descriptor-driven 多维 copy：

```text
GMEM Tensor/Layout
  -> TensorMap descriptor
  -> TMA coordinate tensor
  -> group_modes / tma_partition
  -> elected TMA transaction
```

第 15 章再把本章的 group completion 与第 14 章的 transaction completion 放进统一 stage ring。

## 13.18 检查清单与练习

检查清单：

1. 标明 copy 的 issuer 粒度。
2. 标明 byte count、cache operator 和 alignment。
3. 标明 commit/wait 属于哪个 async mechanism。
4. 单独标明 cross-thread publication barrier。
5. 记录完整 tile/residue contract。
6. 对 `ldmatrix` 写出每 lane carrier word 数。
7. 列出真正提供 row starts 的 lanes。
8. 从 Layout 推导 physical row address。
9. 分开记录 `ldmatrix` 与 `stmatrix` 的最低架构。
10. 用 PTX 证明指令 specialization，而不只检查结果。

练习：

1. 把 `cp_async_roundtrip.py` 的 `BLOCK` 改为 64，重新证明 tile divisibility。
2. 把 consumer permutation改为前一 lane，并证明仍是双射。
3. 为最后一个不完整 tile 设计 zero-fill 协议，但保持所有 threads 参与 barrier。
4. 手算 x2 时 lane 0、7、8、15 对应的 row start。
5. 给 b16 x4 的 512-byte SMEM region 画出 lane→i32 carrier 表。
6. 加入 `MMALayout.COL` 路径，并把 reference 改为显式 transpose。
7. 设计双 buffer 版本，说明哪些 independent work 真正与 copy overlap。

