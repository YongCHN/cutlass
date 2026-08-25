# 第 17 章：MMA Atom、TiledMMA 与 partition 公共模型

> 范围标签：`[通用]`（公共抽象）；配套执行核使用 `[SM80+]`
>
> 前置章节：第 4–8 章 Layout/Tensor/TensorSSA、第 10 章 TiledCopy、第 13 章 matrix fragment
>
> 配套代码：[`17_mma_model/`](../code/17_mma_model/README.md)

## 17.1 本章要回答的问题

Tensor Core API 最容易被一串相似名字淹没。本章只建立一条可跨架构复用的推理链：

1. `MmaOp → MmaAtom → TiledMma → ThrMma` 每层增加了什么？
2. 一条 `m16n8k16` 指令与一个 `32×32×16` CTA footprint 怎样连接？
3. `atom_layout_mnk`、`permutation_mnk` 和 operand tile shape 为什么不能混为一谈？
4. `partition_A/B/C` 返回的 value/repeat modes 表示什么？
5. A/B 的重复 ownership 为什么通常是复用，而 C 的重复 ownership 却常是错误？
6. `make_fragment_A/B/C` 为什么不能简单翻译成“分配寄存器”？
7. `cute.gemm` 的 `D,A,B,C` 参数和 accumulator 更新语义是什么？
8. 怎样用 identity Tensor 对 coverage、multiplicity 和无重叠做机器验证？

本章的目标不是记住某代 Tensor Core 的所有指令，而是形成一个检查表。第 18–20 章只替换硬件 op、operand storage 与同步协议，这条骨架保持不变。

## 17.2 先固定数学坐标约定

本教程统一使用：

```text
A: (M,K)
B: (N,K)
C: (M,N)

C[m,n] = sum_k A[m,k] * B[n,k]
```

所以 PyTorch reference 是：

```python
reference = a.float() @ b.float().T
```

`B(N,K)` 不是数学公式写错，而是 CuTe dense GEMM 常用的 operand-B logical view。SM80 `mma.sync...row.col` 的 B operand 固定为 K-major；使用 `B(N,K)` 后，A/B 都能以连续 K 存放。若外部 API 接收 `B(K,N)`，host wrapper 应显式转置 view/stride，而不是让正文在两套坐标中来回切换。

## 17.3 四层对象各负责什么

```text
MmaOp
  └─ make_mma_atom
       └─ MmaAtom
            └─ make_tiled_mma(atom_layout, permutation)
                 └─ TiledMma
                      └─ get_slice(thread_idx)
                           └─ ThrMma
```

### 17.3.1 `MmaOp`：选择硬件操作

`MmaOp` 是一个 Python 层描述，确定：

- 指令家族和目标架构；
- A/B/accumulator dtype；
- instruction shape `(M,N,K)`；
- operand layout、参与者数和 storage 约束；
- op-specific runtime state。

本章使用：

```python
op = cute.nvgpu.warp.MmaF16BF16Op(
    cutlass.Float16,
    cutlass.Float32,
    (16, 8, 16),
)
```

它选择 SM80+ warp collective `mma.sync.m16n8k16`。它还没有决定 CTA 用几个 warp，也没有绑定任何 A/B/C Tensor。

### 17.3.2 `MmaAtom`：把 op 与指令 Layout metadata 结合

`cute.make_mma_atom(op)` 产生 Atom。Atom 的 trait 携带：

- `thr_id`：指令参与者映射；
- `shape_mnk`：单 atom 的数学 footprint；
- `tv_layout_A/B/C`：`(thread,value) → operand coordinate` 映射。

Atom 是“最小不可再分的协作操作”，不是“一条线程指令”。warp MMA atom 由 32 lanes 共同执行；WGMMA atom 由 warpgroup 参与；某些 UMMA atom 还关联 CTA group。

### 17.3.3 `TiledMma`：在 MNK 空间复制 atom

```python
tiled_mma = cute.make_tiled_mma(
    op,
    cute.make_layout((2, 4, 1)),
)
```

`(2,4,1)` 表示把 atom 分别沿 M/N/K 复制 2/4/1 次。对 `m16n8k16`：

```text
natural footprint = (2*16, 4*8, 1*16)
                  = (32, 32, 16)
participants      = 2*4*1 warps
                  = 256 threads
```

它回答的是“多少个 atom 在空间上协作”，不是 K-loop 跑多少次。K-loop repeat 来自 operand tile 相对于这个 footprint 的外层 mode。

### 17.3.4 `ThrMma`：当前线程看到的 slice

```python
tid = cute.arch.thread_idx()[0]
thr_mma = tiled_mma.get_slice(tid)
```

`ThrMma` 绑定 runtime thread index。只有到这一层，才能对具体 Tensor 调用：

```python
t_a = thr_mma.partition_A(tile_a)
t_b = thr_mma.partition_B(tile_b)
t_c = thr_mma.partition_C(tile_c)
```

它不是新硬件对象，也不会单独执行 MMA；它是 `TiledMma` ownership map 的一个 thread slice。

## 17.4 Instruction shape、atom layout、permutation 是三件事

| 对象 | 回答的问题 | 示例 |
|---|---|---|
| instruction shape | 一个硬件 atom 算多大的 MNK？ | `(16,8,16)` |
| atom layout | atom/participant 怎样在 MNK 空间复制？ | `(2,4,1)` |
| permutation tiler | value space 的目标 footprint/排列怎样扩展？ | `(32,32,16)` |

没有显式 permutation 时，TiledMMA 使用 atom shape 与 atom layout 形成 natural tiling。显式 `permutation_mnk` 可以规定更大的 value footprint 或排列，但它不是把 A/B 指针“自动转置”，也不是任意改变硬件指令的固定 operand layout。

生产 Ampere GEMM 常写：

```python
permutation_mnk = (
    atom_m * mma_m,
    atom_n * mma_n * 2,
    atom_k * mma_k,
)
```

额外的 N factor 会让一个 warp 在 value space 中覆盖更多 N repeats，以匹配更大的 coalesced matrix load/store。是否这样做要由 SMEM layout、`ldmatrix` copy atom 和 epilogue共同证明，不能孤立抄一个 tuple。

## 17.5 从 `Thr Layout VMNK` 读 participant 空间

本章 `2×4×1` inspector 打印：

```text
Thr Layout VMNK: (32,2,4,1):(1,32,64,0)
```

可按四个 mode 阅读：

```text
V: 32 lanes，stride 1
M: 2 个 warp positions，stride 32
N: 4 个 warp positions，stride 64
K: 1 个 position，因此 stride 0
```

thread offset 是：

```text
lane + 32*warp_m + 64*warp_n
```

它完整覆盖 `[0,255]`。这正是第 4–5 章 Layout 的应用：participant hierarchy 最终仍是 coordinate-to-index function。

## 17.6 TV Layout 是指令 ABI

对单 warp `m16n8k16`，CuTe 4.7.0 打印：

```text
TV Layout A: ((4,8),(2,2,2)):((32,1),(16,8,128))
TV Layout B: ((4,8),(2,2)):((16,1),(8,64))
TV Layout C: ((4,8),(2,2)):((32,1),(16,8))
```

不要把这三行只当调试字符串。它们编码了硬件 ABI：

- thread mode 怎样分解 32 lanes；
- 每 lane 有多少 logical values；
- thread/value pair 对应 operand 的哪个元素；
- 哪些 bit/mode 适合 `ldmatrix` 的 lane-to-register layout。

由 value shape 可直接得到：

```text
A: 2*2*2 = 8 values/lane
B: 2*2   = 4 values/lane
C: 2*2   = 4 values/lane
```

总量验证：

```text
32*8 = 256 = 16*16  (A)
32*4 = 128 = 8*16   (B)
32*4 = 128 = 16*8   (C)
```

总量相等只是必要条件。真正的 coverage 还要枚举坐标，排除两个 TV points 指向同一元素且漏掉另一个元素。

## 17.7 `partition_A/B/C` 的 shape

对恰好一个 instruction footprint 的 operand，inspector 打印：

```text
partition A shape: ((2,2,2), 1, 1)
partition B shape: ((2,2),   1, 1)
partition C shape: ((2,2),   1, 1)
```

概念上可标为：

```text
A: (MMA-value, repeat-M, repeat-K)
B: (MMA-value, repeat-N, repeat-K)
C: (MMA-value, repeat-M, repeat-N)
```

第一个 mode 保留 Atom 的嵌套 value shape；后两个 modes 是更大 tile 相对 TiledMMA footprint 的 repeats。本例 tile 恰好等于 footprint，所以 repeats 都为 1。

若第 18 章 A tile 是 `(32,64)`，而 TiledMMA 的 K footprint 是 16，则 per-thread A partition 的 K repeat 为 4。mainloop 的 `k_block` 就是在选择这一 mode：

```python
r_a[None, None, k_block]
```

因此“partition 后第 2/3 mode 是什么”必须从 operand role 与输入 tile shape 推导，不能只看 mode 编号猜含义。

## 17.8 Ownership 不等于所有 operand 都无重复

`2×4×1` atom layout 对 `(32,32,16)` footprint 的实测是：

| Operand | thread-values | logical domain | multiplicity |
|---|---:|---:|---:|
| A | `256×8=2048` | `32×16=512` | 4 |
| B | `256×4=1024` | `32×16=512` | 2 |
| C | `256×4=1024` | `32×32=1024` | 1 |

解释：

- 同一个 A tile 被 4 个 N-position warps读取；
- 同一个 B tile 被 2 个 M-position warps读取；
- 每个 C element 只属于一个 accumulator owner。

A/B multiplicity 是数据复用，不是 race，因为它们是 read operands。C 若 multiplicity > 1，则通常需要 split-K reduction/atomic 或别的显式合并协议；在普通 dense tile 中它意味着 output ownership 有问题。

所以 coverage proof 应分别检查：

```text
A/B: complete coverage + expected read multiplicity
C:   complete coverage + multiplicity exactly one
```

## 17.9 用 identity Tensor 导出映射

memory Tensor 与 coordinate Tensor 经过同一个 partition：

```python
t_s_a = thr_mma.partition_A(s_a)
t_c_a = thr_mma.partition_A(cute.make_identity_tensor((tile_m, tile_k)))
```

两者 layout 相同，但 `t_c_a[i]` 的值是 `(m,k)` 坐标。配套代码把每线程坐标写到：

```text
map_A[thread, value, 2]
map_B[thread, value, 2]
map_C[thread, value, 2]
```

host 侧再检查 set coverage 与每坐标出现次数。这比肉眼看 pretty-print 更稳健，也能在更换 atom layout 后成为 regression test。

## 17.10 `partition` 只定义 ownership，不移动数据

下面三行都只创建 views：

```python
t_s_a = thr_mma.partition_A(s_a)
t_s_b = thr_mma.partition_B(s_b)
t_g_c = thr_mma.partition_C(g_c)
```

它们不会发出 load/store，也不会创建 register fragment。真正的 SMEM → RMEM movement 需要第 13 章的 copy path：

```python
ld_atom = cute.make_copy_atom(
    cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), dtype
)
copy_a = cute.make_tiled_copy_A(ld_atom, tiled_mma)
thr_copy_a = copy_a.get_slice(tid)
cute.copy(
    copy_a,
    thr_copy_a.partition_S(s_a),
    thr_copy_a.retile(r_a),
)
```

`make_tiled_copy_A` 的作用是把 copy instruction 的 TV Layout 与 MMA operand-A 所需 TV Layout 适配起来；`retile` 给同一 register fragment 建立 copy-friendly view。

## 17.11 `make_fragment_A/B/C` 不是统一的存储分配器

公共 API 名称相同，但返回 fragment 的 storage 由 MMA op 决定：

| 路线 | A/B fragment 常见形态 | C fragment 常见形态 |
|---|---|---|
| SM80 warp MMA | ldmatrix/普通 load 产生的 RMEM | RMEM accumulator |
| SM90 WGMMA | A/B 可由 SMEM descriptor 驱动，也有 register variant | RMEM accumulator |
| SM100 UMMA | SMEM descriptor、TMEM 或 op-specific operand | TMEM accumulator |

因此正确写法是：

1. 先从 `partition_A/B/C` 得到 op-compatible shape；
2. 再让当前 `tiled_mma` 构造 fragment；
3. 按 fragment 的 address space 选择 copy/fence/lifetime protocol。

不能假设 `make_fragment_C` 永远“创建 TensorSSA registers”。第 20 章会看到它以 shape/layout 描述 TMEM allocation view。

## 17.12 `cute.gemm` 的 accumulator 语义

公共形式是：

```python
cute.gemm(tiled_mma, D, A, B, C)
```

数学语义：

```text
D = A * B + C
```

经典 in-place accumulation 写成：

```python
r_c.fill(0.0)
for k_block in ...:
    cute.gemm(tiled_mma, r_c, r_a_k, r_b_k, r_c)
```

第一次是 `A*B+0`，后续每次读旧 `r_c` 并写新 `r_c`。看起来是 mutation，但 SM80 RMEM fragment 仍遵循第 8 章的 SSA update。

某些异步 MMA family 还要求显式 accumulate flag、fence、commit/wait 或 completion barrier。公共数学式不替代架构同步协议。

## 17.13 Collective convergence 是 correctness contract

SM80 warp MMA 的 32 lanes 必须 converged 执行同一指令。错误例子：

```python
if lane == 0:
    cute.gemm(...)  # 错：只有一个 lane 进入 warp collective
```

边界处理应 predicate 输入 values，或者让 invalid CTA 在所有 warp collectives 之前整体退出。不要让同一 warp 的部分 lanes 绕过 `ldmatrix`/MMA。

对多 warp TiledMMA，每条硬件 MMA 仍是 warp scope；CTA barrier 则保护跨 warp 共享的 SMEM stage。必须同时审计两个 scope。

## 17.14 Alignment proof 与 ownership proof 分开

identity Tensor 可以证明 logical mapping，却不能证明地址满足 instruction alignment。`ldmatrix.x4` 还要求：

- SMEM base 有足够对齐；
- row/layout 产生合法地址；
- copy atom source pointer alignment 满足 128-bit requirement；
- swizzle 与 address computation 符合 op 约束。

本章通过 `SmemAllocator.allocate_tensor(..., byte_alignment=16)` 建立 base contract，并从 retained PTX 确认 `ldmatrix.sync.aligned...x4`。coverage 与 alignment 是两份证据，缺一不可。

## 17.15 配套 inspector 的执行流程

[`mma_layout_inspector.py`](../code/17_mma_model/mma_layout_inspector.py) 对 FP16/BF16 各执行两组配置：

```text
atom_layout=(1,1,1) -> tile=(16,8,16),  32 threads
atom_layout=(2,4,1) -> tile=(32,32,16), 256 threads
```

每组依次：

1. trace 时打印 `TiledMma`、TV Layout 与 partition shape；
2. 把 A/B 从 GMEM 放入 16-byte-aligned SMEM；
3. 对 identity Tensor 做相同 partition，导出坐标；
4. 用 `ldmatrix.x4` 把 A/B 转为 RMEM fragment；
5. 发出真实 `mma.sync`；
6. direct-store FP32 C；
7. 对照 PyTorch、坐标 domain/multiplicity 和 PTX。

它有意只执行一个 instruction-K step，不做 pipeline 或性能优化。第 18 章在不改变 ownership model 的前提下增加完整 K-loop。

## 17.16 B200 实测结果

CuTe DSL 4.7.0 / B200 上：

```text
FP16, (1,1,1): max_error=9.537e-07
FP16, (2,4,1): max_error=1.907e-06

(2,4,1) A: domain=512,  multiplicity=4
(2,4,1) B: domain=512,  multiplicity=2
(2,4,1) C: domain=1024, multiplicity=1

PTX: ldmatrix.sync.aligned.m8n8.x4.shared.b16
PTX: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
PTX: mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
```

这里只声称 L1/L4 correctness 与 instruction evidence，不声称 B200 对 Ampere MMA 路线的性能代表 SM80。

## 17.17 一个可复用的 MMA 审计表

面对任意新 MMA kernel，按下面顺序记录：

1. **Op**：架构、instruction MNK、dtype、operand layout、participant scope；
2. **Atom**：`thr_id` 与 A/B/C TV Layout；
3. **Tiling**：atom layout、permutation、总 participants、natural footprint；
4. **Partition**：A/B/C 输入 tile shape，逐 mode 语义和 expected multiplicity；
5. **Fragment**：address space、dtype、shape、allocation owner、lifetime；
6. **Copy**：GMEM/SMEM/RMEM/TMEM 每一跳的 copy atom、alignment、predicate；
7. **MMA**：collective convergence、accumulate state、commit/wait/fence；
8. **Epilogue**：C ownership、conversion、coalescing、residue；
9. **Evidence**：reference、mapping enumeration、IR/PTX/SASS。

只要这九项能闭环，复杂 GEMM 也只是更多 repeats、stages 和 roles；如果某一项只能用“应该是”解释，就还没有证明 kernel 正确。

## 17.18 本章小结

- `MmaOp` 选操作，`MmaAtom` 携带指令 TV metadata，`TiledMma` 空间复制，`ThrMma` 绑定 thread slice；
- atom layout 决定 participant replication，permutation 决定 value footprint/排列，K-loop repeats 来自 operand tile；
- `partition` 是 view/ownership 变换，不是数据移动；
- A/B read multiplicity 表示跨 M/N atom 复用，C 通常必须单 owner；
- fragment 的实际 storage 是 op-specific，不能跨 SM80/SM90/SM100 套同一假设；
- identity Tensor + retained PTX 分别证明 logical mapping 与 instruction/alignment evidence。

下一章沿用 `2×4×1` TiledMMA，把单次 K=16 扩展成 `cp.async` 双缓冲、`ldmatrix` 和多轮 `mma.sync` 的 SM80 GEMM。
