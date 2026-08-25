# 第 20 章：Blackwell tcgen05、TMEM 与 UMMA

> 范围标签：`[SM100/SM103][Experimental]`
>
> 前置章节：第 14–15 章 TMA/pipeline、第 17 章 MMA模型、第 19 章 async MMA协议
>
> 配套代码：[`20_blackwell_tcgen05/`](../code/20_blackwell_tcgen05/README.md)

## 20.1 本章数据路径

```text
GMEM A/B
  -> TMA
  -> swizzled SMEM A/B
  -> tcgen05 SMEM descriptors
  -> tcgen05.mma (UMMA)
  -> FP32 TMEM accumulator
  -> tcgen05.ld
  -> RMEM conversion/epilogue
  -> GMEM or SMEM -> TMA store
```

本章回答：

1. TMEM解决什么问题，为什么不是“更多 shared memory”？
2. TMEM allocation address、columns、rows和 holding buffer怎样配合？
3. `tcgen05.mma` 的 issue granularity与 completion协议是什么？
4. `scale_d/ACCUMULATE`怎样避免 accumulator memset？
5. TMEM store/load为何有独立 `wait::st/wait::ld`？
6. allocation permit何时 relinquish，TMEM何时才能 deallocate？
7. CTA_1和 CTA_2的 accumulator owner、TMA completion和 epilogue有何区别？
8. CuTe `TmemAllocator/PipelineTmaUmma` 与底层 primitives怎样对应？

## 20.2 架构范围必须写精确

tcgen05 TMEM/UMMA是 datacenter Blackwell能力，目标是 SM100/SM103。不能把“compute capability >= 10”粗略解释为所有 Blackwell/Thor产品都支持同一指令、TMEM容量和 CTA group。

CuTe DSL API会给出 datacenter Blackwell建议；代码也显式检查当前 B200 capability 10.0。后续 SM120/consumer路线需使用对应 warp MMA或该架构支持的 op，不应调用 tcgen05 allocation。

## 20.3 TMEM是什么

TMEM是 Tensor Core侧的专用存储：

- 主要承载大 MMA accumulator；
- 与普通 per-thread RMEM不同，布局按 rows/columns与 subpartition组织；
- CTA-private allocation通过 tcgen05指令管理；
- 普通 load/store不能直接访问，需 `tcgen05.ld/st/cp`；
- 不是 SMEM，不能用普通指针和 CTA任意线程寻址语义类比。

把大 FP32 accumulator从 RMEM移到 TMEM，可以显著降低每线程 register pressure，为 TMA producer、epilogue和更大 tile腾出 registers。但代价是显式 allocation、drain和 lifetime protocol。

## 20.4 地址模型：row与column

TMEM base是一个 32-bit地址，可分解：

```text
high bits -> row_id
low bits  -> col_id
```

`TmemAddr` 封装该编码：

```python
base = prims.TmemAddr(raw)
ptr = prims.TmemAddr.from_row_col(
    base.row_id + row_offset,
    base.col_id + col_offset,
).as_ptr(cutlass.Float32)
```

`32x32b` load表示一次覆盖32 rows，每 row读取32 bits。一个 warp的 lane对应一 row；`x32` repetition使每 lane获得32个 FP32 values，即32 columns。

## 20.5 Column allocation granularity

最小 allocation通常是32 columns，并要求列数满足硬件粒度。第一个示例：

```text
32 lanes × 32 FP32 values/lane
TMEM shape = 32 rows × 32 columns × 32 bits
allocation = 32 columns
```

GEMM accumulator需要更多列。固定 CTA_1 `128×128` FP32 accumulator使用：

```text
NUM_TMEM_COLS = (N/8)*32 = 512
```

这不是简单 `N`；TMEM accumulator layout由 MMA atom规定。生产代码应从 `make_fragment_C/partition_shape_C` layout推导，而不是手写公式。

## 20.6 Holding buffer与 allocation publication

allocation instruction把 TMEM start address写入一个 SMEM holding buffer：

```python
tmem_addr = cutlass.Array(Int32, 1, space=smem)
tcgen05_alloc(tmem_addr, columns)
barrier_cta_sync()
raw = tmem_addr.load()
```

只有负责 allocation的 warp执行 collective；其他 warps必须等地址写入可见。CuTe `utils.TmemAllocator` 把 holding buffer、named barrier和 pointer retrieval封装为：

```text
allocate -> wait_for_alloc -> retrieve_ptr
```

底层教学例保留显式 barrier，便于看到 publication requirement。

## 20.7 Allocation permit不是 allocation lifetime

两个概念必须分开：

- **allocation permit**：控制 allocation machinery/资源准入；
- **allocated TMEM region**：保存 accumulator，直到 dealloc。

取得 TMEM pointer后可以尽早：

```python
tcgen05_relinquish_alloc_permit()
```

这允许其他工作申请 allocation，但不会释放当前 columns。真正释放必须在所有 MMA、TMEM load/store和 epilogue readers结束后调用 dealloc。

过晚 relinquish降低并发；过早 dealloc造成数据破坏。

## 20.8 TMEM store/load roundtrip

[`tmem_roundtrip.py`](../code/20_blackwell_tcgen05/tmem_roundtrip.py) 的协议：

```text
alloc 32 cols
CTA publication
tcgen05.st 32x32b.x32
tcgen05.wait::st
tcgen05.ld 32x32b.x32
tcgen05.wait::ld
write GMEM
fence::before_thread_sync
CTA barrier
dealloc
relinquish permit
```

store wait证明此前 TMEM writes完成，load wait证明 vector result可消费。普通 CTA barrier不能替代 tcgen05 operation completion wait。

B200实测输出 `32×32` FP32与 `[0,1023]`逐位一致。

## 20.9 SMEM descriptor

底层 descriptor：

```python
desc = prims.Tcgen05SmemDesc.build(
    start_address=s_a,
    leading_byte_offset=16,
    stride_byte_offset=8*K*element_bytes,
    layout=SWIZZLE_128B,
)
```

包含 base、leading/stride offset与 swizzle。对 K-major FP16、K=64：

```text
row bytes = 64*2 = 128
TMA swizzle = 128B
K granule = 16 elements = 32 bytes
```

每次 MMA K-block通过 `advance_start_address(32*kb)`选择下一个 K granule，不必重建 descriptor。

## 20.10 Instruction descriptor

`Tcgen05InstrDesc` 描述：

- accumulator dtype；
- A/B dtype family；
- logical M/N；
- transpose/scale/block-scale等 instruction fields。

固定例：

```python
idesc = Tcgen05InstrDesc.build(
    c_dtype=Float32,
    a_dtype=Float16,
    b_dtype=Float16,
    m_dim=128,
    n_dim=128,
)
```

instruction descriptor和 SMEM descriptor职责不同：前者描述计算格式/shape，后者描述 operand storage。混用 offset单位是常见错误。

## 20.11 CTA_1 issue granularity

tcgen05 MMA instruction由一个 elected lane发出：

```python
if warp == 0 and elect_sync():
    tcgen05_mma(CTA_1, tmem, desc_a, desc_b, idesc, scale_d)
```

“one elected lane issue”不代表只有该线程拥有结果。硬件 collective读取 SMEM descriptors并把整个 tile写入 TMEM。

与 WGMMA的 128-thread uniform issue不同，Blackwell生产 kernel常让一个 MMA warp负责 issue，其余 warps负责 TMA/epilogue。participant/issuer概念必须按 op重新确认。

## 20.12 `scale_d` / `ACCUMULATE`

固定 FP16 K=64需要四个 K=16 MMAs：

```text
kb=0: scale_d=False -> overwrite accumulator
kb=1: scale_d=True  -> accumulate
kb=2: scale_d=True
kb=3: scale_d=True
```

首个 issue overwrite TMEM，避免单独 memset整个 accumulator。跨 outer K tiles时只有整个 problem的第一个 K-block使用 False；后续全部 True。

CuTe高层 atom用 `tcgen05.Field.ACCUMULATE`表达同一状态：

```python
tiled_mma.set(Field.ACCUMULATE, True)
```

## 20.13 MMA completion：`tcgen05.commit` 到 mbarrier

`tcgen05.mma`是异步操作。结束一个 accumulator generation后，由 elected issuer：

```python
tcgen05_commit(acc_mbar, group=CTA_1)
```

commit把此前 tcgen05 MMA completion关联到 mbarrier arrival。epilogue warps等待 barrier phase，才能 `tcgen05.ld` accumulator。

这不是 Hopper `wgmma.commit_group/wait_group`；名字相似但 completion对象不同：

| Hopper | Blackwell |
|---|---|
| WGMMA group queue | tcgen05 commit to mbarrier |
| wait_group(N) | mbarrier phase wait |
| accumulator RMEM | accumulator TMEM |

## 20.14 CTA_1固定 GEMM

[`tcgen05_1cta_gemm.py`](../code/20_blackwell_tcgen05/tcgen05_1cta_gemm.py)：

```text
A/B: FP16 (128,64), K-major
TMA: one full A + B tile
MMA: four CTA_1 m128n128k16 issues
ACC: FP32 TMEM
drain: four warps × four 32-column subtiles
C: FP32 (128,128)
```

输入使用小整数表示，FP16乘法与 FP32 sum都可精确表示。B200与 PyTorch reference bit-exact。

## 20.15 TMEM epilogue的 subpartition映射

128 M rows由四个 warps drain：

```text
warp 0 -> rows   0..31
warp 1 -> rows  32..63
warp 2 -> rows  64..95
warp 3 -> rows 96..127
lane   -> row within 32-row subpartition
```

N=128拆成四个32-column subtiles。每 warp/lane每次 `tcgen05.ld.x32`得到一整段 row values，然后 vector store到 GMEM。

physical warp id与 TMEM subpartition绑定；若把 epilogue warp range平移到 warps 2..5，要用 `warp_idx % 4`映射 SP，而不能使用逻辑 `warp_idx-start`。

## 20.16 Fence before thread synchronization

在 CTA barrier/dealloc之前：

```python
tcgen05_fence(BEFORE_THREAD_SYNC)
barrier_cta_sync()
dealloc()
```

fence把此前 tcgen05 operations排序到 thread synchronization之前，barrier确保所有 drain warps完成使用，随后 allocation owner才能释放。缺少任一环都可能让 TMEM仍被访问时回收。

## 20.17 高层 CuTe TMEM模型

Blackwell `fp16_gemm_0.py` 展示推荐的组合：

```text
TmemAllocator
PipelineTmaUmma
PipelineUmmaAsync
tiled_mma.make_fragment_A/B(SMEM)
tiled_mma.make_fragment_C(acc_shape) -> TMEM layout
make_tensor(tmem_ptr, fragment.layout)
tcgen05.make_tmem_copy
```

公共第 17 章骨架仍成立，但 fragment C不是 RMEM allocation：

```python
acc_shape = tiled_mma.partition_shape_C((M,N))
t_c = tiled_mma.make_fragment_C(acc_shape)
t_c = cute.make_tensor(tmem_ptr, t_c.layout)
```

它先生成 TMEM layout，再绑定 runtime allocation pointer。

## 20.18 TMA-UMMA 与 UMMA-async pipelines

production 1CTA kernel通常有两组 pipelines：

```text
PipelineTmaUmma:
  EMPTY SMEM stage -> TMA writing -> UMMA reading -> EMPTY

PipelineUmmaAsync:
  EMPTY accumulator -> UMMA writing TMEM -> epilogue reading -> EMPTY
```

前者的 completion来自 TMA transaction和 UMMA consume；后者的 completion来自 `tcgen05.commit`/epilogue release。把所有状态压进一个 barrier会丢失 producer/consumer generation关系。

## 20.19 CTA_2：两个 CTAs组成一个 MMA group

CTA_2 示例使用 cluster `(2,1,1)`，两个 CTAs协作计算 `256×256×64`：

```text
CTA 0 loads A rows   0..127 and B rows   0..127
CTA 1 loads A rows 128..255 and B rows 128..255
CTA_2 tcgen05 MMA combines both contributions
leader CTA owns collective TMEM accumulator
both CTAs drain their output M half
```

这里 A/B分别按 group lane切片。它不是普通 split-K；两个 CTAs共同构成一条 CTA_2 MMA的 operands/shape。

## 20.20 CTA_2 TMEM allocation

两个 CTAs都调用：

```text
tcgen05.alloc.cta_group::2
```

allocation由 group协同，pointer publication还需要 cluster formation barrier与 CTA barrier。leader持有/管理 collective accumulator，但 peer需要一致的 pointer/protocol view。

dealloc也必须使用 `cta_group::2`，不能用 CTA_1释放 CTA_2 allocation。

## 20.21 CTA_2 TMA completion routing

每 CTA发出自己的 A/B TMA load，但 `group=CTA_2` 将 complete_tx路由到 leader CTA的 mbarrier。leader transaction count覆盖：

```text
2 CTAs × (A bytes + B bytes)
```

leader等待所有四份 copies完成后才能 issue collective MMA。若按单 CTA bytes设置，会过早读 peer未完成的 SMEM。

例子没有做数据 multicast，因为每 CTA读取不同 slice；传一个仅包含自己的 multicast mask反而会选择 multicast routing并增加开销。

## 20.22 CTA_2 commit multicast

MMA完成与 SMEM-consumed信号需要投递给两个 group members：

```text
tcgen05.commit.cta_group::2 ... multicast mask=0b11
```

用途：

- 通知两个 CTAs A/B buffers可复用；
- 通知两个 CTAs accumulator已完成，可进入 epilogue；
- 保持 phase/generation一致。

mask与 cluster rank/group位置有关。单 group `0b11`最简单，多 group cluster必须按 leader rank移位。

## 20.23 CTA_2 canonical runner

[`tcgen05_2cta_gemm.py`](../code/20_blackwell_tcgen05/tcgen05_2cta_gemm.py) 不复制500行 experimental cluster kernel，而是加载仓库 canonical：

```text
examples/python/CuTeDSL/experimental/primitives/tcgen05/2cta_mma_basic.py
```

runner固定三组验证：

```text
(256,256,64): 1×1 cluster tile
(512,256,64): 2×1 cluster grid
(256,512,64): 1×2 cluster grid
```

这样教程只维护 validation/PTX contract，canonical protocol继续随 CuTe experimental API演进。

## 20.24 1CTA与2CTA对照

| 维度 | CTA_1 | CTA_2 |
|---|---|---|
| launch | 普通 CTA | 2-CTA cluster |
| operand source | 本 CTA SMEM | 两 CTA各自 SMEM slice |
| TMA completion | CTA-local barrier | routed到 group leader |
| TMEM allocation | `cta_group::1` | `cta_group::2` |
| MMA issuer | elected lane | leader elected lane |
| accumulator | CTA-private TMEM | collective/leader-owned TMEM |
| completion | commit local mbarrier | commit + multicast group members |
| epilogue | 四 warps drain 128 rows | 两 CTAs各 drain 128-row half |
| lifetime | CTA barrier | CTA + cluster/group protocol |

## 20.25 B200验证结果

### TMEM roundtrip

```text
shape=(32,32), exact roundtrip: PASS
PTX: alloc, st.x32, wait::st, ld.x32, wait::ld, dealloc, relinquish
```

### CTA_1 GEMM

```text
shape=(128,128,64), bit-exact FP32 output: PASS
PTX: TMA, alloc.cta_group::1, mma.kind::f16,
     commit, ld.x32, dealloc.cta_group::1
```

### CTA_2 GEMM

```text
(256,256,64) max_error=0
(512,256,64) max_error=0
(256,512,64) max_error=0
PTX: shared::cluster TMA cta_group::2,
     alloc/mma/commit/dealloc cta_group::2,
     cluster arrive/wait
```

全部在 B200 / CuTe DSL 4.7.0完成 L1/L2/L4。

## 20.26 常见失败模式

- 在非 datacenter Blackwell上调用 TMEM/tcgen05；
- allocation columns不是合法粒度；
- holding buffer地址尚未 publication就读取；
- permit一直不 relinquish，阻塞其他 allocations；
- MMA后未 commit到 completion barrier就读取 TMEM；
- `tcgen05.ld/st` 后缺 `wait::ld/st`；
- physical warp与 TMEM subpartition映射错误；
- dealloc前缺 tcgen05 fence/CTA drain；
- CTA_2用单 CTA transaction bytes；
- CTA_2 commit multicast mask不包含 peer；
- CTA_2 allocation却用 CTA_1 dealloc；
- 把 TMEM当 DSMEM，尝试普通 peer pointer访问。

## 20.27 从 primitive到 production GEMM

本章底层代码隔离 instruction protocol；生产 `fp16_gemm_0/1` 还加入：

1. TiledMMA/fragment C的 TMEM layout推导；
2. multi-stage TMA-UMMA pipeline；
3. multi-stage accumulator/epilogue pipeline；
4. warp specialization与 register budget；
5. swizzled SMEM/TMA atoms；
6. TMEM tiled copy与 vector epilogue；
7. CTA_2 multicast与 cluster grid；
8. residue、persistent scheduler和 TMA store。

下一章会把第 18–20 章的三条 tensor-core路径放回统一 GEMM演化链，逐步比较每个优化减少的成本。

## 20.28 本章小结

- TMEM是 tcgen05 accumulator专用存储，需显式 alloc/address publication/permit/dealloc；
- `tcgen05.mma`异步写 TMEM，通过 commit到 mbarrier通知 epilogue；
- TMEM `ld/st`各有独立 wait，thread barrier不能替代 operation completion；
- CTA_1的 owner/lifetime是 CTA-local，CTA_2引入 cluster formation、completion routing、collective allocation和 multicast commit；
- B200已完成 TMEM roundtrip、CTA_1与 CTA_2数值/PTX闭环；
- production CuTe用 TmemAllocator与两类 pipelines封装相同底层状态机。
