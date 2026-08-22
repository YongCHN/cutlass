# 第 14 章：TMA 从 descriptor 到 partition

> 范围标签：`[SM90+][Experimental]`
>
> 前置章节：第 5–7 章 Layout/Tensor、第 9 章 cluster、第 12 章 mbarrier、第 13 章 async copy
>
> 配套代码：[`14_tma/`](../code/14_tma/README.md)

## 14.1 本章要回答的问题

TMA 的使用表面上仍是 `cute.copy`，但它不是“换一个 CopyOp”这么简单。一次正确 TMA transaction 同时依赖：

```text
GMEM Tensor metadata
+ SMEM Layout
+ CTA tiler / cluster map
+ TensorMap descriptor
+ TMA coordinate tensor
+ partitioned atom domain
+ issuer protocol
+ completion protocol
+ memory-proxy ordering
```

本章回答：

1. TensorMap descriptor 编码了什么，没编码什么？
2. `make_tiled_tma_atom` 为什么返回 `TmaInfo`？
3. 为什么 TMA atom domain 必须放在 mode 0？
4. 怎样逐 mode 推导 `local_tile → group_modes → tma_partition`？
5. TMA load 与 store 为什么使用不同 completion protocol？
6. multicast mask、cluster receiver 和 transaction bytes 怎样配套？
7. online descriptor update、prefetch 和 proxy fence 解决什么问题？

## 14.2 TMA 与 per-thread copy 的根本差别

第 13 章中，每线程显式给出 source/destination address。TMA 则由一个 issuer 给硬件：

- 一个 TensorMap descriptor；
- 一个多维 logical coordinate；
- 一个 SMEM tile address；
- load 时的 mbarrier；
- multicast 时的 receiver mask。

硬件依据 descriptor 生成整块 transaction。线程不再逐元素计算 GMEM 地址，因此 descriptor 必须提前说明多维寻址规则。

## 14.3 TensorMap descriptor 的字段模型

从教程角度，可把 descriptor 理解为下面的静态/半静态记录：

| 字段 | 作用 |
|---|---|
| global base address | Tensor 起始地址 |
| global dimensions | 每个 logical mode 的 extent |
| global strides | 多维坐标到 GMEM byte offset 的规则 |
| element/data format | FP16、BF16、FP8、packed narrow format 等 |
| box dimensions | 一条 TMA 指令处理的多维 tile |
| element strides | box 内采样步长 |
| interleave | descriptor 级数据交织方式 |
| swizzle | TMA 写入/读取 SMEM 时的物理地址变换 |
| OOB fill | 越界 load 的填充值行为 |

descriptor 不保存当前 CTA 的 `(tile_m,tile_n)`。CTA 通过 TMA coordinate tensor 和 partitioned view 选择本次 transaction 的坐标。

## 14.4 `make_tiled_tma_atom` 的三个输入

```python
info = cpasync.make_tiled_tma_atom(
    cpasync.CopyBulkTensorTileG2SOp(),
    gmem_tensor,
    smem_layout,
    cta_tiler,
)
```

三个结构输入各有职责：

- `gmem_tensor`：global shape/stride、element type 和 base-address contract；
- `smem_layout`：tile 到 SMEM 的实际排布，包括 composed swizzle；
- `cta_tiler`：一个 CTA transaction 覆盖的逻辑 tile domain。

这三者必须一致。只改 tile shape 而不改 SMEM layout/descriptor，或只改 swizzle 而沿用旧物理读取公式，都破坏契约。

## 14.5 `TmaInfo` 不只是二元 tuple

4.7.0 返回 `TmaInfo`：

```python
info.atom
info.tma_tensor
info.smem_layout
```

它仍支持兼容式 unpack：

```python
atom, tma_tensor = info
```

但显式属性更能表达含义：

- `atom`：copy operation + non-executable descriptor trait；
- `tma_tensor`：把 CuTe logical coordinate 映射为 TMA 消费坐标的 Tensor；
- `smem_layout`：构造 descriptor 时实际使用的 layout，尤其 staged layout 被切片时很重要。

不要把 `tma_tensor` 当成普通数据 Tensor。它携带的是 descriptor coordinate basis。

## 14.6 从原始 Tensor 到 tiled view

假设输入是 `(M,N)=(256,384)`，tile 是 `(128,128)`：

```python
gsrc = cute.local_tile(
    tma_tensor, (128, 128), (None, None)
)
```

概念 shape 为：

```text
(TileM, TileN, GridM, GridN)
= (128, 128, 2, 3)
```

前两个 modes 是一条 CTA transaction 的 atom domain，后两个 modes 是 CTA 选择的 rest/grid domain。

## 14.7 为什么必须 `group_modes(..., 0, 2)`

`tma_partition` 要求 mode 0 表示完整 TMA atom domain。未分组时，mode 0 只有 `TileM`，`TileN` 仍是并列 mode，不满足输入约定。

```python
grouped_gmem = cute.group_modes(gsrc, 0, 2)
grouped_smem = cute.group_modes(smem, 0, 2)
```

概念 shape 变化：

```text
GMEM: (128,128,2,3) -> ((128,128),2,3)
SMEM: (128,128)     -> ((128,128),)
```

`group_modes` 不是复制数据，也不是 flatten 整个 Tensor。它重新组织 domain hierarchy，让 atom 成为一个 mode，rest modes 原样保留。

## 14.8 `tma_partition` 的五个参数

```python
t_smem, t_gmem = cpasync.tma_partition(
    atom,
    cta_coord,
    cta_layout,
    grouped_smem,
    grouped_gmem,
)
```

逐项含义：

1. `atom`：descriptor-backed copy trait；
2. `cta_coord`：当前 CTA 在 cluster value layout 中的位置；
3. `cta_layout`：cluster CTAs 如何共同覆盖 atom；
4. `grouped_smem`：mode 0 是 SMEM atom domain；
5. `grouped_gmem`：mode 0 是 GMEM atom domain，后面保留 grid modes。

单 CTA、1×1 cluster 使用：

```python
cta_coord = 0
cta_layout = cute.make_layout(1)
```

这不是说 launch grid 只有一个 CTA；它只说明每个 TMA tile 不需要由 cluster 内多个 CTA 分区。

## 14.9 Partition 输出怎样读

对前面的 `(2,3)` tile grid，概念结果是：

```text
t_smem: (TMA-internal-atom)
t_gmem: (TMA-internal-atom, 2, 3)
```

内部 atom mode 可能包含向量化/descriptor coordinate 结构，不应强行当成普通 `(128,128)`。用户需要做的是保留 mode 0，再索引 rest modes：

```python
cta_tile = t_gmem[(None, bidx, bidy)]
```

`None` 表示保留完整 atom mode；`bidx/bidy` 选择本 CTA 的 grid tile。若误写成 `[bidx,bidy]`，就会开始切 atom 本身。

## 14.10 TMA load 的双重完成条件

本章 128×128 FP16 tile 的 transaction bytes 为：

```text
128 × 128 × 2 = 32768 bytes
```

mbarrier 同时追踪：

- software arrival count；
- async transaction byte count。

代码协议是：

```python
with cute.arch.elect_one():
    cute.arch.mbarrier_init(barrier, 1)
    cute.arch.mbarrier_expect_tx(barrier, 32768)

cute.copy(load_atom, gmem_tile, smem_tile, tma_bar_ptr=barrier)

with cute.arch.elect_one():
    cute.arch.mbarrier_arrive(barrier)

cute.arch.mbarrier_wait(barrier, 0)
```

barrier 只有在 software arrival 与 TMA `complete_tx::bytes` 都满足后才完成。普通 CTA barrier既不会替 TMA engine 递减 bytes，也不能替代这次 transaction completion。

## 14.11 Issuer 约束与 `elect_one`

TMA instruction 是 single-issuer operation。高层 TMA Copy Atom 会对一次 warp copy 隐式选出一个 issuer，所以不要再把 `cute.copy` 包进第二层 `elect_one`。

但以下操作仍需显式 single issuer：

- mbarrier init；
- expect transaction bytes；
- software arrive；
- descriptor update/prefetch（依具体协议）。

更容易忽略的是：`elect_one` 是 warp scope。一个 128-thread CTA 若让所有 4 个 warps 执行同一 TMA path，就可能得到 4 个 issuers。本章转置 kernel 明确限制 warp 0 初始化和发出 TMA，全部 128 threads 只参与 SMEM transpose。

## 14.12 TMA store 的协议不同

TMA store 不用 load-side mbarrier 通知 destination global memory。典型序列是：

```python
cute.arch.barrier()
cute.arch.fence_proxy("async.shared", space="cta")
cute.copy(store_atom, smem_tile, gmem_tile)
cute.arch.cp_async_bulk_commit_group()
cute.arch.cp_async_bulk_wait_group(0)
```

这里有三个独立义务：

1. CTA barrier：所有普通 threads 已写完 output SMEM；
2. proxy fence：generic-proxy SMEM writes 对 async proxy 可见；
3. bulk commit/wait：TMA store 已读完 SMEM，global transaction 已按协议完成。

commit/wait 是 warp-uniform 操作，不应只让 elected lane 执行。

## 14.13 为什么 proxy fence 不能省略

CUDA memory model 中，普通 thread 对 SMEM 的 load/store 属于 generic proxy，TMA 通过 async proxy 访问 SMEM。CTA barrier建立 threads 之间的 rendezvous/order，却不自动桥接全部 proxy domain。

因此转置示例的路径必须是：

```text
TMA async proxy writes smem_src
  -> mbarrier wait
threads read smem_src and generic-store smem_dst
  -> CTA barrier
  -> fence.proxy.async.shared::cta
TMA async proxy reads smem_dst
```

遗漏 fence 可能在简单输入或某次调度下“看起来没问题”，仍属于未完成的 memory protocol。

## 14.14 转置时 grid modes 也要交换

输入 CTA `(tile_m,tile_n)` 处理 source tile 后，目标是 destination tile `(tile_n,tile_m)`：

```python
t_gdst[(None, tile_n, tile_m)]
```

SMEM 内做 `smem_dst[col,row] = smem_src[row,col]` 只交换 tile 内坐标；如果 global rest modes 不交换，多 tile 方阵以外的 case 会暴露错误。

本章用 `(256,128)` 和 `(256,384)` 验证这一点，而不只测单 tile 方阵。

## 14.15 Multicast：一次读取，多 CTA 投递

TMA multicast 允许 leader 发出一次 global read，把同一 tile 投递到 cluster 内多个 CTA 的 SMEM。典型用途是 GEMM 中多个 N-direction CTAs 复用同一 A tile。

协议新增：

- cluster launch shape；
- multicast receiver mask；
- leader/non-leader roles；
- cluster-scope init publication；
- 每个 receiver 的 transaction completion；
- receiver SMEM lifetime。

本章最小示例使用两个 CTAs 和 mask `0b11`。

## 14.16 Multicast transaction bytes 怎样计算

每个 destination CTA 收到一份完整 tile，并向目标 barrier 贡献 completion bytes。因此 leader barrier 期望：

```text
descriptor.global_tx_bytes() × number_of_receivers
```

本章是：

```text
128 × 64 × 2-byte FP16 × 2 CTAs = 32768 transaction bytes
```

只填写单份 tile bytes 会过早完成；填写过多则永久等待。

## 14.17 Cluster mask 与 CTA layout

简单连续 cluster 可以直观看成 bit `rank` 对应 CTA rank。生产 GEMM 通常使用四维 VMNK cluster layout，并用：

```python
cpasync.create_tma_multicast_mask(
    cta_layout_vmnk,
    cta_coord_vmnk,
    mcast_mode,
)
```

计算 image mask。`mcast_mode` 表示沿哪个 tensor/cluster mode 共享，不应通过假设物理 rank 顺序手写复杂 mask。

本章 raw primitive 示例特意固定为 SM100 `CTA_2` routing。它不能直接推广成任意 4/8 CTA cluster；更大 cluster 需要重新设计每 CTA local barrier 与 completion routing。

## 14.18 Swizzle 属于 descriptor/SMEM 双边契约

TMA descriptor 的 swizzle 决定硬件如何把 logical box 映射到 SMEM physical address。consumer 有两种安全方式：

1. 使用与 descriptor 同源的 CuTe composed SMEM Layout；
2. 在专门的低层实验中，明确逆推 physical column。

[`tma_multicast.py`](../code/14_tma/tma_multicast.py) 使用 raw descriptor 的 `s128b` swizzle，因此输出前显式做：

```text
physical_group = logical_group XOR (row mod 8)
```

这段公式只属于当前 element width、row shape 和 swizzle mode。高层 kernel 应优先让 `make_tiled_tma_atom` 保存并返回 `smem_layout`，避免复制硬件地址公式。

## 14.19 Interleave、sub-byte 与 descriptor format

对 FP16，global shape 与 storage byte shape容易一致。对 packed FP4/FP6，它们会分离：

- descriptor dimensions 按 logical elements；
- global strides 依 TensorMap format 的 bit width 换算；
- backing buffer 可能是 byte-packed storage；
- SMEM 展开宽度可能不同于 global packed width。

因此 descriptor 不能只按 `dtype.width` 机械计算。应明确 TensorMap format、packing、box constraint 和 shared representation。低精度的系统化处理留到第 27 章。

## 14.20 Descriptor prefetch

```python
cpasync.prefetch_descriptor(tma_atom)
```

提示硬件提前获取 descriptor，减少第一次 TMA 使用的 descriptor latency。它不搬运 tensor payload，也不替代：

- load transaction 的 mbarrier；
- descriptor update 的 acquire/release；
- pipeline stage acquire。

应在 descriptor 已经稳定且未来确实会使用时发出。

## 14.21 Online descriptor 与 tensormap replace

某些 persistent/grouped workload 的下一份 Tensor shape/address 只有运行时才知道。典型 online descriptor 流程是：

1. 把 atom 持有的 descriptor copy 到对齐 workspace；
2. 更新 global base address、shape、stride 等允许字段；
3. 用 tensormap proxy release 把 generic writes 发布；
4. 使用前执行 acquire；
5. 给 TMA copy 提供 override descriptor pointer；
6. 保证 descriptor 在所有异步使用者结束前不被覆盖。

4.7.0 helper 包括：

```python
cpasync.copy_tensormap(atom, descriptor_ptr)
cpasync.update_tma_descriptor(atom, new_gmem_tensor, descriptor_ptr)
cpasync.fence_tma_desc_release()
cpasync.fence_tma_desc_acquire(descriptor_ptr)
```

还有从 shared descriptor 同步复制到 global descriptor 的 `cp_fence_tma_desc_release`。这些 fence 处理的是 tensormap proxy，不是 SMEM async proxy，不能互换。

## 14.22 Descriptor lifetime 是资源协议

online descriptor 与 SMEM stage 一样需要 lifetime 证明：

```text
free -> writing -> published -> in-use by TMA -> retired -> free
```

尤其不能在 TMA 仍可能读取 descriptor 时，把同一 128-byte workspace 改成下一任务。生产代码常把 descriptor slot 与 work/pipeline state 绑定，而不是用一个全局临时对象反复覆盖。

## 14.23 三个可执行示例

### `tma_copy_v0.py`

覆盖：

- `TmaInfo` 三个属性；
- `local_tile`、`group_modes`、`tma_partition`；
- 1×1 cluster 的 G2S/S2G；
- 32768-byte mbarrier transaction；
- 多 tile dynamic shape；
- TMA load/store 与 completion PTX。

### `tma_transpose_v1.py`

覆盖：

- warp 0 single issuer 与 4-warp compute CTA；
- 两个独立 SMEM buffers；
- tile-internal 和 grid-mode transpose；
- generic→async proxy fence；
- bulk store commit/wait；
- 非方形多 tile reference。

### `tma_multicast.py`

覆盖：

- SM100 two-CTA cluster；
- leader 发出一次 multicast；
- `0b11` receiver mask；
- two-destination transaction byte count；
- leader completion + cluster publication；
- 两个 CTA 得到完全相同 tile；
- multicast PTX qualifier。

运行：

```bash
python discussion/code/14_tma/tma_copy_v0.py
python discussion/code/14_tma/tma_transpose_v1.py
python discussion/code/14_tma/tma_multicast.py
```

## 14.24 常见错误

1. 直接把未分组的 `(TileM,TileN,...)` 传给 `tma_partition`。
2. 索引 partition result 时漏掉保留 atom mode 的 `None`。
3. 把 launch grid layout 与 cluster CTA layout 混为一谈。
4. 128-thread CTA 的每个 warp 都发出同一 TMA copy。
5. mbarrier arrival count 正确，但 transaction bytes 错误。
6. TMA load 后只做 CTA barrier，没有 transaction completion wait。
7. threads 写 SMEM 后直接 TMA store，遗漏 async proxy fence。
8. TMA store 后没有 bulk commit/wait就复用 SMEM或退出相关 lifetime。
9. transpose 只交换 tile 内坐标，不交换 grid modes。
10. descriptor swizzle 与 consumer physical layout 不一致。
11. multicast byte count 没乘 receiver 数。
12. 把 `CTA_2` routing 推广到任意 cluster shape。
13. 更新 descriptor 后遗漏 tensormap proxy release/acquire。
14. descriptor 尚在被 TMA 使用时覆盖 workspace。

## 14.25 与第 15 章的接口

本章每个 CTA 只处理一次 transaction generation，协议是：

```text
initialize
  -> issue one load
  -> wait
  -> compute
  -> issue one store
  -> wait
```

第 15 章会把它扩展为：

```text
prologue: fill stages
steady: producer(stage p) overlaps consumer(stage c)
tail: drain outstanding loads/stores
```

届时 `PipelineTmaAsync`/`PipelineTmaStore` 不是隐藏 descriptor/partition 知识，而是把本章已经证明的 barrier、phase、transaction bytes 和 lifetime 协议封装成状态机。

## 14.26 检查清单与练习

检查清单：

1. 记录 global logical shape、stride 和 element format。
2. 记录 box dims、SMEM layout、swizzle/interleave。
3. 证明 CTA tiler 与 descriptor atom domain 一致。
4. 在每个 Tensor shape 中标注 atom mode 和 rest modes。
5. `group_modes` 后确认 atom 位于 mode 0。
6. 写出 `tma_partition` 的 CTA coordinate/layout。
7. 明确唯一 issuer 是一个 lane、一个 warp还是一个 CTA role。
8. 计算 load transaction bytes。
9. 分开写 load mbarrier 与 store bulk-group completion。
10. thread writes→TMA store 前检查 proxy fence。
11. multicast 同时检查 mask、receiver count 和 completion routing。
12. online descriptor 写出 workspace alignment、proxy fence 和 lifetime。

练习：

1. 手算 `(512,256)`、tile `(128,64)` 的 `local_tile/group_modes` shape。
2. 把基础 copy tile 改成 `(64,128)`，重新计算 SMEM bytes 与 grid。
3. 在 copy kernel 中加入 descriptor prefetch，并检查 PTX。
4. 给 transpose 添加 BF16 case，保持 bit-exact reference。
5. 把 multicast 改写为高层 `CopyBulkTensorTileG2SMulticastOp`，用 VMNK layout 生成 mask。
6. 为 4-CTA multicast 画出 local barrier 与 completion routing，不直接沿用 `CTA_2`。
7. 设计双 descriptor slots 的 online update 状态机，并证明不会覆盖 in-flight descriptor。

