"""TMA load -> thread transpose -> TMA store with two SMEM buffers.

The load completion path uses mbarrier transaction bytes.  Thread writes to
the output buffer use the generic proxy, so a CTA barrier plus async-shared
proxy fence is required before the TMA store may read that buffer.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Type, Union

import torch


ARTIFACT_DIR = Path(__file__).with_name("generated_artifacts") / "transpose_v1"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("CUTE_DSL_KEEP", "ptx")
os.environ.setdefault("CUTE_DSL_DUMP_DIR", str(ARTIFACT_DIR))

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack


TILE = 128
THREADS = 128


class TmaTranspose:
    @cute.jit
    def __call__(self, src: cute.Tensor, dst: cute.Tensor):
        dtype: Type[cutlass.Numeric] = src.element_type
        layout = cute.make_layout((TILE, TILE), stride=(TILE, 1))

        @cute.struct
        class SharedStorage:
            barrier: cute.struct.MemRange[cutlass.Int64, 1]
            src: cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(layout)], 1024]
            dst: cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(layout)], 1024]

        self.shared_storage = SharedStorage
        self.transaction_bytes = cute.size_in_bytes(dtype, layout)
        cta_tiler = cute.product_each(layout.shape)
        src_info = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), src, layout, cta_tiler
        )
        dst_info = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), dst, layout, cta_tiler
        )
        self.kernel(
            src_info.atom,
            src_info.tma_tensor,
            dst_info.atom,
            dst_info.tma_tensor,
            layout,
        ).launch(
            grid=cute.ceil_div((*src.shape, 1), (TILE, TILE)),
            block=(THREADS, 1, 1),
        )

    @cute.kernel
    def kernel(
        self,
        load_atom: cute.CopyAtom,
        load_tensor: cute.Tensor,
        store_atom: cute.CopyAtom,
        store_tensor: cute.Tensor,
        layout: Union[cute.Layout, cute.ComposedLayout],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp = tidx // 32
        tile_m, tile_n, _ = cute.arch.block_idx()
        allocator = utils.SmemAllocator()
        storage = allocator.allocate(self.shared_storage)
        barrier = storage.barrier.data_ptr()
        smem_src = storage.src.get_tensor(layout)
        smem_dst = storage.dst.get_tensor(layout)

        if warp == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(barrier, 1)
                cute.arch.mbarrier_expect_tx(barrier, self.transaction_bytes)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        gsrc = cute.local_tile(load_tensor, (TILE, TILE), (None, None))
        gdst = cute.local_tile(store_tensor, (TILE, TILE), (None, None))
        t_smem_src, t_gsrc = cpasync.tma_partition(
            load_atom, 0, cute.make_layout(1),
            cute.group_modes(smem_src, 0, 2), cute.group_modes(gsrc, 0, 2),
        )
        t_smem_dst, t_gdst = cpasync.tma_partition(
            store_atom, 0, cute.make_layout(1),
            cute.group_modes(smem_dst, 0, 2), cute.group_modes(gdst, 0, 2),
        )

        if warp == 0:
            cute.copy(load_atom, t_gsrc[(None, tile_m, tile_n)], t_smem_src, tma_bar_ptr=barrier)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(barrier)
        cute.arch.mbarrier_wait(barrier, 0)

        # One thread owns one input row and serially writes one output column.
        # This is intentionally simple and bank-conflict prone: Chapter 15 will
        # optimize overlap; the present chapter isolates correctness ordering.
        row = tidx
        for col in cutlass.range_constexpr(TILE):
            smem_dst[col, row] = smem_src[row, col]
        cute.arch.barrier()

        # Generic-proxy stores above must be published to the async proxy before
        # TMA reads smem_dst.  A CTA barrier alone does not cross that boundary.
        cute.arch.fence_proxy("async.shared", space="cta")
        if warp == 0:
            cute.copy(store_atom, t_smem_dst, t_gdst[(None, tile_n, tile_m)])
            # commit/wait is warp-uniform, not nested inside elect_one.
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0)


def as_dynamic(tensor: torch.Tensor):
    return (
        from_dlpack(tensor, assumed_align=16)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=16)
    )


def inspect_ptx() -> dict[str, list[str]]:
    lines: list[str] = []
    for path in ARTIFACT_DIR.glob("*.ptx"):
        lines.extend(line.strip() for line in path.read_text().splitlines())
    patterns = {
        "tma_load": ("cp.async.bulk.tensor", ".shared::cta.global", "mbarrier"),
        "proxy_fence": ("fence.proxy.async.shared::cta",),
        "tma_store": ("cp.async.bulk.tensor", ".global.shared::cta"),
        "store_wait": ("cp.async.bulk.wait_group",),
    }
    return {
        name: [line for line in lines if all(token in line for token in tokens)]
        for name, tokens in patterns.items()
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires an NVIDIA GPU")
    major, minor = torch.cuda.get_device_capability()
    if major < 9:
        raise RuntimeError(f"TMA requires SM90+, got {major}.{minor}")

    exemplar_src = torch.empty((TILE, TILE), dtype=torch.float16, device="cuda")
    exemplar_dst = torch.empty_like(exemplar_src)
    compiled = cute.compile(TmaTranspose(), as_dynamic(exemplar_src), as_dynamic(exemplar_dst))
    for shape in ((128, 128), (256, 128), (256, 384)):
        src = torch.randn(shape, dtype=torch.float16, device="cuda")
        dst = torch.full((shape[1], shape[0]), float("nan"), dtype=torch.float16, device="cuda")
        compiled(as_dynamic(src), as_dynamic(dst))
        torch.cuda.synchronize()
        expected = src.t().contiguous()
        torch.testing.assert_close(dst, expected, rtol=0, atol=0)
        print(f"shape={shape} -> {tuple(dst.shape)}: PASS")

    evidence = inspect_ptx()
    missing = [name for name, matches in evidence.items() if not matches]
    if missing:
        raise AssertionError(f"missing TMA/proxy PTX families: {missing}")
    for name, matches in evidence.items():
        print(f"PTX {name}: {matches[0]}")
    print("PASS")


if __name__ == "__main__":
    main()
