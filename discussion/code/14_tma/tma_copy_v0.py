"""SM90+ TMA copy using make_tiled_tma_atom and tma_partition.

This is intentionally a single-stage kernel.  It exposes descriptor creation,
mode grouping, partitioning, mbarrier transaction completion, and the TMA-store
commit/wait protocol before Chapter 15 adds a reusable pipeline state machine.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Type, Union

import torch


ARTIFACT_DIR = Path(__file__).with_name("generated_artifacts") / "copy_v0"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("CUTE_DSL_KEEP", "ptx")
os.environ.setdefault("CUTE_DSL_DUMP_DIR", str(ARTIFACT_DIR))

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack


TILE_M = 128
TILE_N = 128
THREADS = 32


class TmaCopy:
    @cute.jit
    def __call__(self, src: cute.Tensor, dst: cute.Tensor):
        if cutlass.const_expr(src.element_type != dst.element_type):
            raise TypeError("src and dst element types must match")
        dtype: Type[cutlass.Numeric] = src.element_type
        smem_layout = cute.make_layout((TILE_M, TILE_N), stride=(TILE_N, 1))

        @cute.struct
        class SharedStorage:
            barrier: cute.struct.MemRange[cutlass.Int64, 1]
            data: cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(smem_layout)], 1024
            ]

        self.shared_storage = SharedStorage
        self.transaction_bytes = cute.size_in_bytes(dtype, smem_layout)
        cta_tiler = cute.product_each(smem_layout.shape)

        src_info = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), src, smem_layout, cta_tiler
        )
        dst_info = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), dst, smem_layout, cta_tiler
        )
        self.kernel(
            src_info.atom,
            src_info.tma_tensor,
            dst_info.atom,
            dst_info.tma_tensor,
            src_info.smem_layout,
        ).launch(
            grid=cute.ceil_div((*src.shape, 1), (TILE_M, TILE_N)),
            block=(THREADS, 1, 1),
        )

    @cute.kernel
    def kernel(
        self,
        load_atom: cute.CopyAtom,
        load_tensor: cute.Tensor,
        store_atom: cute.CopyAtom,
        store_tensor: cute.Tensor,
        smem_layout: Union[cute.Layout, cute.ComposedLayout],
    ):
        bidx, bidy, _ = cute.arch.block_idx()
        allocator = utils.SmemAllocator()
        storage = allocator.allocate(self.shared_storage)
        barrier = storage.barrier.data_ptr()
        smem = storage.data.get_tensor(smem_layout)

        # Arrival count counts software arrivals; transaction bytes are a
        # second completion condition decremented by the TMA engine.
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(barrier, 1)
            cute.arch.mbarrier_expect_tx(barrier, self.transaction_bytes)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        # local_tile first exposes tile modes + grid modes.  TMA requires its
        # atom domain to be mode 0, so the two tile modes are grouped together.
        gsrc = cute.local_tile(load_tensor, (TILE_M, TILE_N), (None, None))
        gdst = cute.local_tile(store_tensor, (TILE_M, TILE_N), (None, None))
        t_smem, t_gsrc = cpasync.tma_partition(
            load_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(smem, 0, 2),
            cute.group_modes(gsrc, 0, 2),
        )
        _, t_gdst = cpasync.tma_partition(
            store_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(smem, 0, 2),
            cute.group_modes(gdst, 0, 2),
        )

        cute.copy(
            load_atom,
            t_gsrc[(None, bidx, bidy)],
            t_smem,
            tma_bar_ptr=barrier,
        )
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive(barrier)
        cute.arch.mbarrier_wait(barrier, 0)

        cute.copy(store_atom, t_smem, t_gdst[(None, bidx, bidy)])
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
        "tma_store": ("cp.async.bulk.tensor", ".global.shared::cta"),
        "expect_tx": ("mbarrier.expect_tx",),
        "store_commit": ("cp.async.bulk.commit_group",),
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

    exemplar = torch.empty((TILE_M, TILE_N), dtype=torch.float16, device="cuda")
    compiled = cute.compile(TmaCopy(), as_dynamic(exemplar), as_dynamic(exemplar.clone()))
    for shape in ((128, 128), (256, 128), (256, 384)):
        src = torch.randn(shape, dtype=torch.float16, device="cuda")
        dst = torch.full_like(src, float("nan"))
        compiled(as_dynamic(src), as_dynamic(dst))
        torch.cuda.synchronize()
        torch.testing.assert_close(dst, src, rtol=0, atol=0)
        print(f"shape={shape}, tiles=({shape[0]//TILE_M},{shape[1]//TILE_N}): PASS")

    evidence = inspect_ptx()
    missing = [name for name, matches in evidence.items() if not matches]
    if missing:
        raise AssertionError(f"missing TMA PTX families: {missing}")
    for name, matches in evidence.items():
        print(f"PTX {name}: {matches[0]}")
    print("PASS")


if __name__ == "__main__":
    main()
