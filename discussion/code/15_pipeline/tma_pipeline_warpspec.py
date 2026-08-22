"""SM90+ warp-specialized multi-stage TMA load pipeline.

Warp 0 issues TMA loads; warp 1 drains the swizzled shared-memory stages to
global memory.  The producer has an explicit prologue that fills up to S
stages, a steady-state loop that acquires released slots, and a CTA tail that
keeps barrier/SMEM lifetime valid until the consumer has drained all tiles.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch


ARTIFACT_DIR = Path(__file__).with_name("generated_artifacts") / "tma_warpspec"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("CUTE_DSL_KEEP", "ptx")
os.environ.setdefault("CUTE_DSL_DUMP_DIR", str(ARTIFACT_DIR))

import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.experimental import primitives as prims


TILE_M = 128
TILE_K = 64
TILE_ELEMENTS = TILE_M * TILE_K
THREADS = 64


@cute.jit
def issue_tma_load(descriptor, smem, full_barrier, tile, stage):
    if prims.elect_sync():
        prims.mbarrier_arrive_expect_tx(
            full_barrier, descriptor.global_tx_bytes()
        )
    if prims.elect_sync():
        prims.cp_async_bulk_tensor_shared_cta_global(
            smem.subview(stage * TILE_ELEMENTS),
            descriptor.get_ptr(),
            (tile * TILE_K, cutlass.Int32(0)),
            full_barrier,
        )


@cute.kernel
def tma_pipeline_kernel(
    descriptor: cutlass.GridConstant[cuda.TensorMap],
    dst: cute.Tensor,
    num_tiles: cutlass.Int32,
    STAGES: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    warp = tidx // 32
    lane = tidx % 32

    smem = cutlass.Array(
        cutlass.Float16,
        STAGES * TILE_ELEMENTS,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    barriers = cutlass.Array(
        cutlass.Int64,
        2 * STAGES,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    full = barriers
    empty = barriers.subview(STAGES)

    if warp == 0:
        if prims.elect_sync():
            for stage in cutlass.range_constexpr(STAGES):
                prims.mbarrier_init(full.subview(stage), 1)
                prims.mbarrier_init(empty.subview(stage), 1)
    prims.fence_mbarrier_init()
    prims.barrier_cta_sync(0)
    if warp == 0:
        if prims.elect_sync():
            for stage in cutlass.range_constexpr(STAGES):
                prims.mbarrier_arrive(empty.subview(stage))
    prims.barrier_cta_sync(0)

    if warp == 0:
        # Prologue: each initially free stage can be filled without an acquire.
        for tile in cutlass.range_constexpr(STAGES):
            if tile < num_tiles:
                issue_tma_load(
                    descriptor, smem, full.subview(tile), cutlass.Int32(tile), tile
                )

        # Steady state: stage reuse is guarded by consumer release.
        for tile in cutlass.range(STAGES, num_tiles):
            stage = tile % STAGES
            phase = (tile // STAGES) & 1
            while not prims.mbarrier_try_wait_parity(
                empty.subview(stage), phase, time_limit=10_000_000
            ):
                pass
            issue_tma_load(descriptor, smem, full.subview(stage), tile, stage)

    if warp == 1:
        dst_ptr = dst.iterator.raw_ptr()
        for tile in cutlass.range(num_tiles):
            stage = tile % STAGES
            phase = (tile // STAGES) & 1
            while not prims.mbarrier_try_wait_parity(
                full.subview(stage), phase, time_limit=10_000_000
            ):
                pass

            # One warp drains all 128x64 elements.  Undo the descriptor's
            # 128-byte SMEM swizzle before writing the public row-major tensor.
            for row_group in cutlass.range_constexpr(TILE_M // 32):
                row = row_group * 32 + lane
                for col in cutlass.range_constexpr(TILE_K):
                    col_group = col // 8
                    col_rem = col % 8
                    physical_col = (col_group ^ (row & 7)) * 8 + col_rem
                    smem_index = stage * TILE_ELEMENTS + row * TILE_K + physical_col
                    dst_index = row * (num_tiles * TILE_K) + tile * TILE_K + col
                    (dst_ptr + dst_index).store((smem.data_ptr() + smem_index).load())

            prims.bar_warp_sync(cute.arch.FULL_MASK)
            if prims.elect_sync():
                prims.mbarrier_arrive(empty.subview(stage))

    # Producer tail: all participants rendezvous after the consumer has
    # released every live stage.  Only then can barrier storage be invalidated.
    prims.barrier_cta_sync(0)
    if tidx == 0:
        for stage in cutlass.range_constexpr(STAGES):
            prims.mbarrier_inval(full.subview(stage))
            prims.mbarrier_inval(empty.subview(stage))


@cute.jit
def launch(
    src: cute.Tensor,
    dst: cute.Tensor,
    num_tiles: int,
    STAGES: cutlass.Constexpr[int],
):
    descriptor = cuda.create_tensor_map_tiled(
        global_address=src.iterator.toint(),
        dtype=cutlass.Float16,
        global_dims=[src.shape[1], src.shape[0]],
        global_strides=[src.shape[1] * 2 // 16],
        box_dims=[TILE_K, TILE_M],
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    tma_pipeline_kernel(descriptor, dst, num_tiles, STAGES).launch(
        grid=(1, 1, 1), block=(THREADS, 1, 1)
    )


@lru_cache(maxsize=None)
def compile_kernel(stages: int, num_tiles: int):
    shape = (TILE_M, num_tiles * TILE_K)
    fake_src = make_fake_compact_tensor(
        cutlass.Float16, shape, stride_order=(1, 0), assumed_align=16
    )
    fake_dst = make_fake_compact_tensor(
        cutlass.Float16, shape, stride_order=(1, 0), assumed_align=16
    )
    return cute.compile(
        launch, fake_src, fake_dst, num_tiles, stages, options="--enable-tvm-ffi"
    )


def inspect_ptx() -> dict[str, list[str]]:
    lines: list[str] = []
    for path in ARTIFACT_DIR.glob("*.ptx"):
        lines.extend(line.strip() for line in path.read_text().splitlines())
    patterns = {
        "tma_load": ("cp.async.bulk.tensor", ".shared::cta.global", "complete_tx"),
        "expect_tx": ("mbarrier.arrive.expect_tx",),
        "wait": ("mbarrier.try_wait.parity",),
        "release": ("mbarrier.arrive",),
        "invalidate": ("mbarrier.inval",),
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
        raise RuntimeError(f"TMA pipeline requires SM90+, got {major}.{minor}")

    for stages, tile_cases in ((2, (1, 2, 3, 5)), (3, (1, 3, 4, 7))):
        for num_tiles in tile_cases:
            shape = (TILE_M, num_tiles * TILE_K)
            src = torch.arange(
                shape[0] * shape[1], dtype=torch.float16, device="cuda"
            ).reshape(shape)
            dst = torch.full_like(src, float("nan"))
            compile_kernel(stages, num_tiles)(src, dst, num_tiles)
            torch.cuda.synchronize()
            torch.testing.assert_close(dst, src, rtol=0, atol=0)
            print(
                f"stages={stages}, tiles={num_tiles}, shape={shape}, "
                f"wraps={num_tiles // stages}: PASS"
            )

    evidence = inspect_ptx()
    missing = [name for name, matches in evidence.items() if not matches]
    if missing:
        raise AssertionError(f"missing TMA pipeline PTX families: {missing}")
    for name, matches in evidence.items():
        print(f"PTX {name}: {matches[0]}")
    print("PASS")


if __name__ == "__main__":
    main()

