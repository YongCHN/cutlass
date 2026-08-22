"""Thread-vector -> warp -> CTA shared-memory reduction ladder.

One 128-thread CTA reduces each row.  Every thread first folds a contiguous
Float32 vector, then a shuffle butterfly reduces within each warp, and warp
leaders publish four partials through shared memory for the CTA result.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch


ARTIFACT_DIR = Path(__file__).with_name("generated_artifacts") / "reduction_ladder"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("CUTE_DSL_KEEP", "ptx")
os.environ.setdefault("CUTE_DSL_DUMP_DIR", str(ARTIFACT_DIR))

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.experimental import primitives as prims


WARP_SIZE = 32
WARPS = 4
THREADS = WARP_SIZE * WARPS


@cute.jit
def warp_sum(value):
    for offset in (16, 8, 4, 2, 1):
        value = value + cute.arch.shuffle_sync_bfly(value, offset)
    return value


@cute.jit
def warp_max(value):
    for offset in (16, 8, 4, 2, 1):
        value = cute.math.max(value, cute.arch.shuffle_sync_bfly(value, offset))
    return value


@cute.kernel
def reduction_kernel(
    src: cute.Tensor,
    out_sum: cute.Tensor,
    out_max: cute.Tensor,
    ITEMS_PER_THREAD: cutlass.Constexpr[int],
):
    row, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    warp = tidx // WARP_SIZE
    lane = tidx % WARP_SIZE

    warp_sums = cutlass.Array(
        cutlass.Float32, WARPS, space=cutlass.AddressSpace.smem, alignment=16
    )
    warp_maxes = cutlass.Array(
        cutlass.Float32, WARPS, space=cutlass.AddressSpace.smem, alignment=16
    )

    row_width = THREADS * ITEMS_PER_THREAD
    thread_base = row * row_width + tidx * ITEMS_PER_THREAD
    values = (src.iterator.raw_ptr() + thread_base).load(
        count=ITEMS_PER_THREAD, alignment=ITEMS_PER_THREAD * 4
    )
    local_sum = values.reduce("add")
    local_max = values.reduce("max")
    lane_sum = warp_sum(local_sum)
    lane_max = warp_max(local_max)

    if lane == 0:
        warp_sums[warp] = lane_sum
        warp_maxes[warp] = lane_max
    prims.barrier_cta_sync(0)

    if warp == 0:
        cta_sum = cutlass.Float32(0.0)
        cta_max = cutlass.Float32(-float("inf"))
        if lane < WARPS:
            cta_sum = warp_sums[lane]
            cta_max = warp_maxes[lane]
        cta_sum = warp_sum(cta_sum)
        cta_max = warp_max(cta_max)
        if lane == 0:
            out_sum[row] = cta_sum
            out_max[row] = cta_max


@cute.jit
def launch(
    src: cute.Tensor,
    out_sum: cute.Tensor,
    out_max: cute.Tensor,
    ITEMS_PER_THREAD: cutlass.Constexpr[int],
):
    reduction_kernel(src, out_sum, out_max, ITEMS_PER_THREAD).launch(
        grid=(src.shape[0], 1, 1), block=(THREADS, 1, 1)
    )


@lru_cache(maxsize=None)
def compile_kernel(items_per_thread: int):
    fake_rows = cute.sym_int64()
    width = THREADS * items_per_thread
    fake_src = make_fake_compact_tensor(
        cutlass.Float32,
        (fake_rows, width),
        stride_order=(1, 0),
        assumed_align=max(16, items_per_thread * 4),
    )
    fake_out = make_fake_compact_tensor(cutlass.Float32, (fake_rows,))
    return cute.compile(
        launch,
        fake_src,
        fake_out,
        fake_out,
        items_per_thread,
        options="--enable-tvm-ffi",
    )


def inspect_ptx() -> dict[str, list[str]]:
    lines: list[str] = []
    for path in ARTIFACT_DIR.glob("*.ptx"):
        lines.extend(line.strip() for line in path.read_text().splitlines())
    patterns = {
        "vector_load": "ld.global.v",
        "shuffle": "shfl.sync.bfly",
        "shared_store": "st.shared",
        "cta_barrier": "barrier.sync",
    }
    return {
        name: [line for line in lines if pattern in line]
        for name, pattern in patterns.items()
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires an NVIDIA GPU")

    for items_per_thread in (1, 4, 8):
        width = THREADS * items_per_thread
        compiled = compile_kernel(items_per_thread)
        for rows in (1, 7, 33):
            values = torch.arange(rows * width, device="cuda", dtype=torch.float32)
            src = ((values % 31) - 15).reshape(rows, width).contiguous()
            out_sum = torch.empty(rows, dtype=torch.float32, device="cuda")
            out_max = torch.empty_like(out_sum)
            compiled(src, out_sum, out_max)
            torch.cuda.synchronize()
            expected_sum = src.sum(dim=1)
            expected_max = src.max(dim=1).values
            torch.testing.assert_close(out_sum, expected_sum, rtol=0, atol=0)
            torch.testing.assert_close(out_max, expected_max, rtol=0, atol=0)
            print(f"rows={rows}, width={width}, items/thread={items_per_thread}: PASS")

    evidence = inspect_ptx()
    # ITEMS_PER_THREAD=1 is scalar, but x4/x8 must provide vector-load evidence.
    missing = [name for name, matches in evidence.items() if not matches]
    if missing:
        raise AssertionError(f"missing reduction PTX families: {missing}")
    for name, matches in evidence.items():
        print(f"PTX {name}: {matches[0]}")
    print("PASS")


if __name__ == "__main__":
    main()

