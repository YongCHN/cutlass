"""Two-warp, multi-stage producer/consumer protocol using SM90+ mbarrier.

Each stage has a full barrier and an empty barrier.  The producer waits for
empty, fills the stage, and signals full.  The consumer waits for full,
accumulates the stage, and signals empty.  Runtime tile counts intentionally
wrap the two-stage ring so both parity values are exercised.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch


ARTIFACT_DIR = Path(__file__).with_name("generated_artifacts") / "mbarrier_pingpong"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("CUTE_DSL_KEEP", "ptx")
os.environ.setdefault("CUTE_DSL_DUMP_DIR", str(ARTIFACT_DIR))

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.experimental import primitives as prims


WARP_SIZE = 32
BLOCK_THREADS = 64
STAGES = 2


@cute.kernel
def pingpong_kernel(
    src: cute.Tensor,
    dst: cute.Tensor,
    num_tiles: cutlass.Int32,
):
    tidx, _, _ = cute.arch.thread_idx()
    lane = tidx % WARP_SIZE
    warp = tidx // WARP_SIZE

    smem = cutlass.Array(
        cutlass.Float32,
        STAGES * WARP_SIZE,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )
    barriers = cutlass.Array(
        cutlass.Int64,
        2 * STAGES,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    full = barriers
    empty = barriers.subview(STAGES)

    # One elected producer lane initializes every barrier object.  Publishing
    # initialization and then rendezvousing the CTA are separate obligations.
    if warp == 0:
        if prims.elect_sync():
            for stage in cutlass.range_constexpr(STAGES):
                prims.mbarrier_init(full.subview(stage), 1)
                prims.mbarrier_init(empty.subview(stage), 1)
    prims.fence_mbarrier_init()
    prims.barrier_cta_sync(0)

    # Mark every stage empty before the first producer iteration.  A fresh
    # barrier starts at parity 0; one arrive completes it and flips to 1.
    if warp == 0:
        if prims.elect_sync():
            for stage in cutlass.range_constexpr(STAGES):
                prims.mbarrier_arrive(empty.subview(stage))
    prims.barrier_cta_sync(0)

    if warp == 0:
        for tile in cutlass.range(num_tiles):
            stage = tile % STAGES
            phase = (tile // STAGES) & 1
            while not prims.mbarrier_try_wait_parity(
                empty.subview(stage), phase, time_limit=10_000_000
            ):
                pass

            smem[stage * WARP_SIZE + lane] = src[tile * WARP_SIZE + lane]
            # All producer lanes must publish their values before the elected
            # lane is allowed to signal stage completion.
            prims.bar_warp_sync(cute.arch.FULL_MASK)
            if prims.elect_sync():
                prims.mbarrier_arrive(full.subview(stage))

    if warp == 1:
        accum = cutlass.Float32(0.0)
        for tile in cutlass.range(num_tiles):
            stage = tile % STAGES
            phase = (tile // STAGES) & 1
            while not prims.mbarrier_try_wait_parity(
                full.subview(stage), phase, time_limit=10_000_000
            ):
                pass

            accum = accum + smem[stage * WARP_SIZE + lane]
            # Ensure every consumer has finished reading before one elected
            # lane releases the buffer back to the producer.
            prims.bar_warp_sync(cute.arch.FULL_MASK)
            if prims.elect_sync():
                prims.mbarrier_arrive(empty.subview(stage))
        dst[lane] = accum

    # No barrier object is invalidated while either role can still use it.
    prims.barrier_cta_sync(0)
    if tidx == 0:
        for stage in cutlass.range_constexpr(STAGES):
            prims.mbarrier_inval(full.subview(stage))
            prims.mbarrier_inval(empty.subview(stage))


@cute.jit
def launch(
    src: cute.Tensor,
    dst: cute.Tensor,
    num_tiles: cutlass.Int32,
):
    pingpong_kernel(src, dst, num_tiles).launch(
        grid=(1, 1, 1), block=(BLOCK_THREADS, 1, 1)
    )


def compile_kernel():
    symbolic_src = cute.sym_int64(divisibility=WARP_SIZE)
    fake_src = make_fake_compact_tensor(cutlass.Float32, (symbolic_src,))
    fake_dst = make_fake_compact_tensor(cutlass.Float32, (WARP_SIZE,))
    return cute.compile(
        launch,
        fake_src,
        fake_dst,
        cutlass.Int32(0),
        options="--enable-tvm-ffi",
    )


def inspect_ptx() -> dict[str, list[str]]:
    ptx_files = sorted(ARTIFACT_DIR.glob("*.ptx"), key=lambda p: p.stat().st_mtime)
    if not ptx_files:
        raise RuntimeError("CUTE_DSL_KEEP=ptx did not produce a PTX artifact")
    lines: list[str] = []
    for path in ptx_files:
        lines.extend(line.strip() for line in path.read_text().splitlines())
    patterns = {
        "init": "mbarrier.init",
        "init_fence": "fence.mbarrier_init",
        "arrive": "mbarrier.arrive",
        "wait_parity": "mbarrier.try_wait.parity",
        "invalidate": "mbarrier.inval",
    }
    return {
        name: [line for line in lines if pattern in line]
        for name, pattern in patterns.items()
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires an NVIDIA GPU")
    major, minor = torch.cuda.get_device_capability()
    if major < 9:
        raise RuntimeError(f"mbarrier requires SM90+, got {major}.{minor}")

    compiled = compile_kernel()
    for num_tiles in (1, 2, 3, 5, 8):
        src = torch.arange(
            num_tiles * WARP_SIZE, dtype=torch.float32, device="cuda"
        )
        dst = torch.full((WARP_SIZE,), float("nan"), device="cuda")
        compiled(src, dst, num_tiles)
        torch.cuda.synchronize()
        expected = src.reshape(num_tiles, WARP_SIZE).sum(dim=0)
        torch.testing.assert_close(dst, expected, rtol=0, atol=0)
        print(
            f"num_tiles={num_tiles}, stages={STAGES}, wraps={num_tiles // STAGES}, "
            f"dst0={dst[0].item():.1f}, max_abs_error="
            f"{(dst - expected).abs().max().item():.1e}"
        )

    evidence = inspect_ptx()
    missing = [name for name, lines in evidence.items() if not lines]
    if missing:
        raise AssertionError(f"missing expected PTX mbarrier families: {missing}")
    for name, lines in evidence.items():
        print(f"PTX {name}: {lines[0]}")
    print("PASS")


if __name__ == "__main__":
    main()
