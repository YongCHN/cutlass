"""Warp collectives and named CTA barriers in one verifiable program.

Target: SM90+ for elect.sync; the individual vote/shuffle/warp-barrier
operations have wider architecture support.  Validated target is B200/SM100.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch


ARTIFACT_DIR = Path(__file__).with_name("generated_artifacts") / "warp_collectives"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("CUTE_DSL_KEEP", "ptx")
os.environ.setdefault("CUTE_DSL_DUMP_DIR", str(ARTIFACT_DIR))

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.experimental import primitives as prims


WARP_SIZE = 32
WARP_OUT = 39
CTA_OUT = 3
FULL_MASK = 0xFFFFFFFF


@cute.kernel
def warp_collectives_kernel(out: cute.Tensor):
    lane, _, _ = cute.arch.thread_idx()
    smem = cutlass.Array(
        cutlass.Int32, 1, space=cutlass.AddressSpace.smem, alignment=4
    )

    # One writer followed by a warp-scoped acquire-release rendezvous.
    if lane == 0:
        smem[0] = cutlass.Int32(1234)
    prims.bar_warp_sync(FULL_MASK)

    # elect.sync returns one Boolean winner.  It does not promise lane 0.
    elected = prims.elect_sync()
    out[lane] = cutlass.Int32(1) if elected else cutlass.Int32(0)
    if elected:
        out[32] = lane

    predicate = lane < 8
    any_true = prims.vote_sync(FULL_MASK, predicate, prims.VoteSync.ANY)
    all_true = prims.vote_sync(FULL_MASK, predicate, prims.VoteSync.ALL)
    ballot = prims.vote_sync(FULL_MASK, predicate, prims.VoteSync.BALLOT)

    # Indexed shuffle broadcasts the value owned by source lane 7.
    lane_value = lane * cutlass.Int32(10)
    broadcast = prims.shfl_sync(
        FULL_MASK,
        lane_value,
        cutlass.Int32(7),
        0x1F,
        prims.Shfl.IDX,
    )
    warp_sum = prims.redux_sync(
        lane,
        prims.ReductionKind.ADD,
        FULL_MASK,
    )

    if lane == 0:
        out[33] = any_true
        out[34] = all_true
        out[35] = ballot
        out[36] = broadcast
        out[37] = warp_sum
        out[38] = smem[0]


@cute.kernel
def cta_barriers_kernel(out: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    warp = tidx // WARP_SIZE
    lane = tidx % WARP_SIZE
    smem = cutlass.Array(
        cutlass.Int32, 1, space=cutlass.AddressSpace.smem, alignment=4
    )

    # Named slot 1 has exactly 64 participants.  Warp 0 arrives without
    # waiting; warp 1 waits.  Every lane in each warp participates.
    if warp == 0:
        if lane == 0:
            smem[0] = cutlass.Int32(0xC0DE)
        prims.barrier_cta_arrive(barrier_id=1, thread_count=2 * WARP_SIZE)
    else:
        prims.barrier_cta_sync(barrier_id=1, thread_count=2 * WARP_SIZE)
        if lane == 0:
            out[0] = smem[0]

    # Independent slots avoid mixing reduction and non-reduction operations in
    # the same active barrier generation.
    any_lane_zero = prims.barrier_cta_red(
        tidx == 0,
        barrier_id=2,
        kind="or",
        thread_count=2 * WARP_SIZE,
    )
    even_count = prims.barrier_cta_red(
        tidx % 2 == 0,
        barrier_id=3,
        kind="popc",
        thread_count=2 * WARP_SIZE,
    )
    if tidx == 0:
        out[1] = any_lane_zero
        out[2] = even_count


@cute.jit
def launch_warp(out: cute.Tensor):
    warp_collectives_kernel(out).launch(
        grid=(1, 1, 1), block=(WARP_SIZE, 1, 1)
    )


@cute.jit
def launch_cta(out: cute.Tensor):
    cta_barriers_kernel(out).launch(
        grid=(1, 1, 1), block=(2 * WARP_SIZE, 1, 1)
    )


def inspect_ptx() -> dict[str, list[str]]:
    ptx_files = sorted(ARTIFACT_DIR.glob("*.ptx"), key=lambda p: p.stat().st_mtime)
    if not ptx_files:
        raise RuntimeError("CUTE_DSL_KEEP=ptx did not produce a PTX artifact")
    lines: list[str] = []
    for path in ptx_files:
        lines.extend(line.strip() for line in path.read_text().splitlines())

    patterns = {
        "elect": ("elect.sync",),
        "vote": ("vote.sync",),
        "shuffle": ("shfl.sync",),
        "redux": ("redux.sync",),
        "warp_barrier": ("bar.warp.sync",),
        # Depending on the PTX ISA spelling selected by the backend, named CTA
        # operations can appear as barrier.{arrive,sync,red}, barrier.cta.*, or
        # the legacy bar.* aliases.  Match the semantic family, not one spelling.
        "cta_arrive": ("barrier.cta.arrive", "barrier.arrive", "bar.arrive"),
        "cta_sync": ("barrier.cta.sync", "barrier.sync", "bar.sync"),
        "cta_red": ("barrier.cta.red", "barrier.red", "bar.red"),
    }
    return {
        name: [line for line in lines if any(pattern in line for pattern in choices)]
        for name, choices in patterns.items()
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires an NVIDIA GPU")
    major, minor = torch.cuda.get_device_capability()
    if major < 9:
        raise RuntimeError(f"elect.sync requires SM90+, got {major}.{minor}")

    warp_out = torch.full((WARP_OUT,), -1, dtype=torch.int32, device="cuda")
    cta_out = torch.full((CTA_OUT,), -1, dtype=torch.int32, device="cuda")
    warp_arg = from_dlpack(warp_out)
    cta_arg = from_dlpack(cta_out)
    warp_fn = cute.compile(launch_warp, warp_arg)
    cta_fn = cute.compile(launch_cta, cta_arg)
    warp_fn(warp_arg)
    cta_fn(cta_arg)
    torch.cuda.synchronize()

    flags = warp_out[:WARP_SIZE].cpu()
    elected_lane = int(warp_out[32].item())
    if int(flags.sum().item()) != 1 or int(flags[elected_lane].item()) != 1:
        raise AssertionError("elect.sync must select exactly one participating lane")
    expected_warp_tail = torch.tensor(
        [1, 0, 0xFF, 70, 496, 1234], dtype=torch.int32, device="cuda"
    )
    torch.testing.assert_close(warp_out[33:], expected_warp_tail, rtol=0, atol=0)
    expected_cta = torch.tensor(
        [0xC0DE, 1, 32], dtype=torch.int32, device="cuda"
    )
    torch.testing.assert_close(cta_out, expected_cta, rtol=0, atol=0)

    evidence = inspect_ptx()
    missing = [name for name, lines in evidence.items() if not lines]
    if missing:
        raise AssertionError(f"missing expected PTX instruction families: {missing}")

    print(
        f"warp PASS: elected_lane={elected_lane}, ballot=0x{int(warp_out[35]):x}, "
        f"broadcast={int(warp_out[36])}, redux_sum={int(warp_out[37])}"
    )
    print(
        f"cta PASS: handoff=0x{int(cta_out[0]):x}, "
        f"any_lane_zero={int(cta_out[1])}, even_count={int(cta_out[2])}"
    )
    for name, lines in evidence.items():
        print(f"PTX {name}: {lines[0]}")
    print("PASS")


if __name__ == "__main__":
    main()
