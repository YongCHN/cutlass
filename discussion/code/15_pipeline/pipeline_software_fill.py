"""Warp-specialized S-stage software-fill pipeline using SM90+ mbarrier.

Warp 0 is the producer and warp 1 is the consumer.  Every shared-memory stage
has a full barrier and an empty barrier.  Runtime tile counts exercise partial
fill, the first wrap, repeated steady-state reuse, and the final drain.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch


ARTIFACT_DIR = Path(__file__).with_name("generated_artifacts") / "software_fill"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("CUTE_DSL_KEEP", "ptx")
os.environ.setdefault("CUTE_DSL_DUMP_DIR", str(ARTIFACT_DIR))

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.experimental import primitives as prims


WARP_SIZE = 32
THREADS = 64
TILE = 128


@cute.kernel
def pipeline_kernel(
    src: cute.Tensor,
    dst: cute.Tensor,
    num_tiles: cutlass.Int32,
    STAGES: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    warp = tidx // WARP_SIZE
    lane = tidx % WARP_SIZE

    data = cutlass.Array(
        cutlass.Float32,
        STAGES * TILE,
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

    if warp == 0:
        if prims.elect_sync():
            for stage in cutlass.range_constexpr(STAGES):
                prims.mbarrier_init(full.subview(stage), 1)
                prims.mbarrier_init(empty.subview(stage), 1)
    prims.fence_mbarrier_init()
    prims.barrier_cta_sync(0)

    # Fresh barriers have parity 0.  Pre-arriving every empty barrier flips it
    # to 1, so the producer's first wait(old parity=0) succeeds immediately.
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

            global_base = tile * TILE
            stage_base = stage * TILE
            for item in cutlass.range_constexpr(TILE // WARP_SIZE):
                offset = lane + item * WARP_SIZE
                data[stage_base + offset] = src[global_base + offset]
            prims.bar_warp_sync(cute.arch.FULL_MASK)
            if prims.elect_sync():
                prims.mbarrier_arrive(full.subview(stage))

    if warp == 1:
        for tile in cutlass.range(num_tiles):
            stage = tile % STAGES
            phase = (tile // STAGES) & 1
            while not prims.mbarrier_try_wait_parity(
                full.subview(stage), phase, time_limit=10_000_000
            ):
                pass

            global_base = tile * TILE
            stage_base = stage * TILE
            for item in cutlass.range_constexpr(TILE // WARP_SIZE):
                offset = lane + item * WARP_SIZE
                dst[global_base + offset] = data[stage_base + offset]
            prims.bar_warp_sync(cute.arch.FULL_MASK)
            if prims.elect_sync():
                prims.mbarrier_arrive(empty.subview(stage))

    # This is the drain/tail: neither role may invalidate storage until both
    # loops have retired every stage generation they can still reference.
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
    STAGES: cutlass.Constexpr[int],
):
    pipeline_kernel(src, dst, num_tiles, STAGES).launch(
        grid=(1, 1, 1), block=(THREADS, 1, 1)
    )


@lru_cache(maxsize=None)
def compile_kernel(stages: int):
    symbolic_n = cute.sym_int64(divisibility=TILE)
    fake_src = make_fake_compact_tensor(cutlass.Float32, (symbolic_n,))
    fake_dst = make_fake_compact_tensor(cutlass.Float32, (symbolic_n,))
    return cute.compile(
        launch,
        fake_src,
        fake_dst,
        cutlass.Int32(0),
        stages,
        options="--enable-tvm-ffi",
    )


def inspect_ptx() -> dict[str, list[str]]:
    lines: list[str] = []
    for path in ARTIFACT_DIR.glob("*.ptx"):
        lines.extend(line.strip() for line in path.read_text().splitlines())
    patterns = {
        "init": "mbarrier.init",
        "arrive": "mbarrier.arrive",
        "wait": "mbarrier.try_wait.parity",
        "warp_publish": "bar.warp.sync",
        "invalidate": "mbarrier.inval",
    }
    return {
        name: [line for line in lines if pattern in line]
        for name, pattern in patterns.items()
    }


def state_trace(stages: int, num_tiles: int) -> str:
    return " ".join(
        f"{tile}:(s{tile % stages},p{(tile // stages) & 1})"
        for tile in range(num_tiles)
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires an NVIDIA GPU")
    major, minor = torch.cuda.get_device_capability()
    if major < 9:
        raise RuntimeError(f"mbarrier pipeline requires SM90+, got {major}.{minor}")

    for stages in (2, 3, 4):
        compiled = compile_kernel(stages)
        tile_cases = sorted({1, stages - 1, stages, stages + 1, 2 * stages + 1})
        for num_tiles in tile_cases:
            n = num_tiles * TILE
            src = torch.arange(n, dtype=torch.float32, device="cuda")
            dst = torch.full_like(src, float("nan"))
            compiled(src, dst, num_tiles)
            torch.cuda.synchronize()
            torch.testing.assert_close(dst, src, rtol=0, atol=0)
            print(
                f"stages={stages}, tiles={num_tiles}, "
                f"trace=[{state_trace(stages, num_tiles)}]: PASS"
            )

    evidence = inspect_ptx()
    missing = [name for name, matches in evidence.items() if not matches]
    if missing:
        raise AssertionError(f"missing software pipeline PTX families: {missing}")
    for name, matches in evidence.items():
        print(f"PTX {name}: {matches[0]}")
    print("PASS")


if __name__ == "__main__":
    main()

