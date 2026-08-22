"""SM80+ per-thread cp.async roundtrip: GMEM -> SMEM -> GMEM.

Each thread owns one aligned 16-byte vector.  The example deliberately has no
residue path: N must be a multiple of BLOCK * 4.  That keeps the instruction,
group-completion, and CTA-publication contracts visible without mixing them
with boundary handling.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch


ARTIFACT_DIR = Path(__file__).with_name("generated_artifacts") / "cp_async"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("CUTE_DSL_KEEP", "ptx")
os.environ.setdefault("CUTE_DSL_DUMP_DIR", str(ARTIFACT_DIR))

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.experimental import primitives as prims


BLOCK = 128
ELEMENTS_PER_THREAD = 4
COPY_BYTES = 16
TILE_ELEMENTS = BLOCK * ELEMENTS_PER_THREAD


@cute.kernel
def copy_kernel(src: cute.Tensor, dst: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    smem = cutlass.Array(
        cutlass.Float32,
        TILE_ELEMENTS,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )

    tile_base = bidx * TILE_ELEMENTS
    thread_base = tidx * ELEMENTS_PER_THREAD

    # One thread issues exactly one 16-byte SM80 cp.async instruction.
    prims.cp_async_shared_global(
        smem.data_ptr() + thread_base,
        src.iterator.raw_ptr() + tile_base + thread_base,
        COPY_BYTES,
        "cg",
    )

    # wait_group(0) completes each lane's preceding copy group.  The CTA
    # barrier is still needed because the following reader is allowed to be a
    # different thread in the general producer/consumer pattern.
    prims.cp_async_commit_group()
    prims.cp_async_wait_group(0)
    prims.barrier_cta_sync(0)

    # Read the next lane's vector and write it to that vector's global slot.
    # The permutation keeps dst == src while making the CTA barrier genuinely
    # load-bearing: no lane can consume until every producer has completed.
    consumer_thread = (tidx + 1) % BLOCK
    consumer_base = consumer_thread * ELEMENTS_PER_THREAD
    values = (smem.data_ptr() + consumer_base).load(count=ELEMENTS_PER_THREAD)
    (dst.iterator.raw_ptr() + tile_base + consumer_base).store(values)


@cute.jit
def launch(src: cute.Tensor, dst: cute.Tensor):
    copy_kernel(src, dst).launch(
        grid=(src.shape[0] // TILE_ELEMENTS, 1, 1),
        block=(BLOCK, 1, 1),
    )


@lru_cache(maxsize=1)
def compile_kernel():
    symbolic_n = cute.sym_int64(divisibility=TILE_ELEMENTS)
    fake_src = make_fake_compact_tensor(cutlass.Float32, (symbolic_n,))
    fake_dst = make_fake_compact_tensor(cutlass.Float32, (symbolic_n,))
    return cute.compile(launch, fake_src, fake_dst, options="--enable-tvm-ffi")


def inspect_ptx() -> dict[str, list[str]]:
    lines: list[str] = []
    for path in ARTIFACT_DIR.glob("*.ptx"):
        lines.extend(line.strip() for line in path.read_text().splitlines())
    return {
        "copy": [line for line in lines if "cp.async.cg.shared.global" in line],
        "commit": [line for line in lines if "cp.async.commit_group" in line],
        "wait": [
            line
            for line in lines
            if "cp.async.wait_group" in line and "0" in line
        ],
        "cta_barrier": [
            line
            for line in lines
            if "bar.sync" in line
            or "barrier.sync" in line
            or "barrier.cta.sync" in line
        ],
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires an NVIDIA GPU")
    major, minor = torch.cuda.get_device_capability()
    if major < 8:
        raise RuntimeError(f"cp.async requires SM80+, got {major}.{minor}")

    compiled = compile_kernel()
    for n in (TILE_ELEMENTS, 4 * TILE_ELEMENTS, 16 * TILE_ELEMENTS):
        src = torch.arange(n, dtype=torch.float32, device="cuda")
        dst = torch.full_like(src, float("nan"))
        compiled(src, dst)
        torch.cuda.synchronize()
        torch.testing.assert_close(dst, src, rtol=0, atol=0)
        print(f"n={n}, tile={TILE_ELEMENTS}, max_abs_error={(dst-src).abs().max().item():.1e}")

    evidence = inspect_ptx()
    missing = [name for name, matches in evidence.items() if not matches]
    if missing:
        raise AssertionError(f"missing PTX instruction families: {missing}")
    for name, matches in evidence.items():
        print(f"PTX {name}: {matches[0]}")
    print("PASS")


if __name__ == "__main__":
    main()
