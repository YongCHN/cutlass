"""Blackwell TMEM allocation, store, load, wait, and deallocation roundtrip.

Target: datacenter Blackwell SM100/SM103.
Each of 32 lanes stores and reloads 32 FP32 values in TMEM.  The allocation is
32 columns, the minimum hardware allocation granularity.
"""

from __future__ import annotations

import argparse
import os
from functools import lru_cache
from pathlib import Path

import torch


ARTIFACT_DIR = Path(__file__).with_name("generated_artifacts") / "tmem"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("CUTE_DSL_KEEP", "ptx")
os.environ.setdefault("CUTE_DSL_DUMP_DIR", str(ARTIFACT_DIR))

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.experimental import primitives as prims


WARP = 32
COLS = 32


@cute.kernel
def kernel(out: cute.Tensor):
    lane, _, _ = cute.arch.thread_idx()
    tmem_addr_smem = cutlass.Array(
        cutlass.Int32, 1, space=cutlass.AddressSpace.smem, alignment=4
    )

    # alloc/dealloc are warp collective.  The address is returned through
    # SMEM and becomes visible after the CTA barrier.
    prims.tcgen05_alloc(tmem_addr_smem, COLS, group="cta_1")
    prims.barrier_cta_sync(0)
    tmem_ptr = cutlass.inttoptr(tmem_addr_smem.load(), 6, cutlass.Float32)

    values = cutlass.Vector.from_elements(
        tuple(cutlass.Float32(lane * COLS + i) for i in range(COLS)),
        dtype=cutlass.Float32,
    )
    prims.tcgen05_st("32x32b", tmem_ptr, values)
    prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
    result = prims.tcgen05_ld("32x32b", tmem_ptr, num=COLS)
    prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)

    for i in cutlass.range_constexpr(COLS):
        out[lane, i] = result[i]

    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    prims.barrier_cta_sync(0)
    prims.tcgen05_dealloc(tmem_ptr, COLS, group="cta_1")
    prims.tcgen05_relinquish_alloc_permit(group="cta_1")


@cute.jit
def launch(out: cute.Tensor):
    kernel(out).launch(grid=(1, 1, 1), block=(WARP, 1, 1))


@lru_cache(maxsize=1)
def compile_kernel():
    fake = make_fake_compact_tensor(
        cutlass.Float32, (WARP, COLS), stride_order=(1, 0), assumed_align=16
    )
    return cute.compile(launch, fake, options="--enable-tvm-ffi")


def inspect_ptx() -> None:
    ptx = "\n".join(path.read_text() for path in ARTIFACT_DIR.glob("*.ptx"))
    patterns = {
        "alloc": "tcgen05.alloc",
        "store": "tcgen05.st.sync.aligned.32x32b",
        "store wait": "tcgen05.wait::st",
        "load": "tcgen05.ld.sync.aligned.32x32b",
        "load wait": "tcgen05.wait::ld",
        "dealloc": "tcgen05.dealloc",
        "relinquish": "tcgen05.relinquish_alloc_permit",
    }
    for name, needle in patterns.items():
        lines = [line.strip() for line in ptx.splitlines() if needle in line]
        if not lines:
            raise AssertionError(f"missing {name} PTX containing {needle!r}")
        print(f"PTX {name}: {lines[0]}")


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires an NVIDIA GPU")
    major, minor = torch.cuda.get_device_capability()
    if major != 10:
        raise RuntimeError(f"TMEM requires datacenter Blackwell SM100/103, got {major}.{minor}")
    out = torch.empty((WARP, COLS), device="cuda", dtype=torch.float32)
    compile_kernel()(out)
    torch.cuda.synchronize()
    expected = torch.arange(WARP * COLS, device="cuda", dtype=torch.float32).reshape(
        WARP, COLS
    )
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    print(f"shape={tuple(out.shape)}, exact roundtrip: PASS")
    inspect_ptx()
    print("PASS")


if __name__ == "__main__":
    main()
