"""Fixed-shape Blackwell CTA_1 tcgen05 FP16 GEMM.

Target: datacenter Blackwell SM100/SM103.
Data path: GMEM TMA -> swizzled SMEM descriptors -> tcgen05 MMA -> FP32 TMEM
           -> tcgen05 load -> FP32 GMEM.
Shape: A[128,64], B[128,64], C[128,128] = A @ B.T.
"""

from __future__ import annotations

import argparse
import os
from functools import lru_cache
from pathlib import Path

import torch


ARTIFACT_DIR = Path(__file__).with_name("generated_artifacts") / "cta1"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("CUTE_DSL_KEEP", "ptx")
os.environ.setdefault("CUTE_DSL_DUMP_DIR", str(ARTIFACT_DIR))

import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.experimental import primitives as prims
from cutlass.experimental.primitives import Tcgen05InstrDesc


M = 128
N = 128
K = 64
K_GRANULE = 16
THREADS = 128
NUM_TMEM_COLS = (N // 8) * 32


@cute.kernel
def kernel(
    tma_a: cutlass.GridConstant[cuda.TensorMap],
    tma_b: cutlass.GridConstant[cuda.TensorMap],
    c: cute.Tensor,
):
    tid, _, _ = cute.arch.thread_idx()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    s_a = cutlass.Array(
        cutlass.Float16,
        M * K,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    s_b = cutlass.Array(
        cutlass.Float16,
        N * K,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    full_bar = cutlass.Array(
        cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8
    )
    acc_bar = cutlass.Array(
        cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8
    )
    tmem_addr = cutlass.Array(
        cutlass.Int32, 1, space=cutlass.AddressSpace.smem, alignment=4
    )

    if warp == 0:
        if prims.elect_sync():
            prims.mbarrier_init(full_bar, 1)
            prims.mbarrier_init(acc_bar, 1)
    prims.fence_mbarrier_init()
    prims.barrier_cta_sync(0)

    if warp == 0:
        prims.tcgen05_alloc(tmem_addr, NUM_TMEM_COLS, group="cta_1")
    prims.barrier_cta_sync(0)
    tmem_ptr = cutlass.inttoptr(tmem_addr.load(), 6, cutlass.Float32)
    prims.tcgen05_relinquish_alloc_permit(group="cta_1")

    idesc = Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cutlass.Float16,
        b_dtype=cutlass.Float16,
        n_dim=N,
        m_dim=M,
    )

    if warp == 0:
        desc_a = prims.Tcgen05SmemDesc.build(
            start_address=s_a,
            leading_byte_offset=16,
            stride_byte_offset=8 * K * 2,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
        )
        desc_b = prims.Tcgen05SmemDesc.build(
            start_address=s_b,
            leading_byte_offset=16,
            stride_byte_offset=8 * K * 2,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
        )
        if prims.elect_sync():
            tx_bytes = tma_a.global_tx_bytes() + tma_b.global_tx_bytes()
            prims.mbarrier_arrive_expect_tx(full_bar, tx_bytes)
            prims.cp_async_bulk_tensor_shared_cta_global(
                s_a, tma_a.get_ptr(), (0, 0), full_bar
            )
            prims.cp_async_bulk_tensor_shared_cta_global(
                s_b, tma_b.get_ptr(), (0, 0), full_bar
            )

        while not prims.mbarrier_try_wait_parity(full_bar, 0, time_limit=10_000_000):
            pass

        for kb in cutlass.range_constexpr(K // K_GRANULE):
            byte_offset: cutlass.Constexpr[int] = kb * K_GRANULE * 2
            if prims.elect_sync():
                prims.tcgen05_mma(
                    prims.Tcgen05MMAKind.F16,
                    prims.CTAGroup.CTA_1,
                    tmem_ptr,
                    desc_a.advance_start_address(byte_offset),
                    desc_b.advance_start_address(byte_offset),
                    idesc,
                    kb != 0,
                )
        if prims.elect_sync():
            prims.tcgen05_commit(acc_bar, group=prims.CTAGroup.CTA_1)

    prims.barrier_cta_sync(0)
    while not prims.mbarrier_try_wait_parity(acc_bar, 0, time_limit=10_000_000):
        pass

    # Four warps drain the 128 TMEM rows.  Each tcgen05.ld returns 32
    # contiguous columns for one row per lane.
    lane = tid & cutlass.Int32(31)
    tmem_sp = warp % cutlass.Int32(4)
    base = prims.TmemAddr(tmem_addr.load())
    tmem_row = base.row_id + tmem_sp * cutlass.Int32(32)
    row = tmem_sp * cutlass.Int32(32) + lane
    g_c = c.iterator.raw_ptr()
    for subtile in cutlass.range_constexpr(N // 32):
        src = prims.TmemAddr.from_row_col(
            tmem_row, base.col_id + subtile * 32
        ).as_ptr(cutlass.Float32)
        values = prims.tcgen05_ld("32x32b", src, num=32)
        prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
        (g_c + row * N + subtile * 32).store(values, alignment=16)

    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    prims.barrier_cta_sync(0)
    if warp == 0:
        prims.tcgen05_dealloc(tmem_ptr, NUM_TMEM_COLS, group="cta_1")


@cute.jit
def launch(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    tma_a = cuda.create_tensor_map_tiled(
        global_address=a.iterator.toint(),
        dtype=cutlass.Float16,
        global_dims=[K, M],
        global_strides=[K * 2 // 16],
        box_dims=[K, M],
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    tma_b = cuda.create_tensor_map_tiled(
        global_address=b.iterator.toint(),
        dtype=cutlass.Float16,
        global_dims=[K, N],
        global_strides=[K * 2 // 16],
        box_dims=[K, N],
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    kernel(tma_a, tma_b, c).launch(
        grid=(1, 1, 1), block=(THREADS, 1, 1)
    )


@lru_cache(maxsize=1)
def compile_kernel():
    fake_a = make_fake_compact_tensor(
        cutlass.Float16, (M, K), stride_order=(1, 0), assumed_align=16
    )
    fake_b = make_fake_compact_tensor(
        cutlass.Float16, (N, K), stride_order=(1, 0), assumed_align=16
    )
    fake_c = make_fake_compact_tensor(
        cutlass.Float32, (M, N), stride_order=(1, 0), assumed_align=16
    )
    return cute.compile(
        launch, fake_a, fake_b, fake_c, options="--enable-tvm-ffi"
    )


def inspect_ptx() -> None:
    ptx = "\n".join(path.read_text() for path in ARTIFACT_DIR.glob("*.ptx"))
    patterns = {
        "TMA": ("cp.async.bulk.tensor", ".shared::cta.global"),
        "alloc": ("tcgen05.alloc.cta_group::1",),
        "MMA": ("tcgen05.mma.cta_group::1.kind::f16",),
        "commit": ("tcgen05.commit.cta_group::1",),
        "TMEM load": ("tcgen05.ld.sync.aligned.32x32b",),
        "dealloc": ("tcgen05.dealloc.cta_group::1",),
    }
    for name, tokens in patterns.items():
        lines = [
            line.strip()
            for line in ptx.splitlines()
            if all(token in line for token in tokens)
        ]
        if not lines:
            raise AssertionError(f"missing {name} PTX tokens {tokens}")
        print(f"PTX {name}: {lines[0]}")


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires an NVIDIA GPU")
    major, minor = torch.cuda.get_device_capability()
    if major != 10:
        raise RuntimeError(f"tcgen05 requires datacenter Blackwell, got {major}.{minor}")
    torch.manual_seed(20)
    a = torch.randint(-2, 3, (M, K), device="cuda", dtype=torch.int32).to(torch.float16)
    b = torch.randint(-2, 3, (N, K), device="cuda", dtype=torch.int32).to(torch.float16)
    c = torch.empty((M, N), device="cuda", dtype=torch.float32)
    compile_kernel()(a, b, c)
    torch.cuda.synchronize()
    reference = a.float() @ b.float().T
    torch.testing.assert_close(c, reference, rtol=0, atol=0)
    print(f"shape=({M},{N},{K}), bit-exact FP32 output: PASS")
    inspect_ptx()
    print("PASS")


if __name__ == "__main__":
    main()
