"""Minimal Hopper TMA -> SMEM descriptor -> WGMMA -> RMEM GEMM.

Target: SM90a only (H100/H200).
Input:  A[64,16], B[64,16], FP16, contiguous and 16-byte aligned.
Output: C[64,64] = A @ B.T in FP32.

The program always AOT-compiles and checks retained SM90a PTX.  It performs
numerical execution only on compute capability 9.x: an SM90a cubin cannot be
launched on B200/SM100 (CUDA error 209), even though the B200 development
machine can compile and inspect it.
"""

from __future__ import annotations

import argparse
import os
from functools import lru_cache
from pathlib import Path

import torch


ARTIFACT_DIR = Path(__file__).with_name("generated_artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
os.environ["CUTE_DSL_ARCH"] = "sm_90a"
os.environ.setdefault("CUTE_DSL_KEEP", "ptx")
os.environ.setdefault("CUTE_DSL_DUMP_DIR", str(ARTIFACT_DIR))

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import make_fake_compact_tensor


M = 64
N = 64
K = 16
THREADS = 128
DTYPE = cutlass.Float16
ACC_DTYPE = cutlass.Float32


class HopperWgmmaGemm:
    @cute.jit
    def __call__(self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
        major = cute.nvgpu.OperandMajorMode.K
        tiled_mma = utils.sm90_make_trivial_tiled_mma(
            DTYPE,
            DTYPE,
            major,
            major,
            ACC_DTYPE,
            (1, 1, 1),
            (M, N),
        )

        # WGMMA descriptors encode a swizzled SMEM layout.  The composed
        # layout's swizzle is moved onto the pointer by MemRange.get_tensor.
        a_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW32, DTYPE
        )
        b_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW32, DTYPE
        )
        s_a_layout = cute.tile_to_shape(a_atom, (M, K), order=(0, 1))
        s_b_layout = cute.tile_to_shape(b_atom, (N, K), order=(0, 1))

        @cute.struct
        class SharedStorage:
            barrier: cute.struct.MemRange[cutlass.Int64, 1]
            a: cute.struct.Align[
                cute.struct.MemRange[DTYPE, cute.cosize(s_a_layout)], 1024
            ]
            b: cute.struct.Align[
                cute.struct.MemRange[DTYPE, cute.cosize(s_b_layout)], 1024
            ]

        self.storage = SharedStorage
        self.tx_bytes = cute.size_in_bytes(DTYPE, s_a_layout) + cute.size_in_bytes(
            DTYPE, s_b_layout
        )

        tma_a = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), a, s_a_layout, (M, K)
        )
        tma_b = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), b, s_b_layout, (N, K)
        )
        self.kernel(
            tma_a.atom,
            tma_a.tma_tensor,
            tma_b.atom,
            tma_b.tma_tensor,
            c,
            tiled_mma,
            tma_a.smem_layout,
            tma_b.smem_layout,
        ).launch(grid=(1, 1, 1), block=(THREADS, 1, 1))

    @cute.kernel
    def kernel(
        self,
        tma_a: cute.CopyAtom,
        g_a: cute.Tensor,
        tma_b: cute.CopyAtom,
        g_b: cute.Tensor,
        g_c: cute.Tensor,
        tiled_mma: cute.TiledMma,
        s_a_layout: cute.ComposedLayout,
        s_b_layout: cute.ComposedLayout,
    ):
        tid, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.storage)
        barrier = storage.barrier.data_ptr()
        s_a = storage.a.get_tensor(s_a_layout.outer, swizzle=s_a_layout.inner)
        s_b = storage.b.get_tensor(s_b_layout.outer, swizzle=s_b_layout.inner)

        # Only warp 0 initializes/issues TMA.  All 128 threads wait for the
        # transaction-counted completion before WGMMA may read descriptors.
        if warp == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(barrier, 1)
                cute.arch.mbarrier_expect_tx(barrier, self.tx_bytes)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        t_s_a, t_g_a = cpasync.tma_partition(
            tma_a,
            0,
            cute.make_layout(1),
            cute.group_modes(s_a, 0, 2),
            cute.group_modes(g_a, 0, 2),
        )
        t_s_b, t_g_b = cpasync.tma_partition(
            tma_b,
            0,
            cute.make_layout(1),
            cute.group_modes(s_b, 0, 2),
            cute.group_modes(g_b, 0, 2),
        )
        if warp == 0:
            cute.copy(tma_a, t_g_a, t_s_a, tma_bar_ptr=barrier)
            cute.copy(tma_b, t_g_b, t_s_b, tma_bar_ptr=barrier)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(barrier)
        cute.arch.mbarrier_wait(barrier, 0)

        # A single warpgroup is one TiledMMA participant.  A/B fragments are
        # SMEM descriptors; only C is a per-thread RMEM fragment.
        thr_mma = tiled_mma.get_slice(0)
        t_s_a_mma = thr_mma.partition_A(s_a)
        t_s_b_mma = thr_mma.partition_B(s_b)
        t_g_c_mma = thr_mma.partition_C(g_c)
        desc_a = tiled_mma.make_fragment_A(t_s_a_mma)
        desc_b = tiled_mma.make_fragment_B(t_s_b_mma)
        accum = cute.make_rmem_tensor(t_g_c_mma.shape, ACC_DTYPE)
        accum.fill(0.0)

        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
        cute.nvgpu.warpgroup.fence()
        cute.gemm(tiled_mma, accum, desc_a, desc_b, accum)
        cute.nvgpu.warpgroup.commit_group()
        cute.nvgpu.warpgroup.wait_group(0)

        store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), ACC_DTYPE)
        cute.copy(store, accum, t_g_c_mma)


@lru_cache(maxsize=1)
def compile_kernel():
    fake_a = make_fake_compact_tensor(
        DTYPE, (M, K), stride_order=(1, 0), assumed_align=16
    )
    fake_b = make_fake_compact_tensor(
        DTYPE, (N, K), stride_order=(1, 0), assumed_align=16
    )
    fake_c = make_fake_compact_tensor(
        ACC_DTYPE, (M, N), stride_order=(1, 0), assumed_align=16
    )
    return cute.compile(
        HopperWgmmaGemm(), fake_a, fake_b, fake_c, options="--enable-tvm-ffi"
    )


def inspect_ptx() -> None:
    ptx = "\n".join(path.read_text() for path in ARTIFACT_DIR.glob("*.ptx"))
    patterns = {
        "TMA load": ("cp.async.bulk.tensor", ".shared::cta.global", "mbarrier"),
        "WGMMA fence": ("wgmma.fence",),
        "WGMMA MMA": ("wgmma.mma_async.sync.aligned.m64n64k16",),
        "WGMMA commit": ("wgmma.commit_group",),
        "WGMMA wait": ("wgmma.wait_group.sync.aligned",),
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
        raise RuntimeError("This example requires an NVIDIA GPU environment")
    compiled = compile_kernel()
    inspect_ptx()
    major, minor = torch.cuda.get_device_capability()
    if major != 9:
        print(
            f"COMPILE-ONLY PASS: built sm_90a WGMMA on device {major}.{minor}; "
            "numerical launch requires H100/H200 (SM90a)"
        )
        return

    torch.manual_seed(19)
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((N, K), device="cuda", dtype=torch.float16)
    c = torch.empty((M, N), device="cuda", dtype=torch.float32)
    compiled(a, b, c)
    torch.cuda.synchronize()
    reference = a.float() @ b.float().T
    torch.testing.assert_close(c, reference, rtol=3e-3, atol=3e-3)
    error = float((c - reference).abs().max())
    print(f"shape=({M},{N},{K}), max_error={error:.3e}: PASS")


if __name__ == "__main__":
    main()
