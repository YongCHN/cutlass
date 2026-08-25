"""A compact SM80 tensor-op GEMM with a two-stage cp.async mainloop.

Target: SM80+ (validated on NVIDIA B200 / SM100)
Input:  A[M,K], B[N,K], FP16 or BF16, contiguous and 16-byte aligned
Output: C[M,N] = A @ B.T, FP32
Shape contract: M/N are multiples of 32; K is a multiple of 64

This is a teaching kernel, not a replacement for CUTLASS's tuned Ampere GEMM.
It deliberately keeps a fixed 32x32x64 CTA tile so the entire data path is
visible: 128-bit cp.async -> double-buffered SMEM -> ldmatrix -> registers ->
eight warp-level mma.sync atoms -> direct FP32 epilogue.
"""

from __future__ import annotations

import argparse
import os
from functools import lru_cache
from pathlib import Path

import torch


ARTIFACT_DIR = Path(__file__).with_name("generated_artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("CUTE_DSL_KEEP", "ptx")
os.environ.setdefault("CUTE_DSL_DUMP_DIR", str(ARTIFACT_DIR))

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor


BM = 32
BN = 32
BK = 64
STAGES = 2
ATOM_LAYOUT_MNK = (2, 4, 1)
THREADS = 32 * ATOM_LAYOUT_MNK[0] * ATOM_LAYOUT_MNK[1]
COPY_BITS = 128
COPY_ELEMS = COPY_BITS // 16


@cute.kernel
def gemm_kernel(
    m_a: cute.Tensor,
    m_b: cute.Tensor,
    m_c: cute.Tensor,
    tiled_copy_a: cute.TiledCopy,
    tiled_copy_b: cute.TiledCopy,
    tiled_mma: cute.TiledMma,
    AB_DTYPE: cutlass.Constexpr[type],
):
    tid, _, _ = cute.arch.thread_idx()
    tile_m, tile_n, _ = cute.arch.block_idx()

    # CuTe uses B[N,K].  This makes both operands K-major in memory and means
    # the public mathematical operation is A @ B.T.
    g_a = cute.local_tile(m_a, (BM, BK), (tile_m, None))
    g_b = cute.local_tile(m_b, (BN, BK), (tile_n, None))
    g_c = cute.local_tile(m_c, (BM, BN), (tile_m, tile_n))

    smem = cutlass.utils.SmemAllocator()
    s_a = smem.allocate_tensor(
        AB_DTYPE,
        cute.make_layout((BM, BK, STAGES), stride=(BK, 1, BM * BK)),
        byte_alignment=16,
    )
    s_b = smem.allocate_tensor(
        AB_DTYPE,
        cute.make_layout((BN, BK, STAGES), stride=(BK, 1, BN * BK)),
        byte_alignment=16,
    )

    thr_copy_a = tiled_copy_a.get_slice(tid)
    thr_copy_b = tiled_copy_b.get_slice(tid)
    t_ag_a = thr_copy_a.partition_S(g_a)
    t_as_a = thr_copy_a.partition_D(s_a)
    t_bg_b = thr_copy_b.partition_S(g_b)
    t_bs_b = thr_copy_b.partition_D(s_b)

    # The fixed shape contract makes every 64-bit copy valid.  Chapter 21
    # removes this contract with identity-tensor predicates and zero filling.
    k_tiles = cute.size(g_a, mode=[2])
    cute.copy(tiled_copy_a, t_ag_a[None, None, None, 0], t_as_a[None, None, None, 0])
    cute.copy(tiled_copy_b, t_bg_b[None, None, None, 0], t_bs_b[None, None, None, 0])
    cute.arch.cp_async_commit_group()
    if k_tiles > 1:
        cute.copy(tiled_copy_a, t_ag_a[None, None, None, 1], t_as_a[None, None, None, 1])
        cute.copy(tiled_copy_b, t_bg_b[None, None, None, 1], t_bs_b[None, None, None, 1])
        cute.arch.cp_async_commit_group()

    thr_mma = tiled_mma.get_slice(tid)
    t_cs_a = thr_mma.partition_A(s_a)
    t_cs_b = thr_mma.partition_B(s_b)
    t_cg_c = thr_mma.partition_C(g_c)
    r_a = tiled_mma.make_fragment_A(t_cs_a[None, None, None, 0])
    r_b = tiled_mma.make_fragment_B(t_cs_b[None, None, None, 0])
    r_c = tiled_mma.make_fragment_C(t_cg_c)
    r_c.fill(0.0)

    ld_atom = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), AB_DTYPE
    )
    copy_s2r_a = cute.make_tiled_copy_A(ld_atom, tiled_mma)
    copy_s2r_b = cute.make_tiled_copy_B(ld_atom, tiled_mma)
    thr_s2r_a = copy_s2r_a.get_slice(tid)
    thr_s2r_b = copy_s2r_b.get_slice(tid)
    t_cs_a_copy = thr_s2r_a.partition_S(s_a)
    t_cs_b_copy = thr_s2r_b.partition_S(s_b)
    t_cr_a_copy = thr_s2r_a.retile(r_a)
    t_cr_b_copy = thr_s2r_b.retile(r_b)

    # At most one newer group may remain outstanding.  The wait plus CTA
    # barrier publishes the current stage before any warp issues ldmatrix.
    read_stage = cutlass.Int32(0)
    for k_tile in range(k_tiles):
        if k_tile + 1 < k_tiles:
            cute.arch.cp_async_wait_group(1)
        else:
            cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        for k_block in cutlass.range_constexpr(BK // 16):
            cute.copy(
                copy_s2r_a,
                t_cs_a_copy[None, None, k_block, read_stage],
                t_cr_a_copy[None, None, k_block],
            )
            cute.copy(
                copy_s2r_b,
                t_cs_b_copy[None, None, k_block, read_stage],
                t_cr_b_copy[None, None, k_block],
            )
            cute.gemm(
                tiled_mma,
                r_c,
                r_a[None, None, k_block],
                r_b[None, None, k_block],
                r_c,
            )

        # No thread may overwrite this stage until every warp has finished
        # reading it.  Reuse the consumed stage for tile k+2.
        cute.arch.sync_threads()
        if k_tile + STAGES < k_tiles:
            cute.copy(
                tiled_copy_a,
                t_ag_a[None, None, None, k_tile + STAGES],
                t_as_a[None, None, None, read_stage],
            )
            cute.copy(
                tiled_copy_b,
                t_bg_b[None, None, None, k_tile + STAGES],
                t_bs_b[None, None, None, read_stage],
            )
            cute.arch.cp_async_commit_group()
        read_stage = (read_stage + 1) % STAGES

    store_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), m_c.element_type)
    cute.copy(store_atom, r_c, t_cg_c)


@cute.jit
def launch(
    m_a: cute.Tensor,
    m_b: cute.Tensor,
    m_c: cute.Tensor,
    AB_DTYPE: cutlass.Constexpr[type],
):
    cp_atom = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL),
        AB_DTYPE,
        num_bits_per_copy=COPY_BITS,
    )
    # 32 rows x 8 K-groups, one four-element value vector per thread.
    copy_thr = cute.make_layout((BM, BK // COPY_ELEMS), stride=(BK // COPY_ELEMS, 1))
    copy_val = cute.make_layout((1, COPY_ELEMS))
    tiled_copy_a = cute.make_tiled_copy_tv(cp_atom, copy_thr, copy_val)
    tiled_copy_b = cute.make_tiled_copy_tv(cp_atom, copy_thr, copy_val)

    mma_op = cute.nvgpu.warp.MmaF16BF16Op(
        AB_DTYPE, cutlass.Float32, (16, 8, 16)
    )
    tiled_mma = cute.make_tiled_mma(
        mma_op,
        cute.make_layout(ATOM_LAYOUT_MNK),
        permutation_mnk=(BM, BN, 16),
    )
    gemm_kernel(
        m_a, m_b, m_c, tiled_copy_a, tiled_copy_b, tiled_mma, AB_DTYPE
    ).launch(
        grid=(cute.size(m_a, mode=[0]) // BM, cute.size(m_b, mode=[0]) // BN, 1),
        block=(THREADS, 1, 1),
    )


@lru_cache(maxsize=None)
def compile_kernel(ab_dtype: type, m: int, n: int, k: int):
    fake_a = make_fake_compact_tensor(
        ab_dtype, (m, k), stride_order=(1, 0), assumed_align=16
    )
    fake_b = make_fake_compact_tensor(
        ab_dtype, (n, k), stride_order=(1, 0), assumed_align=16
    )
    fake_c = make_fake_compact_tensor(
        cutlass.Float32, (m, n), stride_order=(1, 0), assumed_align=16
    )
    return cute.compile(
        launch, fake_a, fake_b, fake_c, ab_dtype, options="--enable-tvm-ffi"
    )


def validate_shape(m: int, n: int, k: int) -> None:
    for label, value, quantum in (("M", m, BM), ("N", n, BN), ("K", k, BK)):
        if value <= 0 or value % quantum:
            raise ValueError(
                f"{label} must be a positive multiple of {quantum}, got {value}"
            )


def inspect_ptx(dtype_name: str) -> None:
    ptx = "\n".join(path.read_text() for path in ARTIFACT_DIR.glob("*.ptx"))
    mma_suffix = "bf16.bf16.f32" if dtype_name == "bf16" else "f16.f16.f32"
    needles = {
        "cp.async": "cp.async.cg.shared.global",
        "ldmatrix": "ldmatrix.sync.aligned",
        "mma": f"mma.sync.aligned.m16n8k16.row.col.f32.{mma_suffix}",
        "cta barrier": "bar.sync",
    }
    for label, needle in needles.items():
        matches = [line.strip() for line in ptx.splitlines() if needle in line]
        if not matches:
            raise AssertionError(f"missing {label} PTX containing {needle!r}")
        print(f"PTX {label}: {matches[0]}")


def run_case(
    m: int,
    n: int,
    k: int,
    ab_dtype: type,
    torch_dtype: torch.dtype,
    name: str,
) -> None:
    validate_shape(m, n, k)
    torch.manual_seed(m * 10000 + n * 100 + k)
    a = torch.randn((m, k), device="cuda", dtype=torch_dtype)
    b = torch.randn((n, k), device="cuda", dtype=torch_dtype)
    c = torch.empty((m, n), device="cuda", dtype=torch.float32)
    compile_kernel(ab_dtype, m, n, k)(a, b, c)
    torch.cuda.synchronize()
    reference = a.float() @ b.float().T
    torch.testing.assert_close(c, reference, rtol=3e-3, atol=3e-3)
    max_error = float((c - reference).abs().max())
    tiles = (m // BM, n // BN, k // BK)
    print(
        f"dtype={name}, shape=({m},{n},{k}), CTA tiles={tiles}, "
        f"max_error={max_error:.3e}: PASS"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "all"), default="all")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires an NVIDIA GPU")
    major, minor = torch.cuda.get_device_capability()
    if major < 8:
        raise RuntimeError(f"this kernel requires SM80+, got {major}.{minor}")

    cases = {
        "fp16": (cutlass.Float16, torch.float16),
        "bf16": (cutlass.BFloat16, torch.bfloat16),
    }
    selected = cases if args.dtype == "all" else {args.dtype: cases[args.dtype]}
    shapes = (
        ((args.m, args.n, args.k),)
        if args.m is not None and args.n is not None and args.k is not None
        else ((32, 32, 64), (64, 96, 128), (96, 64, 192))
    )
    for name, (cute_dtype, torch_dtype) in selected.items():
        for shape in shapes:
            run_case(*shape, cute_dtype, torch_dtype, name)
        inspect_ptx(name)
    print("PASS")


if __name__ == "__main__":
    main()
