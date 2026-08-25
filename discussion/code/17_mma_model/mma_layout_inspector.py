"""Inspect one SM80 warp MMA and prove its thread/value ownership.

Target: SM80+ (validated on NVIDIA B200 / SM100)
Input:  one-warp or 2x4-warp A/B tiles, FP16 or BF16, 16-byte aligned
Output: C = A @ B.T in FP32

Besides executing one ``mma.sync.m16n8k16`` instruction, the kernel exports
the A/B/C logical coordinates seen by every lane after ``partition_A/B/C``.
The host checks fragment sizes, coordinate coverage, C non-overlap, numerical
correctness, and retained PTX.
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


WARP_SIZE = 32
INST_M, INST_N, INST_K = 16, 8, 16
A_VALUES_PER_LANE = 8
B_VALUES_PER_LANE = 4
C_VALUES_PER_LANE = 4


@cute.kernel
def inspect_kernel(
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    map_a: cute.Tensor,
    map_b: cute.Tensor,
    map_c: cute.Tensor,
    tiled_mma: cute.TiledMma,
    AB_DTYPE: cutlass.Constexpr[type],
    ATOM_M: cutlass.Constexpr[int],
    ATOM_N: cutlass.Constexpr[int],
):
    lane, _, _ = cute.arch.thread_idx()
    tile_m = INST_M * ATOM_M
    tile_n = INST_N * ATOM_N
    thread_count = WARP_SIZE * ATOM_M * ATOM_N

    # Ordinary row-major SMEM tensors.  For B, CuTe's logical operand shape is
    # (N,K), so the mathematical GEMM is A @ B.T.
    smem = cutlass.utils.SmemAllocator()
    s_a = smem.allocate_tensor(
        AB_DTYPE,
        cute.make_layout((tile_m, INST_K), stride=(INST_K, 1)),
        byte_alignment=16,
    )
    s_b = smem.allocate_tensor(
        AB_DTYPE,
        cute.make_layout((tile_n, INST_K), stride=(INST_K, 1)),
        byte_alignment=16,
    )
    for i in cutlass.range_constexpr((tile_m * INST_K) // thread_count):
        flat = lane + i * thread_count
        s_a[flat // INST_K, flat % INST_K] = a[flat // INST_K, flat % INST_K]
    for i in cutlass.range_constexpr((tile_n * INST_K) // thread_count):
        flat = lane + i * thread_count
        s_b[flat // INST_K, flat % INST_K] = b[flat // INST_K, flat % INST_K]
    cute.arch.sync_threads()

    thr_mma = tiled_mma.get_slice(lane)
    t_s_a = thr_mma.partition_A(s_a)
    t_s_b = thr_mma.partition_B(s_b)
    t_g_c = thr_mma.partition_C(c)

    # Partitioning an identity tensor produces the same thread/value shape,
    # but its values are logical coordinates rather than matrix elements.
    t_c_a = thr_mma.partition_A(cute.make_identity_tensor((tile_m, INST_K)))
    t_c_b = thr_mma.partition_B(cute.make_identity_tensor((tile_n, INST_K)))
    t_c_c = thr_mma.partition_C(cute.make_identity_tensor((tile_m, tile_n)))
    for i in cutlass.range_constexpr(A_VALUES_PER_LANE):
        map_a[lane, i, 0] = t_c_a[i][0]
        map_a[lane, i, 1] = t_c_a[i][1]
    for i in cutlass.range_constexpr(B_VALUES_PER_LANE):
        map_b[lane, i, 0] = t_c_b[i][0]
        map_b[lane, i, 1] = t_c_b[i][1]
    for i in cutlass.range_constexpr(C_VALUES_PER_LANE):
        map_c[lane, i, 0] = t_c_c[i][0]
        map_c[lane, i, 1] = t_c_c[i][1]

    # ldmatrix is not inferred from partition alone.  The copy atom adapts the
    # instruction's lane/value layout to the fragment layout required by MMA.
    ld_atom = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), AB_DTYPE
    )
    copy_a = cute.make_tiled_copy_A(ld_atom, tiled_mma)
    copy_b = cute.make_tiled_copy_B(ld_atom, tiled_mma)
    thr_copy_a = copy_a.get_slice(lane)
    thr_copy_b = copy_b.get_slice(lane)

    r_a = tiled_mma.make_fragment_A(t_s_a)
    r_b = tiled_mma.make_fragment_B(t_s_b)
    r_c = tiled_mma.make_fragment_C(t_g_c)
    r_c.fill(0.0)
    cute.copy(copy_a, thr_copy_a.partition_S(s_a), thr_copy_a.retile(r_a))
    cute.copy(copy_b, thr_copy_b.partition_S(s_b), thr_copy_b.retile(r_b))
    cute.gemm(tiled_mma, r_c, r_a, r_b, r_c)

    # The warp MMA accumulator has one disjoint four-value fragment per lane.
    store_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), c.element_type)
    cute.copy(store_atom, r_c, t_g_c)


@cute.jit
def launch(
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    map_a: cute.Tensor,
    map_b: cute.Tensor,
    map_c: cute.Tensor,
    AB_DTYPE: cutlass.Constexpr[type],
    ATOM_M: cutlass.Constexpr[int],
    ATOM_N: cutlass.Constexpr[int],
):
    tile_m = INST_M * ATOM_M
    tile_n = INST_N * ATOM_N
    op = cute.nvgpu.warp.MmaF16BF16Op(
        AB_DTYPE, cutlass.Float32, (INST_M, INST_N, INST_K)
    )
    tiled_mma = cute.make_tiled_mma(
        op, cute.make_layout((ATOM_M, ATOM_N, 1))
    )

    # These are compile-time objects.  Printing them during tracing is a useful
    # inspector in its own right; the coordinate tensors below make the same
    # mapping machine-checkable at runtime.
    print(tiled_mma)
    print("partition A shape:", tiled_mma.partition_shape_A((tile_m, INST_K)))
    print("partition B shape:", tiled_mma.partition_shape_B((tile_n, INST_K)))
    print("partition C shape:", tiled_mma.partition_shape_C((tile_m, tile_n)))

    inspect_kernel(
        a,
        b,
        c,
        map_a,
        map_b,
        map_c,
        tiled_mma,
        AB_DTYPE,
        ATOM_M,
        ATOM_N,
    ).launch(
        grid=(1, 1, 1), block=(WARP_SIZE * ATOM_M * ATOM_N, 1, 1)
    )


@lru_cache(maxsize=None)
def compile_kernel(ab_dtype: type, atom_m: int, atom_n: int):
    tile_m = INST_M * atom_m
    tile_n = INST_N * atom_n
    threads = WARP_SIZE * atom_m * atom_n
    fake_a = make_fake_compact_tensor(
        ab_dtype, (tile_m, INST_K), stride_order=(1, 0), assumed_align=16
    )
    fake_b = make_fake_compact_tensor(
        ab_dtype, (tile_n, INST_K), stride_order=(1, 0), assumed_align=16
    )
    fake_c = make_fake_compact_tensor(
        cutlass.Float32, (tile_m, tile_n), stride_order=(1, 0), assumed_align=16
    )
    fake_map_a = make_fake_compact_tensor(
        cutlass.Int32, (threads, A_VALUES_PER_LANE, 2), stride_order=(2, 1, 0)
    )
    fake_map_b = make_fake_compact_tensor(
        cutlass.Int32, (threads, B_VALUES_PER_LANE, 2), stride_order=(2, 1, 0)
    )
    fake_map_c = make_fake_compact_tensor(
        cutlass.Int32, (threads, C_VALUES_PER_LANE, 2), stride_order=(2, 1, 0)
    )
    return cute.compile(
        launch,
        fake_a,
        fake_b,
        fake_c,
        fake_map_a,
        fake_map_b,
        fake_map_c,
        ab_dtype,
        atom_m,
        atom_n,
        options="--enable-tvm-ffi",
    )


def coordinates(tensor: torch.Tensor) -> list[tuple[int, int]]:
    return [tuple(int(x) for x in coord) for coord in tensor.reshape(-1, 2).cpu()]


def verify_partition(
    name: str,
    mapping: torch.Tensor,
    expected_domain: set[tuple[int, int]],
    expected_multiplicity: int,
) -> None:
    coords = coordinates(mapping)
    actual = set(coords)
    if actual != expected_domain:
        raise AssertionError(f"{name}: coordinate coverage mismatch")
    counts = {coord: coords.count(coord) for coord in actual}
    if set(counts.values()) != {expected_multiplicity}:
        raise AssertionError(
            f"{name}: expected multiplicity {expected_multiplicity}, got "
            f"{sorted(set(counts.values()))}"
        )
    print(
        f"{name}: threads={mapping.shape[0]}, values/thread={mapping.shape[1]}, "
        f"domain={len(actual)}, multiplicity={expected_multiplicity}: PASS"
    )


def inspect_ptx(dtype_name: str) -> None:
    ptx = "\n".join(path.read_text() for path in ARTIFACT_DIR.glob("*.ptx"))
    mma_suffix = "bf16.bf16.f32" if dtype_name == "bf16" else "f16.f16.f32"
    required = {
        "ldmatrix": "ldmatrix.sync.aligned",
        "mma": f"mma.sync.aligned.m16n8k16.row.col.f32.{mma_suffix}",
    }
    for label, needle in required.items():
        matches = [line.strip() for line in ptx.splitlines() if needle in line]
        if not matches:
            raise AssertionError(f"missing {label} PTX containing {needle!r}")
        print(f"PTX {label}: {matches[0]}")


def run_case(
    ab_dtype: type,
    torch_dtype: torch.dtype,
    name: str,
    atom_m: int,
    atom_n: int,
) -> None:
    torch.manual_seed(17)
    tile_m = INST_M * atom_m
    tile_n = INST_N * atom_n
    threads = WARP_SIZE * atom_m * atom_n
    a = torch.randn((tile_m, INST_K), device="cuda", dtype=torch_dtype)
    b = torch.randn((tile_n, INST_K), device="cuda", dtype=torch_dtype)
    c = torch.empty((tile_m, tile_n), device="cuda", dtype=torch.float32)
    map_a = torch.full(
        (threads, A_VALUES_PER_LANE, 2), -1, device="cuda", dtype=torch.int32
    )
    map_b = torch.full(
        (threads, B_VALUES_PER_LANE, 2), -1, device="cuda", dtype=torch.int32
    )
    map_c = torch.full(
        (threads, C_VALUES_PER_LANE, 2), -1, device="cuda", dtype=torch.int32
    )

    compile_kernel(ab_dtype, atom_m, atom_n)(a, b, c, map_a, map_b, map_c)
    torch.cuda.synchronize()
    reference = a.float() @ b.float().T
    torch.testing.assert_close(c, reference, rtol=2e-3, atol=2e-3)
    max_error = float((c - reference).abs().max())
    print(
        f"dtype={name}, atom_layout=({atom_m},{atom_n},1), "
        f"shape=({tile_m},{tile_n},{INST_K}), max_error={max_error:.3e}: PASS"
    )

    verify_partition(
        "A",
        map_a,
        {(m, k) for m in range(tile_m) for k in range(INST_K)},
        expected_multiplicity=atom_n,
    )
    verify_partition(
        "B",
        map_b,
        {(n, k) for n in range(tile_n) for k in range(INST_K)},
        expected_multiplicity=atom_m,
    )
    verify_partition(
        "C",
        map_c,
        {(m, n) for m in range(tile_m) for n in range(tile_n)},
        expected_multiplicity=1,
    )
    inspect_ptx(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "all"), default="all")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires an NVIDIA GPU")
    major, minor = torch.cuda.get_device_capability()
    if major < 8:
        raise RuntimeError(f"MmaF16BF16Op requires SM80+, got {major}.{minor}")

    cases = {
        "fp16": (cutlass.Float16, torch.float16),
        "bf16": (cutlass.BFloat16, torch.bfloat16),
    }
    selected = cases if args.dtype == "all" else {args.dtype: cases[args.dtype]}
    for name, (cute_dtype, torch_dtype) in selected.items():
        for atom_m, atom_n in ((1, 1), (2, 4)):
            run_case(cute_dtype, torch_dtype, name, atom_m, atom_n)
    print("PASS")


if __name__ == "__main__":
    main()
