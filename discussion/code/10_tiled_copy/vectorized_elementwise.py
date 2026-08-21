"""Canonical TiledCopy elementwise add with aligned and residue paths.

Target: any NVIDIA architecture supported by CuTe DSL 4.7.0.
Validated: NVIDIA B200 (SM100), FP32 row-major tensors.

The aligned path uses an explicit 128-bit CopyUniversal atom and accepts only
full (1, 512) CTA tiles.  The general path uses per-value predicates derived
from an identity Tensor; it is correct for arbitrary positive M/N but does not
promise that a partially valid vector remains a single 128-bit instruction.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch


ARTIFACT_DIR = Path(__file__).with_name("generated_artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)
os.environ.setdefault("CUTE_DSL_KEEP", "ptx")
os.environ.setdefault("CUTE_DSL_DUMP_DIR", str(ARTIFACT_DIR))

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


THREADS = 128
VALUES_PER_THREAD = 4
COPY_BITS = 128
TILE_M = 1
TILE_N = THREADS * VALUES_PER_THREAD


def make_tv_layout():
    """Map 128 threads x 4 values onto one contiguous row of 512 FP32s."""
    thr_layout = cute.make_ordered_layout((1, THREADS), order=(1, 0))
    val_layout = cute.make_ordered_layout((1, VALUES_PER_THREAD), order=(1, 0))
    return thr_layout, val_layout


@cute.kernel
def aligned_tiled_copy_kernel(
    g_a: cute.Tensor,
    g_b: cute.Tensor,
    g_c: cute.Tensor,
    thr_layout: cute.Layout,
    val_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    block_coord = ((None, None), bidx)
    block_a = g_a[block_coord]
    block_b = g_b[block_coord]
    block_c = g_c[block_coord]

    load_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        g_a.element_type,
        num_bits_per_copy=COPY_BITS,
    )
    store_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        g_c.element_type,
        num_bits_per_copy=COPY_BITS,
    )
    tiled_load = cute.make_tiled_copy_tv(load_atom, thr_layout, val_layout)
    tiled_store = cute.make_tiled_copy_tv(store_atom, thr_layout, val_layout)

    thread_load = tiled_load.get_slice(tidx)
    thread_store = tiled_store.get_slice(tidx)
    thread_a = thread_load.partition_S(block_a)
    thread_b = thread_load.partition_S(block_b)
    thread_c = thread_store.partition_D(block_c)

    fragment_a = cute.make_fragment_like(thread_a)
    fragment_b = cute.make_fragment_like(thread_b)
    fragment_c = cute.make_fragment_like(thread_c)

    cute.copy(load_atom, thread_a, fragment_a)
    cute.copy(load_atom, thread_b, fragment_b)
    fragment_c.store(fragment_a.load() + fragment_b.load())
    cute.copy(store_atom, fragment_c, thread_c)


@cute.kernel
def masked_tiled_copy_kernel(
    g_a: cute.Tensor,
    g_b: cute.Tensor,
    g_c: cute.Tensor,
    g_coord: cute.Tensor,
    problem_shape: cute.Shape,
    thr_layout: cute.Layout,
    val_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    block_coord = ((None, None), bidx)
    block_a = g_a[block_coord]
    block_b = g_b[block_coord]
    block_c = g_c[block_coord]
    block_coord_tensor = g_coord[block_coord]

    # Leaving num_bits_per_copy unspecified lets the compiler choose a safe
    # width under a per-value predicate.  Interior ownership is still the same
    # 4-value TV Layout as the explicit 128-bit path.
    load_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), g_a.element_type)
    store_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), g_c.element_type)
    tiled_load = cute.make_tiled_copy_tv(load_atom, thr_layout, val_layout)
    tiled_store = cute.make_tiled_copy_tv(store_atom, thr_layout, val_layout)

    thread_load = tiled_load.get_slice(tidx)
    thread_store = tiled_store.get_slice(tidx)
    thread_a = thread_load.partition_S(block_a)
    thread_b = thread_load.partition_S(block_b)
    thread_c = thread_store.partition_D(block_c)
    thread_coord = thread_store.partition_D(block_coord_tensor)

    fragment_a = cute.make_fragment_like(thread_a)
    fragment_b = cute.make_fragment_like(thread_b)
    fragment_c = cute.make_fragment_like(thread_c)
    predicate = cute.make_rmem_tensor(thread_coord.shape, cutlass.Boolean)

    # Invalid load lanes get a neutral value before the predicated copy.  This
    # prevents an undefined register from entering the TensorSSA addition even
    # though the matching store lane will also be masked.
    fragment_a.fill(0.0)
    fragment_b.fill(0.0)
    for i in cutlass.range_constexpr(cute.size(predicate)):
        predicate[i] = cute.elem_less(thread_coord[i], problem_shape)

    cute.copy(load_atom, thread_a, fragment_a, pred=predicate)
    cute.copy(load_atom, thread_b, fragment_b, pred=predicate)
    fragment_c.store(fragment_a.load() + fragment_b.load())
    cute.copy(store_atom, fragment_c, thread_c, pred=predicate)


@cute.jit
def aligned_elementwise(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    # This specialization is deliberately strict: all CTAs own full tiles and
    # each row/tile origin stays 16-byte aligned for FP32 vector operations.
    assert a.shape == b.shape and b.shape == c.shape
    assert a.shape[1] % TILE_N == 0

    thr_layout, val_layout = make_tv_layout()
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    g_a = cute.zipped_divide(a, tiler_mn)
    g_b = cute.zipped_divide(b, tiler_mn)
    g_c = cute.zipped_divide(c, tiler_mn)

    print(f"[aligned] tiler_mn={tiler_mn}")
    print(f"[aligned] tv_layout={tv_layout}")
    aligned_tiled_copy_kernel(g_a, g_b, g_c, thr_layout, val_layout).launch(
        grid=(cute.size(g_c, mode=[1]), 1, 1),
        block=(cute.size(tv_layout, mode=[0]), 1, 1),
    )


@cute.jit
def masked_elementwise(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    problem_shape = c.shape
    thr_layout, val_layout = make_tv_layout()
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    g_a = cute.zipped_divide(a, tiler_mn)
    g_b = cute.zipped_divide(b, tiler_mn)
    g_c = cute.zipped_divide(c, tiler_mn)
    identity = cute.make_identity_tensor(problem_shape)
    g_coord = cute.zipped_divide(identity, tiler_mn)

    masked_tiled_copy_kernel(
        g_a,
        g_b,
        g_c,
        g_coord,
        problem_shape,
        thr_layout,
        val_layout,
    ).launch(
        grid=(cute.size(g_c, mode=[1]), 1, 1),
        block=(cute.size(tv_layout, mode=[0]), 1, 1),
    )


def as_dynamic(tensor: torch.Tensor):
    return from_dlpack(tensor, assumed_align=16).mark_layout_dynamic(leading_dim=1)


def inspect_vector_ptx() -> tuple[int, int, list[str]]:
    """Return explicit vector load/store evidence from generated PTX."""
    ptx_files = sorted(ARTIFACT_DIR.glob("*.ptx"), key=lambda p: p.stat().st_mtime)
    if not ptx_files:
        raise RuntimeError("CUTE_DSL_KEEP=ptx did not produce a PTX artifact")

    lines: list[str] = []
    for path in ptx_files:
        lines.extend(path.read_text().splitlines())
    # PTX may spell one 128-bit FP32 transaction as either v4.b32 or v2.b64.
    # The B200/CuTe DSL 4.7.0 toolchain currently chooses v2.b64.
    vector_loads = [
        line.strip()
        for line in lines
        if "ld.global.v4.b32" in line or "ld.global.v2.b64" in line
    ]
    vector_stores = [
        line.strip()
        for line in lines
        if "st.global.v4.b32" in line or "st.global.v2.b64" in line
    ]
    evidence = (vector_loads[:2] + vector_stores[:1])
    return len(vector_loads), len(vector_stores), evidence


def run_aligned(seed: int) -> None:
    torch.manual_seed(seed)
    shape = (4, 1024)
    a = torch.randn(shape, dtype=torch.float32, device="cuda")
    b = torch.randn_like(a)
    c = torch.empty_like(a)
    args = tuple(from_dlpack(t, assumed_align=16) for t in (a, b, c))

    compiled = cute.compile(aligned_elementwise, *args)
    compiled(*args)
    torch.cuda.synchronize()
    expected = a + b
    torch.testing.assert_close(c, expected, rtol=0, atol=0)

    vector_loads, vector_stores, evidence = inspect_vector_ptx()
    if vector_loads < 2 or vector_stores < 1:
        raise AssertionError(
            "expected at least two 128-bit global loads and one 128-bit global store"
        )
    print(
        f"aligned shape={shape}, max_abs_error={(c - expected).abs().max().item():.1e}, "
        f"ptx_vector_loads={vector_loads}, ptx_vector_stores={vector_stores}"
    )
    for line in evidence:
        print(f"PTX: {line}")


def run_masked(shapes: list[tuple[int, int]], seed: int) -> None:
    torch.manual_seed(seed + 1)
    exemplar = shapes[0]
    exemplar_tensors = [
        torch.empty(exemplar, dtype=torch.float32, device="cuda") for _ in range(3)
    ]
    compiled = cute.compile(masked_elementwise, *map(as_dynamic, exemplar_tensors))

    for m, n in shapes:
        a = torch.randn((m, n), dtype=torch.float32, device="cuda")
        b = torch.randn_like(a)
        c = torch.full_like(a, float("nan"))
        compiled(as_dynamic(a), as_dynamic(b), as_dynamic(c))
        torch.cuda.synchronize()
        expected = a + b
        torch.testing.assert_close(c, expected, rtol=0, atol=0)
        print(
            f"masked shape=({m},{n}), cta_tiles={m * ((n + TILE_N - 1) // TILE_N)}, "
            f"tail_values={n % TILE_N}, max_abs_error={(c - expected).abs().max().item():.1e}"
        )


def parse_shapes(text: str) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for item in text.split(","):
        m_text, n_text = item.lower().split("x", maxsplit=1)
        m, n = int(m_text), int(n_text)
        if m <= 0 or n <= 0:
            raise ValueError("shape extents must be positive")
        result.append((m, n))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CuTe DSL TiledCopy vectorized and masked elementwise add"
    )
    parser.add_argument(
        "--masked-shapes",
        default="1x1,3x511,3x512,5x513,7x1003",
        help="comma-separated dynamic MxN residue cases",
    )
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This example requires an NVIDIA GPU")

    run_aligned(args.seed)
    run_masked(parse_shapes(args.masked_shapes), args.seed)
    print("PASS")


if __name__ == "__main__":
    main()
