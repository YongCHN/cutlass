"""Shared-memory tiled transpose with compact and padded layouts.

Target: any architecture supported by the synchronous CuTe DSL path.
Validation target: NVIDIA B200 (SM100), CuTe DSL 4.7.0.

The kernel deliberately processes two input tiles per CTA so that the same
shared-memory buffer is reused.  Consequently both barriers are semantically
necessary: the first publishes the newly loaded tile, and the second prevents
the next producer round from overwriting data that consumers still read.
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


TILE = 32
BLOCK_Y = 8
THREADS = TILE * BLOCK_Y
TILES_PER_CTA = 2


@cute.kernel
def transpose_kernel(
    src: cute.Tensor,
    dst: cute.Tensor,
    launch_smem_bytes: cute.Tensor,
    PAD: cutlass.Constexpr,
):
    """Transpose two 32x32 tiles while reusing one SMEM allocation."""
    tx, ty, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()

    # Row-major logical layout.  PAD=0 gives stride 32; PAD=1 gives stride 33.
    # The logical data tile is still 32x32.  The extra column only changes the
    # physical stride used by the transposed (column-wise) read.
    smem_layout = cute.make_ordered_layout((TILE, TILE + PAD), order=(1, 0))
    allocator = cutlass.utils.SmemAllocator()
    smem = allocator.allocate_tensor(
        cutlass.Float32,
        smem_layout,
        byte_alignment=16,
        swizzle=None,
    )

    if bx == 0 and by == 0 and tx == 0 and ty == 0:
        # This is the launch-time dynamic-SMEM region size selected after the
        # allocator pass, not a runtime-sized allocation request.
        launch_smem_bytes[0] = cute.arch.dynamic_smem_size()

    problem_shape = src.shape

    # Each CTA owns two consecutive row tiles and one column tile.  A static
    # two-round loop makes buffer reuse visible without introducing pipelines.
    for tile_round in cutlass.range_constexpr(TILES_PER_CTA):
        tile_row = by * TILES_PER_CTA + tile_round

        # 256 threads load 4 values each: (ty + j, tx), j={0,8,16,24}.
        # Invalid residue coordinates write a neutral value, so every thread
        # still reaches the CTA-wide barriers in uniform control flow.
        for j in cutlass.range_constexpr(0, TILE, BLOCK_Y):
            local_row = ty + j
            global_row = tile_row * TILE + local_row
            global_col = bx * TILE + tx
            if cute.elem_less((global_row, global_col), problem_shape):
                smem[local_row, tx] = src[global_row, global_col]
            else:
                smem[local_row, tx] = cutlass.Float32(0.0)

        # Producer completion: all SMEM writes are visible before any thread
        # starts reading the tile through the transposed coordinate system.
        cute.arch.sync_threads()

        # Across one warp, tx varies from 0..31.  The access smem[tx, k]
        # therefore has a 32-word stride when PAD=0 and a 33-word stride when
        # PAD=1.  This is the classic bank-conflict comparison.
        for j in cutlass.range_constexpr(0, TILE, BLOCK_Y):
            output_row = bx * TILE + ty + j
            output_col = tile_row * TILE + tx
            if cute.elem_less((output_row, output_col), dst.shape):
                value = smem[tx, ty + j]
                dst[output_row, output_col] = value

        # Consumer completion: without this barrier, fast producer lanes from
        # the next round could overwrite SMEM while slower lanes still read it.
        cute.arch.sync_threads()


@cute.jit
def transpose(
    src: cute.Tensor,
    dst: cute.Tensor,
    launch_smem_bytes: cute.Tensor,
    PAD: cutlass.Constexpr,
):
    # src/dst transposed-shape compatibility is a runtime host contract.  The
    # extents are staged values here and therefore cannot appear in a Python
    # assert; run_variant constructs the matching destination explicitly.
    tiles_m = cute.ceil_div(src.shape[0], TILE)
    tiles_n = cute.ceil_div(src.shape[1], TILE)
    transpose_kernel(src, dst, launch_smem_bytes, PAD).launch(
        grid=(tiles_n, cute.ceil_div(tiles_m, TILES_PER_CTA), 1),
        block=(TILE, BLOCK_Y, 1),
        # Omitted smem= means SmemAllocator usage is inferred automatically.
    )


def as_dynamic(tensor: torch.Tensor):
    """Preserve the row-major unit-stride mode while making shape/LD dynamic."""
    return from_dlpack(tensor, assumed_align=16).mark_layout_dynamic(leading_dim=1)


def compile_variant(pad: int, exemplar: tuple[int, int]):
    m, n = exemplar
    src = torch.empty((m, n), dtype=torch.float32, device="cuda")
    dst = torch.empty((n, m), dtype=torch.float32, device="cuda")
    meta = torch.empty((1,), dtype=torch.int32, device="cuda")
    return cute.compile(
        transpose,
        as_dynamic(src),
        as_dynamic(dst),
        from_dlpack(meta),
        pad,
    )


def inspect_ptx() -> tuple[int, int, int, list[str]]:
    """Collect SMEM load/store and CTA-barrier evidence from retained PTX."""
    ptx_files = sorted(ARTIFACT_DIR.glob("*.ptx"), key=lambda p: p.stat().st_mtime)
    if not ptx_files:
        raise RuntimeError("CUTE_DSL_KEEP=ptx did not produce a PTX artifact")

    lines: list[str] = []
    for path in ptx_files:
        lines.extend(path.read_text().splitlines())
    loads = [line.strip() for line in lines if "ld.shared" in line]
    stores = [line.strip() for line in lines if "st.shared" in line]
    barriers = [
        line.strip()
        for line in lines
        if "bar.sync" in line or "barrier.cta.sync" in line
    ]
    evidence = loads[:1] + stores[:1] + barriers[:2]
    return len(loads), len(stores), len(barriers), evidence


def parse_shapes(text: str) -> list[tuple[int, int]]:
    shapes: list[tuple[int, int]] = []
    for item in text.split(","):
        m_text, n_text = item.lower().split("x", maxsplit=1)
        m, n = int(m_text), int(n_text)
        if m <= 0 or n <= 0:
            raise ValueError("all shape extents must be positive")
        shapes.append((m, n))
    return shapes


def run_variant(
    pad: int,
    shapes: list[tuple[int, int]],
    seed: int,
) -> None:
    compiled = compile_variant(pad, shapes[0])
    expected_bytes = TILE * (TILE + pad) * 4

    for case, (m, n) in enumerate(shapes):
        torch.manual_seed(seed + case)
        src = torch.randn((m, n), dtype=torch.float32, device="cuda")
        dst = torch.full((n, m), float("nan"), dtype=torch.float32, device="cuda")
        meta = torch.full((1,), -1, dtype=torch.int32, device="cuda")
        compiled(as_dynamic(src), as_dynamic(dst), from_dlpack(meta))
        torch.cuda.synchronize()

        expected = src.transpose(0, 1).contiguous()
        torch.testing.assert_close(dst, expected, rtol=0, atol=0)
        actual_bytes = int(meta.item())
        if actual_bytes != expected_bytes:
            raise AssertionError(
                f"PAD={pad}: expected {expected_bytes} SMEM bytes, got {actual_bytes}"
            )
        print(
            f"pad={pad}, shape=({m},{n}), smem_bytes={actual_bytes}, "
            f"grid=({(n + TILE - 1) // TILE},"
            f"{((m + TILE - 1) // TILE + TILES_PER_CTA - 1) // TILES_PER_CTA}), "
            f"max_abs_error={(dst - expected).abs().max().item():.1e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CuTe DSL shared-memory tiled transpose"
    )
    parser.add_argument(
        "--shapes",
        default="1x1,31x33,32x32,65x67,97x129",
        help="comma-separated dynamic MxN cases",
    )
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This example requires an NVIDIA GPU")

    shapes = parse_shapes(args.shapes)
    for pad in (0, 1):
        run_variant(pad, shapes, args.seed)

    loads, stores, barriers, evidence = inspect_ptx()
    if loads == 0 or stores == 0 or barriers < 2:
        raise AssertionError("expected shared loads/stores and at least two CTA barriers")
    print(f"PTX shared_loads={loads}, shared_stores={stores}, barriers={barriers}")
    for line in evidence:
        print(f"PTX: {line}")
    print("PASS")


if __name__ == "__main__":
    main()
