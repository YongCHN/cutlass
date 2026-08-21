"""Boundary-safe 2-D vector add using a congruent identity Tensor.

Target: any NVIDIA architecture supported by CuTe DSL 4.7.0.
Validated: NVIDIA B200 (SM100), FP32 row-major tensors.

One CTA owns a logical (1, 128) tile.  The data and coordinate tensors are
tiled identically; the coordinate tile supplies the exact global predicate for
residue columns.  One compiled handle is reused for several dynamic shapes.
"""

from __future__ import annotations

import argparse

import torch

import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


TILE_M = 1
TILE_N = 128
THREADS = TILE_N


@cute.kernel
def masked_vector_add_kernel(
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    problem_shape: cute.Shape,
):
    tidx, _, _ = cute.arch.thread_idx()
    block_x, block_y, _ = cute.arch.block_idx()

    tile_shape = (TILE_M, TILE_N)
    tile_coord = (block_y, block_x)

    a_tile = cute.local_tile(a, tile_shape, tile_coord)
    b_tile = cute.local_tile(b, tile_shape, tile_coord)
    c_tile = cute.local_tile(c, tile_shape, tile_coord)

    identity = cute.make_identity_tensor(problem_shape)
    coord_tile = cute.local_tile(identity, tile_shape, tile_coord)

    # TILE_M is one, so adjacent thread IDs map to adjacent row-major columns.
    # This is still a scalar copy; Chapter 10 will give each thread multiple
    # values through a TV Layout and a TiledCopy.
    local_coord = (0, tidx)
    global_coord = coord_tile[local_coord]
    if cute.elem_less(global_coord, problem_shape):
        c_tile[local_coord] = a_tile[local_coord] + b_tile[local_coord]


@cute.jit
def masked_vector_add(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    problem_shape = c.shape
    tiles_m = cute.ceil_div(problem_shape[0], TILE_M)
    tiles_n = cute.ceil_div(problem_shape[1], TILE_N)

    masked_vector_add_kernel(a, b, c, problem_shape).launch(
        # CUDA x traverses column tiles; y traverses rows.
        grid=(tiles_n, tiles_m, 1),
        block=(THREADS, 1, 1),
    )


def as_dynamic(tensor: torch.Tensor):
    """Keep the row-major unit-stride mode while making shape/LD dynamic."""
    return from_dlpack(tensor).mark_layout_dynamic(leading_dim=1)


def reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + b


def parse_shapes(text: str) -> list[tuple[int, int]]:
    shapes: list[tuple[int, int]] = []
    for item in text.split(","):
        m_text, n_text = item.lower().split("x", maxsplit=1)
        m, n = int(m_text), int(n_text)
        if m <= 0 or n <= 0:
            raise ValueError("all shape extents must be positive")
        shapes.append((m, n))
    return shapes


def run(shapes: list[tuple[int, int]], seed: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires an NVIDIA GPU")

    torch.manual_seed(seed)
    exemplar_shape = shapes[0]
    exemplar_a = torch.empty(exemplar_shape, dtype=torch.float32, device="cuda")
    exemplar_b = torch.empty_like(exemplar_a)
    exemplar_c = torch.empty_like(exemplar_a)
    compiled = cute.compile(
        masked_vector_add,
        as_dynamic(exemplar_a),
        as_dynamic(exemplar_b),
        as_dynamic(exemplar_c),
    )

    for m, n in shapes:
        a = torch.randn((m, n), dtype=torch.float32, device="cuda")
        b = torch.randn_like(a)
        # NaN makes a missed valid store immediately visible to verification.
        c = torch.full_like(a, float("nan"))
        args = (as_dynamic(a), as_dynamic(b), as_dynamic(c))
        compiled(*args)
        torch.cuda.synchronize()

        expected = reference(a, b)
        torch.testing.assert_close(c, expected, rtol=0, atol=0)
        tail = n % TILE_N
        print(
            f"shape=({m},{n}), grid=({(n + TILE_N - 1) // TILE_N},{m}), "
            f"tail_columns={tail}, max_abs_error={(c - expected).abs().max().item():.1e}"
        )

    print("PASS")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Boundary-safe CuTe DSL vector add with identity Tensor predicates"
    )
    parser.add_argument(
        "--shapes",
        default="1x1,3x127,3x128,5x129,7x1003",
        help="comma-separated MxN cases; one compiled handle is reused",
    )
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    run(parse_shapes(args.shapes), args.seed)


if __name__ == "__main__":
    main()
