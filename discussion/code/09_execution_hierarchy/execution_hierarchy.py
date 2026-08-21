"""Map CUDA's 3-D execution hierarchy to stable linear participant IDs.

Target: any NVIDIA architecture supported by CuTe DSL 4.7.0.
Validated: NVIDIA B200 (SM100), one 2-D grid of 2-D CTAs.

The kernel writes one row per thread so the host can verify thread, lane, warp,
warpgroup, CTA, and grid flattening without relying on device printf ordering.
"""

from __future__ import annotations

import torch

import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


GRID = (2, 2, 1)
BLOCK = (32, 8, 1)

# Column meanings in the output table.  Keeping the schema in one place makes
# the CPU reference and the printed sample unambiguous.
TX = 0
TY = 1
BX = 2
BY = 3
LANE = 4
WARP = 5
WARPGROUP = 6
THREAD_LINEAR = 7
BLOCK_LINEAR = 8
GLOBAL_LINEAR = 9
NUM_FIELDS = 10


@cute.kernel
def hierarchy_kernel(out: cute.Tensor):
    tx, ty, tz = cute.arch.thread_idx()
    bx, by, bz = cute.arch.block_idx()
    block_x, block_y, _ = cute.arch.block_dim()
    grid_x, grid_y, _ = cute.arch.grid_dim()

    # CUDA flattens x first, then y, then z.  The logical warp index returned
    # by warp_idx follows this stable CTA-local flattening.
    thread_linear = tx + ty * block_x + tz * block_x * block_y
    block_linear = bx + by * grid_x + bz * grid_x * grid_y
    threads_per_cta = block_x * block_y
    global_linear = block_linear * threads_per_cta + thread_linear

    lane = cute.arch.lane_idx()
    warp = cute.arch.warp_idx()
    warpgroup = warp // 4

    out[global_linear, TX] = tx
    out[global_linear, TY] = ty
    out[global_linear, BX] = bx
    out[global_linear, BY] = by
    out[global_linear, LANE] = lane
    out[global_linear, WARP] = warp
    out[global_linear, WARPGROUP] = warpgroup
    out[global_linear, THREAD_LINEAR] = thread_linear
    out[global_linear, BLOCK_LINEAR] = block_linear
    out[global_linear, GLOBAL_LINEAR] = global_linear


@cute.jit
def launch(out: cute.Tensor):
    hierarchy_kernel(out).launch(grid=GRID, block=BLOCK)


def reference() -> torch.Tensor:
    """Build the execution mapping independently on the CPU."""
    rows: list[list[int]] = []
    for by in range(GRID[1]):
        for bx in range(GRID[0]):
            block_linear = bx + by * GRID[0]
            for ty in range(BLOCK[1]):
                for tx in range(BLOCK[0]):
                    thread_linear = tx + ty * BLOCK[0]
                    global_linear = block_linear * (BLOCK[0] * BLOCK[1]) + thread_linear
                    rows.append(
                        [
                            tx,
                            ty,
                            bx,
                            by,
                            thread_linear % 32,
                            thread_linear // 32,
                            (thread_linear // 32) // 4,
                            thread_linear,
                            block_linear,
                            global_linear,
                        ]
                    )
    return torch.tensor(rows, dtype=torch.int32)


def run() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires an NVIDIA GPU")

    total_threads = GRID[0] * GRID[1] * BLOCK[0] * BLOCK[1]
    out = torch.empty((total_threads, NUM_FIELDS), dtype=torch.int32, device="cuda")
    out_cute = from_dlpack(out)

    compiled = cute.compile(launch, out_cute)
    compiled(out_cute)
    torch.cuda.synchronize()

    expected = reference()
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)

    # These rows cross a lane, warp, warpgroup, and CTA boundary.
    sample_rows = (0, 31, 32, 127, 128, 255, 256, 1023)
    print("columns=tx,ty,bx,by,lane,warp,warpgroup,thread_linear,block_linear,global_linear")
    for row in sample_rows:
        print(f"row {row}: {out[row].cpu().tolist()}")
    print("PASS")


if __name__ == "__main__":
    run()
