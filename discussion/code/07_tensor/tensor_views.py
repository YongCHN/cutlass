"""Exercise Tensor tiling, slicing, domain offsets, and coordinate tensors."""

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


ROWS = 4
COLS = 6
TILE_ROWS = 2
TILE_COLS = 3
THREADS = TILE_ROWS * TILE_COLS


@cute.kernel
def tensor_views_kernel(src: cute.Tensor, dst: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    block_x, block_y, _ = cute.arch.block_idx()

    tile_shape = (TILE_ROWS, TILE_COLS)
    tile_coord = (block_x, block_y)

    # local_tile performs zipped_divide and then selects one coordinate in
    # the remainder modes.  Applying it to the data and identity tensors in
    # parallel keeps their shapes congruent.
    src_tile = cute.local_tile(src, tile_shape, tile_coord)
    dst_tile = cute.local_tile(dst, tile_shape, tile_coord)
    identity = cute.make_identity_tensor(src.shape)
    coord_tile = cute.local_tile(identity, tile_shape, tile_coord)

    local_coord = cute.idx2crd(tidx, tile_shape)
    global_coord = coord_tile[local_coord]
    row = global_coord[0]
    col = global_coord[1]
    dst_tile[local_coord] = src_tile[local_coord] + cutlass.Float32(row * 10 + col)


@cute.jit
def launch(src: cute.Tensor, dst: cute.Tensor):
    tensor_views_kernel(src, dst).launch(
        grid=(ROWS // TILE_ROWS, COLS // TILE_COLS, 1),
        block=(THREADS, 1, 1),
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires an NVIDIA GPU")

    src = torch.arange(ROWS * COLS, dtype=torch.float32, device="cuda").reshape(
        ROWS, COLS
    )
    dst = torch.empty_like(src)

    src_cute = from_dlpack(src)
    dst_cute = from_dlpack(dst)
    compiled = cute.compile(launch, src_cute, dst_cute)
    compiled(src_cute, dst_cute)
    torch.cuda.synchronize()

    row_coord = torch.arange(ROWS, device="cuda", dtype=torch.float32)[:, None]
    col_coord = torch.arange(COLS, device="cuda", dtype=torch.float32)[None, :]
    expected = src + row_coord * 10 + col_coord
    torch.testing.assert_close(dst, expected, rtol=0, atol=0)

    print(f"src=\n{src.cpu()}")
    print(f"dst=\n{dst.cpu()}")
    print("PASS")


if __name__ == "__main__":
    main()
