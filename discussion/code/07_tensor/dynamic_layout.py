"""Compile once for dynamic 1-D layouts and run several shapes/strides."""

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


THREADS = 128


@cute.kernel
def scale_kernel(src: cute.Tensor, dst: cute.Tensor, n: cutlass.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    idx = bidx * THREADS + tidx
    if idx < n:
        dst[idx] = src[idx] * cutlass.Float32(2.0)


@cute.jit
def scale(src: cute.Tensor, dst: cute.Tensor):
    n = cute.size(src)
    scale_kernel(src, dst, n).launch(
        grid=(cute.ceil_div(n, THREADS), 1, 1),
        block=(THREADS, 1, 1),
    )


def as_dynamic(tensor: torch.Tensor):
    """Expose shape and non-leading strides as runtime layout values."""
    return from_dlpack(tensor).mark_layout_dynamic(leading_dim=0)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires an NVIDIA GPU")

    # Compile with one runtime length, then reuse the same handle for two other
    # lengths.  All tensors are contiguous 1-D views, so leading_dim=0 keeps
    # the unit stride static while the extent remains dynamic.
    exemplar_src = torch.arange(17, dtype=torch.float32, device="cuda")
    exemplar_dst = torch.empty_like(exemplar_src)
    compiled = cute.compile(scale, as_dynamic(exemplar_src), as_dynamic(exemplar_dst))

    for n in (17, 257, 1003):
        src = torch.arange(n, dtype=torch.float32, device="cuda") - 3.0
        dst = torch.empty_like(src)
        compiled(as_dynamic(src), as_dynamic(dst))
        torch.cuda.synchronize()
        torch.testing.assert_close(dst, src * 2, rtol=0, atol=0)
        print(f"n={n}, first={dst[0].item()}, last={dst[-1].item()}")

    print("PASS")


if __name__ == "__main__":
    main()
