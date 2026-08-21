"""Minimal CuTe DSL vector addition for NVIDIA GPUs.

This example computes ``c[i] = a[i] + b[i]`` for a one-dimensional FP32
tensor.  It intentionally uses direct tensor indexing so the relationship
between CUDA's execution hierarchy and CuTe tensors stays visible.

Validated environment:
  - NVIDIA B200 (SM100)
  - nvidia-cutlass-dsl 4.7.0 (also previously checked with 4.4.2)
  - PyTorch 2.9.1+cu129
"""

from __future__ import annotations

import argparse

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


THREADS_PER_BLOCK = 256


@cute.kernel
def vector_add_kernel(
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    num_elements: cutlass.Int32,
):
    """Add one pair of elements per CUDA thread."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    linear_idx = bidx * THREADS_PER_BLOCK + tidx
    if linear_idx < num_elements:
        c[linear_idx] = a[linear_idx] + b[linear_idx]


@cute.jit
def vector_add(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    """Configure and launch ``vector_add_kernel``."""
    num_elements = cute.size(c)
    num_blocks = cute.ceil_div(num_elements, THREADS_PER_BLOCK)

    vector_add_kernel(a, b, c, num_elements).launch(
        grid=(num_blocks, 1, 1),
        block=(THREADS_PER_BLOCK, 1, 1),
    )


def run(num_elements: int, seed: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires an NVIDIA GPU")
    if num_elements <= 0:
        raise ValueError("num_elements must be positive")

    torch.manual_seed(seed)
    device = torch.device("cuda:0")
    a = torch.randn(num_elements, device=device, dtype=torch.float32)
    b = torch.randn(num_elements, device=device, dtype=torch.float32)
    c = torch.empty_like(a)

    a_cute = from_dlpack(a)
    b_cute = from_dlpack(b)
    c_cute = from_dlpack(c)

    print("Compiling CuTe DSL vector-add kernel...")
    compiled_vector_add = cute.compile(vector_add, a_cute, b_cute, c_cute)

    print(f"Launching with {num_elements} elements...")
    compiled_vector_add(a_cute, b_cute, c_cute)
    torch.cuda.synchronize()

    expected = a + b
    torch.testing.assert_close(c, expected, rtol=1e-5, atol=1e-6)

    max_abs_error = (c - expected).abs().max().item()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"max_abs_error: {max_abs_error:.3e}")
    print(f"first five results: {c[:5].cpu().tolist()}")
    print("PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CuTe DSL vector addition")
    parser.add_argument(
        "--num-elements",
        type=int,
        default=1_000_003,
        help="Vector length; the default deliberately exercises the tail predicate",
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.num_elements, args.seed)
