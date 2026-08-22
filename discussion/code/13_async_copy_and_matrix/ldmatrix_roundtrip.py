"""Warp-level b16 ldmatrix/stmatrix roundtrip for x1, x2, and x4.

The public tensors use ordinary row-major coordinates.  The interesting part
is the register boundary: the warp collectively loads N independent 8x8 b16
matrices, keeps N packed 32-bit words per lane, and stores them back.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch


ARTIFACT_DIR = Path(__file__).with_name("generated_artifacts") / "ldmatrix"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("CUTE_DSL_KEEP", "ptx")
os.environ.setdefault("CUTE_DSL_DUMP_DIR", str(ARTIFACT_DIR))

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.experimental import primitives as prims


WARP_SIZE = 32
ROWS_PER_MATRIX = 8
COLS = 8


@cute.kernel
def matrix_kernel(
    src: cute.Tensor,
    dst: cute.Tensor,
    DTYPE: cutlass.Constexpr[type],
    NUM_MATRICES: cutlass.Constexpr[int],
):
    lane, _, _ = cute.arch.thread_idx()
    rows = NUM_MATRICES * ROWS_PER_MATRIX
    smem_in = cutlass.Array(
        DTYPE, (rows, COLS), space=cutlass.AddressSpace.smem, alignment=128
    )
    smem_out = cutlass.Array(
        DTYPE, (rows, COLS), space=cutlass.AddressSpace.smem, alignment=128
    )

    # 64*N elements / 32 lanes = 2*N elements per lane.
    for i in cutlass.range_constexpr(2 * NUM_MATRICES):
        flat = lane + i * WARP_SIZE
        smem_in[flat // COLS, flat % COLS] = src[flat // COLS, flat % COLS]
    prims.barrier_cta_sync(0)

    # For xN b16, lanes 0..8*N-1 contribute one 16-byte row address.
    # The remaining lanes still participate in the warp instruction.
    row_ptr = smem_in.data_ptr() + lane * COLS
    registers = prims.ldmatrix(row_ptr, NUM_MATRICES, prims.MMALayout.ROW)
    prims.stmatrix(
        smem_out.data_ptr() + lane * COLS,
        registers,
        prims.MMALayout.ROW,
    )
    prims.barrier_cta_sync(0)

    for i in cutlass.range_constexpr(2 * NUM_MATRICES):
        flat = lane + i * WARP_SIZE
        dst[flat // COLS, flat % COLS] = smem_out[flat // COLS, flat % COLS]


@cute.jit
def launch(
    src: cute.Tensor,
    dst: cute.Tensor,
    DTYPE: cutlass.Constexpr[type],
    NUM_MATRICES: cutlass.Constexpr[int],
):
    matrix_kernel(src, dst, DTYPE, NUM_MATRICES).launch(
        grid=(1, 1, 1), block=(WARP_SIZE, 1, 1)
    )


@lru_cache(maxsize=None)
def compile_kernel(dtype: type, num_matrices: int):
    shape = (num_matrices * ROWS_PER_MATRIX, COLS)
    fake_src = make_fake_compact_tensor(dtype, shape, stride_order=(1, 0), assumed_align=16)
    fake_dst = make_fake_compact_tensor(dtype, shape, stride_order=(1, 0), assumed_align=16)
    return cute.compile(
        launch, fake_src, fake_dst, dtype, num_matrices, options="--enable-tvm-ffi"
    )


def inspect_ptx() -> dict[str, set[str]]:
    lines: list[str] = []
    for path in ARTIFACT_DIR.glob("*.ptx"):
        lines.extend(line.strip() for line in path.read_text().splitlines())
    result: dict[str, set[str]] = {"ldmatrix": set(), "stmatrix": set()}
    for line in lines:
        if "ldmatrix.sync" in line:
            result["ldmatrix"].add(line)
        if "stmatrix.sync" in line:
            result["stmatrix"].add(line)
    return result


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires an NVIDIA GPU")
    major, minor = torch.cuda.get_device_capability()
    if major < 9:
        raise RuntimeError(
            f"this roundtrip uses SM90+ stmatrix (ldmatrix itself is older), got {major}.{minor}"
        )

    dtype_cases = ((cutlass.Float16, torch.float16, "fp16"), (cutlass.BFloat16, torch.bfloat16, "bf16"))
    for cute_dtype, torch_dtype, name in dtype_cases:
        for num_matrices in (1, 2, 4):
            shape = (num_matrices * ROWS_PER_MATRIX, COLS)
            src = torch.arange(shape[0] * shape[1], device="cuda", dtype=torch.float32).reshape(shape).to(torch_dtype)
            dst = torch.empty_like(src)
            compile_kernel(cute_dtype, num_matrices)(src, dst)
            torch.cuda.synchronize()
            torch.testing.assert_close(dst, src, rtol=0, atol=0)
            print(f"dtype={name}, x{num_matrices}, shape={shape}, words_per_lane={num_matrices}: PASS")

    evidence = inspect_ptx()
    if not evidence["ldmatrix"] or not evidence["stmatrix"]:
        raise AssertionError("missing ldmatrix/stmatrix in retained PTX")
    for family, lines in evidence.items():
        for line in sorted(lines)[:3]:
            print(f"PTX {family}: {line}")
    print("PASS")


if __name__ == "__main__":
    main()

