"""CuTe DSL scalar types, explicit conversion, and constexpr specialization."""

from __future__ import annotations

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def scalar_kernel(
    out_i32: cute.Tensor,
    out_f32: cute.Tensor,
    x: cutlass.Int32,
    scale: cutlass.Float32,
    square_result: cutlass.Constexpr,
):
    """Write values whose types and staging decisions are explicit."""
    integer_value = x * cutlass.Int32(3) + cutlass.Int32(1)
    floating_value = cutlass.Float32(integer_value) * scale

    # This branch is resolved while specializing the kernel.
    if cutlass.const_expr(square_result):
        floating_value = floating_value * floating_value

    out_i32[0] = integer_value
    out_f32[0] = floating_value

    # This predicate depends on a runtime value and becomes GPU control flow.
    if integer_value % cutlass.Int32(2) == cutlass.Int32(0):
        out_i32[1] = cutlass.Int32(1)
    else:
        out_i32[1] = cutlass.Int32(0)


@cute.jit
def launch_scalar_demo(
    out_i32: cute.Tensor,
    out_f32: cute.Tensor,
    x: cutlass.Int32,
    scale: cutlass.Float32,
    square_result: cutlass.Constexpr,
):
    scalar_kernel(out_i32, out_f32, x, scale, square_result).launch(
        grid=(1, 1, 1),
        block=(1, 1, 1),
    )


def run_case(square_result: bool) -> None:
    out_i32 = torch.empty(2, dtype=torch.int32, device="cuda")
    out_f32 = torch.empty(1, dtype=torch.float32, device="cuda")
    out_i32_cute = from_dlpack(out_i32)
    out_f32_cute = from_dlpack(out_f32)

    x = 5
    scale = 0.5
    compiled = cute.compile(
        launch_scalar_demo,
        out_i32_cute,
        out_f32_cute,
        x,
        scale,
        square_result,
    )

    # square_result is static and therefore absent from the runtime ABI.
    compiled(out_i32_cute, out_f32_cute, x, scale)
    torch.cuda.synchronize()

    expected_float = 64.0 if square_result else 8.0
    torch.testing.assert_close(
        out_i32.cpu(), torch.tensor([16, 1], dtype=torch.int32), rtol=0, atol=0
    )
    torch.testing.assert_close(
        out_f32.cpu(),
        torch.tensor([expected_float], dtype=torch.float32),
        rtol=0,
        atol=0,
    )
    print(
        f"square_result={square_result}: "
        f"i32={out_i32.cpu().tolist()}, f32={out_f32.cpu().tolist()}"
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires an NVIDIA GPU")
    run_case(False)
    run_case(True)
    print("PASS")


if __name__ == "__main__":
    main()
