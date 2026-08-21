"""Compile-time unrolling, dynamic loops, branches, and loop-carried values."""

from __future__ import annotations

import os
from pathlib import Path

import torch

# IR is opt-in in CuTe DSL 4.7.0. Keep it beside this example so the generated
# control-flow structure can be inspected after the run.
IR_DUMP_DIR = Path(__file__).with_name("generated_ir")
IR_DUMP_DIR.mkdir(exist_ok=True)
os.environ.setdefault("CUTE_DSL_KEEP", "ir-debug")
os.environ.setdefault("CUTE_DSL_DUMP_DIR", str(IR_DUMP_DIR))

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def control_flow_kernel(
    out: cute.Tensor,
    bound: cutlass.Int32,
    add_hundred: cutlass.Constexpr,
):
    constexpr_sum = cutlass.Int32(0)
    for i in cutlass.range_constexpr(4):
        constexpr_sum = constexpr_sum + cutlass.Int32(i)

    dynamic_sum = cutlass.Int32(0)
    for i in range(bound):
        dynamic_sum = dynamic_sum + i

    # The compiler emits a dynamic branch because bound is a runtime value.
    if bound % cutlass.Int32(2) == cutlass.Int32(0):
        dynamic_sum = dynamic_sum + cutlass.Int32(10)
    else:
        dynamic_sum = dynamic_sum + cutlass.Int32(20)

    # This branch disappears during specialization.
    if cutlass.const_expr(add_hundred):
        dynamic_sum = dynamic_sum + cutlass.Int32(100)

    countdown = bound
    while countdown > cutlass.Int32(0):
        countdown = countdown - cutlass.Int32(1)

    unrolled_sum = cutlass.Int32(0)
    for i in cutlass.range(bound, unroll=2):
        unrolled_sum = unrolled_sum + i * cutlass.Int32(2)

    out[0] = constexpr_sum
    out[1] = dynamic_sum
    out[2] = countdown
    out[3] = unrolled_sum


@cute.jit
def launch_control_flow(
    out: cute.Tensor,
    bound: cutlass.Int32,
    add_hundred: cutlass.Constexpr,
):
    control_flow_kernel(out, bound, add_hundred).launch(
        grid=(1, 1, 1),
        block=(1, 1, 1),
    )


def expected(bound: int, add_hundred: bool) -> torch.Tensor:
    dynamic_sum = sum(range(bound))
    dynamic_sum += 10 if bound % 2 == 0 else 20
    dynamic_sum += 100 if add_hundred else 0
    return torch.tensor(
        [6, dynamic_sum, 0, 2 * sum(range(bound))],
        dtype=torch.int32,
    )


def run_specialization(add_hundred: bool, bounds: tuple[int, ...]) -> None:
    out = torch.empty(4, dtype=torch.int32, device="cuda")
    out_cute = from_dlpack(out)

    compiled = cute.compile(
        launch_control_flow,
        out_cute,
        bounds[0],
        add_hundred,
    )
    for bound in bounds:
        compiled(out_cute, bound)
        torch.cuda.synchronize()
        torch.testing.assert_close(out.cpu(), expected(bound, add_hundred), rtol=0, atol=0)
        print(
            f"add_hundred={add_hundred}, bound={bound}: "
            f"result={out.cpu().tolist()}"
        )

    mlir = compiled.__mlir__
    if not mlir:
        candidates = sorted(
            IR_DUMP_DIR.glob("*.mlir"), key=lambda path: path.stat().st_mtime
        )
        if not candidates:
            raise RuntimeError("CuTe DSL did not expose or dump the requested MLIR")
        mlir = candidates[-1].read_text()
    print(
        f"specialization add_hundred={add_hundred}: "
        f"mlir_chars={len(mlir)}, scf.for={mlir.count('scf.for')}, "
        f"scf.while={mlir.count('scf.while')}, scf.if={mlir.count('scf.if')}"
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires an NVIDIA GPU")

    # Dynamic bounds reuse a compiled handle; the static flag needs another one.
    run_specialization(True, (5, 6))
    run_specialization(False, (5,))
    print("PASS")


if __name__ == "__main__":
    main()
