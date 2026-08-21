"""Executable negative tests for three CuTe DSL staging/type constraints."""

from __future__ import annotations

from collections.abc import Callable

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def dynamic_constexpr_kernel(out: cute.Tensor, value: cutlass.Int32):
    # Invalid: a runtime predicate cannot be forced into compile-time Python.
    if cutlass.const_expr(value > cutlass.Int32(0)):
        out[0] = value


@cute.jit
def launch_dynamic_constexpr(out: cute.Tensor, value: cutlass.Int32):
    dynamic_constexpr_kernel(out, value).launch(
        grid=(1, 1, 1), block=(1, 1, 1)
    )


@cute.kernel
def dynamic_list_index_kernel(out: cute.Tensor, index: cutlass.Int32):
    values = [cutlass.Int32(10), cutlass.Int32(20)]
    # Invalid: list structure and indexing live in Python; index lives on GPU.
    out[0] = values[index]


@cute.jit
def launch_dynamic_list_index(out: cute.Tensor, index: cutlass.Int32):
    dynamic_list_index_kernel(out, index).launch(
        grid=(1, 1, 1), block=(1, 1, 1)
    )


@cute.kernel
def branch_type_change_kernel(
    out: cute.Tensor,
    predicate: cutlass.Boolean,
):
    value = cutlass.Int32(1)
    if predicate:
        # Invalid: a dynamic region cannot change an outer value from i32 to f32.
        value = cutlass.Float32(2.0)
    out[0] = value


@cute.jit
def launch_branch_type_change(out: cute.Tensor, predicate: cutlass.Boolean):
    branch_type_change_kernel(out, predicate).launch(
        grid=(1, 1, 1), block=(1, 1, 1)
    )


def expect_compile_failure(name: str, compile_action: Callable[[], object]) -> None:
    try:
        compile_action()
    except Exception as error:  # The diagnostic class/text can change across releases.
        first_line = str(error).strip().splitlines()[0]
        print(f"{name}: expected failure: {type(error).__name__}: {first_line}")
        return
    raise AssertionError(f"{name}: compilation unexpectedly succeeded")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires an NVIDIA GPU")

    out_i32 = from_dlpack(torch.empty(1, dtype=torch.int32, device="cuda"))
    out_f32 = from_dlpack(torch.empty(1, dtype=torch.float32, device="cuda"))

    expect_compile_failure(
        "dynamic value passed to const_expr",
        lambda: cute.compile(launch_dynamic_constexpr, out_i32, 1),
    )
    expect_compile_failure(
        "dynamic Python-list index",
        lambda: cute.compile(launch_dynamic_list_index, out_i32, 1),
    )
    expect_compile_failure(
        "dynamic branch changes value type",
        lambda: cute.compile(launch_branch_type_change, out_f32, True),
    )
    print("PASS (all failures were expected)")


if __name__ == "__main__":
    main()
