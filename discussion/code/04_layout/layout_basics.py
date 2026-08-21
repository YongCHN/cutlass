"""CuTe Layout basics: coordinate mapping, hierarchy, and dynamic values."""

from __future__ import annotations

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.jit
def inspect_static_layouts():
    """Build static layouts and verify their coordinate-to-index functions."""
    shape = (3, 4)
    left_major = cute.make_layout(shape)  # default compact stride: (1, 3)
    row_major = cute.make_layout(shape, stride=(4, 1))
    padded = cute.make_layout(shape, stride=(8, 1))
    broadcast_rows = cute.make_layout(shape, stride=(0, 1))

    ordered_left = cute.make_ordered_layout(shape, order=(0, 1))
    ordered_row = cute.make_ordered_layout(shape, order=(1, 0))
    identity = cute.make_identity_layout(shape)

    print(f"left_major={left_major}")
    print(f"row_major={row_major}")
    print(f"padded={padded}")
    print(f"broadcast_rows={broadcast_rows}")
    print(f"ordered_left={ordered_left}")
    print(f"ordered_row={ordered_row}")
    print(f"identity={identity}, identity((1, 2))={identity((1, 2))}")

    assert left_major((1, 2)) == 7
    assert row_major((1, 2)) == 6
    assert padded((1, 2)) == 10
    assert broadcast_rows((1, 2)) == 2
    assert ordered_left((1, 2)) == left_major((1, 2))
    assert ordered_row((1, 2)) == row_major((1, 2))
    assert cute.crd2idx((1, 2), row_major) == row_major((1, 2))

    # An integer coordinate is first unfolded using the shape's leftmost-fast
    # logical enumeration. For shape (3, 4), logical index 5 is coord (2, 1).
    assert cute.idx2crd(5, shape) == (2, 1)
    assert left_major(5) == 5
    assert row_major(5) == 9

    assert cute.rank(row_major) == 2
    assert cute.depth(row_major) == 1
    assert cute.size(row_major) == 12
    assert cute.cosize(row_major) == 12
    assert cute.cosize(padded) == 20
    assert cute.cosize(broadcast_rows) == 4

    hierarchical = cute.make_layout(
        ((2, 3), 4),
        stride=((1, 2), 6),
    )
    print(f"hierarchical={hierarchical}")
    print(
        "hierarchical: "
        f"rank={cute.rank(hierarchical)}, "
        f"depth={cute.depth(hierarchical)}, "
        f"size={cute.size(hierarchical)}, "
        f"cosize={cute.cosize(hierarchical)}"
    )
    assert cute.rank(hierarchical) == 2
    assert cute.depth(hierarchical) == 2
    assert cute.rank(hierarchical.shape[0]) == 2
    assert cute.size(hierarchical) == 24
    assert cute.cosize(hierarchical) == 24
    assert hierarchical(((1, 2), 3)) == 23

    print("row-major offset table")
    for row in cutlass.range_constexpr(3):
        print(
            f"row {row}: "
            f"{row_major((row, 0))}, {row_major((row, 1))}, "
            f"{row_major((row, 2))}, {row_major((row, 3))}"
        )

    print("padded offset table")
    for row in cutlass.range_constexpr(3):
        print(
            f"row {row}: "
            f"{padded((row, 0))}, {padded((row, 1))}, "
            f"{padded((row, 2))}, {padded((row, 3))}"
        )

    print("broadcast-row offset table")
    for row in cutlass.range_constexpr(3):
        print(
            f"row {row}: "
            f"{broadcast_rows((row, 0))}, {broadcast_rows((row, 1))}, "
            f"{broadcast_rows((row, 2))}, {broadcast_rows((row, 3))}"
        )


@cute.kernel
def dynamic_layout_kernel(
    offsets: cute.Tensor,
    rows: cutlass.Int32,
    columns: cutlass.Int32,
    leading_dimension: cutlass.Int32,
):
    """Evaluate one runtime-created padded row-major Layout per thread."""
    layout = cute.make_layout(
        (rows, columns),
        stride=(leading_dimension, cutlass.Int32(1)),
    )
    tidx, _, _ = cute.arch.thread_idx()
    count = rows * columns

    if tidx < count:
        row = tidx // columns
        column = tidx % columns
        offsets[tidx] = cute.crd2idx((row, column), layout)

    if tidx == 0:
        cute.printf("dynamic layout={}", layout)


@cute.jit
def launch_dynamic_layout(
    offsets: cute.Tensor,
    rows: cutlass.Int32,
    columns: cutlass.Int32,
    leading_dimension: cutlass.Int32,
):
    dynamic_layout_kernel(offsets, rows, columns, leading_dimension).launch(
        grid=(1, 1, 1),
        block=(32, 1, 1),
    )


def expected_offsets(rows: int, columns: int, leading_dimension: int) -> torch.Tensor:
    values = [
        row * leading_dimension + column
        for row in range(rows)
        for column in range(columns)
    ]
    return torch.tensor(values, dtype=torch.int32)


def run_dynamic_cases() -> None:
    # Keep the Tensor's own layout fixed at 12 elements. Only the Layout being
    # evaluated inside the kernel is dynamic, so one compiled handle is reusable.
    offsets = torch.full((12,), -1, dtype=torch.int32, device="cuda")
    offsets_cute = from_dlpack(offsets)
    compiled = cute.compile(launch_dynamic_layout, offsets_cute, 3, 4, 8)

    for rows, columns, leading_dimension in ((3, 4, 8), (2, 5, 7)):
        offsets.fill_(-1)
        compiled(offsets_cute, rows, columns, leading_dimension)
        torch.cuda.synchronize()

        expected = expected_offsets(rows, columns, leading_dimension)
        actual = offsets[: rows * columns].cpu()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        print(
            f"dynamic case shape=({rows}, {columns}), "
            f"stride=({leading_dimension}, 1): {actual.tolist()}"
        )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires an NVIDIA GPU")

    cutlass.cuda.initialize_cuda_context()
    compiled_inspector = cute.compile(inspect_static_layouts)
    compiled_inspector()
    run_dynamic_cases()
    print("PASS")


if __name__ == "__main__":
    main()
