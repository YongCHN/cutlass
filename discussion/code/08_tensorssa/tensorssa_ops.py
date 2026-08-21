"""TensorSSA load/store, broadcasting, slicing, and reduction in one thread."""

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def tensorssa_kernel(
    src: cute.Tensor,
    dst: cute.Tensor,
    row_sums: cute.Tensor,
    selected_column: cute.Tensor,
):
    # A memory Tensor load creates one immutable, thread-local TensorSSA value.
    src_value = src.load()

    row_bias = cute.make_rmem_tensor((2, 1), cutlass.Float32)
    row_bias[0] = 10.0
    row_bias[1] = 20.0

    column_bias = cute.make_rmem_tensor((1, 3), cutlass.Float32)
    column_bias[0] = 0.5
    column_bias[1] = 1.0
    column_bias[2] = 1.5

    # Shapes (2,3), (2,1), and (1,3) broadcast to (2,3).  Every operator
    # creates a new TensorSSA value; none of these expressions mutate src.
    result = src_value * 2.0 + row_bias.load() + column_bias.load()

    # make_fragment_like creates register-backed storage with a congruent
    # shape.  store/load are the explicit value <-> storage boundaries.
    fragment = cute.make_fragment_like(dst)
    fragment.store(result)
    round_trip = fragment.load()
    dst.store(round_trip)

    # reduction_profile=(None, 1) preserves mode 0 and reduces mode 1.
    sums = round_trip.reduce(
        cute.ReductionOp.ADD,
        0.0,
        reduction_profile=(None, 1),
    )
    row_sums.store(sums)

    # None preserves a mode; integer 1 fixes mode 1 at column coordinate 1.
    selected_column.store(round_trip[(None, 1)])


@cute.jit
def launch(
    src: cute.Tensor,
    dst: cute.Tensor,
    row_sums: cute.Tensor,
    selected_column: cute.Tensor,
):
    tensorssa_kernel(src, dst, row_sums, selected_column).launch(
        grid=(1, 1, 1),
        block=(1, 1, 1),
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires an NVIDIA GPU")

    src = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=torch.float32,
        device="cuda",
    )
    dst = torch.empty_like(src)
    row_sums = torch.empty(2, dtype=torch.float32, device="cuda")
    selected_column = torch.empty(2, dtype=torch.float32, device="cuda")

    args = tuple(
        from_dlpack(tensor)
        for tensor in (src, dst, row_sums, selected_column)
    )
    compiled = cute.compile(launch, *args)
    compiled(*args)
    torch.cuda.synchronize()

    expected = torch.tensor(
        [[12.5, 15.0, 17.5], [28.5, 31.0, 33.5]],
        dtype=torch.float32,
        device="cuda",
    )
    torch.testing.assert_close(dst, expected, rtol=0, atol=0)
    torch.testing.assert_close(row_sums, expected.sum(dim=1), rtol=0, atol=0)
    torch.testing.assert_close(selected_column, expected[:, 1], rtol=0, atol=0)

    print(f"dst=\n{dst.cpu()}")
    print(f"row_sums={row_sums.cpu().tolist()}")
    print(f"selected_column={selected_column.cpu().tolist()}")
    print("PASS")


if __name__ == "__main__":
    main()
