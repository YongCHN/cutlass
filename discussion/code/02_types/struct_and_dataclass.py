"""Frozen dataclass arguments and a shared-memory ``cute.struct`` layout."""

from dataclasses import dataclass

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


WIDTH = 4


@dataclass(frozen=True)
class KernelParams:
    """A read-only pytree mixing static structure and dynamic leaves."""

    width: int
    src: cute.Tensor
    dst: cute.Tensor
    scale: cutlass.Float32
    bias: cutlass.Float32


@cute.struct
class SharedRecord:
    """A byte layout allocated in shared memory, not a by-value DSL object.

    ``@cute.struct`` needs concrete annotations at class creation time in 4.7.0,
    so this module deliberately does not enable postponed annotations.
    """

    values: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, WIDTH],
        16,
    ]
    checksum: cutlass.Float32


@cute.kernel
def transform_kernel(params: KernelParams):
    # One thread is sufficient here: the point is representation, not parallelism.
    storage = cutlass.utils.SmemAllocator().allocate(SharedRecord)
    checksum = cutlass.Float32(0.0)

    # range_constexpr requires a code-read-time Python integer; a field reached
    # through a value tree is not accepted as its bound in CuTe DSL 4.7.0.
    for i in cutlass.range_constexpr(WIDTH):
        value = params.src[i] * params.scale + params.bias
        storage.values[i] = value
        checksum = checksum + value

    storage.checksum = checksum

    for i in cutlass.range_constexpr(WIDTH):
        params.dst[i] = storage.values[i]
    params.dst[params.width] = storage.checksum.ptr.load()


@cute.jit
def launch_transform(
    src: cute.Tensor,
    dst: cute.Tensor,
    scale: cutlass.Float32,
    bias: cutlass.Float32,
    width: cutlass.Constexpr,
):
    params = KernelParams(
        width=width,
        src=src,
        dst=dst,
        scale=scale,
        bias=bias,
    )
    transform_kernel(params).launch(grid=(1, 1, 1), block=(1, 1, 1))


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires an NVIDIA GPU")

    src = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")
    dst = torch.empty(WIDTH + 1, dtype=torch.float32, device="cuda")
    src_cute = from_dlpack(src)
    dst_cute = from_dlpack(dst)
    scale = 1.5
    bias = -0.25

    compiled = cute.compile(
        launch_transform,
        src_cute,
        dst_cute,
        scale,
        bias,
        WIDTH,
    )
    compiled(src_cute, dst_cute, scale, bias)
    torch.cuda.synchronize()

    transformed = src * scale + bias
    expected = torch.cat((transformed, transformed.sum().reshape(1)))
    torch.testing.assert_close(dst, expected, rtol=0, atol=0)
    print(f"result={dst.cpu().tolist()}")
    print("PASS")


if __name__ == "__main__":
    main()
