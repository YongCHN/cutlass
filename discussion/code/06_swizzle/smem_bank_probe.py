"""Round-trip values through a physically XOR-swizzled SMEM address map."""

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


THREADS = 32
SMEM_ELEMENTS = 1024


@cute.kernel
def swizzled_smem_kernel(dst: cute.Tensor):
    lane, _, _ = cute.arch.thread_idx()
    smem = cutlass.Array(
        cutlass.Float32,
        SMEM_ELEMENTS,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    # Pointer swizzles operate on byte addresses. Use the canonical 128-byte
    # SMEM swizzle here; the visualizer separately uses an educational
    # element-offset swizzle so that its bank table stays easy to hand-check.
    swizzle = cutlass.Swizzle.from_name("s128b")

    logical_ptr = smem.data_ptr() + lane * 4
    physical_ptr = logical_ptr.apply_swizzle(swizzle)
    physical_ptr.store(cutlass.Float32(lane))
    cute.arch.sync_threads()

    # Keep this scalar probe entirely at the physical-pointer level.  The
    # higher-level load_swizzled API constructs a CuTe Tensor and remaps whole
    # 128-bit vectors; it should be paired with the corresponding vector store
    # path rather than with this per-lane scalar physical store.
    dst[lane] = logical_ptr.apply_swizzle(swizzle).load()


@cute.jit
def launch(dst: cute.Tensor):
    swizzled_smem_kernel(dst).launch(
        grid=(1, 1, 1),
        block=(THREADS, 1, 1),
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires an NVIDIA GPU")

    dst = torch.empty(THREADS, dtype=torch.float32, device="cuda")
    dst_cute = from_dlpack(dst)
    compiled = cute.compile(launch, dst_cute)
    compiled(dst_cute)
    torch.cuda.synchronize()

    expected = torch.arange(THREADS, dtype=torch.float32, device="cuda")
    print(f"dst={dst.cpu().tolist()}")
    torch.testing.assert_close(dst, expected, rtol=0, atol=0)
    print("PASS")


if __name__ == "__main__":
    main()
