"""Visualize an XOR Swizzle as a ComposedLayout and a bank mapping."""

import cutlass
import cutlass.cute as cute


@cute.jit
def visualize():
    # Logical lane i addresses FP32 element i*32. Without a transform, all 32
    # lanes select bank 0. Swizzle<5,0,5> XORs bits [5,10) into bits [0,5).
    outer = cute.make_layout(32, stride=32)
    swizzle = cute.make_swizzle(5, 0, 5)
    composed = cute.make_composed_layout(swizzle, 0, outer)

    print(f"outer={outer}")
    print(f"swizzle={swizzle}")
    print(f"composed={composed}")
    print("lane: logical_offset -> swizzled_offset -> bank")

    for lane in cutlass.range_constexpr(32):
        logical_offset = outer(lane)
        physical_offset = composed(lane)
        bank = physical_offset % 32
        print(f"{lane}: {logical_offset} -> {physical_offset} -> {bank}")
        assert logical_offset % 32 == 0
        assert physical_offset == (lane * 32) ^ lane
        assert bank == lane
        assert physical_offset ^ lane == logical_offset

    # The last assertion is the involution property for this concrete mapping:
    # applying the same XOR mask again restores the logical offset.
    print("PASS")


def main() -> None:
    cutlass.cuda.initialize_cuda_context()
    compiled = cute.compile(visualize)
    compiled()


if __name__ == "__main__":
    main()
