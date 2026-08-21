"""Property-driven laboratory for CuTe Layout algebra."""

import cutlass
import cutlass.cute as cute


@cute.jit
def layout_algebra_lab():
    nested = cute.make_layout(
        ((2, (3, 4)), (3, 2), 1),
        stride=((4, (8, 24)), (2, 6), 12),
    )
    coalesced = cute.coalesce(nested)
    print(f"coalesce: {nested} -> {coalesced}")
    assert cute.size(coalesced) == cute.size(nested)
    assert cute.depth(coalesced) <= 1
    for i in cutlass.range_constexpr(cute.size(nested)):
        assert coalesced(i) == nested(i)

    outer = cute.make_layout((6, 2), stride=(8, 2))
    inner = cute.make_layout((4, 3), stride=(3, 1))
    composed = cute.composition(outer, inner)
    print(f"composition: {outer} o {inner} -> {composed}")
    for i in cutlass.range_constexpr(cute.size(inner)):
        assert composed(i) == outer(inner(i))

    target = cute.make_layout((8, 6), stride=(6, 1))
    tiler = (2, 3)
    logical_d = cute.logical_divide(target, tiler)
    zipped_d = cute.zipped_divide(target, tiler)
    tiled_d = cute.tiled_divide(target, tiler)
    flat_d = cute.flat_divide(target, tiler)
    print(f"target={target}, tiler={tiler}")
    print(f"logical_divide={logical_d}")
    print(f"zipped_divide={zipped_d}")
    print(f"tiled_divide={tiled_d}")
    print(f"flat_divide={flat_d}")
    assert cute.size(logical_d) == cute.size(target)
    assert cute.size(zipped_d) == cute.size(target)
    assert cute.size(tiled_d) == cute.size(target)
    assert cute.size(flat_d) == cute.size(target)

    block = cute.make_layout((2, 2), stride=(2, 1))
    replication = cute.make_layout((3, 2), stride=(1, 3))
    logical_p = cute.logical_product(block, replication)
    zipped_p = cute.zipped_product(block, replication)
    tiled_p = cute.tiled_product(block, replication)
    flat_p = cute.flat_product(block, replication)
    print(f"block={block}, replication={replication}")
    print(f"logical_product={logical_p}")
    print(f"zipped_product={zipped_p}")
    print(f"tiled_product={tiled_p}")
    print(f"flat_product={flat_p}")
    expected_product_size = cute.size(block) * cute.size(replication)
    assert cute.size(logical_p) == expected_product_size
    assert cute.size(zipped_p) == expected_product_size
    assert cute.size(tiled_p) == expected_product_size
    assert cute.size(flat_p) == expected_product_size

    vector = cute.make_layout(4, stride=1)
    complement = cute.complement(vector, 24)
    print(f"complement({vector}, 24)={complement}")
    assert cute.size(complement) == 6
    for i in cutlass.range_constexpr(6):
        assert complement(i) == i * 4

    permutation = cute.make_layout((2, 3), stride=(3, 1))
    right_inv = cute.right_inverse(permutation)
    left_inv = cute.left_inverse(permutation)
    print(f"permutation={permutation}")
    print(f"right_inverse={right_inv}")
    print(f"left_inverse={left_inv}")
    for i in cutlass.range_constexpr(cute.size(permutation)):
        assert permutation(right_inv(i)) == i
        assert left_inv(permutation(i)) == i

    base = cute.make_layout((2, 3, 4), stride=(12, 4, 1))
    selected = cute.select(base, mode=[0, 2])
    grouped = cute.group_modes(base, 0, 2)
    flattened = cute.flatten(grouped)
    sliced = cute.slice_(base, (1, None, None))
    print(f"base={base}")
    print(f"select modes [0,2]={selected}")
    print(f"group modes [0,2)={grouped}")
    print(f"flatten(grouped)={flattened}")
    print(f"slice first mode at 1={sliced}")
    assert cute.size(selected) == 8
    assert cute.rank(grouped) == 2
    assert cute.depth(grouped) == 2
    assert cute.rank(flattened) == 3
    assert cute.size(sliced) == 12

    atom = cute.make_layout((2, 2), stride=(2, 1))
    tiled_to_shape = cute.tile_to_shape(atom, (4, 6), order=(1, 0))
    print(f"tile_to_shape({atom}, (4,6))={tiled_to_shape}")
    assert cute.size(tiled_to_shape) == 24
    print("PASS")


def main() -> None:
    cutlass.cuda.initialize_cuda_context()
    compiled = cute.compile(layout_algebra_lab)
    compiled()


if __name__ == "__main__":
    main()
