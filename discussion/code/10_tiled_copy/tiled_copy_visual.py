"""Print and verify Thread-Value ownership for a small TiledCopy.

Target: architecture-independent CuTe Layout/TiledCopy construction.
Validated: CuTe DSL 4.7.0 in the B200 Linux environment (L0).

The example uses only six logical threads and four values per thread so every
entry in the (4, 6) tile can be inspected by hand.
"""

import cutlass
import cutlass.cute as cute


@cute.jit
def inspect_tiled_copy():
    # thr_layout: tile coordinate -> thread ID
    # val_layout: per-thread tile coordinate -> value ID
    thr_layout = cute.make_layout((2, 3), stride=(3, 1))
    val_layout = cute.make_layout((2, 2), stride=(2, 1))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        cutlass.Float32,
        num_bits_per_copy=32,
    )
    tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
    identity = cute.make_identity_tensor(tiler_mn)

    print(f"thr_layout={thr_layout}")
    print(f"val_layout={val_layout}")
    print(f"tiler_mn={tiler_mn}")
    print(f"tv_layout={tv_layout}")
    print(f"tiled_copy_tiler={tiled_copy.tiler_mn}")
    print("thread,value -> logical_index -> coordinate")

    thread_count = cute.size(thr_layout)
    value_count = cute.size(val_layout)
    assert cute.size(tiler_mn) == thread_count * value_count

    for tid in cutlass.range_constexpr(thread_count):
        thr_copy = tiled_copy.get_slice(tid)
        thr_coord = thr_copy.partition_S(identity)
        assert cute.size(thr_coord) == value_count
        print(f"thread {tid}: partition_shape={thr_coord.shape}")

        for vid in cutlass.range_constexpr(value_count):
            logical_index = tv_layout((tid, vid))
            coordinate = cute.idx2crd(logical_index, tiler_mn)
            partition_coordinate = thr_coord[vid]
            print(f"  ({tid},{vid}) -> {logical_index} -> {coordinate}")
            assert partition_coordinate == coordinate

            # Prove injectivity by comparing with all lexicographically earlier
            # (thread,value) pairs.  Size equality plus injectivity proves exact
            # coverage of the 24-entry tile.
            for earlier_tid in cutlass.range_constexpr(tid + 1):
                earlier_value_limit = vid if earlier_tid == tid else value_count
                for earlier_vid in cutlass.range_constexpr(earlier_value_limit):
                    assert tv_layout((earlier_tid, earlier_vid)) != logical_index

    print("PASS")


def main() -> None:
    cutlass.cuda.initialize_cuda_context()
    compiled = cute.compile(inspect_tiled_copy)
    compiled()


if __name__ == "__main__":
    main()
