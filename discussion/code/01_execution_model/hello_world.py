"""The smallest CuTe DSL kernel/host/Python call chain."""

import cutlass
import cutlass.cute as cute


@cute.kernel
def hello_world_kernel(meta_value: cutlass.Constexpr, runtime_value: cutlass.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    if tidx == 0:
        cute.printf(
            "GPU says: tidx={}, meta_value={}, runtime_value={}",
            tidx,
            meta_value,
            runtime_value,
        )


@cute.jit
def launch_hello_world(
    meta_value: cutlass.Constexpr,
    runtime_value: cutlass.Int32,
):
    hello_world_kernel(meta_value, runtime_value).launch(
        grid=(1, 1, 1),
        block=(1, 1, 1),
    )


def main() -> None:
    cutlass.cuda.initialize_cuda_context()
    launch_hello_world(7, 11)
    cutlass.cuda.stream_sync(cutlass.cuda.default_stream())
    print("PASS")


if __name__ == "__main__":
    main()
