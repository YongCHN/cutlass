"""Show compile-time Python values and runtime GPU values side by side."""

import cutlass
import cutlass.cute as cute


@cute.kernel
def phase_kernel(static_value: cutlass.Constexpr, dynamic_value: cutlass.Int32):
    # Native Python print runs while the DSL traces/compiles this kernel.
    print(f"[compile:kernel] static_value={static_value}")
    print(f"[compile:kernel] dynamic_value={dynamic_value}")

    tidx, _, _ = cute.arch.thread_idx()
    if tidx == 0:
        # cute.printf is emitted into generated GPU code.
        cute.printf(
            "[runtime:gpu] static_value={}, dynamic_value={}, sum={}",
            static_value,
            dynamic_value,
            static_value + dynamic_value,
        )


@cute.jit
def launch_phase_demo(
    static_value: cutlass.Constexpr,
    dynamic_value: cutlass.Int32,
):
    print(f"[compile:jit] static_value={static_value}")
    print(f"[compile:jit] dynamic_value={dynamic_value}")
    phase_kernel(static_value, dynamic_value).launch(
        grid=(1, 1, 1),
        block=(1, 1, 1),
    )


def main() -> None:
    cutlass.cuda.initialize_cuda_context()

    print("compile specialization: static_value=7")
    compiled_static_7 = cute.compile(launch_phase_demo, 7, 11)

    print("first runtime call: dynamic_value=11")
    compiled_static_7(11)
    cutlass.cuda.stream_sync(cutlass.cuda.default_stream())

    print("second runtime call: reuse specialization, dynamic_value=13")
    compiled_static_7(13)
    cutlass.cuda.stream_sync(cutlass.cuda.default_stream())

    print("compile another specialization: static_value=8")
    compiled_static_8 = cute.compile(launch_phase_demo, 8, 13)
    print("third runtime call: dynamic_value=13")
    compiled_static_8(13)
    cutlass.cuda.stream_sync(cutlass.cuda.default_stream())
    print("PASS")


if __name__ == "__main__":
    main()
