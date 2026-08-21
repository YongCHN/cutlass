"""Report whether the current machine can run this CuTe DSL tutorial.

This is a host-side environment probe.  It does not compile a GPU kernel and
does not modify the Python or CUDA environment.
"""

from __future__ import annotations

import platform
import sys


def main() -> None:
    print(f"python={platform.python_version()}")
    print(f"platform={platform.system()} {platform.machine()}")

    try:
        import cutlass
    except Exception as exc:
        print(f"cutlass=UNAVAILABLE ({exc})")
        raise SystemExit(1) from exc

    print(f"cutlass={cutlass.__version__}")
    print(f"cutlass_path={cutlass.__file__}")

    try:
        import torch
    except Exception as exc:
        print(f"torch=UNAVAILABLE ({exc})")
        raise SystemExit(1) from exc

    print(f"torch={torch.__version__}")
    print(f"torch_cuda={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("runnable_tracks=host-only")
        raise SystemExit(1)

    device = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device)
    print(f"gpu={torch.cuda.get_device_name(device)}")
    print(f"compute_capability={major}.{minor}")

    tracks = ["language", "layout", "tensor", "tiled-copy"]
    if major >= 8:
        tracks.append("sm80-warp-mma")
    if major >= 9:
        tracks.append("sm90-tma-wgmma")
    if major >= 10:
        tracks.append("sm100-tcgen05-tmem")
    if (major, minor) >= (12, 0):
        tracks.append("sm120-warp-mma")

    print("runnable_tracks=" + ",".join(tracks))
    print("PASS")


if __name__ == "__main__":
    main()
