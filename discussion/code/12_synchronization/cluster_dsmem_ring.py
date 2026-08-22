"""SM90+ cluster ring using distributed shared memory and mapa.

Every CTA writes its rank to local SMEM, signals its predecessor through a
remote mbarrier, and reads the successor's SMEM value after that successor has
signalled its local barrier.  This keeps cluster launch, initialization
publication, remote signalling, and remote access in one small protocol.
"""

from __future__ import annotations

import argparse
import os
from functools import lru_cache
from pathlib import Path

import torch


ARTIFACT_DIR = Path(__file__).with_name("generated_artifacts") / "cluster_dsmem_ring"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("CUTE_DSL_KEEP", "ptx")
os.environ.setdefault("CUTE_DSL_DUMP_DIR", str(ARTIFACT_DIR))

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.experimental import primitives as prims


@cute.kernel
def cluster_ring_kernel(
    dst: cute.Tensor,
    CLUSTER_SIZE: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    my_rank = cute.arch.block_idx_in_cluster()
    successor = (my_rank + 1) % cutlass.Int32(CLUSTER_SIZE)
    predecessor = (
        my_rank + cutlass.Int32(CLUSTER_SIZE) - cutlass.Int32(1)
    ) % cutlass.Int32(CLUSTER_SIZE)

    local_value = cutlass.Array(
        cutlass.Int32, 1, space=cutlass.AddressSpace.smem, alignment=4
    )
    local_barrier = cutlass.Array(
        cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8
    )

    # Every CTA owns a distinct physical barrier object at the same local SMEM
    # address.  The cluster rendezvous publishes all initializations before any
    # CTA maps and signals a peer object.
    if prims.elect_sync():
        prims.mbarrier_init(local_barrier, 1)
    prims.fence_mbarrier_init()
    prims.barrier_cluster_arrive_relaxed()
    prims.barrier_cluster_wait()

    if tidx == 0:
        local_value[0] = my_rank
    prims.barrier_cta_sync(0)

    # Signal the predecessor's barrier through a shared::cluster pointer.  We
    # are that CTA's successor, so our arrival proves that the value it will
    # read from us has already been written and published.
    if tidx == 0:
        predecessor_barrier = prims.mapa(local_barrier, predecessor)
        prims.mbarrier_arrive(predecessor_barrier, scope=prims.MemScope.CLUSTER)

    # Our successor signals our local barrier after writing its local value.
    # The wait provides the acquire side before we map and load that value.
    while not prims.mbarrier_try_wait_parity(
        local_barrier, 0, time_limit=10_000_000
    ):
        pass

    if tidx == 0:
        successor_ptr = prims.mapa(local_value.data_ptr(), successor)
        dst[my_rank] = successor_ptr[0]

    # Keep every CTA and all DSMEM users alive until remote reads finish.
    prims.barrier_cluster_arrive_relaxed()
    prims.barrier_cluster_wait()
    if tidx == 0:
        prims.mbarrier_inval(local_barrier)


@cute.jit
def launch(dst: cute.Tensor, CLUSTER_SIZE: cutlass.Constexpr):
    cluster_ring_kernel(dst, CLUSTER_SIZE).launch(
        grid=(CLUSTER_SIZE, 1, 1),
        block=(32, 1, 1),
        cluster=(CLUSTER_SIZE, 1, 1),
    )


@lru_cache(maxsize=None)
def compile_kernel(cluster_size: int):
    if cluster_size < 2:
        raise ValueError("cluster_size must be at least two")
    fake_dst = make_fake_compact_tensor(cutlass.Int32, (cluster_size,))
    return cute.compile(
        launch,
        fake_dst,
        cluster_size,
        options="--enable-tvm-ffi",
    )


def inspect_ptx() -> dict[str, list[str]]:
    ptx_files = sorted(ARTIFACT_DIR.glob("*.ptx"), key=lambda p: p.stat().st_mtime)
    if not ptx_files:
        raise RuntimeError("CUTE_DSL_KEEP=ptx did not produce a PTX artifact")
    lines: list[str] = []
    for path in ptx_files:
        lines.extend(line.strip() for line in path.read_text().splitlines())
    patterns = {
        "mapa": "mapa.shared::cluster",
        "cluster_arrive": "barrier.cluster.arrive",
        "cluster_wait": "barrier.cluster.wait",
    }
    result = {
        name: [line for line in lines if pattern in line]
        for name, pattern in patterns.items()
    }
    # Scope/order qualifiers are inserted between `arrive` and the address
    # space on current PTX (for example
    # mbarrier.arrive.release.cluster.shared::cluster.b64).
    result["remote_arrive"] = [
        line
        for line in lines
        if "mbarrier.arrive" in line and "shared::cluster" in line
    ]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="CuTe DSL DSMEM cluster ring")
    parser.add_argument("--cluster-size", type=int, default=2)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This example requires an NVIDIA GPU")
    major, minor = torch.cuda.get_device_capability()
    if major < 9:
        raise RuntimeError(f"CTA clusters require SM90+, got {major}.{minor}")

    compiled = compile_kernel(args.cluster_size)
    dst = torch.full(
        (args.cluster_size,), -1, dtype=torch.int32, device="cuda"
    )
    compiled(dst)
    torch.cuda.synchronize()
    expected = torch.tensor(
        [(rank + 1) % args.cluster_size for rank in range(args.cluster_size)],
        dtype=torch.int32,
        device="cuda",
    )
    torch.testing.assert_close(dst, expected, rtol=0, atol=0)

    evidence = inspect_ptx()
    missing = [name for name, lines in evidence.items() if not lines]
    if missing:
        raise AssertionError(f"missing expected cluster PTX families: {missing}")
    print(f"cluster_size={args.cluster_size}, dst={dst.tolist()}, PASS")
    for name, lines in evidence.items():
        print(f"PTX {name}: {lines[0]}")


if __name__ == "__main__":
    main()
