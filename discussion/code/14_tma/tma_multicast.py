"""Broadcast one TMA tile to both CTAs in a two-block cluster (SM100).

Only CTA rank 0 issues the load.  Its mbarrier expects descriptor_bytes * 2
because two multicast destinations each send transaction completion.  A
cluster barrier publishes the ready state to the non-leader CTA.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch


ARTIFACT_DIR = Path(__file__).with_name("generated_artifacts") / "multicast"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("CUTE_DSL_KEEP", "ptx")
os.environ.setdefault("CUTE_DSL_DUMP_DIR", str(ARTIFACT_DIR))

import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.experimental import primitives as prims


TILE_M = 128
TILE_K = 64
THREADS = 128
CLUSTER_SIZE = 2


@cute.kernel
def multicast_kernel(
    descriptor: cutlass.GridConstant[cuda.TensorMap],
    dst: cute.Tensor,
):
    smem = cutlass.Array(
        cutlass.Float16,
        TILE_M * TILE_K,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    barrier = cutlass.Array(
        cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8
    )
    tidx, _, _ = cute.arch.thread_idx()
    warp = tidx // 32
    lane = tidx % 32
    rank = cute.arch.block_idx_in_cluster()
    leader = rank == cutlass.Int32(0)

    # This example intentionally uses CTA_2 routing: both destinations report
    # completion to the leader CTA's barrier.
    if leader:
        if warp == 0:
            if prims.elect_sync():
                prims.mbarrier_init(barrier, 1)
    prims.fence_mbarrier_init()
    prims.barrier_cluster_arrive_relaxed()
    prims.barrier_cluster_wait()

    if leader:
        if warp == 0:
            if prims.elect_sync():
                prims.mbarrier_arrive_expect_tx(
                    barrier, descriptor.global_tx_bytes() * CLUSTER_SIZE
                )
            if prims.elect_sync():
                prims.cp_async_bulk_tensor_shared_cluster_global(
                    smem,
                    descriptor.get_ptr(),
                    (cutlass.Int32(0), cutlass.Int32(0)),
                    barrier,
                    [],
                    multicast_mask=cutlass.Int32(0b11),
                    group=prims.CTAGroup.CTA_2,
                )

    if leader:
        if warp == 0:
            while not prims.mbarrier_try_wait_parity(
                barrier, cutlass.Int32(0), time_limit=10_000_000
            ):
                pass

    # The non-leader does not wait on the remote mbarrier directly.  Cluster
    # rendezvous tells it that the leader observed both transaction completions.
    prims.barrier_cluster_arrive_relaxed()
    prims.barrier_cluster_wait()

    # TMA used 128-byte swizzling.  Convert physical SMEM columns back to the
    # public row-major coordinate system while each CTA writes its own slice.
    for col in cutlass.range_constexpr(TILE_K):
        row = warp * 32 + lane
        group = col // 8
        within = col % 8
        physical_col = (
            (cutlass.Int32(group) ^ (row & cutlass.Int32(7))) * cutlass.Int32(8)
            + cutlass.Int32(within)
        )
        value = (smem.data_ptr() + row * TILE_K + physical_col).load()
        dst[rank, row * TILE_K + col] = value


@cute.jit
def launch(src: cute.Tensor, dst: cute.Tensor, cluster_size: int):
    descriptor = cuda.create_tensor_map_tiled(
        global_address=src.iterator.toint(),
        dtype=cutlass.Float16,
        global_dims=[TILE_K, TILE_M],
        global_strides=[TILE_K * 2 // 16],
        box_dims=[TILE_K, TILE_M],
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    multicast_kernel(descriptor, dst).launch(
        grid=(cluster_size, 1, 1),
        block=(THREADS, 1, 1),
        cluster=(cluster_size, 1, 1),
    )


@lru_cache(maxsize=1)
def compile_kernel():
    fake_src = make_fake_compact_tensor(
        cutlass.Float16, (TILE_M, TILE_K), stride_order=(1, 0), assumed_align=16
    )
    fake_dst = make_fake_compact_tensor(
        cutlass.Float16,
        (CLUSTER_SIZE, TILE_M * TILE_K),
        stride_order=(1, 0),
        assumed_align=16,
    )
    return cute.compile(
        launch, fake_src, fake_dst, CLUSTER_SIZE, options="--enable-tvm-ffi"
    )


def inspect_ptx() -> dict[str, list[str]]:
    lines: list[str] = []
    for path in ARTIFACT_DIR.glob("*.ptx"):
        lines.extend(line.strip() for line in path.read_text().splitlines())
    patterns = {
        "multicast": ("cp.async.bulk.tensor", "multicast::cluster"),
        "expect_tx": ("mbarrier.arrive.expect_tx",),
        "cluster_arrive": ("barrier.cluster.arrive",),
        "cluster_wait": ("barrier.cluster.wait",),
    }
    return {
        name: [line for line in lines if all(token in line for token in tokens)]
        for name, tokens in patterns.items()
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires an NVIDIA GPU")
    major, minor = torch.cuda.get_device_capability()
    if major < 10:
        raise RuntimeError(
            f"this CTA_2 routing example targets SM100+, got {major}.{minor}"
        )

    src = torch.arange(TILE_M * TILE_K, dtype=torch.float16, device="cuda").reshape(TILE_M, TILE_K)
    dst = torch.full((CLUSTER_SIZE, TILE_M * TILE_K), float("nan"), dtype=torch.float16, device="cuda")
    compile_kernel()(src, dst, CLUSTER_SIZE)
    torch.cuda.synchronize()
    expected = src.reshape(1, -1).expand(CLUSTER_SIZE, -1)
    torch.testing.assert_close(dst, expected, rtol=0, atol=0)
    print(f"cluster={CLUSTER_SIZE}, tile=({TILE_M},{TILE_K}), both CTA copies exact")

    evidence = inspect_ptx()
    missing = [name for name, matches in evidence.items() if not matches]
    if missing:
        raise AssertionError(f"missing multicast PTX families: {missing}")
    for name, matches in evidence.items():
        print(f"PTX {name}: {matches[0]}")
    print("PASS")


if __name__ == "__main__":
    main()
