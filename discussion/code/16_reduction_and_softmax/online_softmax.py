"""Masked online Softmax with hierarchical (thread/warp/CTA) pair reduction.

The associative reduction state is (m, l), where m is the running maximum and
l is the sum of exp(x-m).  Runtime row lengths model causal/ragged masks; all
threads still participate in warp and CTA collectives with the identity pair.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch


ARTIFACT_DIR = Path(__file__).with_name("generated_artifacts") / "online_softmax"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("CUTE_DSL_KEEP", "ptx")
os.environ.setdefault("CUTE_DSL_DUMP_DIR", str(ARTIFACT_DIR))

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.experimental import primitives as prims


WARP_SIZE = 32
WARPS = 4
THREADS = WARP_SIZE * WARPS
NEG_LARGE = -3.402823466e38


@cute.jit
def combine_pair(m_a, l_a, m_b, l_b):
    m = cute.math.max(m_a, m_b)
    l = l_a * cute.math.exp(m_a - m, fastmath=True)
    l = l + l_b * cute.math.exp(m_b - m, fastmath=True)
    return m, l


@cute.jit
def warp_reduce_pair(m, l):
    for offset in (16, 8, 4, 2, 1):
        other_m = cute.arch.shuffle_sync_bfly(m, offset)
        other_l = cute.arch.shuffle_sync_bfly(l, offset)
        m, l = combine_pair(m, l, other_m, other_l)
    return m, l


@cute.kernel
def softmax_kernel(
    src: cute.Tensor,
    lengths: cute.Tensor,
    dst: cute.Tensor,
    stats: cute.Tensor,
):
    row, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    warp = tidx // WARP_SIZE
    lane = tidx % WARP_SIZE
    cols = src.shape[1]
    valid_cols = lengths[row]

    warp_max = cutlass.Array(
        cutlass.Float32, WARPS, space=cutlass.AddressSpace.smem, alignment=16
    )
    warp_sum = cutlass.Array(
        cutlass.Float32, WARPS, space=cutlass.AddressSpace.smem, alignment=16
    )

    # Thread-local online scan.  Masked positions are skipped, so the thread
    # retains the identity pair (NEG_LARGE, 0) when it owns no valid elements.
    m = cutlass.Float32(NEG_LARGE)
    l = cutlass.Float32(0.0)
    for col in cutlass.range(tidx, cols, cutlass.Int32(THREADS)):
        if col < valid_cols:
            x = src[row, col]
            new_m = cute.math.max(m, x)
            l = l * cute.math.exp(m - new_m, fastmath=True)
            l = l + cute.math.exp(x - new_m, fastmath=True)
            m = new_m

    m, l = warp_reduce_pair(m, l)
    if lane == 0:
        warp_max[warp] = m
        warp_sum[warp] = l
    prims.barrier_cta_sync(0)

    if warp == 0:
        cta_m = cutlass.Float32(NEG_LARGE)
        cta_l = cutlass.Float32(0.0)
        if lane < WARPS:
            cta_m = warp_max[lane]
            cta_l = warp_sum[lane]
        cta_m, cta_l = warp_reduce_pair(cta_m, cta_l)
        if lane == 0:
            warp_max[0] = cta_m
            warp_sum[0] = cta_l
            stats[row, 0] = cta_m
            stats[row, 1] = cta_l
    prims.barrier_cta_sync(0)

    row_m = warp_max[0]
    row_l = warp_sum[0]
    for col in cutlass.range(tidx, cols, cutlass.Int32(THREADS)):
        if col < valid_cols:
            dst[row, col] = cute.math.exp(src[row, col] - row_m, fastmath=True) / row_l
        else:
            dst[row, col] = cutlass.Float32(0.0)


@cute.jit
def launch(
    src: cute.Tensor,
    lengths: cute.Tensor,
    dst: cute.Tensor,
    stats: cute.Tensor,
):
    softmax_kernel(src, lengths, dst, stats).launch(
        grid=(src.shape[0], 1, 1), block=(THREADS, 1, 1)
    )


@lru_cache(maxsize=1)
def compile_kernel():
    rows = cute.sym_int64()
    cols = cute.sym_int64()
    fake_src = make_fake_compact_tensor(
        cutlass.Float32, (rows, cols), stride_order=(1, 0), assumed_align=16
    )
    fake_lengths = make_fake_compact_tensor(cutlass.Int32, (rows,), assumed_align=16)
    fake_dst = make_fake_compact_tensor(
        cutlass.Float32, (rows, cols), stride_order=(1, 0), assumed_align=16
    )
    fake_stats = make_fake_compact_tensor(
        cutlass.Float32, (rows, 2), stride_order=(1, 0), assumed_align=16
    )
    return cute.compile(
        launch,
        fake_src,
        fake_lengths,
        fake_dst,
        fake_stats,
        options="--enable-tvm-ffi",
    )


def inspect_ptx() -> dict[str, list[str]]:
    lines: list[str] = []
    for path in ARTIFACT_DIR.glob("*.ptx"):
        lines.extend(line.strip() for line in path.read_text().splitlines())
    patterns = {
        "shuffle": "shfl.sync.bfly",
        "exp": "ex2.approx",
        "shared_store": "st.shared",
        "cta_barrier": "barrier.sync",
    }
    return {
        name: [line for line in lines if pattern in line]
        for name, pattern in patterns.items()
    }


def reference(src: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cols = src.shape[1]
    mask = torch.arange(cols, device=src.device).reshape(1, cols) < lengths.reshape(-1, 1)
    masked = src.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(masked, dim=1).masked_fill(~mask, 0.0)
    row_max = masked.max(dim=1).values
    row_sum = torch.exp(masked - row_max[:, None]).masked_fill(~mask, 0.0).sum(dim=1)
    return probs, torch.stack((row_max, row_sum), dim=1)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires an NVIDIA GPU")

    compiled = compile_kernel()
    for rows, cols in ((1, 1), (5, 31), (7, 128), (9, 257), (4, 1000)):
        torch.manual_seed(2026 + cols)
        src = (torch.randn((rows, cols), device="cuda", dtype=torch.float32) * 7).contiguous()
        lengths = torch.linspace(1, cols, rows, device="cuda").round().to(torch.int32)
        dst = torch.full_like(src, float("nan"))
        stats = torch.full((rows, 2), float("nan"), dtype=torch.float32, device="cuda")
        compiled(src, lengths, dst, stats)
        torch.cuda.synchronize()
        expected, expected_stats = reference(src, lengths)
        torch.testing.assert_close(dst, expected, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(stats, expected_stats, rtol=2e-5, atol=2e-5)
        row_sums = dst.sum(dim=1)
        torch.testing.assert_close(row_sums, torch.ones_like(row_sums), rtol=2e-5, atol=2e-5)
        print(
            f"shape=({rows},{cols}), lengths={lengths.tolist()}, "
            f"max_error={(dst-expected).abs().max().item():.3e}: PASS"
        )

    evidence = inspect_ptx()
    missing = [name for name, matches in evidence.items() if not matches]
    if missing:
        raise AssertionError(f"missing online Softmax PTX families: {missing}")
    for name, matches in evidence.items():
        print(f"PTX {name}: {matches[0]}")
    print("PASS")


if __name__ == "__main__":
    main()

