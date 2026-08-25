"""Run and audit CUTLASS's fixed-shape CTA_2 tcgen05 teaching kernel.

The kernel implementation intentionally remains the repository's canonical
``experimental/primitives/tcgen05/2cta_mma_basic.py`` rather than duplicating
its cluster protocol here.  This runner fixes the validation matrix, reuses
one compiled specialization, checks numerical results, and requires retained
PTX evidence for the CTA_2 lifecycle.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

import torch


ARTIFACT_DIR = Path(__file__).with_name("generated_artifacts") / "cta2"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("CUTE_DSL_KEEP", "ptx")
os.environ.setdefault("CUTE_DSL_DUMP_DIR", str(ARTIFACT_DIR))


REPO_ROOT = Path(__file__).resolve().parents[3]
UPSTREAM = (
    REPO_ROOT
    / "examples/python/CuTeDSL/experimental/primitives/tcgen05/2cta_mma_basic.py"
)


def load_upstream():
    if not UPSTREAM.exists():
        raise FileNotFoundError(
            f"missing canonical CTA_2 example: {UPSTREAM}; run from a full CUTLASS checkout"
        )
    spec = importlib.util.spec_from_file_location("cutedsl_tcgen05_cta2", UPSTREAM)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {UPSTREAM}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inspect_ptx() -> None:
    ptx = "\n".join(path.read_text() for path in ARTIFACT_DIR.glob("*.ptx"))
    patterns = {
        "cluster TMA": ("cp.async.bulk.tensor", ".shared::cluster.global"),
        "CTA_2 alloc": ("tcgen05.alloc.cta_group::2",),
        "CTA_2 MMA": ("tcgen05.mma.cta_group::2.kind::f16",),
        "CTA_2 commit": ("tcgen05.commit.cta_group::2",),
        "TMEM load": ("tcgen05.ld.sync.aligned.32x32b",),
        "CTA_2 dealloc": ("tcgen05.dealloc.cta_group::2",),
        "cluster arrive": ("barrier.cluster.arrive",),
        "cluster wait": ("barrier.cluster.wait",),
    }
    for name, tokens in patterns.items():
        lines = [
            line.strip()
            for line in ptx.splitlines()
            if all(token in line for token in tokens)
        ]
        if not lines:
            raise AssertionError(f"missing {name} PTX tokens {tokens}")
        print(f"PTX {name}: {lines[0]}")


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires an NVIDIA GPU")
    major, minor = torch.cuda.get_device_capability()
    if major != 10:
        raise RuntimeError(f"CTA_2 tcgen05 requires datacenter Blackwell, got {major}.{minor}")

    upstream = load_upstream()
    compiled = upstream.compile(64)
    for m, n, k in ((256, 256, 64), (512, 256, 64), (256, 512, 64)):
        torch.manual_seed(m + n + k)
        a = torch.randint(-2, 2, (m, k), device="cuda", dtype=torch.int32).to(
            torch.float16
        )
        b = torch.randint(-2, 2, (n, k), device="cuda", dtype=torch.int32).to(
            torch.float16
        )
        c = torch.zeros((m, n), device="cuda", dtype=torch.float16)
        compiled(a, b, c, (m, n, k))
        torch.cuda.synchronize()
        reference = (a.float() @ b.float().T).to(torch.float16)
        torch.testing.assert_close(c, reference, rtol=1e-5, atol=0.1)
        max_error = float((c.float() - reference.float()).abs().max())
        clusters = (m // 256, n // 256)
        print(
            f"shape=({m},{n},{k}), cluster tiles={clusters}, "
            f"max_error={max_error:.3e}: PASS"
        )
    inspect_ptx()
    print("PASS")


if __name__ == "__main__":
    main()
