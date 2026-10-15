"""Compute every pointwise value C^(k)_alpha(x) of the paper from the stored weights.

For each of the eight networks (m in {8, 32}, seeds 0-3), each of the 501 recorded states
(epoch 0 = initialization), each of the 5120 points (sample_id 0-4095 train, 4096-5119 test),
each order k in {1, 2, 3} and each alpha in {1/2, 1, 2, inf}. No training is needed: the weights
of every epoch are in data/runs/*/parameters.npy (float32, promoted to float64 here).

Output: outputs/C/C_m{width}_seed{seed}.npy, float64, shape (501, 5120, 3, 4),
        indexed C[epoch, sample_id, k - 1, alpha index] with alpha index 0..3 = 1/2, 1, 2, inf.

Checks at every point: 0 <= C_{1/2} <= C_1 <= C_2 <= C_inf <= 1 (Proposition on monotonicity in alpha).
Resumable: a network whose output file exists is skipped.

Usage, from the repository root:
  python analysis/compute_concentrations.py [--workers 4] [--out outputs/C]
Run time on a laptop CPU: about one minute per network with m = 8 and about half an hour per
network with m = 32 (4960 minors of order 3 per point).
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from rc_metrics import ALPHAS, KS, pointwise_concentrations, unpack

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "data" / "runs"
WIDTHS = (8, 32)
SEEDS = (0, 1, 2, 3)
TOL = 1e-12


def run_network(width, seed, out):
    target = out / f"C_m{width}_seed{seed}.npy"
    if target.exists():
        return {"width": width, "seed": seed, "status": "already computed"}
    run = RUNS / f"m{width}_seed{seed}"
    layout = json.loads((run / "parameter_layout.json").read_text(encoding="utf-8"))
    with np.load(run / "dataset.npz") as d:
        x = d["x"].astype(np.float64)
    theta = np.load(run / "parameters.npy")
    began = time.monotonic()
    C = np.empty((theta.shape[0], x.shape[0], len(KS), len(ALPHAS)))
    for epoch in range(theta.shape[0]):
        w = unpack(theta[epoch], layout)
        C[epoch] = pointwise_concentrations(w["fc1.weight"], w["fc1.bias"], x)
    if not np.isfinite(C).all():
        raise ArithmeticError(f"m{width} seed{seed}: non-finite concentration")
    if C.min() < -TOL or C.max() > 1 + TOL:
        raise ArithmeticError(f"m{width} seed{seed}: concentration outside [0, 1]")
    if np.diff(C, axis=-1).min() < -TOL:
        raise ArithmeticError(f"m{width} seed{seed}: monotonicity in alpha violated")
    tmp = target.with_suffix(".tmp.npy")
    np.save(tmp, C)
    tmp.replace(target)
    return {"width": width, "seed": seed, "status": "computed", "seconds": round(time.monotonic() - began, 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "C")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tasks = [(w, s) for w in (32, 8) for s in SEEDS]       # largest networks first
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for res in pool.map(run_network, *zip(*tasks), [args.out] * len(tasks)):
            print(json.dumps(res), flush=True)
    print("OK")


if __name__ == "__main__":
    main()
