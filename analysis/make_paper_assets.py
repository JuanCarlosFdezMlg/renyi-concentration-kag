"""Build the tables, figures and reported numbers of Section 4 (Computational experiments).

Inputs (read only):
  data/runs/m{8,32}_seed{0..3}/   datasets, predictions and weights of every epoch
  <c-dir>/C_m{w}_seed{s}.npy      pointwise C^(k)_alpha(x), from analysis/compute_concentrations.py
                                  or downloaded from the release of this repository
Outputs:
  results/tables/table_R2.tex          rows of Table 2 (R^2 at epochs 0 and 500)
  results/tables/table_C_example.tex   rows of Table 3 (C^(k)_alpha and PR at x* and pooled)
  results/figures/C_mean_epochs_m32_k3.pdf   Figure 1
  results/figures/R2_magma_m32_k3.pdf        Figure 2
  results/paper_numbers.json           every number quoted in Section 4

Statistics:
  * R^2: mean +/- standard deviation across the four seeds (ddof = 1).
  * C_alpha and PR: pooled over the 1024 test points of the four seeds (4096 values per epoch),
    standard deviation with ddof = 1 in Table 3. The halos of Figure 2 use the population variance
    (ddof = 0), normalized by its maximum, so they do not depend on ddof.
  * PR is obtained at each point from C_2 through PR = N_k^{(1 - C_2)/2}. As an independent check, it is
    compared with PR = ||J^(k)||_1 / ||J^(k)||_2 computed from explicit k x k determinants of the
    Jacobian, rebuilt from the stored weights.

Usage, from the repository root:
  python analysis/make_paper_assets.py [--c-dir outputs/C]
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "data" / "runs"
OUT = ROOT / "results"
FIG = OUT / "figures"
TAB = OUT / "tables"

ALPHAS = ["0.5", "1", "2", "inf"]       # last axis of the C arrays
ALPHA_TEX = ["1/2", "1", "2", r"\infty"]
SEEDS = [0, 1, 2, 3]
WIDTHS = [8, 32]
KS = [1, 2, 3]
EPOCHS = np.arange(501)
EXAMPLE = 4096          # x*: first test point of seed 0, identical for m = 8 and m = 32
TEST = slice(4096, 5120)  # test sample_ids of every run
NTEST = 1024
SMALL_ROW = 1e-8        # hidden neurons whose input weights are below 1e-8 times the largest entry of A

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--c-dir", type=Path, default=ROOT / "outputs" / "C")
args = parser.parse_args()
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
numbers = {}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
    "legend.fontsize": 8.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.facecolor": "white", "pdf.fonttype": 42, "axes.unicode_minus": False,
})


def load_C(width, seed):
    """C[epoch, sample_id, k index, alpha index], alpha index 0..3 = 1/2, 1, 2, inf."""
    path = args.c_dir / f"C_m{width}_seed{seed}.npy"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run analysis/compute_concentrations.py "
                                "or download the C arrays from the release of this repository")
    C = np.load(path, mmap_mode="r")
    assert C.shape == (501, 5120, 3, 4), C.shape
    return C


# ---------------------------------------------------------------- Table 2: R^2
r2 = {}
for width in WIDTHS:
    for seed in SEEDS:
        run = RUNS / f"m{width}_seed{seed}"
        with np.load(run / "dataset.npz") as data:
            yy = data["y"].astype(np.float64)
        pp = np.load(run / "predictions.npy", mmap_mode="r")
        for split, sl in (("train", slice(0, 4096)), ("test", TEST)):
            y = yy[sl]
            r2[width, seed, split] = 1 - np.square(pp[:, sl].astype(np.float64) - y).sum(axis=1) \
                / np.square(y - y.mean()).sum()

split_name = {"train": "Training (4096)", "test": "Test (1024)"}
lines = []
numbers["R2"] = {}
for epoch in (0, 500):
    lines.append(rf"\multirow{{2}}{{*}}{{{epoch}}}")
    for i, split in enumerate(("train", "test")):
        cells = []
        for width in WIDTHS:
            vals = np.array([r2[width, s, split][epoch] for s in SEEDS])
            mu, sd = vals.mean(), vals.std(ddof=1)
            numbers["R2"][f"m{width}_{split}_epoch{epoch}"] = [mu, sd, vals.tolist()]
            cells.append(rf"${mu:.4f} \pm {sd:.4f}$")
        lines.append(("" if i == 0 else " ") + rf"& {split_name[split]} & " + " & ".join(cells) + r" \\")
    if epoch == 0:
        lines.append(r"\midrule")
(TAB / "table_R2.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

# ---------------------------------------------------------------- Table 3: C_alpha and PR at x* and pooled
with np.load(RUNS / "m8_seed0" / "dataset.npz") as d8, np.load(RUNS / "m32_seed0" / "dataset.npz") as d32:
    assert np.array_equal(d8["x"][EXAMPLE], d32["x"][EXAMPLE]) and d8["split"][EXAMPLE] == 1
    numbers["example"] = {"seed": 0, "sample_id": EXAMPLE, "x": d8["x"][EXAMPLE].astype(float).tolist(),
                          "y": float(d8["y"][EXAMPLE])}

pooled = {}     # (width, k, epoch) -> (mean[4], std[4]) over the 4096 pooled test values
point = {}      # (width, k, epoch) -> C(x*)[4]
pr_pooled = {}  # (width, k, epoch) -> (mean, std) of the pointwise PR
pr_point = {}   # (width, k, epoch) -> PR(x*)
pr_check = {}   # max relative error between PR from C_2 and PR from explicit determinants


def n_minors(width, k):
    return math.comb(width, k) * math.comb(3, k)


def pr_from_c2(c2, width, k):
    # Pointwise identity PR = N_k^{(1 - C_2)/2}, applied point by point, never to averaged values.
    return np.exp(0.5 * (1.0 - c2) * math.log(n_minors(width, k)))


_erf = np.vectorize(math.erf)


def direct_pr(width, seed, epoch, k):
    """PR = ||J^(k)||_1 / ||J^(k)||_2 from explicit k x k determinants of the Jacobian (test points)."""
    run = RUNS / f"m{width}_seed{seed}"
    theta = np.load(run / "parameters.npy", mmap_mode="r")[epoch].astype(np.float64)
    A = theta[:3 * width].reshape(width, 3)
    a = theta[3 * width:4 * width]
    with np.load(run / "dataset.npz") as d:
        x = d["x"][TEST].astype(np.float64)
    z = x @ A.T + a
    dsig = 0.5 * (1.0 + _erf(z / math.sqrt(2.0))) + z * np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    J = dsig[:, :, None] * A[None, :, :]                       # (points, m, n)
    rows = list(itertools.combinations(range(width), k))
    cols = list(itertools.combinations(range(3), k))
    out = np.empty(len(x))
    for start in range(0, len(x), 256):
        Jb = J[start:start + 256]
        sub = Jb[:, np.array(rows)[:, None, :, None], np.array(cols)[None, :, None, :]]
        dets = np.abs(np.linalg.det(sub)).reshape(len(Jb), -1)
        out[start:start + 256] = dets.sum(axis=1) / np.sqrt((dets ** 2).sum(axis=1))
    return out


for width in WIDTHS:
    point.update({(width, k, e): np.asarray(load_C(width, 0)[e, EXAMPLE, k - 1], dtype=np.float64)
                  for k in KS for e in (0, 500)})
    for k in KS:
        for e in (0, 500):
            vals = np.concatenate([np.asarray(load_C(width, s)[e, TEST, k - 1, :]) for s in SEEDS])
            assert vals.shape == (4 * NTEST, 4) and np.isfinite(vals).all()
            pooled[width, k, e] = (vals.mean(axis=0), vals.std(axis=0, ddof=1))
            prs = pr_from_c2(vals[:, 2], width, k)
            assert np.all(prs >= 1 - 1e-9) and np.all(prs <= math.sqrt(n_minors(width, k)) + 1e-9)
            pr_pooled[width, k, e] = (prs.mean(), prs.std(ddof=1))
            pr_point[width, k, e] = float(pr_from_c2(point[width, k, e][2], width, k))
            err = 0.0
            for s in SEEDS:
                direct = direct_pr(width, s, e, k)
                from_c = pr_from_c2(np.asarray(load_C(width, s)[e, TEST, k - 1, 2]), width, k)
                err = max(err, float(np.max(np.abs(direct / from_c - 1))))
            pr_check[f"m{width}_k{k}_epoch{e}"] = err
            assert err < 1e-5, pr_check


def fmt(v, scale):
    return f"{v * scale:.4f}"


def sci(v, e):
    return f"{v / 10 ** e:.2f}"


def cell_value(v):
    """Four decimals, exact zeros as 0 and values below 5e-5 in scientific notation."""
    if v == 0:
        return "0"
    if v < 5e-5:
        e = int(np.floor(np.log10(v)))
        return rf"${sci(v, e)} \times 10^{{{e}}}$"
    return fmt(v, 1.0)


def cell_pair(mu, sd):
    if mu == 0 and sd == 0:
        return "$0$"
    return rf"${fmt(mu, 1.0)} \pm {fmt(sd, 1.0)}$"


lines = []
numbers["C_example"] = {}
for e in (0, 500):
    lines.append(rf"\multicolumn{{6}}{{l}}{{\textit{{Epoch {e}{' (initialization)' if e == 0 else ' (end of training)'}}}}} \\")
    lines.append(r"\addlinespace[2pt]")
    for k in KS:
        for ai, a in enumerate(ALPHA_TEX):
            lab = rf"$C^{{({k})}}_{{{a}}}$"
            cells = []
            for width in WIDTHS:
                mu, sd = pooled[width, k, e]
                cx = point[width, k, e][ai]
                cells.append(cell_value(cx))
                cells.append(cell_pair(mu[ai], sd[ai]))
                numbers["C_example"][f"m{width}_k{k}_epoch{e}_alpha{ALPHAS[ai]}"] = {
                    "C_x": cx, "pooled_mean": mu[ai], "pooled_std_ddof1": sd[ai]}
            first = rf"\multirow{{5}}{{*}}{{{k}}}" if ai == 0 else ""
            lines.append(f"{first} & {lab} & " + " & ".join(cells) + r" \\")
        cells = []
        for width in WIDTHS:
            mu, sd = pr_pooled[width, k, e]
            cells.append(f"{pr_point[width, k, e]:.4f}")
            cells.append(rf"${mu:.4f} \pm {sd:.4f}$")
            numbers["C_example"][f"m{width}_k{k}_epoch{e}_PR"] = {
                "PR_x": pr_point[width, k, e], "pooled_mean": mu, "pooled_std_ddof1": sd,
                "upper_bound_sqrtN": math.sqrt(n_minors(width, k))}
        lines.append(rf" & $\mathrm{{PR}}^{{({k})}}$ & " + " & ".join(cells) + r" \\")
        if k < 3:
            lines.append(r"\cmidrule(lr){1-6}")
    if e == 0:
        lines.append(r"\midrule\midrule")  # double rule between the epoch-0 and epoch-500 blocks
(TAB / "table_C_example.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

# ---------------------------------------------------------------- trajectories m = 32, k = 3 (Figures 1 and 2)
mean = np.zeros((501, 4))
var = np.zeros((501, 4))       # population variance (ddof = 0) of the 4096 pooled test values
per_mu = np.zeros((4, 501, 4))
Cs = [load_C(32, s) for s in SEEDS]
for e in EPOCHS:
    vals = np.concatenate([np.asarray(C[e, TEST, 2, :]) for C in Cs])
    mean[e] = vals.mean(axis=0)
    var[e] = vals.var(axis=0)
    for s, C in enumerate(Cs):
        per_mu[s, e] = np.asarray(C[e, TEST, 2, :]).mean(axis=0)
r2_all = np.stack([r2[32, s, "test"] for s in SEEDS])
r2_mean = r2_all.mean(axis=0)

traj = {}
for ai, a in enumerate(ALPHAS):
    traj[a] = {"epoch0": mean[0, ai], "epoch500": mean[500, ai], "max": mean[:, ai].max(),
               "argmax": int(mean[:, ai].argmax()), "std0": np.sqrt(var[0, ai]),
               "std500": np.sqrt(var[500, ai]), "var_max": var[:, ai].max(),
               "var_argmax": int(var[:, ai].argmax())}
traj["R2_test_mean"] = {f"epoch{e}": r2_mean[e] for e in (0, 10, 19, 20, 25, 30, 35, 40, 50, 100, 200, 300, 400, 500)}
traj["C_at_epochs"] = {f"epoch{e}": mean[e].tolist() for e in (19, 20, 100, 200)}
traj["gap_alpha1_alphainf"] = {"epoch0": mean[0, 3] - mean[0, 1], "epoch500": mean[500, 3] - mean[500, 1],
                               "min": (mean[:, 3] - mean[:, 1]).min(), "max": (mean[:, 3] - mean[:, 1]).max()}
traj["seed_test_means_epoch500"] = {f"seed{s}": {"R2_test": r2[32, s, "test"][500],
                                                 "C2_test_mean": per_mu[s, 500, 2]} for s in SEEDS}
traj["R2_first_epoch_above"] = {str(t): int(np.argmax(r2_mean > t)) for t in (0.1, 0.25, 0.5, 0.75, 0.9)}
traj["C2_first_epoch_above"] = {str(t): int(np.argmax(mean[:, 2] > t)) for t in (0.2, 0.3, 0.4, 0.5, 0.6)}

# hidden neurons whose input weights are all below SMALL_ROW times the largest entry of A (m = 32)
small = np.zeros(501)
for s in SEEDS:
    theta = np.load(RUNS / f"m32_seed{s}" / "parameters.npy").astype(np.float64)
    A = np.abs(theta[:, :96].reshape(501, 32, 3))
    small += (A.max(axis=2) < SMALL_ROW * A.max(axis=(1, 2))[:, None]).sum(axis=1)
small /= len(SEEDS)
traj["small_input_weight_neurons"] = {
    "threshold_relative_to_max_abs_A": SMALL_ROW, "mean_over_seeds_at": {f"epoch{e}": small[e] for e in (227, 500)},
    "first_epoch_with_any": int(np.argmax(small > 0)), "max": small.max(), "argmax": int(small.argmax())}
numbers["trajectories_m32_k3"] = traj

colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
fig, ax = plt.subplots(figsize=(6.6, 3.7), layout="constrained")
for ai, a in enumerate(ALPHA_TEX):
    ax.plot(EPOCHS, mean[:, ai], color=colors[ai], ls="-", lw=1.6, label=rf"$\alpha={a}$")
ax.set(xlabel="Epoch", ylabel=r"Mean of $C^{(3)}_{\alpha}$", xlim=(0, 500), ylim=(-0.015, 0.85))
ax.grid(alpha=0.2, lw=0.5)
ax.legend(loc="lower right", ncols=4, frameon=True, framealpha=0.95, handlelength=1.6, columnspacing=1.2)
fig.savefig(FIG / "C_mean_epochs_m32_k3.pdf", metadata={"CreationDate": None})
plt.close(fig)

cmap = plt.get_cmap("magma")
norm = Normalize(0.0, 1.0)
prev = np.concatenate([mean[:1], mean[:-1]])   # halo colour: mean C_alpha of the previous epoch
var_max = var.max(axis=0)
AREA = 70.0
fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.1), sharex=True, sharey=True, layout="constrained")
for ai, ax in enumerate(axes.flat):
    halo = AREA * var[:, ai] / var_max[ai]
    ax.scatter(EPOCHS, r2_mean, s=halo, c=cmap(norm(prev[:, ai])), alpha=0.10, edgecolors="none",
               zorder=1, rasterized=True)
    ax.scatter(EPOCHS, r2_mean, s=2.6, c=mean[:, ai], cmap=cmap, norm=norm, edgecolors="none",
               zorder=2, rasterized=True)
    ax.set_title(rf"$\alpha={ALPHA_TEX[ai]}$")
    ax.set(xlim=(-8, 508), ylim=(-0.18, 1.05))
    ax.grid(alpha=0.15, lw=0.5)
    if ai >= 2:
        ax.set_xlabel("Epoch")
    if ai % 2 == 0:
        ax.set_ylabel(r"Mean test $R^2$")
    fr = [0.25, 0.5, 1.0]
    handles = [ax.scatter([], [], s=AREA * f, color="#68636f", alpha=0.25, edgecolors="none") for f in fr]
    ax.legend(handles, [f"{var_max[ai] * f:.2g}" for f in fr], title=r"Halo area: $\mathrm{Var}(C^{(3)}_{\alpha})$",
              loc="lower right", fontsize=7, title_fontsize=7, framealpha=0.95, labelspacing=0.6, borderpad=0.6)
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, shrink=0.92, pad=0.02, aspect=30)
cb.set_label(r"Mean of $C^{(3)}_{\alpha}$ (point colour)")
fig.savefig(FIG / "R2_magma_m32_k3.pdf", dpi=400, metadata={"CreationDate": None})
plt.close(fig)


def conv(o):
    if isinstance(o, dict):
        return {k: conv(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [conv(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


numbers["PR_direct_check_max_rel_error"] = pr_check
numbers["alpha_values"] = ALPHAS
(OUT / "paper_numbers.json").write_text(json.dumps(conv(numbers), indent=2), encoding="utf-8")
print("OK")
