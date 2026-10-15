"""Exhaustive pointwise Renyi concentration of the hidden GeLU Jacobian.

For J(x) = diag(GeLU'(Ax+a)) A, each k-minor factors into its k
row derivatives and a k-minor of A. Every row/column combination is used.
All measurement arithmetic is float64, on the model's device. The ambient
N = binom(m,k)*binom(3,k) includes exact zero minors. No mass is thresholded
or clipped. Null exterior distributions have undefined C and return NaN.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import math

import torch


ALPHAS = (0.00039, 1.0, 2.0, 278.0)
KS = (1, 2, 3)


@dataclass(frozen=True)
class MinorGeometry:
    k: int
    row_combos: torch.Tensor
    col_combos: torch.Tensor
    log_abs_minors: torch.Tensor
    N: int
    zero_minor_count: torch.Tensor


@dataclass(frozen=True)
class Geometry:
    weight: torch.Tensor
    bias: torch.Tensor
    orders: tuple[MinorGeometry, ...]

    @property
    def N(self) -> tuple[int, ...]:
        return tuple(order.N for order in self.orders)


def log_abs_gelu_derivative(z: torch.Tensor) -> torch.Tensor:
    """Stable exact-GeLU derivative magnitude, including its negative tail.

    The erfcx formula is the one validated in run_xor_alpha.py. Clamping
    here only makes the inactive branch safe; it never clips derivatives.
    Both branches use fixed-shape operations to avoid CUDA mask syncs.
    """
    zn = z.clamp_max(0)
    ratio = math.sqrt(math.pi / 2) * torch.special.erfcx(-zn / math.sqrt(2))
    negative = -zn.square() / 2 - math.log(2 * math.pi) / 2 + (ratio + zn).abs().log()
    zp = z.clamp_min(0)
    gp = 0.5 * torch.erfc(-zp / math.sqrt(2)) + zp * torch.exp(-zp.square() / 2) / math.sqrt(2 * math.pi)
    return torch.where(z < 0, negative, gp.abs().log())


def _combos(n: int, k: int, device: torch.device) -> torch.Tensor:
    return torch.tensor(list(itertools.combinations(range(n), k)), dtype=torch.long, device=device)


@torch.no_grad()
def prepare_geometry(model: torch.nn.Module) -> Geometry:
    """Snapshot fc1 parameters and factor all weight minors once/checkpoint.

    The snapshot is detached and cloned, and is unaffected by subsequent
    optimizer steps. Expected model: three inputs, width 8 or 32, GeLU.
    """
    weight = model.fc1.weight.detach().to(dtype=torch.float64).clone()
    bias = model.fc1.bias.detach().to(dtype=torch.float64).clone()
    if weight.ndim != 2 or weight.shape[1] != 3 or weight.shape[0] not in (8, 32):
        raise ValueError("Expected fc1 weight shape (8,3) or (32,3)")
    if bias.shape != (weight.shape[0],):
        raise ValueError("Unexpected hidden bias shape")
    if not bool(torch.isfinite(weight).all() & torch.isfinite(bias).all()):
        raise ValueError("Cannot evaluate a nonfinite model checkpoint")
    orders = []
    for k in KS:
        rows = _combos(weight.shape[0], k, weight.device)
        cols = _combos(weight.shape[1], k, weight.device)
        if k == 1:
            abs_minors = weight.abs()
        else:
            selected_rows = weight[rows, :]
            abs_minors = torch.stack([
                torch.linalg.det(selected_rows[:, :, col]).abs() for col in cols
            ], dim=-1)
        n = math.comb(weight.shape[0], k) * math.comb(weight.shape[1], k)
        orders.append(MinorGeometry(k, rows, cols, abs_minors.log(), n, (abs_minors == 0).sum()))
    return Geometry(weight, bias, tuple(orders))


@torch.no_grad()
def concentrations_from_logmasses(log_masses: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Evaluate the four fixed orders in one normalization pass.

    Accepts float64 (B,N), N>1, on CPU/CUDA. Exact zeros are -inf.
    Diagnostics retain device tensors; the caller chooses when to sync.
    Invalid rows (NaN or +inf masses) and null rows return four NaNs.
    """
    if log_masses.ndim != 2 or log_masses.shape[1] <= 1:
        raise ValueError("Expected log masses of shape (B,N), N>1")
    if log_masses.dtype != torch.float64:
        raise ValueError("Measurement log masses must be float64")
    maximum = log_masses.amax(dim=1)
    null = torch.isneginf(maximum)
    invalid = torch.isnan(maximum) | torch.isposinf(maximum)
    valid = ~(null | invalid)
    centered = log_masses - torch.where(valid, maximum, 0)[:, None]
    log_total = torch.logsumexp(centered, dim=1)
    logp = centered - torch.where(valid, log_total, 0)[:, None]
    log_n = math.log(log_masses.shape[1])

    # Sequential order evaluation avoids a (B,4,N) temporary allocation.
    result = torch.empty((logp.shape[0], 4), dtype=torch.float64, device=logp.device)
    result[:, 0] = 1 - torch.logsumexp(ALPHAS[0] * logp, dim=1) / ((1 - ALPHAS[0]) * log_n)
    p = logp.exp()
    safe_logp = torch.where(torch.isfinite(logp), logp, 0)
    result[:, 1] = 1 + (p * safe_logp).sum(dim=1) / log_n
    result[:, 2] = 1 + p.square().sum(dim=1).log() / log_n
    max_logp = logp.amax(dim=1)
    # max centering keeps the upper-order exponent at most zero.
    centered_p = logp - torch.where(valid, max_logp, 0)[:, None]
    residual = torch.logsumexp(ALPHAS[3] * centered_p, dim=1)
    result[:, 3] = 1 + max_logp / log_n + (max_logp + residual) / ((ALPHAS[3] - 1) * log_n)
    result.masked_fill_(~valid[:, None], math.nan)

    diagnostics = {
        "support": torch.isfinite(log_masses).sum(dim=1),
        "null_mask": null,
        "invalid_mask": invalid,
        "probability_normalization_error": torch.where(valid, (p.sum(dim=1) - 1).abs(), 0),
    }
    return result, diagnostics


@torch.no_grad()
def pointwise_metrics(geometry: Geometry, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return exhaustive C[B,k,alpha] and batch audit tensors.

    k axis is (1,2,3); alpha axis is ALPHAS. x must be a nonempty batch
    with three columns, on the checkpoint device. Batch 512/1024 keeps
    temporary memory bounded on A100/B200 GPUs even for m=32,k=3.
    """
    if x.ndim != 2 or x.shape[1] != 3 or x.shape[0] == 0:
        raise ValueError("Expected a nonempty input batch of shape (B,3)")
    if x.device != geometry.weight.device:
        raise ValueError("Inputs and geometry must share a device")
    z = x.to(dtype=torch.float64) @ geometry.weight.T + geometry.bias
    log_g = log_abs_gelu_derivative(z)
    values = []
    audits = []
    for order in geometry.orders:
        if order.k == 1:
            row_log_g = log_g
        else:
            # Sum selected columns without allocating (B,rows,k).
            row_log_g = log_g.index_select(1, order.row_combos[:, 0])
            for j in range(1, order.k):
                row_log_g.add_(log_g.index_select(1, order.row_combos[:, j]))
        log_masses = (row_log_g[:, :, None] + order.log_abs_minors[None, :, :]).reshape(x.shape[0], order.N)
        c, audit = concentrations_from_logmasses(log_masses)
        values.append(c)
        audits.append(audit)

    null_mask = torch.stack([a["null_mask"] for a in audits], dim=1)
    invalid_mask = torch.stack([a["invalid_mask"] for a in audits], dim=1)
    diagnostics = {
        "support_min": torch.stack([a["support"].min() for a in audits]),
        "support_max": torch.stack([a["support"].max() for a in audits]),
        "null_distribution_count": null_mask.sum(dim=0),
        "invalid_distribution_count": invalid_mask.sum(dim=0),
        "weight_minor_exact_zeros": torch.stack([o.zero_minor_count for o in geometry.orders]),
        "probability_normalization_max_abs_error": torch.stack([a["probability_normalization_error"].max() for a in audits]),
        "null_mask": null_mask,
        "invalid_mask": invalid_mask,
        "preactivation_min": z.min(),
        "preactivation_max": z.max(),
    }
    return torch.stack(values, dim=1), diagnostics
