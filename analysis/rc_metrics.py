"""Pointwise alpha-RC metrics C^(k)_alpha(x) of a single-hidden-layer GeLU MLP.

For the MLP f(x) = B sigma(Ax + a) + b with the exact GeLU sigma(t) = t F(t), the Jacobian of the
inner map is J(x) = diag(sigma'(z(x))) A with z(x) = Ax + a, so every k x k minor factorizes as

    [J^(k)(x)]_{J,I} = (prod_{j in J} sigma'(z_j(x))) det(A_{J,I})

The determinants det(A_{J,I}) are computed once per network state and reused for every point.
All the minors are enumerated (no sampling, no threshold), in float64 and in logarithmic scale,
so that even the smallest minors keep full relative precision. With p_i = |minor_i| / sum_j |minor_j|
and N_k = C(m, k) C(n, k):

    C_alpha = 1 - H_alpha(p) / log N_k
    alpha = 1/2 : H = 2 log sum_i sqrt(p_i)
    alpha = 1   : H = -sum_i p_i log p_i            (Shannon)
    alpha = 2   : H = -log sum_i p_i^2              (collision)
    alpha = inf : H = -log max_i p_i                (min-entropy)
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.special import erfc, erfcx

ALPHAS = ("0.5", "1", "2", "inf")   # order of the last axis of every C array
KS = (1, 2, 3)                      # order of the k axis
BATCH = 512


def log_abs_gelu_derivative(z):
    """log |sigma'(z)| for the exact GeLU, sigma'(z) = F(z) + z f(z), stable in the negative tail."""
    zn = np.minimum(z, 0.0)
    ratio = math.sqrt(math.pi / 2) * erfcx(-zn / math.sqrt(2))
    with np.errstate(divide="ignore"):
        negative = -zn ** 2 / 2 - math.log(2 * math.pi) / 2 + np.log(np.abs(ratio + zn))
        zp = np.maximum(z, 0.0)
        gp = 0.5 * erfc(-zp / math.sqrt(2)) + zp * np.exp(-zp ** 2 / 2) / math.sqrt(2 * math.pi)
        positive = np.log(np.abs(gp))
    return np.where(z < 0, negative, positive)


def minor_geometry(weight):
    """For each k: row subsets J, log |det A_{J,I}| with shape (rows, cols), and N_k."""
    out = []
    for k in KS:
        rows = np.array(list(itertools.combinations(range(weight.shape[0]), k)))
        cols = np.array(list(itertools.combinations(range(weight.shape[1]), k)))
        if k == 1:
            minors = np.abs(weight)
        else:
            minors = np.stack([np.abs(np.linalg.det(weight[rows][:, :, c])) for c in cols], axis=-1)
        with np.errstate(divide="ignore"):
            log_minors = np.log(minors)
        out.append((k, rows, log_minors, math.comb(weight.shape[0], k) * math.comb(weight.shape[1], k)))
    return out


def concentrations_from_log_masses(log_masses):
    """C[B, 4] for alpha = 1/2, 1, 2, inf from the (B, N_k) log-magnitudes of the minors."""
    n = log_masses.shape[1]
    log_n = math.log(n)
    maximum = log_masses.max(axis=1, keepdims=True)
    if not np.isfinite(maximum).all():
        raise ArithmeticError("null or invalid minor distribution")
    centered = log_masses - maximum              # log(p_i / max_j p_j) <= 0
    e = np.exp(centered)
    total = e.sum(axis=1)
    log_total = np.log(total)                    # log p_i = centered - log_total
    c = np.empty((log_masses.shape[0], 4))
    c[:, 0] = 1 - (2 * np.log(np.sqrt(e).sum(axis=1)) - log_total) / log_n
    with np.errstate(invalid="ignore"):
        plogp = np.where(e > 0, e * (centered - log_total[:, None]), 0.0)
    c[:, 1] = 1 + plogp.sum(axis=1) / total / log_n
    c[:, 2] = 1 + (np.log(np.square(e).sum(axis=1)) - 2 * log_total) / log_n
    c[:, 3] = 1 - log_total / log_n
    return c


def unpack(theta, layout):
    """Weights of one network state from a row of parameters.npy and parameter_layout.json."""
    theta = np.asarray(theta, dtype=np.float64)
    return {item["name"]: theta[item["start"]:item["stop"]].reshape(item["shape"]) for item in layout}


def pointwise_concentrations(weight, bias, x):
    """C[points, k index, alpha index] for the inner map x -> sigma(weight x + bias)."""
    geo = minor_geometry(weight)
    out = np.empty((x.shape[0], len(KS), len(ALPHAS)))
    for start in range(0, x.shape[0], BATCH):
        xb = x[start:start + BATCH]
        log_g = log_abs_gelu_derivative(xb @ weight.T + bias)
        for ki, (k, rows, log_minors, n) in enumerate(geo):
            row_log_g = log_g[:, rows].sum(axis=2) if k > 1 else log_g
            log_masses = (row_log_g[:, :, None] + log_minors[None]).reshape(len(xb), n)
            out[start:start + BATCH, ki] = concentrations_from_log_masses(log_masses)
    return out
