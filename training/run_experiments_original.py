#!/usr/bin/env python3
"""Run the KA-geometry experiments.

The script trains one-hidden-layer GeLU MLPs on the three synthetic regression
tasks xor, linear and random. It measures R2, Zero Rows, Participation Ratio,
Renyi concentration and KAGI on train and test splits.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class DatasetBundle:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor


@dataclass(frozen=True)
class MetricRow:
    function: str
    width: int
    seed: int
    epoch: int
    split: str
    k: int
    train_loss: float
    eval_loss: float
    eval_r2: float
    zero_fraction: float
    pr: float
    c_shannon: float
    c_renyi: float
    threshold: float
    init_zero_fraction: float
    init_c_renyi: float
    kagi_z_component_raw: float
    kagi_c_component_raw: float
    kagi_z_component: float
    kagi_c_component: float
    kagi_k_percent: float
    n_minor_entries: int
    n_minor_rows: int


@dataclass(frozen=True)
class KagiGlobalRow:
    function: str
    width: int
    seed: int
    epoch: int
    split: str
    kagi_global_percent: float


@dataclass(frozen=True)
class TargetMeanRow:
    function: str
    seed: int
    split: str
    y_mean: float


class ShallowMLP(nn.Module):
    """One-hidden-layer MLP matching the notation f_theta(x)=B GeLU(Ax+a)+b."""

    def __init__(self, n_input: int, width: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_input, width)
        self.act = nn.GELU(approximate="none")
        self.fc2 = nn.Linear(width, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.orthogonal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def hidden_preactivation(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc1(x)

    def internal_map(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.hidden_preactivation(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.internal_map(x)).squeeze(-1)


def parse_int_list(raw: str | list[int]) -> list[int]:
    if isinstance(raw, list):
        return raw
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_str_list(raw: str | list[str]) -> list[str]:
    if isinstance(raw, list):
        return raw
    return [part.strip() for part in raw.split(",") if part.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_targets(
    x: torch.Tensor,
    function_name: str,
    generator: torch.Generator,
    linear_c: torch.Tensor | None = None,
    linear_c0: torch.Tensor | None = None,
) -> torch.Tensor:
    if function_name == "xor":
        return torch.prod(torch.sin(math.pi * x), dim=1)
    if function_name == "linear":
        if linear_c is None or linear_c0 is None:
            raise ValueError("linear_c and linear_c0 are required for linear targets")
        return x @ linear_c + linear_c0
    if function_name == "random":
        return 2.0 * torch.rand(x.shape[0], generator=generator).to(x.device) - 1.0
    raise ValueError(f"Unknown function: {function_name}")


def make_dataset(
    function_name: str,
    n_input: int,
    n_train: int,
    n_test: int,
    seed: int,
    device: torch.device,
) -> DatasetBundle:
    generator = torch.Generator()
    generator.manual_seed(seed)

    x_train = (2.0 * torch.rand((n_train, n_input), generator=generator) - 1.0).to(device)
    x_test = (2.0 * torch.rand((n_test, n_input), generator=generator) - 1.0).to(device)

    linear_c = None
    linear_c0 = None
    if function_name == "linear":
        linear_c = (2.0 * torch.rand(n_input, generator=generator) - 1.0).to(device)
        linear_c0 = (2.0 * torch.rand((), generator=generator) - 1.0).to(device)

    y_train = make_targets(x_train, function_name, generator, linear_c, linear_c0)
    y_test = make_targets(x_test, function_name, generator, linear_c, linear_c0)
    return DatasetBundle(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)


def gelu_derivative(z: torch.Tensor) -> torch.Tensor:
    sqrt_two = math.sqrt(2.0)
    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
    normal_cdf = 0.5 * (1.0 + torch.erf(z / sqrt_two))
    normal_pdf = torch.exp(-0.5 * z * z) * inv_sqrt_2pi
    return normal_cdf + z * normal_pdf


def internal_jacobian(model: ShallowMLP, x: torch.Tensor) -> torch.Tensor:
    """Return J(x)=D Psi(x) with shape [batch, m, n]."""

    z = model.hidden_preactivation(x)
    sigma_prime = gelu_derivative(z)
    weight = model.fc1.weight
    return sigma_prime.unsqueeze(-1) * weight.unsqueeze(0)


def combinations_tensor(size: int, order: int, device: torch.device) -> torch.Tensor:
    combos = list(itertools.combinations(range(size), order))
    return torch.tensor(combos, dtype=torch.long, device=device)


def generalized_comb(n_value: float, order: int) -> float:
    if n_value < order:
        return 0.0
    return math.exp(
        math.lgamma(n_value + 1.0)
        - math.lgamma(order + 1.0)
        - math.lgamma(n_value - order + 1.0)
    )


def induced_zero_quantile(width: int, order: int, q_one: float) -> float:
    if order == 1:
        return q_one
    numerator = generalized_comb(width * (1.0 - q_one), order)
    denominator = math.comb(width, order)
    return max(0.0, min(1.0, 1.0 - numerator / denominator))


def empirical_lower_quantile(values: torch.Tensor, q_value: float) -> float:
    """Return an empirical lower-tail quantile without torch.quantile.

    torch.quantile can fail on large tensors in the cluster build. For the zero
    row threshold we only need an order statistic in the lower tail, so kthvalue
    is both simpler and closer to the empirical cutoff used in the paper.
    """

    flat = values.reshape(-1)
    n_values = flat.numel()
    if n_values == 0:
        raise ValueError("Cannot compute a quantile of an empty tensor")
    q_clipped = max(0.0, min(1.0, q_value))
    kth = max(1, min(n_values, math.ceil(q_clipped * n_values)))
    return float(torch.kthvalue(flat, kth).values.item())


def exterior_minors_abs(jacobian: torch.Tensor, order: int) -> torch.Tensor:
    """Return |wedge^k J(x)| with shape [batch, row_combos, col_combos]."""

    batch, width, n_input = jacobian.shape
    if order == 1:
        return jacobian.abs()

    row_combos = combinations_tensor(width, order, jacobian.device)
    col_combos = combinations_tensor(n_input, order, jacobian.device)
    selected_rows = jacobian[:, row_combos, :]
    values = []
    for col_combo in col_combos:
        block = selected_rows[:, :, :, col_combo]
        values.append(torch.linalg.det(block).abs())
    return torch.stack(values, dim=-1).reshape(batch, row_combos.shape[0], col_combos.shape[0])


@torch.no_grad()
def evaluate_loss_and_r2(
    model: ShallowMLP,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    loss_fn: nn.Module,
    batch_size: int,
) -> tuple[float, float]:
    preds = []
    losses = []
    for start in range(0, x_eval.shape[0], batch_size):
        stop = min(start + batch_size, x_eval.shape[0])
        pred = model(x_eval[start:stop])
        target = y_eval[start:stop]
        preds.append(pred)
        losses.append(loss_fn(pred, target).detach() * target.numel())
    y_pred = torch.cat(preds)
    eval_loss = torch.stack(losses).sum() / x_eval.shape[0]
    ss_res = torch.sum((y_eval - y_pred) ** 2)
    ss_tot = torch.sum((y_eval - torch.mean(y_eval)) ** 2).clamp_min(1e-12)
    r2 = 1.0 - ss_res / ss_tot
    return float(eval_loss.cpu()), float(r2.cpu())


@torch.no_grad()
def compute_raw_metric_state(
    model: ShallowMLP,
    x_eval: torch.Tensor,
    orders: Iterable[int],
    batch_size: int,
    thresholds: dict[int, float] | None = None,
) -> dict[int, dict[str, torch.Tensor | int]]:
    accum: dict[int, dict[str, torch.Tensor | float | int]] = {}
    for order in orders:
        width = model.fc1.out_features
        n_input = model.fc1.in_features
        n_rows = math.comb(width, order)
        n_cols = math.comb(n_input, order)
        accum[order] = {
            "row_sum": torch.zeros(n_rows, device=x_eval.device),
            "point_zero_count": torch.zeros(n_rows, device=x_eval.device),
            "pr_sum": 0.0,
            "c_shannon_sum": 0.0,
            "c_renyi_sum": 0.0,
            "samples": 0,
            "n_entries": n_rows * n_cols,
            "n_rows": n_rows,
        }

    eps = 1e-12
    for start in range(0, x_eval.shape[0], batch_size):
        stop = min(start + batch_size, x_eval.shape[0])
        x_batch = x_eval[start:stop]
        jacobian = internal_jacobian(model, x_batch)
        batch_n = x_batch.shape[0]

        for order in orders:
            minors = exterior_minors_abs(jacobian, order)
            row_per_point = minors.mean(dim=2)
            row_mean_batch = row_per_point.mean(dim=0)
            if thresholds is not None:
                accum[order]["point_zero_count"] += (row_per_point <= thresholds[order]).sum(dim=0)

            flat = minors.reshape(batch_n, -1)
            l1 = flat.sum(dim=1).clamp_min(eps)
            l2 = torch.linalg.vector_norm(flat, ord=2, dim=1).clamp_min(eps)
            pr = l1 / l2
            probs = flat / l1.unsqueeze(1)
            log_n = math.log(float(flat.shape[1]))
            h_shannon = -(probs * torch.log(probs.clamp_min(eps))).sum(dim=1)
            h_renyi = -torch.log(torch.sum(probs * probs, dim=1).clamp_min(eps))

            accum[order]["row_sum"] += row_mean_batch * batch_n
            accum[order]["pr_sum"] += float(pr.sum().cpu())
            accum[order]["c_shannon_sum"] += float((1.0 - h_shannon / log_n).sum().cpu())
            accum[order]["c_renyi_sum"] += float((1.0 - h_renyi / log_n).sum().cpu())
            accum[order]["samples"] += batch_n

    result: dict[int, dict[str, torch.Tensor | int]] = {}
    for order, values in accum.items():
        samples = int(values["samples"])
        result[order] = {
            "row_mean": values["row_sum"] / samples,
            "point_zero_fraction": values["point_zero_count"] / samples,
            "pr": torch.tensor(float(values["pr_sum"]) / samples),
            "c_shannon": torch.tensor(float(values["c_shannon_sum"]) / samples),
            "c_renyi": torch.tensor(float(values["c_renyi_sum"]) / samples),
            "n_entries": int(values["n_entries"]),
            "n_rows": int(values["n_rows"]),
        }
    return result


@torch.no_grad()
def compute_zero_thresholds(
    model: ShallowMLP,
    x_eval: torch.Tensor,
    orders: Iterable[int],
    batch_size: int,
    zero_quantile: float,
) -> dict[int, float]:
    """Compute Q^(k) from all initial row means rho_j^(k)(x_s).

    Freedman and Mulligan define zero rows pointwise, so the initialization
    quantile must be estimated over pairs (row, example), not over rows after
    averaging across the dataset.
    """

    values_by_order: dict[int, list[torch.Tensor]] = {order: [] for order in orders}
    for start in range(0, x_eval.shape[0], batch_size):
        stop = min(start + batch_size, x_eval.shape[0])
        jacobian = internal_jacobian(model, x_eval[start:stop])
        for order in orders:
            minors = exterior_minors_abs(jacobian, order)
            row_per_point = minors.mean(dim=2)
            values_by_order[order].append(row_per_point.reshape(-1).detach().cpu())

    thresholds: dict[int, float] = {}
    width = model.fc1.out_features
    for order, chunks in values_by_order.items():
        all_values = torch.cat(chunks)
        q_order = induced_zero_quantile(width, order, zero_quantile)
        thresholds[order] = empirical_lower_quantile(all_values, q_order)
    return thresholds


def zero_summary(raw: dict[str, torch.Tensor | int]) -> float:
    point_zero_fraction = raw["point_zero_fraction"]
    if not isinstance(point_zero_fraction, torch.Tensor):
        raise TypeError("Expected tensor-valued raw metric state")
    zero_fraction = point_zero_fraction.mean()
    return float(zero_fraction.cpu())


def compute_thresholds_and_init(
    model: ShallowMLP,
    x_eval: torch.Tensor,
    orders: list[int],
    metric_batch_size: int,
    zero_quantile: float,
) -> tuple[dict[int, float], dict[int, dict[str, float]]]:
    thresholds = compute_zero_thresholds(model, x_eval, orders, metric_batch_size, zero_quantile)
    init_with_threshold = compute_raw_metric_state(
        model,
        x_eval,
        orders,
        metric_batch_size,
        thresholds=thresholds,
    )
    init_metrics: dict[int, dict[str, float]] = {}
    for order in orders:
        zero_fraction = zero_summary(init_with_threshold[order])
        init_metrics[order] = {
            "zero_fraction": zero_fraction,
            "pr": float(init_with_threshold[order]["pr"].cpu()),
            "c_shannon": float(init_with_threshold[order]["c_shannon"].cpu()),
            "c_renyi": float(init_with_threshold[order]["c_renyi"].cpu()),
        }
    return thresholds, init_metrics


def clip_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def kagi_components(
    zero_fraction: float,
    c_renyi: float,
    init_zero_fraction: float,
    init_c_renyi: float,
    _context: str,
) -> tuple[float, float, float, float, float]:
    z_raw = (zero_fraction - init_zero_fraction) / (1.0 - init_zero_fraction)
    c_raw = (c_renyi - init_c_renyi) / (1.0 - init_c_renyi)
    z_component = clip_unit(z_raw)
    c_component = clip_unit(c_raw)

    return z_raw, c_raw, z_component, c_component, 100.0 * z_component * c_component


@torch.no_grad()
def measure_split(
    model: ShallowMLP,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    loss_fn: nn.Module,
    function_name: str,
    width: int,
    seed: int,
    epoch: int,
    split: str,
    train_loss: float,
    thresholds: dict[int, float],
    init_metrics: dict[int, dict[str, float]],
    orders: list[int],
    batch_size: int,
) -> list[MetricRow]:
    eval_loss, eval_r2 = evaluate_loss_and_r2(model, x_eval, y_eval, loss_fn, batch_size)
    raw = compute_raw_metric_state(model, x_eval, orders, batch_size, thresholds=thresholds)

    rows: list[MetricRow] = []
    for order in orders:
        zero_fraction = zero_summary(raw[order])
        pr = float(raw[order]["pr"].cpu())
        c_shannon = float(raw[order]["c_shannon"].cpu())
        c_renyi = float(raw[order]["c_renyi"].cpu())
        init_zero = init_metrics[order]["zero_fraction"]
        init_c_renyi = init_metrics[order]["c_renyi"]
        context = f"{function_name} m={width} seed={seed} epoch={epoch} split={split} k={order}"
        z_raw, c_raw, z_comp, c_comp, kagi_k_percent = kagi_components(
            zero_fraction,
            c_renyi,
            init_zero,
            init_c_renyi,
            context,
        )
        rows.append(
            MetricRow(
                function=function_name,
                width=width,
                seed=seed,
                epoch=epoch,
                split=split,
                k=order,
                train_loss=train_loss,
                eval_loss=eval_loss,
                eval_r2=eval_r2,
                zero_fraction=zero_fraction,
                pr=pr,
                c_shannon=c_shannon,
                c_renyi=c_renyi,
                threshold=thresholds[order],
                init_zero_fraction=init_zero,
                init_c_renyi=init_c_renyi,
                kagi_z_component_raw=z_raw,
                kagi_c_component_raw=c_raw,
                kagi_z_component=z_comp,
                kagi_c_component=c_comp,
                kagi_k_percent=kagi_k_percent,
                n_minor_entries=int(raw[order]["n_entries"]),
                n_minor_rows=int(raw[order]["n_rows"]),
            )
        )
    return rows


def make_global_kagi_rows(rows: Iterable[MetricRow]) -> list[KagiGlobalRow]:
    grouped: dict[tuple[str, int, int, int, str], list[MetricRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.function, row.width, row.seed, row.epoch, row.split)].append(row)

    out: list[KagiGlobalRow] = []
    for key, group in sorted(grouped.items()):
        numerator = sum(item.k * item.kagi_k_percent for item in group)
        denominator = sum(item.k for item in group)
        out.append(
            KagiGlobalRow(
                function=key[0],
                width=key[1],
                seed=key[2],
                epoch=key[3],
                split=key[4],
                kagi_global_percent=numerator / denominator,
            )
        )
    return out


def should_measure(epoch: int, epochs: int, measure_every: int) -> bool:
    return epoch == 0 or epoch == epochs or epoch % measure_every == 0


def train_one_configuration(
    args: argparse.Namespace,
    function_name: str,
    width: int,
    seed: int,
    device: torch.device,
) -> list[MetricRow]:
    set_seed(seed)
    orders = [order for order in args.orders if order <= min(width, args.n_input)]
    data = make_dataset(
        function_name=function_name,
        n_input=args.n_input,
        n_train=args.n_train,
        n_test=args.n_test,
        seed=seed,
        device=device,
    )

    generator = torch.Generator()
    generator.manual_seed(seed + 10_000)
    train_loader = DataLoader(
        TensorDataset(data.x_train, data.y_train),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )

    model = ShallowMLP(args.n_input, width).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    split_tensors = {
        "train": (data.x_train, data.y_train),
        "test": (data.x_test, data.y_test),
    }
    thresholds_by_split: dict[str, dict[int, float]] = {}
    init_metrics_by_split: dict[str, dict[int, dict[str, float]]] = {}
    for split_name, (x_split, _) in split_tensors.items():
        thresholds, init_metrics = compute_thresholds_and_init(
            model=model,
            x_eval=x_split,
            orders=orders,
            metric_batch_size=args.metric_batch_size,
            zero_quantile=args.zero_quantile,
        )
        thresholds_by_split[split_name] = thresholds
        init_metrics_by_split[split_name] = init_metrics

    all_rows: list[MetricRow] = []

    def measure_all_splits(epoch: int, train_loss: float) -> None:
        for split_name in args.splits:
            x_split, y_split = split_tensors[split_name]
            all_rows.extend(
                measure_split(
                    model=model,
                    x_eval=x_split,
                    y_eval=y_split,
                    loss_fn=loss_fn,
                    function_name=function_name,
                    width=width,
                    seed=seed,
                    epoch=epoch,
                    split=split_name,
                    train_loss=train_loss,
                    thresholds=thresholds_by_split[split_name],
                    init_metrics=init_metrics_by_split[split_name],
                    orders=orders,
                    batch_size=args.metric_batch_size,
                )
            )

    measure_all_splits(epoch=0, train_loss=float("nan"))

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu()) * x_batch.shape[0]
            seen += x_batch.shape[0]
        train_loss = running_loss / max(seen, 1)

        if should_measure(epoch, args.epochs, args.measure_every):
            model.eval()
            measure_all_splits(epoch=epoch, train_loss=train_loss)
            print(
                f"[{function_name} m={width} seed={seed}] epoch={epoch} "
                f"train_loss={train_loss:.6g}",
                flush=True,
            )

    return all_rows


def write_dataclass_rows(path: Path, rows: list[object], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def finite_mean_std(values: list[float]) -> tuple[float, float]:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return float("nan"), float("nan")
    mean = statistics.fmean(finite_values)
    std = statistics.stdev(finite_values) if len(finite_values) >= 2 else 0.0
    return mean, std


def make_target_mean_rows(args: argparse.Namespace, device: torch.device) -> list[TargetMeanRow]:
    rows: list[TargetMeanRow] = []
    for function_name in args.functions:
        for seed in args.seeds:
            data = make_dataset(
                function_name=function_name,
                n_input=args.n_input,
                n_train=args.n_train,
                n_test=args.n_test,
                seed=seed,
                device=device,
            )
            rows.append(
                TargetMeanRow(
                    function=function_name,
                    seed=seed,
                    split="train",
                    y_mean=float(data.y_train.mean().detach().cpu()),
                )
            )
            rows.append(
                TargetMeanRow(
                    function=function_name,
                    seed=seed,
                    split="test",
                    y_mean=float(data.y_test.mean().detach().cpu()),
                )
            )
    return rows


def write_target_mean_summary(path: Path, rows: list[TargetMeanRow]) -> None:
    grouped: dict[tuple[str, str], list[TargetMeanRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.function, row.split)].append(row)

    fieldnames = ["function", "split", "n_seeds", "y_mean_mean", "y_mean_std"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(grouped):
            group = grouped[key]
            mean, std = finite_mean_std([item.y_mean for item in group])
            writer.writerow(
                {
                    "function": key[0],
                    "split": key[1],
                    "n_seeds": len(group),
                    "y_mean_mean": mean,
                    "y_mean_std": std,
                }
            )


def write_metric_summary(path: Path, rows: list[MetricRow]) -> None:
    grouped: dict[tuple[str, int, int, str, int], list[MetricRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.function, row.width, row.epoch, row.split, row.k)].append(row)

    metrics = [
        "train_loss",
        "eval_loss",
        "eval_r2",
        "zero_fraction",
        "pr",
        "c_shannon",
        "c_renyi",
        "kagi_z_component_raw",
        "kagi_c_component_raw",
        "kagi_z_component",
        "kagi_c_component",
        "kagi_k_percent",
    ]
    fieldnames = ["function", "width", "epoch", "split", "k", "n_seeds"]
    for metric in metrics:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std"])

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(grouped):
            group = grouped[key]
            out: dict[str, float | int | str] = {
                "function": key[0],
                "width": key[1],
                "epoch": key[2],
                "split": key[3],
                "k": key[4],
                "n_seeds": len(group),
            }
            for metric in metrics:
                mean, std = finite_mean_std([getattr(item, metric) for item in group])
                out[f"{metric}_mean"] = mean
                out[f"{metric}_std"] = std
            writer.writerow(out)


def write_kagi_global_summary(path: Path, rows: list[KagiGlobalRow]) -> None:
    grouped: dict[tuple[str, int, int, str], list[KagiGlobalRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.function, row.width, row.epoch, row.split)].append(row)

    fieldnames = [
        "function",
        "width",
        "epoch",
        "split",
        "n_seeds",
        "kagi_global_percent_mean",
        "kagi_global_percent_std",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(grouped):
            group = grouped[key]
            mean, std = finite_mean_std([item.kagi_global_percent for item in group])
            writer.writerow(
                {
                    "function": key[0],
                    "width": key[1],
                    "epoch": key[2],
                    "split": key[3],
                    "n_seeds": len(group),
                    "kagi_global_percent_mean": mean,
                    "kagi_global_percent_std": std,
                }
            )


def subset_final_metrics(rows: list[MetricRow], final_epoch: int) -> list[MetricRow]:
    return [row for row in rows if row.epoch == final_epoch]


def subset_final_global(rows: list[KagiGlobalRow], final_epoch: int) -> list[KagiGlobalRow]:
    return [row for row in rows if row.epoch == final_epoch]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--functions", type=parse_str_list, default=["xor", "linear", "random"])
    parser.add_argument("--widths", type=parse_int_list, default=[8, 32])
    parser.add_argument("--seeds", type=parse_int_list, default=[0, 1, 2, 3])
    parser.add_argument("--orders", type=parse_int_list, default=[1, 2, 3])
    parser.add_argument("--splits", type=parse_str_list, default=["train", "test"])
    parser.add_argument("--n-input", type=int, default=3)
    parser.add_argument("--n-train", type=int, default=4096)
    parser.add_argument("--n-test", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--metric-batch-size", type=int, default=256)
    parser.add_argument("--measure-every", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--zero-quantile", type=float, default=0.01)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    invalid_splits = set(args.splits).difference({"train", "test"})
    if invalid_splits:
        raise ValueError(f"Unsupported splits: {sorted(invalid_splits)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **vars(args),
                "output_dir": str(args.output_dir),
                "optimizer": "adam",
                "device_resolved": str(device),
            },
            handle,
            indent=2,
        )

    print(f"Using device: {device}", flush=True)
    all_metric_rows: list[MetricRow] = []
    metric_fieldnames = list(MetricRow.__dataclass_fields__)
    target_mean_rows = make_target_mean_rows(args, device)
    write_dataclass_rows(
        args.output_dir / "target_means.csv",
        target_mean_rows,
        list(TargetMeanRow.__dataclass_fields__),
    )
    write_target_mean_summary(args.output_dir / "target_mean_summary.csv", target_mean_rows)

    for function_name in args.functions:
        for width in args.widths:
            for seed in args.seeds:
                rows = train_one_configuration(args, function_name, width, seed, device)
                all_metric_rows.extend(rows)
                write_dataclass_rows(
                    args.output_dir / "per_seed" / f"metrics_{function_name}_m{width}_seed{seed}.csv",
                    rows,
                    metric_fieldnames,
                )

    global_rows = make_global_kagi_rows(all_metric_rows)
    final_metric_rows = subset_final_metrics(all_metric_rows, args.epochs)
    final_global_rows = subset_final_global(global_rows, args.epochs)

    write_dataclass_rows(args.output_dir / "metrics_by_epoch.csv", all_metric_rows, metric_fieldnames)
    write_metric_summary(args.output_dir / "metrics_by_epoch_summary.csv", all_metric_rows)
    write_dataclass_rows(args.output_dir / "kagi_global_by_epoch.csv", global_rows, list(KagiGlobalRow.__dataclass_fields__))
    write_kagi_global_summary(args.output_dir / "kagi_global_by_epoch_summary.csv", global_rows)

    write_dataclass_rows(args.output_dir / "final_metrics.csv", final_metric_rows, metric_fieldnames)
    write_metric_summary(args.output_dir / "final_metrics_summary.csv", final_metric_rows)
    write_dataclass_rows(args.output_dir / "final_kagi_global.csv", final_global_rows, list(KagiGlobalRow.__dataclass_fields__))
    write_kagi_global_summary(args.output_dir / "final_kagi_global_summary.csv", final_global_rows)

    print(f"Wrote outputs under {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
