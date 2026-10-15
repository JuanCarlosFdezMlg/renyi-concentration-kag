"""Exhaustive pointwise Renyi measurements for one XOR training trajectory."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from exact_metrics import ALPHAS, KS, prepare_geometry, pointwise_metrics


def json_save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def import_original():
    source = Path(__file__).with_name("run_experiments_original.py")
    spec = importlib.util.spec_from_file_location("tfg_original", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, source


def create_array(path, shape, dtype):
    out = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    out[:] = np.nan
    out.flush()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, choices=[8, 32], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metric-batch-size", type=int, default=1024)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite existing run: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    original, source = import_original()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA allocation required")
    original.set_seed(args.seed)
    data = original.make_dataset("xor", 3, 4096, 1024, args.seed, device)
    train_gen = torch.Generator().manual_seed(args.seed + 10000)
    loader = DataLoader(TensorDataset(data.x_train, data.y_train), batch_size=250,
                        shuffle=True, generator=train_gen)
    model = original.ShallowMLP(3, args.width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    x = torch.cat([data.x_train, data.x_test])
    y = torch.cat([data.y_train, data.y_test])
    y_numpy = y.cpu().numpy()
    np.savez(args.output / "dataset.npz", x=x.cpu().numpy(), y=y_numpy,
             sample_id=np.arange(5120, dtype=np.int32),
             split=np.array([0] * 4096 + [1] * 1024, dtype=np.uint8))
    layout = []
    offset = 0
    for name, parameter in model.named_parameters():
        layout.append({"name": name, "shape": list(parameter.shape), "start": offset,
                       "stop": offset + parameter.numel()})
        offset += parameter.numel()
    json_save(args.output / "parameter_layout.json", layout)
    config = {"function": "xor", "target": "product_i sin(pi*x_i)",
              "n_input": 3, "width": args.width, "seed": args.seed,
              "dataset_seed": args.seed, "training_shuffle_seed": args.seed + 10000,
              "epochs": args.epochs, "recorded_epochs": list(range(args.epochs + 1)),
              "n_train": 4096, "n_test": 1024, "batch_size": 250, "lr": 0.01,
              "optimizer": "Adam", "activation": "GeLU approximate=none",
              "k_values": list(KS), "alpha_values": list(ALPHAS),
              "C_axes": ["epoch", "sample_id", "k_index", "alpha_index"],
              "split_encoding": {"0": "train", "1": "test"},
              "train_sample_ids": [0, 4095], "test_sample_ids": [4096, 5119],
              "C_dtype": "float64", "training_dtype": "float32", "predictions_dtype": "float32",
              "R2_accumulation_dtype": "float64", "C_null_policy": "NaN retained and counted",
              "point_sampling": "none", "minor_sampling": "none", "epoch_sampling": "none",
              "source_commit": "30b1897060ba767027402b38941aa6e49bf225dc",
              "source_sha256": digest(source), "metric_code_sha256": digest(Path(__file__).with_name("exact_metrics.py")),
              "metric_batch_size": args.metric_batch_size,
              "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(0), "hostname": platform.node(),
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
              "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
              "tf32_matmul": False}
    json_save(args.output / "config.json", config)
    concentrations = create_array(args.output / "C.npy", (args.epochs + 1, 5120, 3, 4), np.float64)
    predictions = create_array(args.output / "predictions.npy", (args.epochs + 1, 5120), np.float32)
    weights = create_array(args.output / "parameters.npy", (args.epochs + 1, offset), np.float32)
    summaries = []
    audit_rows = []
    started = time.perf_counter()
    measurement_seconds = 0.0
    training_seconds = 0.0

    @torch.no_grad()
    def measure(epoch, train_loss):
        nonlocal measurement_seconds
        begin = time.perf_counter()
        model.eval()
        geometry = prepare_geometry(model)
        chunks = []
        pred_chunks = []
        batch_diagnostics = []
        for start in range(0, 5120, args.metric_batch_size):
            xb = x[start:start + args.metric_batch_size]
            c, diagnostics = pointwise_metrics(geometry, xb)
            chunks.append(c)
            pred_chunks.append(model(xb))
            batch_diagnostics.append(diagnostics)
        c_numpy = torch.cat(chunks).cpu().numpy()
        pred_numpy = torch.cat(pred_chunks).cpu().numpy()
        if int(torch.stack([d["invalid_distribution_count"] for d in batch_diagnostics]).sum().cpu()) != 0:
            raise ArithmeticError("Invalid nonfinite minor distribution; distinguish from a null distribution")
        if c_numpy.shape != (5120, 3, 4):
            raise AssertionError(c_numpy.shape)
        if not np.isfinite(pred_numpy).all():
            raise ArithmeticError("Nonfinite predictions")
        finite = np.isfinite(c_numpy)
        if np.isinf(c_numpy).any():
            raise ArithmeticError("Infinite concentration")
        if finite.any() and (c_numpy[finite].min() < -1e-10 or c_numpy[finite].max() > 1 + 1e-10):
            raise ArithmeticError("Concentration outside [0,1]")
        differences = np.diff(c_numpy, axis=-1)
        minimum_increment = float(np.nanmin(differences)) if np.isfinite(differences).any() else None
        if minimum_increment is not None and minimum_increment < -1e-10:
            raise ArithmeticError("Pointwise alpha monotonicity violation")
        concentrations[epoch] = c_numpy
        predictions[epoch] = pred_numpy
        weights[epoch] = torch.cat([p.detach().flatten() for p in model.parameters()]).cpu().numpy()
        for split, lower, upper in (("train", 0, 4096), ("test", 4096, 5120)):
            yy = y_numpy[lower:upper].astype(np.float64)
            pp = pred_numpy[lower:upper].astype(np.float64)
            ss_res = np.square(yy - pp).sum()
            ss_tot = np.square(yy - yy.mean()).sum()
            row = {"width": args.width, "seed": args.seed, "epoch": epoch, "split": split,
                   "n_samples": upper - lower, "r2": 1 - ss_res / ss_tot,
                   "mse": ss_res / (upper - lower), "training_pass_loss": train_loss}
            for ki, k in enumerate(KS):
                for ai, a in enumerate(ALPHAS):
                    values = c_numpy[lower:upper, ki, ai]
                    row[f"C_k{k}_a{a:g}_mean"] = float(np.nanmean(values)) if np.isfinite(values).any() else None
                    row[f"C_k{k}_a{a:g}_std_points"] = float(np.nanstd(values, ddof=1)) if np.isfinite(values).sum() > 1 else None
                    row[f"C_k{k}_a{a:g}_undefined"] = int(np.isnan(values).sum())
            summaries.append(row)
        audit = {"epoch": epoch, "points": 5120, "C_values_written": int(c_numpy.size),
                 "nonfinite_C_values": int((~finite).sum()),
                 "undefined_point_k_indices": np.argwhere(~finite[:, :, 0]).tolist(),
                 "minimum_alpha_increment": minimum_increment}
        # Keep diagnostic tensor names and values without depending on implementation internals.
        def serial(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().tolist()
            if isinstance(value, dict):
                return {str(k): serial(v) for k, v in value.items() if not str(k).endswith("_mask")}
            if isinstance(value, (tuple, list)):
                return [serial(v) for v in value]
            return value
        audit["batches"] = serial(batch_diagnostics)
        audit_rows.append(audit)
        torch.cuda.synchronize()
        measurement_seconds += time.perf_counter() - begin
        if epoch % 25 == 0 or epoch == args.epochs:
            concentrations.flush()
            predictions.flush()
            weights.flush()
            json_save(args.output / "progress.json", {"last_complete_epoch": epoch,
                "target_epoch": args.epochs, "elapsed_seconds": time.perf_counter() - started,
                "measurement_seconds": measurement_seconds, "training_seconds": training_seconds})
            print(json.dumps({"width": args.width, "seed": args.seed, "epoch": epoch,
                "test_r2": summaries[-1]["r2"], "elapsed": time.perf_counter() - started}), flush=True)

    measure(0, None)
    for epoch in range(1, args.epochs + 1):
        begin = time.perf_counter()
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(xb)
        torch.cuda.synchronize()
        training_seconds += time.perf_counter() - begin
        measure(epoch, total_loss / 4096)

    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    json_save(args.output / "audit.json", audit_rows)
    counts = {"epochs": args.epochs + 1, "samples_per_epoch": 5120,
              "C_values": int(concentrations.size), "R2_values": len(summaries),
              "undefined_C_values": sum(r["nonfinite_C_values"] for r in audit_rows),
              "measurement_seconds": measurement_seconds, "training_seconds": training_seconds,
              "total_seconds": time.perf_counter() - started,
              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "status": "COMPLETE"}
    json_save(args.output / "complete.json", counts)
    paths = sorted(p for p in args.output.iterdir() if p.is_file())
    manifest = "".join(f"{digest(p)}  {p.name}\n" for p in paths)
    (args.output / "SHA256SUMS.txt").write_text(manifest, encoding="utf-8")
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
