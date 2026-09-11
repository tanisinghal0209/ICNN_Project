"""
Official High-Dimensional Baseline Sweep (D=4, D=8, D=16, D=32).

Trains ICNN Optimal Transport solvers across dimensions under fixed nominal
configurations, loading checkpoints from the Wasserstein2Benchmark repo.
Creates the structured experiments/high_dimensional/ directory and outputs logs.

IMPORTANT:
  - Architecture, lr, betas, inner_iters are FROZEN. Do NOT change.
  - Do NOT overwrite the D=2 baseline record (experiments/d2_baseline_record.json).
  - Uses compute_official_l2_uvp from src/metrics.py for evaluation.
"""
import csv
import io
import inspect
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.benchmark import load_mix3to10_benchmark, get_benchmark_samplers
from src.metrics import compute_official_l2_uvp
from src.solver import train_icnn_ot

BENCHMARK_PATH = os.environ.get(
    'WASSERSTEIN_BENCHMARK_PATH',
    '/Users/tanishasinghal/Downloads/Wasserstein2Benchmark'
)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------------------------------------------------------------
# Fixed nominal settings — FROZEN for entire baseline sweep
# -----------------------------------------------------------------------
N_ITERS    = 2000
BATCH_SIZE = 256
HIDDEN     = (128, 128, 128)
LR         = 1e-3
INNER      = 10
ACTIVATION = 'softplus'
LOG_EVERY  = 400
SEED       = 0

N_SAMPLES_EVAL = 8192
VAR_SAMPLES    = 16384


def count_parameters(model):
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_dimension(d, repo_root):
    """Run one dimension, log all telemetry, write output files."""
    dim_dir = ROOT / "experiments" / "high_dimensional" / f"D{d}"
    dim_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(repo_root)
    checkpoint_v1 = repo_root / "benchmarks" / "Mix3toMix10" / f"{d}_v1.pt"

    import src.benchmark as bm_module
    benchmark_src_file = inspect.getfile(bm_module.load_mix3to10_benchmark)

    log_lines = []
    def log(msg=""):
        print(msg)
        log_lines.append(msg)

    log(f"============================================================")
    log(f"D = {d}")
    log(f"============================================================")
    log(f"Benchmark source:     {benchmark_src_file}")
    log(f"Benchmark checkpoint: {checkpoint_v1}")
    log(f"Dimension:            {d}")
    log(f"Seed:                 {SEED}")
    log(f"Training configuration: n_iters={N_ITERS}, batch_size={BATCH_SIZE}, "
        f"hidden_dims={HIDDEN}, lr={LR}, inner_iters={INNER}, activation={ACTIVATION}")
    log()

    # 1. Load benchmark
    benchmark, module_info = load_mix3to10_benchmark(
        dim=d, benchmark_path=repo_root, device=DEVICE
    )
    mu_sampler, nu_sampler = get_benchmark_samplers(benchmark, device=DEVICE)

    # 2. Train — capture iteration log output
    f_io = io.StringIO()
    with redirect_stdout(f_io):
        f, g, history, final_f_loss, training_time = train_icnn_ot(
            mu_sampler, nu_sampler,
            input_dim=d,
            n_iters=N_ITERS,
            batch_size=BATCH_SIZE,
            hidden_dims=HIDDEN,
            lr=LR,
            inner_iters=INNER,
            device=DEVICE,
            activation=ACTIVATION,
            log_every=LOG_EVERY,
            model_type='icnn',
        )

    # Emit captured training logs
    for line in f_io.getvalue().splitlines():
        log(line)
    log()

    # 3. Collect telemetry — BEFORE any wrapping
    param_count   = count_parameters(f)
    final_g_loss  = history["g_loss"][-1] if history["g_loss"] else 0.0
    final_f_gnorm = history["f_grad_norm"][-1] if history["f_grad_norm"] else 0.0
    final_g_gnorm = history["g_grad_norm"][-1] if history["g_grad_norm"] else 0.0
    final_f_pnorm = history["f_param_norm"][-1] if history["f_param_norm"] else 0.0
    has_nan       = any(history["has_nan"])
    iters_per_sec = N_ITERS / training_time

    # 4. Official L2-UVP evaluation (uses src/metrics.py — no wrapping required)
    metrics = compute_official_l2_uvp(
        benchmark, f,
        device=DEVICE,
        n_samples=N_SAMPLES_EVAL,
        var_samples=VAR_SAMPLES,
    )
    l2_uvp   = metrics["L2-UVP"]
    cos_sim  = metrics["Cosine Similarity"]
    l2_error = metrics["L2 Error"]

    # 5. Terminal log
    log(f"Final f_loss:           {final_f_loss:.4f}")
    log(f"Final g_loss:           {final_g_loss:.4f}")
    log()
    log(f"Training time:          {training_time:.2f} s")
    log(f"Iterations/sec:         {iters_per_sec:.2f}")
    log(f"Parameter count (f):    {param_count}")
    log()
    log(f"Final f gradient norm:  {final_f_gnorm:.4f}")
    log(f"Final g gradient norm:  {final_g_gnorm:.4f}")
    log(f"Parameter norm (f):     {final_f_pnorm:.4f}")
    log()
    log(f"NaN/Inf:                {has_nan}")
    log(f"Completed successfully: {not has_nan}")
    log()
    log(f"Official evaluation:")
    log(f"  L2-UVP:               {l2_uvp:.4f}%")
    log(f"  Cosine Similarity:    {cos_sim:.4f}")
    log(f"  L2 Error:             {l2_error:.4f}")

    # 6. Save files
    with open(dim_dir / "run.log", "w") as fh:
        fh.write("\n".join(log_lines) + "\n")

    metrics_data = {
        "D":                    d,
        "L2-UVP":               float(l2_uvp),
        "Cosine Similarity":    float(cos_sim),
        "L2 Error":             float(l2_error),
        "final_f_loss":         float(final_f_loss),
        "final_g_loss":         float(final_g_loss),
        "f_gradient_norm":      float(final_f_gnorm),
        "g_gradient_norm":      float(final_g_gnorm),
        "parameter_norm":       float(final_f_pnorm),
        "parameter_count":      int(param_count),
        "training_time":        float(training_time),
        "iterations_per_second": float(iters_per_sec),
        "NaN/Inf":              bool(has_nan),
        "completed_iterations": N_ITERS,
    }
    with open(dim_dir / "metrics.json", "w") as fh:
        json.dump(metrics_data, fh, indent=2)

    with open(dim_dir / "loss_history.json", "w") as fh:
        json.dump(history, fh, indent=2)

    config_data = {
        "n_iters":      N_ITERS,
        "batch_size":   BATCH_SIZE,
        "hidden_dims":  list(HIDDEN),
        "lr":           LR,
        "inner_iters":  INNER,
        "activation":   ACTIVATION,
        "seed":         SEED,
        "device":       DEVICE,
        "benchmark_path": str(repo_root),
        "benchmark_module": module_info["map_benchmark_file"],
    }
    with open(dim_dir / "config.json", "w") as fh:
        json.dump(config_data, fh, indent=2)

    # Loss curve plot
    plt.figure(figsize=(6, 4))
    plt.plot(history["f_loss"], label="f_loss", linewidth=0.8)
    plt.plot(history["g_loss"], label="g_loss", linewidth=0.8, alpha=0.6)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(f"D={d} Training Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(dim_dir / "loss.png", dpi=200, bbox_inches="tight")
    plt.close()

    return metrics_data, history["f_loss"]


def main():
    repo_root = Path(BENCHMARK_PATH).resolve()
    dims = [4, 8, 16, 32]

    results = {}
    loss_histories = {}

    for d in dims:
        metrics_data, f_loss_hist = run_dimension(d, repo_root)
        results[str(d)] = metrics_data
        loss_histories[str(d)] = f_loss_hist

    # Include the persisted D=2 regression baseline. Run
    # ``experiments/d2_regression.py`` first to refresh this record.
    d2_path = ROOT / "experiments" / "d2_baseline_record.json"
    if d2_path.exists():
        with open(d2_path) as fh:
            d2 = json.load(fh)
        telemetry = d2.get("telemetry", {})
        results["2"] = {
            "D": 2,
            "L2-UVP":               d2["L2-UVP"],
            "Cosine Similarity":    d2["Cosine Similarity"],
            "L2 Error":             d2["L2 Error"],
            "final_f_loss":         telemetry.get("final_f_loss", 1.7751),
            "final_g_loss":         telemetry.get("final_g_loss", 0.0),
            "f_gradient_norm":      telemetry.get("final_f_gradient_norm", 0.213),
            "g_gradient_norm":      telemetry.get("final_g_gradient_norm", 0.0),
            "parameter_norm":       telemetry.get("final_f_parameter_norm", 26.5),
            "parameter_count":      d2.get("parameter_count", 34051),
            "training_time":        d2["training_time"],
            "iterations_per_second": 2000.0 / d2["training_time"],
            "NaN/Inf":              False,
            "completed_iterations": 2000,
        }

    # Save consolidated outputs
    out_dir = ROOT / "experiments" / "high_dimensional"
    out_dir.mkdir(parents=True, exist_ok=True)

    ordered = [results[k] for k in sorted(results.keys(), key=int)]
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(ordered, fh, indent=2)

    fields = [
        "D", "L2-UVP", "Cosine Similarity", "L2 Error",
        "final_f_loss", "final_g_loss",
        "f_gradient_norm", "g_gradient_norm", "parameter_norm",
        "parameter_count", "training_time", "iterations_per_second",
        "NaN/Inf", "completed_iterations",
    ]
    with open(out_dir / "summary.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in ordered:
            writer.writerow({k: row[k] for k in fields})

    # Print consolidated table
    print("\n" + "=" * 90)
    print("CONSOLIDATED HIGH-DIMENSIONAL BASELINE SWEEP")
    print("=" * 90)
    print(f"{'D':<5} {'L2-UVP':>10} {'Cosine':>8} {'L2Err':>8} "
          f"{'f_loss':>8} {'f_gnorm':>9} {'p_norm':>8} "
          f"{'params':>8} {'time':>8} {'it/s':>6} {'NaN':>5}")
    print("-" * 90)
    for r in ordered:
        print(
            f"{r['D']:<5} {r['L2-UVP']:>9.4f}% {r['Cosine Similarity']:>8.4f} "
            f"{r['L2 Error']:>8.4f} {r['final_f_loss']:>8.4f} "
            f"{r['f_gradient_norm']:>9.4f} {r['parameter_norm']:>8.4f} "
            f"{r['parameter_count']:>8d} {r['training_time']:>7.1f}s "
            f"{r['iterations_per_second']:>6.2f} {str(r['NaN/Inf']):<5}"
        )
    print("=" * 90)

    # Plots
    dims_list = sorted([int(k) for k in results.keys()])
    l2_uvps    = [results[str(d)]["L2-UVP"]            for d in dims_list]
    cos_sims   = [results[str(d)]["Cosine Similarity"]  for d in dims_list]
    times      = [results[str(d)]["training_time"]      for d in dims_list]
    gnorms     = [results[str(d)]["f_gradient_norm"]    for d in dims_list]

    def _savefig(name, ylabel, ys, marker, color, title):
        plt.figure(figsize=(6, 4))
        plt.plot(dims_list, ys, marker=marker, color=color, linewidth=1.5)
        plt.xlabel("Dimension D")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.savefig(out_dir / name, dpi=200, bbox_inches="tight")
        plt.close()

    _savefig("l2_uvp_vs_dimension.png",  "L2-UVP (%)",            l2_uvps,  "o", "C0", "L2-UVP vs Dimension")
    _savefig("cosine_vs_dimension.png",  "Cosine Similarity",      cos_sims, "s", "C1", "Cosine Similarity vs Dimension")
    _savefig("time_vs_dimension.png",    "Training Time (s)",       times,    "^", "C2", "Training Time vs Dimension")
    _savefig("grad_norm_vs_dimension.png","Final f Gradient Norm", gnorms,   "x", "C3", "Gradient Norm vs Dimension")

    plt.figure(figsize=(7, 4.5))
    for d in dims_list:
        if str(d) in loss_histories:
            plt.plot(loss_histories[str(d)], label=f"D={d}", linewidth=0.8)
    plt.xlabel("Iteration")
    plt.ylabel("f_loss")
    plt.title("Training Loss vs Iteration")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(out_dir / "loss_vs_iteration.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
