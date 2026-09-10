"""Oracle-supervised ICNN transport-map diagnostic.

This module is intentionally independent of the adversarial OT training loop.
For each sample ``x`` from the official Mix3ToMix10 input distribution, it uses
the benchmark's known forward map as a target and trains one ICNN potential:

    MSE(grad_x f(x), benchmark.map_fwd(x)).

The default configuration exactly matches the D=16/D=32 high-dimensional
baseline architecture and optimizer settings.  It writes only under
``experiments/oracle_regression`` and never changes existing experiment output.
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.benchmark import load_mix3to10_benchmark
from src.icnn import ICNN
from src.metrics import compute_official_l2_uvp


BENCHMARK_PATH = os.environ.get(
    "WASSERSTEIN_BENCHMARK_PATH",
    "/Users/tanishasinghal/Downloads/Wasserstein2Benchmark",
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# These match experiments/run_high_d_sweep.py for a controlled comparison.
DEFAULT_DIMS = (16, 32)
N_ITERS = 2000
BATCH_SIZE = 256
HIDDEN_DIMS = (128, 128, 128)
LR = 1e-3
BETAS = (0.5, 0.9)
ACTIVATION = "softplus"
SEED = 0
LOG_EVERY = 400
N_SAMPLES_EVAL = 8192
VAR_SAMPLES = 16384


def count_parameters(model):
    """Return the number of trainable model parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def compute_grad_norm(model):
    """Return the L2 norm of the currently populated parameter gradients."""
    squared_norm = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            squared_norm += parameter.grad.detach().norm(2).item() ** 2
    return squared_norm ** 0.5


def compute_param_norm(model):
    """Return the L2 norm of all model parameters."""
    squared_norm = 0.0
    for parameter in model.parameters():
        squared_norm += parameter.detach().norm(2).item() ** 2
    return squared_norm ** 0.5


def has_nan_or_inf(model, loss=None):
    """Check the loss, model parameters, and populated gradients for non-finite values."""
    if loss is not None and not torch.isfinite(loss).all():
        return True
    for parameter in model.parameters():
        if not torch.isfinite(parameter).all():
            return True
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            return True
    return False


def transport_gradient(model, x):
    """Compute grad_x f(x) while retaining its graph for parameter backpropagation."""
    if not x.requires_grad:
        raise ValueError("x must require gradients to compute the transport map.")
    potential = model(x)
    return torch.autograd.grad(potential.sum(), x, create_graph=True)[0]


def sample_oracle_batch(mu_sampler, target_map, batch_size, device):
    """Sample ``(x, T*(x))`` with a detached oracle target.

    The benchmark computes its map by differentiating its own fixed potential, so
    it receives a distinct leaf tensor.  This keeps the benchmark calculation
    out of the trainable ICNN graph while preserving the required graph through
    ``grad_x f(x)``.
    """
    x = mu_sampler(batch_size).to(device).detach().requires_grad_(True)
    target_input = x.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        target = target_map(target_input)
    target = target.detach()

    if target.shape != x.shape:
        raise ValueError(
            f"Oracle target shape {tuple(target.shape)} does not match input shape {tuple(x.shape)}."
        )
    return x, target


def train_oracle_icnn(
    mu_sampler,
    target_map,
    input_dim,
    n_iters=N_ITERS,
    batch_size=BATCH_SIZE,
    hidden_dims=HIDDEN_DIMS,
    lr=LR,
    betas=BETAS,
    device=None,
    activation=ACTIVATION,
    seed=SEED,
    log_every=LOG_EVERY,
):
    """Fit one clipped ICNN to the benchmark's paired oracle samples.

    Returns ``(f, history, final_oracle_mse, training_time)``.  The return
    contract deliberately contains only the potential being trained.
    """
    device = torch.device(DEVICE if device is None else device)
    torch.manual_seed(seed)

    f = ICNN(input_dim, hidden_dims, activation=activation).to(device)
    optimizer = torch.optim.Adam(f.parameters(), lr=lr, betas=betas)
    history = {
        "iteration": [],
        "oracle_mse": [],
        "f_grad_norm": [],
        "f_param_norm": [],
        "iteration_seconds": [],
        "elapsed_seconds": [],
        "has_nan": [],
    }
    start_time = time.perf_counter()

    for iteration in range(n_iters):
        iteration_start = time.perf_counter()
        x, target = sample_oracle_batch(mu_sampler, target_map, batch_size, device)
        prediction = transport_gradient(f, x)
        oracle_mse = F.mse_loss(prediction, target)

        optimizer.zero_grad(set_to_none=True)
        oracle_mse.backward()
        f_grad_norm = compute_grad_norm(f)
        non_finite = has_nan_or_inf(f, oracle_mse)

        # Stop before an optimizer update that would corrupt the saved model.
        if non_finite:
            elapsed = time.perf_counter() - start_time
            history["iteration"].append(iteration)
            history["oracle_mse"].append(float(oracle_mse.detach().cpu()))
            history["f_grad_norm"].append(float(f_grad_norm))
            history["f_param_norm"].append(float(compute_param_norm(f)))
            history["iteration_seconds"].append(time.perf_counter() - iteration_start)
            history["elapsed_seconds"].append(elapsed)
            history["has_nan"].append(True)
            print(f"Iteration {iteration:4d} | non-finite oracle MSE/gradient detected; stopping.")
            break

        optimizer.step()
        f.clip_weights()

        elapsed = time.perf_counter() - start_time
        history["iteration"].append(iteration)
        history["oracle_mse"].append(float(oracle_mse.detach().cpu()))
        history["f_grad_norm"].append(float(f_grad_norm))
        history["f_param_norm"].append(float(compute_param_norm(f)))
        history["iteration_seconds"].append(time.perf_counter() - iteration_start)
        history["elapsed_seconds"].append(elapsed)
        history["has_nan"].append(False)

        if iteration % log_every == 0 or iteration == n_iters - 1:
            print(
                f"Iteration {iteration:4d} | oracle_mse = {oracle_mse.item():.6f} "
                f"| f_gnorm = {f_grad_norm:.4f}"
            )

    training_time = time.perf_counter() - start_time
    if not history["oracle_mse"]:
        raise RuntimeError("Oracle training completed without recording a loss.")
    return f, history, history["oracle_mse"][-1], training_time


def gradient_statistics(history):
    """Summarize an f-gradient-norm trajectory without losing the full log."""
    values = torch.tensor(history["f_grad_norm"], dtype=torch.float64)
    return {
        "peak_f_gradient_norm": float(values.max()),
        "mean_f_gradient_norm": float(values.mean()),
        "std_f_gradient_norm": float(values.std(unbiased=False)),
        "final_f_gradient_norm": float(values[-1]),
    }


def _save_trajectory_plot(values, ylabel, title, output_path):
    plt.figure(figsize=(6.5, 4.25))
    plt.plot(range(len(values)), values, linewidth=0.9)
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _load_baseline(dimension):
    """Load existing baseline results read-only, if their files are present."""
    baseline_dir = ROOT / "experiments" / "high_dimensional" / f"D{dimension}"
    metrics_path = baseline_dir / "metrics.json"
    history_path = baseline_dir / "loss_history.json"
    if not metrics_path.exists() or not history_path.exists():
        return None
    with open(metrics_path) as handle:
        metrics = json.load(handle)
    with open(history_path) as handle:
        history = json.load(handle)
    return metrics, history


def _save_gradient_comparison(dimension, oracle_history, plots_dir):
    """Plot comparable f-parameter gradient norms when the baseline exists."""
    baseline = _load_baseline(dimension)
    if baseline is None:
        return False

    _, baseline_history = baseline
    plt.figure(figsize=(6.5, 4.25))
    plt.plot(baseline_history["f_grad_norm"], label="baseline two-potential training", alpha=0.8)
    plt.plot(oracle_history["f_grad_norm"], label="oracle MSE", alpha=0.8)
    plt.xlabel("Iteration")
    plt.ylabel("f parameter gradient norm")
    plt.title(f"D={dimension}: f-gradient norm comparison")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plots_dir / "f_grad_norm_vs_baseline.png", dpi=200)
    plt.close()
    return True


def run_dimension(
    dimension,
    benchmark_path,
    device=DEVICE,
    n_iters=N_ITERS,
    batch_size=BATCH_SIZE,
    hidden_dims=HIDDEN_DIMS,
    lr=LR,
    betas=BETAS,
    activation=ACTIVATION,
    seed=SEED,
    log_every=LOG_EVERY,
    eval_samples=N_SAMPLES_EVAL,
    var_samples=VAR_SAMPLES,
):
    """Run and save one D-dimensional oracle-regression diagnostic."""
    output_dir = ROOT / "experiments" / "oracle_regression" / f"D{dimension}"
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 76)
    print(f"ORACLE ICNN REGRESSION | D={dimension}")
    print("=" * 76)
    benchmark, module_info = load_mix3to10_benchmark(
        dim=dimension, benchmark_path=benchmark_path, device=device
    )

    def mu_sampler(size):
        return benchmark.input_sampler.sample(size)

    def target_map(x):
        return benchmark.map_fwd(x, nograd=True)

    f, history, final_oracle_mse, training_time = train_oracle_icnn(
        mu_sampler=mu_sampler,
        target_map=target_map,
        input_dim=dimension,
        n_iters=n_iters,
        batch_size=batch_size,
        hidden_dims=hidden_dims,
        lr=lr,
        betas=betas,
        device=device,
        activation=activation,
        seed=seed,
        log_every=log_every,
    )

    # This is the unchanged official evaluation protocol used by the baseline.
    official_metrics = compute_official_l2_uvp(
        benchmark,
        f,
        device=device,
        n_samples=eval_samples,
        var_samples=var_samples,
    )
    gradient_stats = gradient_statistics(history)
    completed_iterations = len(history["oracle_mse"])
    metrics = {
        "D": dimension,
        "training_objective": "MSE(grad_x f(x), benchmark.map_fwd(x))",
        "L2-UVP": float(official_metrics["L2-UVP"]),
        "Cosine Similarity": float(official_metrics["Cosine Similarity"]),
        "L2 Error": float(official_metrics["L2 Error"]),
        "final_oracle_mse": float(final_oracle_mse),
        **gradient_stats,
        "final_f_parameter_norm": float(history["f_param_norm"][-1]),
        "parameter_count_f": int(count_parameters(f)),
        "training_time": float(training_time),
        "iterations_per_second": float(completed_iterations / training_time),
        "NaN/Inf": bool(any(history["has_nan"])),
        "completed_iterations": completed_iterations,
    }
    config = {
        "D": dimension,
        "n_iters_requested": n_iters,
        "batch_size": batch_size,
        "hidden_dims": list(hidden_dims),
        "activation": activation,
        "optimizer": "Adam",
        "lr": lr,
        "betas": list(betas),
        "weight_clipping": "ICNN.Wz weights clamped to >= 0 after each step",
        "seed": seed,
        "device": str(device),
        "benchmark_path": str(Path(benchmark_path).resolve()),
        "benchmark_module": module_info["map_benchmark_file"],
        "oracle_target": "benchmark.map_fwd(x, nograd=True)",
        "transport_prediction": "grad_x f(x), create_graph=True",
        "evaluation_samples": eval_samples,
        "variance_samples": var_samples,
    }

    with open(output_dir / "config.json", "w") as handle:
        json.dump(config, handle, indent=2)
    with open(output_dir / "metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2)
    with open(output_dir / "training_log.json", "w") as handle:
        json.dump(history, handle, indent=2)
    torch.save(f.state_dict(), output_dir / "f_final.pt")

    _save_trajectory_plot(
        history["oracle_mse"], "Oracle MSE", f"D={dimension}: Oracle MSE", plots_dir / "oracle_mse.png"
    )
    _save_trajectory_plot(
        history["f_grad_norm"], "f parameter gradient norm", f"D={dimension}: Gradient Norm", plots_dir / "f_grad_norm.png"
    )
    _save_trajectory_plot(
        history["f_param_norm"], "f parameter norm", f"D={dimension}: Parameter Norm", plots_dir / "f_param_norm.png"
    )
    metrics["baseline_gradient_comparison_available"] = _save_gradient_comparison(
        dimension, history, plots_dir
    )
    with open(output_dir / "metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2)

    print(
        f"D={dimension} | L2-UVP: {metrics['L2-UVP']:.4f}% | "
        f"oracle MSE: {metrics['final_oracle_mse']:.6f} | "
        f"peak f grad: {metrics['peak_f_gradient_norm']:.4f} | "
        f"time: {metrics['training_time']:.1f}s"
    )
    return metrics


def _baseline_comparison_row(oracle_metrics):
    """Build one read-only side-by-side comparison row for the summary files."""
    dimension = oracle_metrics["D"]
    baseline = _load_baseline(dimension)
    row = {
        "D": dimension,
        "oracle_L2-UVP": oracle_metrics["L2-UVP"],
        "oracle_Cosine Similarity": oracle_metrics["Cosine Similarity"],
        "oracle_L2 Error": oracle_metrics["L2 Error"],
        "oracle_final_MSE": oracle_metrics["final_oracle_mse"],
        "oracle_peak_f_gradient_norm": oracle_metrics["peak_f_gradient_norm"],
        "oracle_mean_f_gradient_norm": oracle_metrics["mean_f_gradient_norm"],
        "oracle_NaN/Inf": oracle_metrics["NaN/Inf"],
    }
    if baseline is None:
        return row

    baseline_metrics, baseline_history = baseline
    baseline_gradients = torch.tensor(baseline_history["f_grad_norm"], dtype=torch.float64)
    row.update({
        "baseline_L2-UVP": baseline_metrics["L2-UVP"],
        "baseline_Cosine Similarity": baseline_metrics["Cosine Similarity"],
        "baseline_L2 Error": baseline_metrics["L2 Error"],
        "baseline_final_f_loss": baseline_metrics["final_f_loss"],
        "baseline_peak_f_gradient_norm": float(baseline_gradients.max()),
        "baseline_mean_f_gradient_norm": float(baseline_gradients.mean()),
        "baseline_NaN/Inf": baseline_metrics["NaN/Inf"],
    })
    return row


def save_summary(results):
    """Save machine-readable and concise human-readable experiment summaries."""
    output_dir = ROOT / "experiments" / "oracle_regression"
    output_dir.mkdir(parents=True, exist_ok=True)
    comparisons = [_baseline_comparison_row(result) for result in results]
    with open(output_dir / "summary.json", "w") as handle:
        json.dump(results, handle, indent=2)

    fieldnames = [
        "D", "oracle_L2-UVP", "oracle_Cosine Similarity", "oracle_L2 Error",
        "oracle_final_MSE", "oracle_peak_f_gradient_norm", "oracle_mean_f_gradient_norm",
        "oracle_NaN/Inf", "baseline_L2-UVP", "baseline_Cosine Similarity",
        "baseline_L2 Error", "baseline_final_f_loss", "baseline_peak_f_gradient_norm",
        "baseline_mean_f_gradient_norm", "baseline_NaN/Inf",
    ]
    with open(output_dir / "comparison.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(comparisons)

    report_lines = [
        "# Oracle-supervised ICNN diagnostic",
        "",
        "The D=16/D=32 ICNN uses the baseline 3x128 Softplus architecture, clipping, Adam "
        "(lr=1e-3, betas=(0.5, 0.9)), batch size 256, and 2,000 requested iterations. "
        "Only the objective is changed to MSE(grad_x f(x), T*(x)).",
        "",
        "| D | Method | L2-UVP (%) | Cosine | L2 Error | Final objective | Peak f-grad | Mean f-grad | NaN/Inf |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for oracle, comparison in zip(results, comparisons):
        report_lines.append(
            f"| {oracle['D']} | baseline two-potential training | "
            f"{comparison.get('baseline_L2-UVP', float('nan')):.4f} | "
            f"{comparison.get('baseline_Cosine Similarity', float('nan')):.4f} | "
            f"{comparison.get('baseline_L2 Error', float('nan')):.4f} | "
            f"{comparison.get('baseline_final_f_loss', float('nan')):.6f} | "
            f"{comparison.get('baseline_peak_f_gradient_norm', float('nan')):.4f} | "
            f"{comparison.get('baseline_mean_f_gradient_norm', float('nan')):.4f} | "
            f"{comparison.get('baseline_NaN/Inf', 'unavailable')} |"
        )
        report_lines.append(
            f"| {oracle['D']} | oracle MSE | {oracle['L2-UVP']:.4f} | "
            f"{oracle['Cosine Similarity']:.4f} | {oracle['L2 Error']:.4f} | "
            f"{oracle['final_oracle_mse']:.6f} | {oracle['peak_f_gradient_norm']:.4f} | "
            f"{oracle['mean_f_gradient_norm']:.4f} | {oracle['NaN/Inf']} |"
        )
    report_lines.extend([
        "",
        "Interpretation should use the complete trajectories and all three official transport metrics. "
        "Low oracle error with finite, materially calmer gradients supports the training-dynamics hypothesis; "
        "persistent oracle fitting failure without the second potential implicates the clipped ICNN setup as well.",
        "",
        "The two objectives have different units, so their losses are not compared directly. The transport metrics "
        "and f-parameter gradient-norm trajectories are the controlled comparisons.",
    ])
    with open(output_dir / "summary.md", "w") as handle:
        handle.write("\n".join(report_lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", type=int, nargs="+", default=list(DEFAULT_DIMS))
    parser.add_argument("--benchmark-path", default=BENCHMARK_PATH)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--n-iters", type=int, default=N_ITERS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--log-every", type=int, default=LOG_EVERY)
    parser.add_argument("--eval-samples", type=int, default=N_SAMPLES_EVAL)
    parser.add_argument("--var-samples", type=int, default=VAR_SAMPLES)
    return parser.parse_args()


def main():
    args = parse_args()
    benchmark_path = Path(args.benchmark_path).resolve()
    results = [
        run_dimension(
            dimension=dimension,
            benchmark_path=benchmark_path,
            device=args.device,
            n_iters=args.n_iters,
            batch_size=args.batch_size,
            lr=args.lr,
            log_every=args.log_every,
            eval_samples=args.eval_samples,
            var_samples=args.var_samples,
        )
        for dimension in args.dims
    ]
    save_summary(results)


if __name__ == "__main__":
    main()
