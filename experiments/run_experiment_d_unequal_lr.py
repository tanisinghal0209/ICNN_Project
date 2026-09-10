"""Experiment D: minimax stabilization through unequal player learning rates.

The only varied hyperparameter is the learning rate of the maximising ICNN
potential.  This script does not alter the baseline training implementation or
any prior experiment output.  It recreates the baseline loop locally so that
each configuration differs only in ``g_lr``.
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.benchmark import get_benchmark_samplers, load_mix3to10_benchmark
from src.icnn import ICNN
from src.losses import minimax_loss
from src.metrics import compute_official_l2_uvp
from src.solver import check_nan_inf, compute_grad_norm, compute_param_norm


BENCHMARK_PATH = os.environ.get(
    "WASSERSTEIN_BENCHMARK_PATH",
    "/Users/tanishasinghal/Downloads/Wasserstein2Benchmark",
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# The default is the original D=16 high-dimensional baseline.  ``--dimension``
# supports a controlled replication at another existing baseline dimension.
DIM = 16
N_ITERS = 2000
BATCH_SIZE = 256
HIDDEN_DIMS = (128, 128, 128)
ACTIVATION = "softplus"
BETAS = (0.5, 0.9)
INNER_ITERS = 10
SEED = 0
LOG_EVERY = 400
N_SAMPLES_EVAL = 8192
VAR_SAMPLES = 16384


CONFIGS = (
    ("Baseline", 1e-3, 1e-3),
    ("G-slower-2x", 1e-3, 5e-4),
    ("G-slower-4x", 1e-3, 2.5e-4),
)


def make_optimizers(f, g, f_lr, g_lr):
    """Create the two baseline Adam optimizers with independently set rates."""
    opt_f = torch.optim.Adam(f.parameters(), lr=f_lr, betas=BETAS)
    opt_g = torch.optim.Adam(g.parameters(), lr=g_lr, betas=BETAS)
    return opt_f, opt_g


def train_icnn_ot_unequal_lr(
    mu_sampler,
    nu_sampler,
    input_dim,
    f_lr,
    g_lr,
    n_iters=N_ITERS,
    batch_size=BATCH_SIZE,
    hidden_dims=HIDDEN_DIMS,
    inner_iters=INNER_ITERS,
    device=None,
    activation=ACTIVATION,
    seed=SEED,
    log_every=LOG_EVERY,
):
    """Run the original two-potential loop, varying only f/g optimizer rates.

    Returns ``(f, g, history, final_f_loss, training_time)`` so the output is
    directly comparable with :func:`src.solver.train_icnn_ot`.
    """
    device = torch.device(DEVICE if device is None else device)
    torch.manual_seed(seed)

    f = ICNN(input_dim, hidden_dims, activation=activation).to(device)
    g = ICNN(input_dim, hidden_dims, activation=activation).to(device)
    opt_f, opt_g = make_optimizers(f, g, f_lr=f_lr, g_lr=g_lr)
    history = {
        "f_loss": [],
        "g_loss": [],
        "f_grad_norm": [],
        "g_grad_norm": [],
        "f_param_norm": [],
        "g_param_norm": [],
        "iteration_seconds": [],
        "elapsed_seconds": [],
        "has_nan": [],
    }
    start_time = time.perf_counter()

    for iteration in range(n_iters):
        iteration_start = time.perf_counter()
        last_g_loss = 0.0
        g_grad_norm = 0.0

        # This is the existing inner maximisation loop, with only opt_g.lr changed.
        for inner_index in range(inner_iters):
            x_mu = mu_sampler(batch_size).to(device)
            y_nu = nu_sampler(batch_size).to(device)
            _, g_loss = minimax_loss(f, g, x_mu, y_nu)
            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            if inner_index == inner_iters - 1:
                g_grad_norm = compute_grad_norm(g)
                last_g_loss = g_loss.item()
            opt_g.step()
            g.clip_weights()

        # This is the existing outer minimisation loop, with only opt_f.lr changed.
        x_mu = mu_sampler(batch_size).to(device)
        y_nu = nu_sampler(batch_size).to(device)
        f_loss, _ = minimax_loss(f, g, x_mu, y_nu)
        opt_f.zero_grad(set_to_none=True)
        f_loss.backward()
        f_grad_norm = compute_grad_norm(f)
        opt_f.step()
        f.clip_weights()

        elapsed = time.perf_counter() - start_time
        history["f_loss"].append(float(f_loss.item()))
        history["g_loss"].append(float(last_g_loss))
        history["f_grad_norm"].append(float(f_grad_norm))
        history["g_grad_norm"].append(float(g_grad_norm))
        history["f_param_norm"].append(float(compute_param_norm(f)))
        history["g_param_norm"].append(float(compute_param_norm(g)))
        history["iteration_seconds"].append(float(time.perf_counter() - iteration_start))
        history["elapsed_seconds"].append(float(elapsed))
        history["has_nan"].append(bool(check_nan_inf(f) or check_nan_inf(g)))

        if iteration % log_every == 0 or iteration == n_iters - 1:
            print(
                f"Iteration {iteration:4d} | f_loss = {f_loss.item():.4f} | "
                f"g_loss = {last_g_loss:.4f} | f_gnorm = {f_grad_norm:.4f} | "
                f"g_gnorm = {g_grad_norm:.4f}"
            )

    training_time = time.perf_counter() - start_time
    return f, g, history, history["f_loss"][-1], training_time


def trajectory_statistics(values, prefix):
    """Return final, peak, mean, and population standard deviation for a trace."""
    trace = torch.tensor(values, dtype=torch.float64)
    return {
        f"final_{prefix}": float(trace[-1]),
        f"peak_{prefix}": float(trace.max()),
        f"mean_{prefix}": float(trace.mean()),
        f"std_{prefix}": float(trace.std(unbiased=False)),
    }


def _save_plot(values, ylabel, title, output_path):
    plt.figure(figsize=(6.5, 4.25))
    plt.plot(range(len(values)), values, linewidth=0.9)
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _save_per_run_plots(history, name, output_dir, dimension):
    plot_specs = (
        ("f_loss", "f loss", f"D={dimension} {name}: f Loss", "f_loss.png"),
        ("g_loss", "g loss", f"D={dimension} {name}: g Loss", "g_loss.png"),
        ("f_grad_norm", "f parameter gradient norm", f"D={dimension} {name}: f Gradient Norm", "f_grad_norm.png"),
        ("g_grad_norm", "g parameter gradient norm", f"D={dimension} {name}: g Gradient Norm", "g_grad_norm.png"),
    )
    for history_key, ylabel, title, filename in plot_specs:
        _save_plot(history[history_key], ylabel, title, output_dir / filename)


def run_config(
    name,
    f_lr,
    g_lr,
    benchmark_path,
    dimension=DIM,
    device=DEVICE,
    n_iters=N_ITERS,
    batch_size=BATCH_SIZE,
    log_every=LOG_EVERY,
    eval_samples=N_SAMPLES_EVAL,
    var_samples=VAR_SAMPLES,
):
    """Run one controlled configuration and write its isolated artifacts."""
    output_dir = ROOT / "experiments" / f"stabilization_unequal_lr_d{dimension}" / name
    output_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 80)
    print(f"EXPERIMENT D | D={dimension} | {name} | f_lr={f_lr:g}, g_lr={g_lr:g}")
    print("=" * 80)

    benchmark, module_info = load_mix3to10_benchmark(
        dim=dimension, benchmark_path=benchmark_path, device=device
    )
    mu_sampler, nu_sampler = get_benchmark_samplers(benchmark, device=device)
    f, g, history, final_f_loss, training_time = train_icnn_ot_unequal_lr(
        mu_sampler=mu_sampler,
        nu_sampler=nu_sampler,
        input_dim=dimension,
        f_lr=f_lr,
        g_lr=g_lr,
        n_iters=n_iters,
        batch_size=batch_size,
        device=device,
        log_every=log_every,
    )
    official_metrics = compute_official_l2_uvp(
        benchmark,
        f,
        device=device,
        n_samples=eval_samples,
        var_samples=var_samples,
    )

    completed_iterations = len(history["f_loss"])
    metrics = {
        "config_name": name,
        "D": dimension,
        "f_lr": f_lr,
        "g_lr": g_lr,
        "L2-UVP": float(official_metrics["L2-UVP"]),
        "Cosine Similarity": float(official_metrics["Cosine Similarity"]),
        "L2 Error": float(official_metrics["L2 Error"]),
        "final_f_loss": float(final_f_loss),
        "final_g_loss": float(history["g_loss"][-1]),
        **trajectory_statistics(history["f_grad_norm"], "f_gradient_norm"),
        **trajectory_statistics(history["g_grad_norm"], "g_gradient_norm"),
        "final_f_parameter_norm": float(history["f_param_norm"][-1]),
        "final_g_parameter_norm": float(history["g_param_norm"][-1]),
        "training_time": float(training_time),
        "iterations_per_second": float(completed_iterations / training_time),
        "NaN/Inf": bool(any(history["has_nan"])),
        "completed_iterations": completed_iterations,
    }
    config = {
        "D": dimension,
        "n_iters": n_iters,
        "batch_size": batch_size,
        "hidden_dims": list(HIDDEN_DIMS),
        "activation": ACTIVATION,
        "optimizer": "Adam",
        "betas": list(BETAS),
        "inner_iters": INNER_ITERS,
        "f_lr": f_lr,
        "g_lr": g_lr,
        "seed": SEED,
        "weight_clipping": "ICNN.Wz weights clamped to >= 0 after each step",
        "device": str(device),
        "benchmark_path": str(Path(benchmark_path).resolve()),
        "benchmark_module": module_info["map_benchmark_file"],
        "evaluation_samples": eval_samples,
        "variance_samples": var_samples,
    }
    with open(output_dir / "config.json", "w") as handle:
        json.dump(config, handle, indent=2)
    with open(output_dir / "metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2)
    with open(output_dir / "loss_history.json", "w") as handle:
        json.dump(history, handle, indent=2)
    torch.save(f.state_dict(), output_dir / "f_final.pt")
    torch.save(g.state_dict(), output_dir / "g_final.pt")
    _save_per_run_plots(history, name, output_dir, dimension)

    print(
        f"{name} | L2-UVP: {metrics['L2-UVP']:.4f}% | "
        f"peak f-grad: {metrics['peak_f_gradient_norm']:.4f} | "
        f"peak g-grad: {metrics['peak_g_gradient_norm']:.4f} | "
        f"time: {metrics['training_time']:.1f}s"
    )
    return metrics, history


def _save_comparison_plot(histories, key, ylabel, title, output_path):
    plt.figure(figsize=(7, 4.5))
    for name, history in histories.items():
        plt.plot(history[key], label=name, linewidth=0.9)
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_summary(results, histories, dimension):
    """Write side-by-side tables and trajectory plots for one dimension."""
    output_dir = ROOT / "experiments" / f"stabilization_unequal_lr_d{dimension}"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "summary.json", "w") as handle:
        json.dump(results, handle, indent=2)

    fields = [
        "config_name", "D", "f_lr", "g_lr", "L2-UVP", "Cosine Similarity", "L2 Error",
        "final_f_loss", "final_g_loss", "final_f_gradient_norm", "peak_f_gradient_norm",
        "mean_f_gradient_norm", "std_f_gradient_norm", "final_g_gradient_norm",
        "peak_g_gradient_norm", "mean_g_gradient_norm", "std_g_gradient_norm",
        "final_f_parameter_norm", "final_g_parameter_norm", "training_time",
        "iterations_per_second", "NaN/Inf", "completed_iterations",
    ]
    with open(output_dir / "summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    _save_comparison_plot(
        histories, "f_loss", "f loss", f"D={dimension} Unequal-LR: f-loss trajectories", output_dir / "f_loss_comparison.png"
    )
    _save_comparison_plot(
        histories, "g_loss", "g loss", f"D={dimension} Unequal-LR: g-loss trajectories", output_dir / "g_loss_comparison.png"
    )
    _save_comparison_plot(
        histories, "f_grad_norm", "f parameter gradient norm",
        f"D={dimension} Unequal-LR: f-gradient trajectories", output_dir / "f_grad_norm_comparison.png"
    )
    _save_comparison_plot(
        histories, "g_grad_norm", "g parameter gradient norm",
        f"D={dimension} Unequal-LR: g-gradient trajectories", output_dir / "g_grad_norm_comparison.png"
    )

    report = [
        f"# Experiment D — Unequal player learning rates at D={dimension}",
        "",
        f"All runs use the fixed D={dimension} baseline ICNN, clipping, batch size, 2,000 iterations, "
        "Adam betas, seed, inner iterations, benchmark, and evaluation protocol. Only `g_lr` changes.",
        "",
        "| Config | f LR | g LR | L2-UVP (%) | Cosine | L2 Error | Peak f-grad | Peak g-grad | NaN/Inf |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        report.append(
            f"| {result['config_name']} | {result['f_lr']:.1e} | {result['g_lr']:.1e} | "
            f"{result['L2-UVP']:.4f} | {result['Cosine Similarity']:.4f} | "
            f"{result['L2 Error']:.4f} | {result['peak_f_gradient_norm']:.4f} | "
            f"{result['peak_g_gradient_norm']:.4f} | {result['NaN/Inf']} |"
        )
    report.extend([
        "",
        "Interpretation is intentionally deferred until the measured changes in official transport metrics "
        "and both gradient trajectories are compared. Lower `g_lr` is not treated as successful solely "
        "because it produces smaller gradients.",
    ])
    with open(output_dir / "summary.md", "w") as handle:
        handle.write("\n".join(report) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-path", default=BENCHMARK_PATH)
    parser.add_argument("--dimension", type=int, default=DIM)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--n-iters", type=int, default=N_ITERS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--log-every", type=int, default=LOG_EVERY)
    parser.add_argument("--eval-samples", type=int, default=N_SAMPLES_EVAL)
    parser.add_argument("--var-samples", type=int, default=VAR_SAMPLES)
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=[config[0] for config in CONFIGS],
        default=[config[0] for config in CONFIGS],
        help="Configuration names to run; defaults to the three controlled runs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    configs = [config for config in CONFIGS if config[0] in args.configs]
    results = []
    histories = {}
    for name, f_lr, g_lr in configs:
        metrics, history = run_config(
            name=name,
            f_lr=f_lr,
            g_lr=g_lr,
            benchmark_path=Path(args.benchmark_path).resolve(),
            dimension=args.dimension,
            device=args.device,
            n_iters=args.n_iters,
            batch_size=args.batch_size,
            log_every=args.log_every,
            eval_samples=args.eval_samples,
            var_samples=args.var_samples,
        )
        results.append(metrics)
        histories[name] = history
    save_summary(results, histories, dimension=args.dimension)


if __name__ == "__main__":
    main()
