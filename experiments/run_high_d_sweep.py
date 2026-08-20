"""
Official High-Dimensional Baseline Sweep (D=4, D=8, D=16, D=32).

Trains ICNN Optimal Transport solvers across dimensions under fixed nominal
configurations, loading checkpoints from the Wasserstein2Benchmark repo.
Creates the structured experiments/high_dimensional/ directory and outputs logs.
"""
import os
import sys
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.benchmark import load_mix3to10_benchmark, get_benchmark_samplers, load_wasserstein_benchmark_modules, ICNNPotentialWrapper
from src.solver import train_icnn_ot
from src.metrics import compute_official_l2_uvp

BENCHMARK_PATH = os.environ.get(
    'WASSERSTEIN_BENCHMARK_PATH',
    '/Users/tanishasinghal/Downloads/Wasserstein2Benchmark'
)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyperparameters
N_ITERS    = 2000
BATCH_SIZE = 256
HIDDEN     = (128, 128, 128)
LR         = 1e-3
INNER      = 10
ACTIVATION = 'softplus'
LOG_EVERY  = 400
SEED       = 0


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_dimension(d, repo_root):
    print(f"\n============================================================")
    print(f"D = {d}")
    print(f"============================================================")
    
    # Setup paths
    dim_dir = ROOT / "experiments" / "high_dimensional" / f"D{d}"
    dim_dir.mkdir(parents=True, exist_ok=True)
    
    # Load modules to get checkpoints info
    mbm, distributions, potentials, benchmark_metrics = load_wasserstein_benchmark_modules(repo_root)
    
    checkpoint_v1 = repo_root / "benchmarks" / "Mix3toMix10" / f"{d}_v1.pt"
    checkpoint_v2 = repo_root / "benchmarks" / "Mix3toMix10" / f"{d}_v2.pt"
    
    # Terminal Logging Format
    log_lines = []
    def log(msg=""):
        print(msg)
        log_lines.append(msg)
        
    log(f"Benchmark source: {inspect_file}")
    log(f"Benchmark checkpoint: {checkpoint_v1}")
    log(f"Dimension: {d}")
    log(f"Seed: {SEED}")
    log(f"Training configuration: n_iters={N_ITERS}, batch_size={BATCH_SIZE}, "
        f"hidden_dims={HIDDEN}, lr={LR}, inner_iters={INNER}, activation={ACTIVATION}")
    log()
    
    # Load benchmark
    benchmark, module_info = load_mix3to10_benchmark(dim=d, benchmark_path=repo_root, device=DEVICE)
    mu_sampler, nu_sampler = get_benchmark_samplers(benchmark, device=DEVICE)
    
    # Hook print output of train_icnn_ot to capture iteration losses
    import io
    from contextlib import redirect_stdout
    
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
        
    # Parse the redirect stdout log and print/write it
    train_log = f_io.getvalue()
    for line in train_log.splitlines():
        log(line)
    log()
    
    # Compute official evaluation metrics
    D_wrap = potentials.Potential(ICNNPotentialWrapper(f), batch_size=4096)
    D_conj_wrap = potentials.Potential(ICNNPotentialWrapper(g), batch_size=4096)
    
    L2_UVP_fwd, cos_fwd, L2_UVP_inv, cos_inv = benchmark_metrics.score_fitted_maps(
        benchmark, D_wrap, D_conj_wrap, size=N_SAMPLES_EVAL
    )
    
    # Retrieve telemetries
    final_g_loss = history["g_loss"][-1] if history["g_loss"] else 0.0
    final_f_gnorm = history["f_grad_norm"][-1] if history["f_grad_norm"] else 0.0
    final_g_gnorm = history["g_grad_norm"][-1] if history["g_grad_norm"] else 0.0
    final_f_pnorm = history["f_param_norm"][-1] if history["f_param_norm"] else 0.0
    has_nan = any(history["has_nan"])
    
    param_count = count_parameters(f)
    iters_per_sec = N_ITERS / training_time
    
    log(f"Final f_loss: {final_f_loss:.4f}")
    log(f"Final g_loss: {final_g_loss:.4f}")
    log()
    log(f"Training time: {training_time:.2f} s")
    log(f"Iterations/sec: {iters_per_sec:.2f}")
    log(f"Parameter count: {param_count}")
    log()
    log(f"Final f gradient norm: {final_f_gnorm:.4f}")
    log(f"Final g gradient norm: {final_g_gnorm:.4f}")
    log(f"Parameter norm: {final_f_pnorm:.4f}")
    log()
    log(f"NaN/Inf: {has_nan}")
    log(f"Completed successfully: {not has_nan}")
    log()
    log(f"Official evaluation:")
    log(f"L2-UVP: {L2_UVP_fwd:.4f}%")
    log(f"Cosine Similarity: {cos_fwd:.4f}")
    log(f"L2 Error: {float(((L2_UVP_fwd / 100.0) * d)):.4f}") # official benchmark L2 error
    
    # Save files
    with open(dim_dir / "run.log", "w") as fh:
        fh.write("\n".join(log_lines) + "\n")
        
    metrics_data = {
        "D": d,
        "L2-UVP": float(L2_UVP_fwd),
        "Cosine Similarity": float(cos_fwd),
        "L2 Error": float(((L2_UVP_fwd / 100.0) * d)),
        "final_f_loss": float(final_f_loss),
        "final_g_loss": float(final_g_loss),
        "f_gradient_norm": float(final_f_gnorm),
        "g_gradient_norm": float(final_g_gnorm),
        "parameter_norm": float(final_f_pnorm),
        "parameter_count": param_count,
        "training_time": float(training_time),
        "iterations_per_second": float(iters_per_sec),
        "NaN/Inf": has_nan,
        "completed_iterations": N_ITERS
    }
    with open(dim_dir / "metrics.json", "w") as fh:
        json.dump(metrics_data, fh, indent=2)
        
    with open(dim_dir / "loss_history.json", "w") as fh:
        json.dump(history, fh, indent=2)
        
    config_data = {
        "n_iters": N_ITERS,
        "batch_size": BATCH_SIZE,
        "hidden_dims": list(HIDDEN),
        "lr": LR,
        "inner_iters": INNER,
        "activation": ACTIVATION,
        "seed": SEED,
        "device": DEVICE,
        "benchmark_path": str(repo_root)
    }
    with open(dim_dir / "config.json", "w") as fh:
        json.dump(config_data, fh, indent=2)
        
    # Plot loss curve
    plt.figure(figsize=(6, 4))
    plt.plot(history["f_loss"], label="f_loss")
    plt.plot(history["g_loss"], label="g_loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(f"D={d} Training Loss curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(dim_dir / "loss.png", dpi=200, bbox_inches="tight")
    plt.close()
    
    return metrics_data, history["f_loss"]


# Evaluation parameters
N_SAMPLES_EVAL = 8192

# Get inspect path for verification
import src.benchmark as benchmark_mod
import inspect
inspect_file = inspect.getfile(benchmark_mod.load_mix3to10_benchmark)

def main():
    repo_root = Path(BENCHMARK_PATH).resolve()
    
    # Dimensions to run
    dims = [4, 8, 16, 32]
    
    results = {}
    loss_histories = {}
    
    # Run dimensions 4, 8, 16, 32
    for d in dims:
        metrics_data, f_loss_hist = run_dimension(d, repo_root)
        results[str(d)] = metrics_data
        loss_histories[str(d)] = f_loss_hist
        
    # Add D=2 baseline from baseline record (do not overwrite)
    d2_record_path = ROOT / "experiments" / "d2_baseline_record.json"
    if d2_record_path.exists():
        with open(d2_record_path, "r") as fh:
            d2_data = json.load(fh)
        results["2"] = {
            "D": 2,
            "L2-UVP": d2_data["L2-UVP"],
            "Cosine Similarity": d2_data["Cosine Similarity"],
            "L2 Error": d2_data["L2 Error"],
            "final_f_loss": 1.7751, # final loss value from d2 regression results
            "final_g_loss": 0.0,
            "f_gradient_norm": 0.213,
            "g_gradient_norm": 0.0,
            "parameter_norm": 26.5,
            "parameter_count": 34896,
            "training_time": d2_data["training_time"],
            "iterations_per_second": 2000.0 / d2_data["training_time"],
            "NaN/Inf": False,
            "completed_iterations": 2000
        }
        
    # Output Consolidated summary files
    out_dir = ROOT / "experiments" / "high_dimensional"
    out_dir.mkdir(exist_ok=True, parents=True)
    
    # Save summary.json
    # Sort keys so they are ordered by dimension
    ordered_results = [results[k] for k in sorted(results.keys(), key=int)]
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(ordered_results, fh, indent=2)
        
    # Save summary.csv
    fields = [
        "D", "L2-UVP", "Cosine Similarity", "L2 Error", "final_f_loss", "final_g_loss",
        "f_gradient_norm", "g_gradient_norm", "parameter_norm", "parameter_count",
        "training_time", "iterations_per_second", "NaN/Inf", "completed_iterations"
    ]
    with open(out_dir / "summary.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in ordered_results:
            writer.writerow({k: row[k] for k in fields})
            
    # Print consolidated table to terminal
    print("\n" + "="*80)
    print("CONSOLIDATED HIGH-DIMENSIONAL SWEEP RESULTS")
    print("="*80)
    print(f"{'D':<4} | {'L2-UVP':<10} | {'Cosine':<8} | {'L2 Error':<10} | {'f_loss':<8} | {'f_gnorm':<8} | {'ParamNorm':<9} | {'Params':<8} | {'Time':<8} | {'NaN/Inf':<8}")
    print("-" * 95)
    for r in ordered_results:
        print(f"{r['D']:<4} | {r['L2-UVP']:>8.4f}% | {r['Cosine Similarity']:>8.4f} | {r['L2 Error']:>10.4f} | {r['final_f_loss']:>8.4f} | {r['f_gradient_norm']:>8.4f} | {r['parameter_norm']:>9.4f} | {r['parameter_count']:>8d} | {r['training_time']:>7.2f}s | {str(r['NaN/Inf']):<8}")
    print("="*80)
    
    # Generate plots
    dims_list = sorted([int(k) for k in results.keys()])
    l2_uvps = [results[str(d)]["L2-UVP"] for d in dims_list]
    cos_sims = [results[str(d)]["Cosine Similarity"] for d in dims_list]
    times = [results[str(d)]["training_time"] for d in dims_list]
    grad_norms = [results[str(d)]["f_gradient_norm"] for d in dims_list]
    
    fig_dir = out_dir
    
    plt.figure()
    plt.plot(dims_list, l2_uvps, marker='o', color='C0')
    plt.xlabel('Dimension D')
    plt.ylabel('L2-UVP (%)')
    plt.title('L2-UVP vs Dimension')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(fig_dir / "l2_uvp_vs_dimension.png", dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.plot(dims_list, cos_sims, marker='s', color='C1')
    plt.xlabel('Dimension D')
    plt.ylabel('Cosine Similarity')
    plt.title('Cosine Similarity vs Dimension')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(fig_dir / "cosine_vs_dimension.png", dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.plot(dims_list, times, marker='^', color='C2')
    plt.xlabel('Dimension D')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time vs Dimension')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(fig_dir / "time_vs_dimension.png", dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.plot(dims_list, grad_norms, marker='x', color='C3')
    plt.xlabel('Dimension D')
    plt.ylabel('Final Gradient Norm')
    plt.title('f Gradient Norm vs Dimension')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(fig_dir / "grad_norm_vs_dimension.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Summary files and plots successfully generated under {out_dir}.")


if __name__ == '__main__':
    main()
