"""
Experiment B: Optimization Stability Ablation at D=16

Evaluates optimization parameters at fixed dimension D=16 and fixed architecture
hidden_dims=(128,128,128):
- Baseline: lr=1e-3, inner_iters=10
- Run A: lr=5e-4, inner_iters=10
- Run B: lr=2.5e-4, inner_iters=10
- Run C: lr=1e-3, inner_iters=20
- Run D: lr=1e-3, inner_iters=50
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

from src.benchmark import load_mix3to10_benchmark, get_benchmark_samplers
from src.metrics import compute_official_l2_uvp
from src.solver import train_icnn_ot

BENCHMARK_PATH = os.environ.get(
    'WASSERSTEIN_BENCHMARK_PATH',
    '/Users/tanishasinghal/Downloads/Wasserstein2Benchmark'
)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Frozen parameters
DIM         = 16
N_ITERS     = 2000
BATCH_SIZE  = 256
HIDDEN_DIMS = (128, 128, 128)
ACTIVATION  = 'softplus'
LOG_EVERY   = 400
SEED        = 0

N_SAMPLES_EVAL = 8192
VAR_SAMPLES    = 16384


def run_opt_config(name, lr, inner_iters, repo_root):
    out_dir = ROOT / "experiments" / "ablation_optimization_d16" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n============================================================")
    print(f"EXPERIMENT B (D=16): Config {name} (lr={lr}, inner_iters={inner_iters})")
    print(f"============================================================")
    
    # 1. Load benchmark
    benchmark, module_info = load_mix3to10_benchmark(dim=DIM, benchmark_path=repo_root, device=DEVICE)
    mu_sampler, nu_sampler = get_benchmark_samplers(benchmark, device=DEVICE)
    
    # 2. Train
    f, g, history, final_f_loss, training_time = train_icnn_ot(
        mu_sampler, nu_sampler,
        input_dim=DIM,
        n_iters=N_ITERS,
        batch_size=BATCH_SIZE,
        hidden_dims=HIDDEN_DIMS,
        lr=lr,
        inner_iters=inner_iters,
        device=DEVICE,
        activation=ACTIVATION,
        log_every=LOG_EVERY,
        model_type='icnn',
    )
    
    # 3. Telemetry
    final_g_loss  = history["g_loss"][-1] if history["g_loss"] else 0.0
    final_f_gnorm = history["f_grad_norm"][-1] if history["f_grad_norm"] else 0.0
    final_g_gnorm = history["g_grad_norm"][-1] if history["g_grad_norm"] else 0.0
    final_f_pnorm = history["f_param_norm"][-1] if history["f_param_norm"] else 0.0
    final_g_pnorm = history["g_param_norm"][-1] if history["g_param_norm"] else 0.0
    has_nan       = any(history["has_nan"])
    iters_per_sec = N_ITERS / training_time
    
    # 4. Evaluation
    metrics = compute_official_l2_uvp(
        benchmark, f,
        device=DEVICE,
        n_samples=N_SAMPLES_EVAL,
        var_samples=VAR_SAMPLES,
    )
    
    res = {
        "config_name": name,
        "lr": lr,
        "inner_iters": inner_iters,
        "L2-UVP": metrics["L2-UVP"],
        "Cosine Similarity": metrics["Cosine Similarity"],
        "L2 Error": metrics["L2 Error"],
        "final_f_loss": final_f_loss,
        "final_g_loss": final_g_loss,
        "f_gradient_norm": final_f_gnorm,
        "g_gradient_norm": final_g_gnorm,
        "f_parameter_norm": final_f_pnorm,
        "g_parameter_norm": final_g_pnorm,
        "training_time": training_time,
        "iterations_per_second": iters_per_sec,
        "NaN/Inf": has_nan,
    }
    
    # 5. Save run files
    with open(out_dir / "metrics.json", "w") as fh:
        json.dump(res, fh, indent=2)
    with open(out_dir / "loss_history.json", "w") as fh:
        json.dump(history, fh, indent=2)
        
    # Plots
    plt.figure(figsize=(6, 4))
    plt.plot(history["f_loss"], label="f_loss")
    plt.plot(history["g_loss"], label="g_loss", alpha=0.6)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(f"D=16 Optimization {name} Loss Curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(out_dir / "loss.png", dpi=200, bbox_inches="tight")
    plt.close()
    
    plt.figure(figsize=(6, 4))
    plt.plot(history["f_grad_norm"], label="f_grad_norm")
    plt.plot(history["g_grad_norm"], label="g_grad_norm", alpha=0.6)
    plt.xlabel("Iteration")
    plt.ylabel("Gradient Norm")
    plt.title(f"D=16 Optimization {name} Gradient Norm Trajectory")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(out_dir / "grad_norm.png", dpi=200, bbox_inches="tight")
    plt.close()
    
    print(f"Config={name:<10} | L2-UVP: {metrics['L2-UVP']:>6.2f}% | Cosine: {metrics['Cosine Similarity']:>6.4f} | f_loss: {final_f_loss:>6.2f} | f_gnorm: {final_f_gnorm:>6.2f} | Time: {training_time:>5.1f}s")
    return res, history


def main():
    repo_root = Path(BENCHMARK_PATH).resolve()
    
    configs = [
        ("Baseline", 1e-3, 10),
        ("LR_5e-4", 5e-4, 10),
        ("LR_2.5e-4", 2.5e-4, 10),
        ("Inner_20", 1e-3, 20),
        ("Inner_50", 1e-3, 50),
    ]
    
    out_base = ROOT / "experiments" / "ablation_optimization_d16"
    out_base.mkdir(parents=True, exist_ok=True)
    
    results = []
    histories = {}
    
    for name, lr, inner in configs:
        res, hist = run_opt_config(name, lr, inner, repo_root)
        results.append(res)
        histories[name] = hist
        
    # Save summary
    with open(out_base / "summary.json", "w") as fh:
        json.dump(results, fh, indent=2)
        
    fields = [
        "config_name", "lr", "inner_iters", "L2-UVP", "Cosine Similarity", "L2 Error",
        "final_f_loss", "final_g_loss", "f_gradient_norm", "g_gradient_norm",
        "f_parameter_norm", "g_parameter_norm", "training_time", "iterations_per_second", "NaN/Inf"
    ]
    with open(out_base / "summary.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fields})
            
    # Comparative Plot: f_loss vs iteration across configurations
    plt.figure(figsize=(7, 4.5))
    for r in results:
        name = r["config_name"]
        plt.plot(histories[name]["f_loss"], label=name, linewidth=1.0)
    plt.xlabel("Iteration")
    plt.ylabel("f_loss")
    plt.title("D=16 Optimization Ablation: f_loss Trajectories")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(out_base / "loss_trajectories_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Comparative Plot: f_grad_norm vs iteration across configurations
    plt.figure(figsize=(7, 4.5))
    for r in results:
        name = r["config_name"]
        plt.plot(histories[name]["f_grad_norm"], label=name, linewidth=1.0)
    plt.xlabel("Iteration")
    plt.ylabel("f Gradient Norm")
    plt.title("D=16 Optimization Ablation: Gradient Norm Trajectories")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(out_base / "grad_norm_trajectories_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("\n" + "="*85)
    print("EXPERIMENT B (OPTIMIZATION STABILITY ABLATION AT D=16) COMPLETE")
    print("="*85)
    print(f"{'Config':<12} | {'LR':<7} | {'Inner':<5} | {'L2-UVP':<10} | {'Cosine':<8} | {'f_loss':<8} | {'f_gnorm':<8} | {'Time':<6}")
    print("-" * 85)
    for r in results:
        print(f"{r['config_name']:<12} | {r['lr']:<7.1e} | {r['inner_iters']:<5d} | {r['L2-UVP']:>8.2f}% | {r['Cosine Similarity']:>8.4f} | {r['final_f_loss']:>8.2f} | {r['f_gradient_norm']:>8.2f} | {r['training_time']:>5.1f}s")
    print("="*85)


if __name__ == '__main__':
    main()
