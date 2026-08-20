"""
Experiment A: Capacity / Expressivity Ablation at D=16

Evaluates hidden layer widths: (64,64,64), (128,128,128), (256,256,256), (512,512,512)
at fixed dimension D=16 and frozen optimization hyperparameters:
n_iters=2000, batch_size=256, lr=1e-3, inner_iters=10, Adam(0.5, 0.9), softplus, seed=0.
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
DIM        = 16
N_ITERS    = 2000
BATCH_SIZE = 256
LR         = 1e-3
INNER      = 10
ACTIVATION = 'softplus'
LOG_EVERY  = 400
SEED       = 0

N_SAMPLES_EVAL = 8192
VAR_SAMPLES    = 16384


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_width(w, repo_root):
    hidden_dims = (w, w, w)
    out_dir = ROOT / "experiments" / "ablation_expressivity_d16" / f"width_{w}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n============================================================")
    print(f"EXPERIMENT A (D=16): Width = {w} {hidden_dims}")
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
        hidden_dims=hidden_dims,
        lr=LR,
        inner_iters=INNER,
        device=DEVICE,
        activation=ACTIVATION,
        log_every=LOG_EVERY,
        model_type='icnn',
    )
    
    # 3. Telemetry
    param_count_f = count_parameters(f)
    param_count_g = count_parameters(g)
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
        "width": w,
        "hidden_dims": list(hidden_dims),
        "L2-UVP": metrics["L2-UVP"],
        "Cosine Similarity": metrics["Cosine Similarity"],
        "L2 Error": metrics["L2 Error"],
        "final_f_loss": final_f_loss,
        "final_g_loss": final_g_loss,
        "f_gradient_norm": final_f_gnorm,
        "g_gradient_norm": final_g_gnorm,
        "f_parameter_norm": final_f_pnorm,
        "g_parameter_norm": final_g_pnorm,
        "parameter_count_f": param_count_f,
        "parameter_count_g": param_count_g,
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
    plt.title(f"D=16 (Width={w}) Loss Curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(out_dir / "loss.png", dpi=200, bbox_inches="tight")
    plt.close()
    
    plt.figure(figsize=(6, 4))
    plt.plot(history["f_grad_norm"], label="f_grad_norm")
    plt.plot(history["g_grad_norm"], label="g_grad_norm", alpha=0.6)
    plt.xlabel("Iteration")
    plt.ylabel("Gradient Norm")
    plt.title(f"D=16 (Width={w}) Gradient Norm Trajectory")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(out_dir / "grad_norm.png", dpi=200, bbox_inches="tight")
    plt.close()
    
    print(f"Width={w:<4} | L2-UVP: {metrics['L2-UVP']:>6.2f}% | Cosine: {metrics['Cosine Similarity']:>6.4f} | f_loss: {final_f_loss:>6.2f} | f_gnorm: {final_f_gnorm:>6.2f} | Time: {training_time:>5.1f}s")
    return res, history


def main():
    repo_root = Path(BENCHMARK_PATH).resolve()
    widths = [64, 128, 256, 512]
    
    out_base = ROOT / "experiments" / "ablation_expressivity_d16"
    out_base.mkdir(parents=True, exist_ok=True)
    
    results = []
    histories = {}
    
    for w in widths:
        res, hist = run_width(w, repo_root)
        results.append(res)
        histories[str(w)] = hist
        
    # Save summary
    with open(out_base / "summary.json", "w") as fh:
        json.dump(results, fh, indent=2)
        
    fields = [
        "width", "L2-UVP", "Cosine Similarity", "L2 Error", "final_f_loss", "final_g_loss",
        "f_gradient_norm", "g_gradient_norm", "f_parameter_norm", "g_parameter_norm",
        "parameter_count_f", "training_time", "iterations_per_second", "NaN/Inf"
    ]
    with open(out_base / "summary.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fields})
            
    # Plots
    w_list = [r["width"] for r in results]
    uvp_list = [r["L2-UVP"] for r in results]
    cos_list = [r["Cosine Similarity"] for r in results]
    gnorm_list = [r["f_gradient_norm"] for r in results]
    time_list = [r["training_time"] for r in results]
    
    plt.figure(figsize=(6, 4))
    plt.plot(w_list, uvp_list, marker='o', color='C0')
    plt.xlabel("Hidden Width (W)")
    plt.ylabel("L2-UVP (%)")
    plt.title("D=16 Expressivity Ablation: L2-UVP vs Width")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(out_base / "uvp_vs_width.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(w_list, gnorm_list, marker='x', color='C3')
    plt.xlabel("Hidden Width (W)")
    plt.ylabel("Final f Gradient Norm")
    plt.title("D=16 Expressivity Ablation: Gradient Norm vs Width")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(out_base / "grad_norm_vs_width.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("\n" + "="*80)
    print("EXPERIMENT A (EXPRESSIVITY ABLATION AT D=16) COMPLETE")
    print("="*80)
    print(f"{'Width':<6} | {'L2-UVP':<10} | {'Cosine':<8} | {'f_loss':<8} | {'f_gnorm':<8} | {'g_gnorm':<8} | {'Params':<8} | {'Time':<6}")
    print("-" * 80)
    for r in results:
        print(f"{r['width']:<6} | {r['L2-UVP']:>8.2f}% | {r['Cosine Similarity']:>8.4f} | {r['final_f_loss']:>8.2f} | {r['f_gradient_norm']:>8.2f} | {r['g_gradient_norm']:>8.4f} | {r['parameter_count_f']:>8d} | {r['training_time']:>5.1f}s")
    print("="*80)


if __name__ == '__main__':
    main()
