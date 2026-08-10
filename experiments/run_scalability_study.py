import json
import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.experiment_runner import make_gaussian_sampler, run_experiment

def main():
    device = "cpu"
    figures_dir = ROOT / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    base_config = {
        "input_dim": 2,
        "n_iters": 500, # 500 iterations for fast scalability comparison
        "batch_size": 256,
        "hidden_dims": [64, 64, 64],
        "lr": 1e-3,
        "inner_iters": 10,
        "activation": "softplus",
        "device": device,
        "log_every": 250,
        "checkpoint_every": 0,
        "plot_n": 256,
        "optimizer": "adam",
        "convexity_coeff": 0.0,
        "model_type": "icnn"
    }

    # ==========================================
    # 1. Dimension Scalability Sweep
    # ==========================================
    print("Running dimension scalability sweep...")
    dims = [2, 8, 32, 64, 128]
    dim_runtimes = []
    dim_losses = []
    
    for d in dims:
        print(f"  Dimension = {d}")
        cfg = dict(base_config)
        cfg["input_dim"] = d
        cfg["exp_name"] = f"scale_dim_{d}"
        
        def make_dim_sampler(dim_val):
            def sampler(batch_size):
                return torch.randn(batch_size, dim_val, device=device)
            return sampler
            
        _, metrics = run_experiment(cfg["exp_name"], cfg, make_dim_sampler(d), make_dim_sampler(d), plotting_fn=None)
        dim_runtimes.append(metrics["runtime_sec"])
        dim_losses.append(metrics["final_f_loss"])
        
    dim_results = {
        "dimensions": dims,
        "runtimes": dim_runtimes,
        "losses": dim_losses
    }
    with open(ROOT / "experiments" / "dim_scalability.json", "w") as fh:
        json.dump(dim_results, fh, indent=2)

    # ==========================================
    # 2. Size Scalability Sweep
    # ==========================================
    print("\nRunning dataset-size scalability sweep...")
    sizes = [100, 1000, 5000, 20000]
    size_runtimes = []
    size_losses = []
    
    for n in sizes:
        print(f"  Dataset Size = {n}")
        cfg = dict(base_config)
        cfg["exp_name"] = f"scale_size_{n}"
        cfg["batch_size"] = min(n, 256)
        
        x_pool = torch.randn(n, 2, device=device)
        y_pool = torch.randn(n, 2, device=device)
        
        def make_finite_sampler(pool):
            def sampler(batch_size):
                idx = torch.randint(0, len(pool), (batch_size,), device=pool.device)
                return pool[idx]
            return sampler
            
        _, metrics = run_experiment(cfg["exp_name"], cfg, make_finite_sampler(x_pool), make_finite_sampler(y_pool), plotting_fn=None)
        size_runtimes.append(metrics["runtime_sec"])
        size_losses.append(metrics["final_f_loss"])
        
    size_results = {
        "dataset_sizes": sizes,
        "runtimes": size_runtimes,
        "losses": size_losses
    }
    with open(ROOT / "experiments" / "size_scalability.json", "w") as fh:
        json.dump(size_results, fh, indent=2)

    # ==========================================
    # 3. Generate Scalability Plots
    # ==========================================
    print("\nGenerating scalability plots...")
    
    # Plot 1: Runtime vs Dimension
    plt.figure(figsize=(6, 4.5))
    plt.plot(dims, dim_runtimes, marker='o', color='C0', linewidth=2)
    plt.title("Runtime vs Input Dimension", fontsize=12, fontweight='bold')
    plt.xlabel("Dimension", fontsize=10)
    plt.ylabel("Runtime (seconds)", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(figures_dir / "scalability_dim_vs_time.png", dpi=150)
    plt.close()
    
    # Plot 2: Runtime vs Dataset Size
    plt.figure(figsize=(6, 4.5))
    plt.plot(sizes, size_runtimes, marker='s', color='C1', linewidth=2)
    plt.title("Runtime vs Dataset Size", fontsize=12, fontweight='bold')
    plt.xlabel("Dataset Size (Number of Samples)", fontsize=10)
    plt.ylabel("Runtime (seconds)", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(figures_dir / "scalability_size_vs_time.png", dpi=150)
    plt.close()
    
    print("Scalability study completed and plots saved.")

if __name__ == "__main__":
    main()
