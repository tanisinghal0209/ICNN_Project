import json
import time
from pathlib import Path
import numpy as np
import torch

import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.experiment_runner import make_gaussian_sampler, run_experiment, plot_transport_experiment

def main():
    device = "cpu"
    
    base_config = {
        "input_dim": 2,
        "n_iters": 1000,
        "batch_size": 256,
        "hidden_dims": [64, 64, 64],
        "lr": 1e-3,
        "inner_iters": 10,
        "activation": "softplus",
        "device": device,
        "log_every": 250,
        "checkpoint_every": 500,
        "plot_n": 256,
        "optimizer": "adam",
        "convexity_coeff": 0.0,
        "model_type": "icnn"
    }

    # ==========================================
    # 1. Very Small Dataset (N = 50)
    # ==========================================
    print("Running failure analysis: Very Small Dataset (N=50)...")
    n_samples = 50
    x_pool = torch.randn(n_samples, 2, device=device)
    y_pool = torch.randn(n_samples, 2, device=device) + torch.tensor([2.0, -1.0], device=device)
    
    def make_finite_sampler(pool):
        def sampler(batch_size):
            idx = torch.randint(0, len(pool), (batch_size,), device=pool.device)
            return pool[idx]
        return sampler

    cfg_small = dict(base_config)
    cfg_small["exp_name"] = "failure_small_data"
    cfg_small["batch_size"] = 16
    
    run_experiment(
        cfg_small["exp_name"], cfg_small,
        make_finite_sampler(x_pool), make_finite_sampler(y_pool),
        plotting_fn=plot_transport_experiment
    )

    # ==========================================
    # 2. Remove Convexity (model_type = mlp)
    # ==========================================
    print("\nRunning failure analysis: Remove Convexity (MLP potentials)...")
    cfg_mlp = dict(base_config)
    cfg_mlp["exp_name"] = "failure_no_convexity"
    cfg_mlp["model_type"] = "mlp"
    
    # We use identity-like or slightly shifted Gaussian to observe the mappings
    mu = [0.0, 0.0]
    nu = [2.0, -1.0]
    mu_sampler = make_gaussian_sampler(mu, 1.0, device=device)
    nu_sampler = make_gaussian_sampler(nu, 0.6, device=device)
    
    run_experiment(
        cfg_mlp["exp_name"], cfg_mlp, mu_sampler, nu_sampler,
        plotting_fn=plot_transport_experiment
    )

    # ==========================================
    # 3. Large Learning Rate (lr = 0.1)
    # ==========================================
    print("\nRunning failure analysis: Large Learning Rate (lr=0.1)...")
    cfg_lr = dict(base_config)
    cfg_lr["exp_name"] = "failure_large_lr"
    cfg_lr["lr"] = 0.1
    
    run_experiment(
        cfg_lr["exp_name"], cfg_lr, mu_sampler, nu_sampler,
        plotting_fn=plot_transport_experiment
    )

    # ==========================================
    # 4. Very High Dimension (dim = 128)
    # ==========================================
    print("\nRunning failure analysis: Very High Dimension (dim=128)...")
    dim = 128
    cfg_dim = dict(base_config)
    cfg_dim["exp_name"] = "failure_high_dim"
    cfg_dim["input_dim"] = dim
    
    def make_dim_sampler(mu_val, sigma, dim_val):
        def sampler(batch_size):
            return torch.randn(batch_size, dim_val, device=device) * sigma + mu_val
        return sampler
        
    mu_sampler_hd = make_dim_sampler(0.0, 1.0, dim)
    nu_sampler_hd = make_dim_sampler(2.0, 0.6, dim)
    
    # Skip plotting since it's high dimensional and simple_transport_plot requires 2D
    run_experiment(cfg_dim["exp_name"], cfg_dim, mu_sampler_hd, nu_sampler_hd, plotting_fn=None)

    # ==========================================
    # 5. Poor Overlap (Shift = 15.0)
    # ==========================================
    print("\nRunning failure analysis: Poor Overlap...")
    cfg_overlap = dict(base_config)
    cfg_overlap["exp_name"] = "failure_poor_overlap"
    
    mu_overlap = [0.0, 0.0]
    nu_overlap = [15.0, 15.0] # far apart
    mu_sampler_overlap = make_gaussian_sampler(mu_overlap, 1.0, device=device)
    nu_sampler_overlap = make_gaussian_sampler(nu_overlap, 1.0, device=device)
    
    run_experiment(
        cfg_overlap["exp_name"], cfg_overlap, mu_sampler_overlap, nu_sampler_overlap,
        plotting_fn=plot_transport_experiment
    )

if __name__ == "__main__":
    main()
