import json
import time
import math
from pathlib import Path
import numpy as np
import torch

import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.experiment_runner import make_gaussian_sampler, make_mixture_sampler, run_experiment, plot_transport_experiment
from experiments.run_disconnected_demo import make_disconnected_sampler

def circle_means(k, radius=2.0):
    angles = np.linspace(0, 2 * math.pi, k, endpoint=False)
    return [[radius * math.cos(a), radius * math.sin(a)] for a in angles]

def estimate_w2(f, g, mu_sampler, nu_sampler, device="cpu", n_eval=5000):
    f.eval()
    g.eval()
    with torch.no_grad():
        x = mu_sampler(n_eval).to(device)
        y = nu_sampler(n_eval).to(device)
        term_mu = f(x).mean().item()
        
    with torch.enable_grad():
        y_for_grad = y.clone().requires_grad_(True)
        grad_g_y = g.grad(y_for_grad)
        term_nu = (y * grad_g_y).sum(dim=1).mean().item() - f(grad_g_y).mean().item()
        
    e_x2 = (x**2).sum(dim=1).mean().item()
    e_y2 = (y**2).sum(dim=1).mean().item()
    
    minimax_loss_val = term_mu + term_nu
    w2_sq = max(0.0, e_x2 + e_y2 - 2.0 * minimax_loss_val)
    return math.sqrt(w2_sq)

def main():
    device = "cpu"
    experiments_dir = ROOT / "experiments"
    
    config = {
        "input_dim": 2,
        "n_iters": 2000,
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
    # 1. Gaussian to Gaussian (Analytical Compare)
    # ==========================================
    print("Running Gaussian -> Gaussian Experiment...")
    mu = [0.0, 0.0]
    sigma_mu = 1.0
    nu = [2.0, -1.0]
    sigma_nu = 0.6
    
    mu_sampler = make_gaussian_sampler(mu, sigma_mu, device=device)
    nu_sampler = make_gaussian_sampler(nu, sigma_nu, device=device)
    
    exp_name = "gaussian_to_gaussian"
    out_dir, metrics = run_experiment(
        exp_name, config, mu_sampler, nu_sampler, plotting_fn=plot_transport_experiment
    )
    
    # Load the trained potentials to evaluate Wasserstein distance
    from src.icnn import ICNN
    f = ICNN(config["input_dim"], config["hidden_dims"], activation=config["activation"]).to(device)
    g = ICNN(config["input_dim"], config["hidden_dims"], activation=config["activation"]).to(device)
    
    f.load_state_dict(torch.load(out_dir / "run" / "checkpoints" / "f_final.pt"))
    g.load_state_dict(torch.load(out_dir / "run" / "checkpoints" / "g_final.pt"))
    
    learned_w2 = estimate_w2(f, g, mu_sampler, nu_sampler, device=device)
    
    # Analytical W2
    # For independent Gaussians: W_2^2 = ||mu_1 - mu_2||^2 + 2 * (sigma_mu - sigma_nu)^2
    analytical_w2_sq = (nu[0] - mu[0])**2 + (nu[1] - mu[1])**2 + 2.0 * (sigma_mu - sigma_nu)**2
    analytical_w2 = math.sqrt(analytical_w2_sq)
    
    metrics.update({
        "analytical_w2": round(analytical_w2, 4),
        "learned_w2": round(learned_w2, 4),
        "w2_error": round(abs(learned_w2 - analytical_w2), 4)
    })
    
    with open(out_dir / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
        
    print(f"Gaussian -> Gaussian metrics: {metrics}")

    # ==========================================
    # 2. Multimodal Mixture
    # ==========================================
    print("\nRunning Multimodal Mixture Experiment...")
    source_mu = [0.0, 0.0]
    source_sigma = 0.8
    mu_sampler_mix = make_gaussian_sampler(source_mu, source_sigma, device=device)
    
    k = 8
    means = circle_means(k, radius=2.0)
    covs = [np.eye(2) * 0.08 for _ in range(k)]
    weights = [1.0 / k] * k
    nu_sampler_mix = make_mixture_sampler(means, covs, weights, device=device)
    
    exp_name_mix = "multimodal"
    config_mix = dict(config)
    config_mix["exp_name"] = exp_name_mix
    out_dir_mix, metrics_mix = run_experiment(
        exp_name_mix, config_mix, mu_sampler_mix, nu_sampler_mix, plotting_fn=plot_transport_experiment
    )
    print(f"Multimodal metrics: {metrics_mix}")

    # ==========================================
    # 3. Disconnected Support
    # ==========================================
    print("\nRunning Disconnected Support Experiment...")
    source_means = [[-2.0, 0.0], [2.0, 0.0]]
    source_sigma = 0.3
    mu_sampler_disc = make_disconnected_sampler(source_means, source_sigma, device=device)
    
    target_means = [[-1.5, -1.5], [1.5, 1.5]]
    target_sigma = 0.5
    nu_sampler_disc = make_disconnected_sampler(target_means, target_sigma, device=device)
    
    exp_name_disc = "disconnected_support"
    config_disc = dict(config)
    config_disc["exp_name"] = exp_name_disc
    out_dir_disc, metrics_disc = run_experiment(
        exp_name_disc, config_disc, mu_sampler_disc, nu_sampler_disc, plotting_fn=plot_transport_experiment
    )
    print(f"Disconnected Support metrics: {metrics_disc}")

if __name__ == "__main__":
    main()
