import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.experiment_runner import make_gaussian_sampler, make_mixture_sampler, run_experiment


ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_ROOT = ROOT / "experiments"


def write_summary(results, summary_path):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)


if __name__ == "__main__":
    results = []
    device = "cpu"

    def run_case(name, config, mu_sampler, nu_sampler, extra=None):
        exp_root, metrics = run_experiment(name, config, mu_sampler, nu_sampler)
        entry = {
            "experiment": name,
            "root": str(exp_root),
            "final_loss": metrics.get("final_f_loss"),
            "runtime_seconds": metrics.get("runtime_sec"),
        }
        if extra:
            entry.update(extra)
        results.append(entry)

    base_cfg = {
        "input_dim": 2,
        "n_iters": 500,
        "batch_size": 256,
        "hidden_dims": [64, 64, 64],
        "lr": 1e-3,
        "inner_iters": 10,
        "activation": "softplus",
        "device": device,
        "log_every": 250,
        "checkpoint_every": 500,
        "plot_n": 256,
        "exp_name": "baseline",
        "model_type": "icnn",
        "optimizer": "adam",
        "convexity_coeff": 0.0,
    }

    run_case("baseline", base_cfg, make_gaussian_sampler([0.0, 0.0], 1.0, device=device), make_gaussian_sampler([0.0, 0.0], 1.0, device=device))

    for layers in [2, 3, 4, 5]:
        cfg = dict(base_cfg)
        cfg["exp_name"] = f"layers_{layers}"
        cfg["hidden_dims"] = [64] * layers
        run_case(cfg["exp_name"], cfg, make_gaussian_sampler([0.0, 0.0], 1.0, device=device), make_gaussian_sampler([0.0, 0.0], 1.0, device=device), {"hidden_layers": layers})

    for width in [64, 128, 256, 512]:
        cfg = dict(base_cfg)
        cfg["exp_name"] = f"width_{width}"
        cfg["hidden_dims"] = [width, width, width]
        run_case(cfg["exp_name"], cfg, make_gaussian_sampler([0.0, 0.0], 1.0, device=device), make_gaussian_sampler([0.0, 0.0], 1.0, device=device), {"hidden_width": width})

    for lr in [1e-2, 1e-3, 1e-4]:
        cfg = dict(base_cfg)
        cfg["exp_name"] = f"lr_{lr}"
        cfg["lr"] = lr
        run_case(cfg["exp_name"], cfg, make_gaussian_sampler([0.0, 0.0], 1.0, device=device), make_gaussian_sampler([0.0, 0.0], 1.0, device=device), {"learning_rate": lr})

    for activation in ["relu", "leaky_relu", "elu"]:
        cfg = dict(base_cfg)
        cfg["exp_name"] = f"activation_{activation}"
        cfg["activation"] = activation
        run_case(cfg["exp_name"], cfg, make_gaussian_sampler([0.0, 0.0], 1.0, device=device), make_gaussian_sampler([0.0, 0.0], 1.0, device=device), {"activation": activation})

    cfg = dict(base_cfg)
    cfg["exp_name"] = "mlp_baseline"
    cfg["model_type"] = "mlp"
    run_case(cfg["exp_name"], cfg, make_gaussian_sampler([0.0, 0.0], 1.0, device=device), make_gaussian_sampler([0.0, 0.0], 1.0, device=device), {"model_type": "mlp"})

    for n_samples in [500, 1000, 5000, 10000]:
        cfg = dict(base_cfg)
        cfg["exp_name"] = f"samples_{n_samples}"
        cfg["batch_size"] = min(n_samples, 256)
        cfg["n_iters"] = 500
        
        # Pre-generate fixed finite pools of size n_samples
        x_pool = torch.randn(n_samples, 2, device=device)
        y_pool = torch.randn(n_samples, 2, device=device)
        
        def make_finite_sampler(pool):
            def sampler(batch_size):
                idx = torch.randint(0, len(pool), (batch_size,), device=pool.device)
                return pool[idx]
            return sampler
            
        run_case(cfg["exp_name"], cfg, make_finite_sampler(x_pool), make_finite_sampler(y_pool), {"sample_count": n_samples})

    for dim in [2, 8, 16, 32]:
        cfg = dict(base_cfg)
        cfg["exp_name"] = f"dim_{dim}"
        cfg["input_dim"] = dim
        def make_dim_sampler(mu, sigma, device, dim=dim):
            def sampler(batch_size):
                return torch.randn(batch_size, dim, device=device) * sigma + torch.tensor([0.0] * dim, device=device)
            return sampler
        run_case(cfg["exp_name"], cfg, make_dim_sampler([0.0] * dim, 1.0, device=device), make_dim_sampler([0.0] * dim, 1.0, device=device), {"dimension": dim})

    mixture_cfg = dict(base_cfg)
    mixture_cfg["exp_name"] = "mixture"
    mixture_cfg["n_iters"] = 2000
    mixture_mu = make_mixture_sampler([[0.0, 0.0], [3.0, 3.0]], [torch.eye(2), torch.eye(2)], [0.5, 0.5], device=device)
    mixture_nu = make_mixture_sampler([[0.0, 0.0], [-3.0, -3.0]], [torch.eye(2), torch.eye(2)], [0.5, 0.5], device=device)
    run_case(mixture_cfg["exp_name"], mixture_cfg, mixture_mu, mixture_nu, {"distribution": "mixture"})

    write_summary(results, EXPERIMENTS_ROOT / "summary.json")
