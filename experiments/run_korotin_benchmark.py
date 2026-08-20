"""Script to run the official Korotin Wasserstein-2 benchmark using the modular src code."""
import json
import sys
from pathlib import Path

import torch

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark import load_mix3to10_benchmark, load_wasserstein_benchmark_modules, ICNNWrapper
from src.solver import train_icnn_ot
from src.config import load_config


def main():
    # Load configuration
    cfg_path = ROOT / "src" / "config.py"  # fallback default config
    config = load_config()
    benchmark_path = config["benchmark_path"]
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    
    dims = [2, 8, 16, 32]
    results = {}

    # Load benchmark modules dynamically to get access to potentials and metrics modules
    mbm, distributions, potentials, benchmark_metrics = load_wasserstein_benchmark_modules(benchmark_path)

    for dim in dims:
        print(f"\n======================================")
        print(f"Running Korotin Benchmark for Dim = {dim} on {device}")
        print(f"======================================")
        
        # Load benchmark instance (handles path switching context internally)
        benchmark = load_mix3to10_benchmark(dim=dim, benchmark_path=benchmark_path, device=device)
        
        # Sampler functions for training
        def mu_sampler(batch_size):
            return benchmark.input_sampler.sample(batch_size)
            
        def nu_sampler(batch_size):
            return benchmark.output_sampler.sample(batch_size)
            
        # Train our ICNN model using our train_icnn_ot function from src.solver
        f, g, history = train_icnn_ot(
            mu_sampler,
            nu_sampler,
            input_dim=dim,
            n_iters=1500,
            batch_size=256,
            hidden_dims=(max(2*dim, 64), max(2*dim, 64), max(dim, 32)),
            lr=1e-3,
            inner_iters=10,
            device=device,
            activation="softplus",
            log_every=250,
            log_dir=None,
            checkpoint_every=0,
            model_type="icnn"
        )
        
        # Wrap our trained networks to match the benchmark interface
        # D is forward potential (f), D_conj is conjugate potential (g)
        D = potentials.Potential(ICNNWrapper(f), batch_size=4096)
        D_conj = potentials.Potential(ICNNWrapper(g), batch_size=4096)
        
        # Score the fitted maps on the benchmark
        # score_fitted_maps returns: L2_UVP_fwd, cos_fwd, L2_UVP_inv, cos_inv
        L2_UVP_fwd, cos_fwd, L2_UVP_inv, cos_inv = benchmark_metrics.score_fitted_maps(
            benchmark, D, D_conj, size=4096
        )
        
        print(f"Dim {dim} results:")
        print(f"  Forward L2-UVP: {L2_UVP_fwd:.4f}% | Cosine Similarity: {cos_fwd:.4f}")
        print(f"  Inverse L2-UVP: {L2_UVP_inv:.4f}% | Cosine Similarity: {cos_inv:.4f}")
        
        results[str(dim)] = {
            "L2_UVP_fwd": float(L2_UVP_fwd),
            "cos_fwd": float(cos_fwd),
            "L2_UVP_inv": float(L2_UVP_inv),
            "cos_inv": float(cos_inv)
        }
        
    # Save the benchmark results to our project's experiments directory
    out_path = ROOT / "experiments" / "korotin_metrics.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nWrote Korotin benchmark results to {out_path}")


if __name__ == "__main__":
    main()
