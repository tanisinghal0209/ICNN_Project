import os
import sys
import json
import math
import torch
import torch.nn as nn
from pathlib import Path

# Add ICNN Project root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Add Wasserstein2Benchmark path
BENCHMARK_PATH = Path("/Users/tanishasinghal/Downloads/Wasserstein2Benchmark")
if str(BENCHMARK_PATH) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_PATH))

# Change directory to Wasserstein2Benchmark/notebooks so relative paths (../benchmarks) resolve correctly
os.chdir(BENCHMARK_PATH / "notebooks")

# Import benchmark modules
import src.map_benchmark as mbm
import src.metrics as metrics
import src.distributions as distributions
import src.potentials as potentials
from src.potentials import Potential

# MONKEYPATCHES for CPU compatibility
# 1. Override RandomGaussianMixSampler to default device to "cpu"
old_rgms_init = distributions.RandomGaussianMixSampler.__init__
def new_rgms_init(self, *args, **kwargs):
    if 'device' not in kwargs:
        kwargs['device'] = 'cpu'
    old_rgms_init(self, *args, **kwargs)
distributions.RandomGaussianMixSampler.__init__ = new_rgms_init

# 2. Override ShiftedPotential to place the shift tensor on the correct device dynamically
old_shifted_init = potentials.ShiftedPotential.__init__
def new_shifted_init(self, potential, shift, batch_size=1024):
    super(potentials.ShiftedPotential, self).__init__(batch_size)
    self.dim = potential.dim
    self.potential = potential
    
    # Check potential parameters for device, fallback to cpu
    dev = 'cpu'
    for p in potential.parameters():
        dev = p.device
        break
    self.shift = torch.tensor(shift, dtype=torch.float32, device=dev)
potentials.ShiftedPotential.__init__ = new_shifted_init


from train import train_icnn_ot

# Define ICNN Wrapper to match benchmark forward pass format (shape batch_size x 1)
class ICNNWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.dim = model.input_dim
        
    def forward(self, x):
        return self.model(x).unsqueeze(-1)

def main():
    device = "cpu"
    dims = [2, 8, 16, 32]
    results = {}
    
    for dim in dims:
        print(f"\n======================================")
        print(f"Running Korotin Benchmark for Dim = {dim}")
        print(f"======================================")
        
        # Load benchmark instance (which loads ground truth potentials from Wasserstein2Benchmark/benchmarks)
        benchmark = mbm.Mix3ToMix10Benchmark(dim=dim, device=device)
        
        # Sampler functions for training
        def mu_sampler(batch_size):
            return benchmark.input_sampler.sample(batch_size)
            
        def nu_sampler(batch_size):
            return benchmark.output_sampler.sample(batch_size)
            
        # Train our ICNN model using our train_icnn_ot function
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
        
        # Wrap our trained networks
        # D is forward potential (f), D_conj is conjugate potential (g)
        D = Potential(ICNNWrapper(f), batch_size=4096)
        D_conj = Potential(ICNNWrapper(g), batch_size=4096)
        
        # Score the fitted maps on the benchmark
        # score_fitted_maps returns: L2_UVP_fwd, cos_fwd, L2_UVP_inv, cos_inv
        L2_UVP_fwd, cos_fwd, L2_UVP_inv, cos_inv = metrics.score_fitted_maps(
            benchmark, D, D_conj, size=4096
        )
        
        print(f"Dim {dim} results:")
        print(f"  Forward L2-UVP: {L2_UVP_fwd:.4f}% | Cosine Similarity: {cos_fwd:.4f}")
        print(f"  Inverse L2-UVP: {L2_UVP_inv:.4f}% | Cosine Similarity: {cos_inv:.4f}")
        
        results[str(dim)] = {
            "L2_UVP_fwd": round(L2_UVP_fwd, 4),
            "cos_fwd": round(cos_fwd, 4),
            "L2_UVP_inv": round(L2_UVP_inv, 4),
            "cos_inv": round(cos_inv, 4)
        }
        
    # Save the benchmark results to our project's experiments directory
    out_path = ROOT / "experiments" / "korotin_metrics.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nWrote Korotin benchmark results to {out_path}")

if __name__ == "__main__":
    main()
