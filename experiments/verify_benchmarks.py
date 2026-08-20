"""
Verify that the official benchmark checkpoints exist and load successfully
for dimensions D = 2, 4, 8, 16, and 32.
"""
import os
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.benchmark import load_mix3to10_benchmark, get_benchmark_samplers, get_benchmark_path

BENCHMARK_PATH = os.environ.get(
    'WASSERSTEIN_BENCHMARK_PATH',
    '/Users/tanishasinghal/Downloads/Wasserstein2Benchmark'
)

def main():
    print("==========================================================")
    print("VERIFYING HIGH-DIMENSIONAL OFFICIAL BENCHMARK AVAILABILITY")
    print("==========================================================")
    
    # Resolve the repository path
    try:
        repo_root = get_benchmark_path(BENCHMARK_PATH)
        print(f"Verified Repository Path: {repo_root}")
    except Exception as e:
        print(f"Error resolving repository path: {e}")
        sys.exit(1)
        
    dims = [2, 4, 8, 16, 32]
    all_ok = True
    
    for d in dims:
        checkpoint_dir = repo_root / "benchmarks" / "Mix3toMix10"
        pt_v1 = checkpoint_dir / f"{d}_v1.pt"
        pt_v2 = checkpoint_dir / f"{d}_v2.pt"
        
        # Verify checkpoint files exist
        if not pt_v1.exists() or not pt_v2.exists():
            print(f"D={d:<2}  ❌ Checkpoints missing (expected {pt_v1.name} and {pt_v2.name})")
            all_ok = False
            continue
            
        try:
            # Try to load benchmark object
            benchmark, module_info = load_mix3to10_benchmark(dim=d, benchmark_path=repo_root, device='cpu')
            
            # Verify input/output dimensions
            assert benchmark.dim == d, f"Dimension mismatch: expected {d}, got {benchmark.dim}"
            
            # Verify sampler shapes
            mu_sampler, nu_sampler = get_benchmark_samplers(benchmark, device='cpu')
            x = mu_sampler(5)
            y = nu_sampler(5)
            assert x.shape == (5, d), f"Input sampler shape incorrect: {x.shape}"
            assert y.shape == (5, d), f"Output sampler shape incorrect: {y.shape}"
            
            # Verify official module path contains Wasserstein2Benchmark
            assert "Wasserstein2Benchmark" in module_info['map_benchmark_file'], \
                f"Not using official benchmark source module: {module_info['map_benchmark_file']}"
                
            print(f"D={d:<2}  ✓ Checkpoints loaded, samplers and dimension verified.")
        except Exception as e:
            print(f"D={d:<2}  ❌ Error loading: {e}")
            all_ok = False
            
    print("==========================================================")
    if all_ok:
        print("STATUS: All Mix3ToMix10 benchmarks (D=2,4,8,16,32) verified successfully.")
    else:
        print("STATUS: Verification FAILED for one or more dimensions.")
        sys.exit(1)

if __name__ == '__main__':
    main()
