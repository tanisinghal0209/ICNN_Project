"""
Official Korotin Wasserstein2Benchmark loader.

Loads the actual benchmark objects from the cloned Wasserstein2Benchmark repository.
The repository path is configured via the WASSERSTEIN_BENCHMARK_PATH environment variable,
or via an explicit argument. This makes the project work identically in local and Colab environments.

Environment setup:
    Local:  export WASSERSTEIN_BENCHMARK_PATH=/Users/you/Downloads/Wasserstein2Benchmark
    Colab:  os.environ['WASSERSTEIN_BENCHMARK_PATH'] = '/content/Wasserstein2Benchmark'

IMPORTANT:
  - No files inside the official Wasserstein2Benchmark repository are ever modified.
  - The working-directory context manager is the ONLY mechanism used to resolve
    relative checkpoint paths (exactly as the notebook did with os.chdir).
  - All benchmark module imports are loaded from the official repository via inspect-verified paths.
"""
import inspect
import os
import sys
import types
import importlib.util
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn


# -----------------------------------------------------------------------
# Environment-based path resolution
# -----------------------------------------------------------------------

def get_benchmark_path(benchmark_path=None):
    """
    Resolve the Wasserstein2Benchmark repository path.

    Priority:
      1. Explicit argument (if provided)
      2. WASSERSTEIN_BENCHMARK_PATH environment variable
      3. Raises an error if neither is set

    Usage:
        Local:  export WASSERSTEIN_BENCHMARK_PATH=/Users/you/Downloads/Wasserstein2Benchmark
        Colab:  os.environ['WASSERSTEIN_BENCHMARK_PATH'] = '/content/Wasserstein2Benchmark'
    """
    if benchmark_path is not None:
        path = Path(benchmark_path).resolve()
    elif 'WASSERSTEIN_BENCHMARK_PATH' in os.environ:
        path = Path(os.environ['WASSERSTEIN_BENCHMARK_PATH']).resolve()
    else:
        raise EnvironmentError(
            "Wasserstein2Benchmark path not configured.\n"
            "Set the WASSERSTEIN_BENCHMARK_PATH environment variable:\n"
            "    Local:  export WASSERSTEIN_BENCHMARK_PATH=/path/to/Wasserstein2Benchmark\n"
            "    Colab:  os.environ['WASSERSTEIN_BENCHMARK_PATH'] = '/content/Wasserstein2Benchmark'"
        )

    if not path.exists():
        raise FileNotFoundError(
            f"Wasserstein2Benchmark directory not found: {path}\n"
            "Please check that the repository has been cloned correctly."
        )

    benchmark_dir = path / "benchmarks" / "Mix3toMix10"
    if not benchmark_dir.exists():
        raise FileNotFoundError(
            f"Mix3toMix10 checkpoints directory not found: {benchmark_dir}\n"
            "Please ensure the official Wasserstein2Benchmark repository is fully cloned."
        )

    return path


# -----------------------------------------------------------------------
# Context manager — temporary working directory (no CWD pollution)
# -----------------------------------------------------------------------

@contextmanager
def temporary_working_directory(path):
    """Temporarily change CWD so relative paths inside map_benchmark resolve correctly."""
    prev_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


# -----------------------------------------------------------------------
# Dynamic benchmark module loading (avoids 'src' namespace collision)
# -----------------------------------------------------------------------

_PKG_NAME = "wasserstein_benchmark"


def _load_benchmark_modules(benchmark_path):
    """
    Dynamically load Wasserstein2Benchmark source modules under a private
    namespace to avoid collision with our own project's `src/` package.

    Returns: (mbm, distributions, potentials, metrics_module)
    """
    benchmark_src = Path(benchmark_path).resolve() / "src"

    if _PKG_NAME not in sys.modules:
        pkg = types.ModuleType(_PKG_NAME)
        pkg.__path__ = [str(benchmark_src)]
        sys.modules[_PKG_NAME] = pkg

    def _load(name):
        full = f"{_PKG_NAME}.{name}"
        if full in sys.modules:
            return sys.modules[full]
        spec = importlib.util.spec_from_file_location(
            full, str(benchmark_src / f"{name}.py")
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod
        spec.loader.exec_module(mod)
        return mod

    # Load in dependency order
    _load("tools")
    distributions = _load("distributions")
    _load("icnn")
    potentials = _load("potentials")
    mbm = _load("map_benchmark")
    metrics_mod = _load("metrics")

    return mbm, distributions, potentials, metrics_mod


# -----------------------------------------------------------------------
# CPU compatibility patches (applied without modifying any repo files)
# -----------------------------------------------------------------------

def _apply_cpu_patches(distributions, potentials):
    """
    Apply minimal patches to fix CUDA-only defaults in the official repo.
    Extracted from notebook Cells 18 & 19 — no repo files are modified.
    """
    # Patch 1: Default device to 'cpu' for RandomGaussianMixSampler
    original_rgms_init = distributions.RandomGaussianMixSampler.__init__

    def patched_rgms_init(self, *args, **kwargs):
        if 'device' not in kwargs:
            kwargs['device'] = 'cpu'
        original_rgms_init(self, *args, **kwargs)

    distributions.RandomGaussianMixSampler.__init__ = patched_rgms_init

    # Patch 2: Place ShiftedPotential.shift tensor on the correct device
    original_shifted_init = potentials.ShiftedPotential.__init__

    def patched_shifted_init(self, potential, shift, batch_size=1024):
        super(potentials.ShiftedPotential, self).__init__(batch_size)
        self.dim = potential.dim
        self.potential = potential
        dev = 'cpu'
        for p in potential.parameters():
            dev = p.device
            break
        self.shift = torch.tensor(shift, dtype=torch.float32, device=dev)

    potentials.ShiftedPotential.__init__ = patched_shifted_init


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

def load_mix3to10_benchmark(dim, benchmark_path=None, device='cpu'):
    """
    Load the official Mix3ToMix10Benchmark from the Wasserstein2Benchmark repository.

    Args:
        dim: Benchmark dimension. Must be one of {2, 4, 8, 16, 32, 64, 128, 256, 512}.
        benchmark_path: Path to Wasserstein2Benchmark repo. If None, uses
                        WASSERSTEIN_BENCHMARK_PATH environment variable.
        device: 'cpu' or 'cuda'.

    Returns:
        benchmark: Official Mix3ToMix10Benchmark object
        module_info: Dict with benchmark module source paths for verification
    """
    bench_root = get_benchmark_path(benchmark_path)
    mbm, distributions, potentials, _ = _load_benchmark_modules(bench_root)
    _apply_cpu_patches(distributions, potentials)

    # Verify that we are loading from the official repository (not recreating locally)
    module_info = {
        "map_benchmark_file": inspect.getfile(mbm.Mix3ToMix10Benchmark),
        "repository_root": str(bench_root),
        "checkpoints_dir": str(bench_root / "benchmarks" / "Mix3toMix10"),
    }

    # Use a temporary CWD so that ../benchmarks/Mix3toMix10/... paths resolve correctly
    # This is exactly what the notebook did with os.chdir(RUN_DIR)
    with temporary_working_directory(bench_root / "notebooks"):
        benchmark = mbm.Mix3ToMix10Benchmark(dim=dim, device=device)

    return benchmark, module_info


def get_benchmark_samplers(benchmark, device='cpu'):
    """Return (mu_sampler, nu_sampler) callables from a loaded benchmark."""
    def mu_sampler(batch_size):
        return benchmark.input_sampler.sample(batch_size).to(device)

    def nu_sampler(batch_size):
        return benchmark.output_sampler.sample(batch_size).to(device)

    return mu_sampler, nu_sampler


# -----------------------------------------------------------------------
# Wrapper for score_fitted_maps interface
# -----------------------------------------------------------------------

class ICNNPotentialWrapper(nn.Module):
    """Wraps our ICNN to match the Potential interface (output shape: batch x 1)."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.dim = model.input_dim

    def forward(self, x):
        return self.model(x).unsqueeze(-1)
