"""
D=2 regression validation script.

Runs the official Mix3ToMix10Benchmark at D=2 using the modular src/ code,
using the exact same settings as the canonical notebook (Cell 20).

Canonical notebook result:
    L2-UVP            = 5.1044%
    Cosine Similarity = 0.8580
    L2 Error          = 0.1027
    Training Time     = 272.71 s

The new modular run must use identical architecture and hyperparameters.
"""
import os
import sys
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.benchmark import load_mix3to10_benchmark, get_benchmark_samplers
from src.solver import train_icnn_ot
from src.metrics import compute_official_l2_uvp

# Configure benchmark path — override with env var in Colab
BENCHMARK_PATH = os.environ.get(
    'WASSERSTEIN_BENCHMARK_PATH',
    '/Users/tanishasinghal/Downloads/Wasserstein2Benchmark'
)
DEVICE = 'cpu'
DIM = 2

# Canonical notebook settings (Cell 10 & 20)
N_ITERS    = 2000
BATCH_SIZE = 256
HIDDEN     = (128, 128, 128)
LR         = 1e-3
INNER      = 10
ACTIVATION = 'softplus'
LOG_EVERY  = 400

# Canonical notebook evaluation settings (Cell 20)
N_SAMPLES_EVAL  = 8192
VAR_SAMPLES     = 16384

# Reference results from the canonical notebook
NOTEBOOK_REFERENCE = {
    "L2-UVP": 5.1044,
    "Cosine Similarity": 0.8580,
    "L2 Error": 0.1027,
    "Training Time": 272.71,
    "note": "n_iters=2000, batch=256, hidden=(128,128,128), seed=0"
}


def main():
    print("=" * 65)
    print(f"D=2 REGRESSION VALIDATION (modular src/)")
    print("=" * 65)

    # 1. Load official benchmark
    print(f"\n[1] Loading official Mix3ToMix10Benchmark (D={DIM})...")
    benchmark, module_info = load_mix3to10_benchmark(
        dim=DIM, benchmark_path=BENCHMARK_PATH, device=DEVICE
    )
    print(f"    Benchmark module: {module_info['map_benchmark_file']}")
    print(f"    Repository root:  {module_info['repository_root']}")
    print(f"    Checkpoints dir:  {module_info['checkpoints_dir']}")
    print(f"    Benchmark dim:    {benchmark.dim}")

    mu_sampler, nu_sampler = get_benchmark_samplers(benchmark, device=DEVICE)

    # Sanity check sampler shapes
    x_check = mu_sampler(4)
    y_check = nu_sampler(4)
    assert x_check.shape == (4, DIM), f"mu shape wrong: {x_check.shape}"
    assert y_check.shape == (4, DIM), f"nu shape wrong: {y_check.shape}"
    print(f"    Sampler shapes:   mu={tuple(x_check.shape)}, nu={tuple(y_check.shape)} ✓")

    # 2. Train using canonical settings
    print(f"\n[2] Training ICNN solver on official benchmark (D={DIM})...")
    print(f"    Settings: n_iters={N_ITERS}, batch={BATCH_SIZE}, "
          f"hidden={HIDDEN}, lr={LR}, inner={INNER}, seed=0")
    print()

    f, g, history, final_f_loss, training_time = train_icnn_ot(
        mu_sampler, nu_sampler,
        input_dim=DIM,
        n_iters=N_ITERS,
        batch_size=BATCH_SIZE,
        hidden_dims=HIDDEN,
        lr=LR,
        inner_iters=INNER,
        device=DEVICE,
        activation=ACTIVATION,
        log_every=LOG_EVERY,
        model_type='icnn',
    )

    print(f"\n    Final f_loss:  {final_f_loss:.4f}")
    print(f"    Training time: {training_time:.2f} s")

    # 3. Compute official metrics
    print(f"\n[3] Computing official L2-UVP metrics...")
    metrics = compute_official_l2_uvp(
        benchmark, f, device=DEVICE,
        n_samples=N_SAMPLES_EVAL,
        var_samples=VAR_SAMPLES
    )

    # 4. Report results
    print("\n" + "=" * 65)
    print("RESULTS COMPARISON")
    print("=" * 65)

    print(f"\n{'Metric':<22} {'This Run':>12} {'Notebook':>12} {'Diff':>12}")
    print("-" * 62)

    for key in ['L2-UVP', 'Cosine Similarity', 'L2 Error']:
        ref = NOTEBOOK_REFERENCE[key]
        val = metrics[key]
        diff = val - ref
        unit = "%" if key == "L2-UVP" else ""
        print(f"{key:<22} {val:>11.4f}{unit} {ref:>11.4f}{unit} {diff:>+11.4f}{unit}")

    print(f"\n{'Training Time':<22} {training_time:>11.2f}s {NOTEBOOK_REFERENCE['Training Time']:>11.2f}s")

    print(f"\nBenchmark module source (verified):")
    print(f"  {module_info['map_benchmark_file']}")

    # 5. Save metrics and full telemetry so the D=2 baseline can be
    # consolidated without hand-written values in later reports.
    telemetry = {
        "initial_f_loss": history["f_loss"][0],
        "final_f_loss": history["f_loss"][-1],
        "initial_g_loss": history["g_loss"][0],
        "final_g_loss": history["g_loss"][-1],
        "peak_f_gradient_norm": max(history["f_grad_norm"]),
        "final_f_gradient_norm": history["f_grad_norm"][-1],
        "peak_g_gradient_norm": max(history["g_grad_norm"]),
        "final_g_gradient_norm": history["g_grad_norm"][-1],
        "final_f_parameter_norm": history["f_param_norm"][-1],
        "final_g_parameter_norm": history["g_param_norm"][-1],
        "NaN/Inf": any(history["has_nan"]),
    }
    parameter_count = sum(p.numel() for p in f.parameters())
    out = {
        "modular_run": {**metrics, "training_time": training_time, **telemetry},
        "notebook_reference": NOTEBOOK_REFERENCE,
        "difference": {k: metrics[k] - NOTEBOOK_REFERENCE[k] for k in ['L2-UVP', 'Cosine Similarity', 'L2 Error']},
        "benchmark_module": module_info['map_benchmark_file'],
        "settings": {
            "dim": DIM, "n_iters": N_ITERS, "batch_size": BATCH_SIZE,
            "hidden_dims": list(HIDDEN), "lr": LR, "inner_iters": INNER,
            "activation": ACTIVATION, "seed": 0,
        }
    }
    out_path = ROOT / "experiments" / "d2_regression_results.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)

    record = {
        "dimension": DIM,
        "seed": 0,
        "n_iters": N_ITERS,
        "batch_size": BATCH_SIZE,
        "hidden_dims": list(HIDDEN),
        "learning_rate": LR,
        "inner_iters": INNER,
        **metrics,
        "training_time": training_time,
        "parameter_count": parameter_count,
        "benchmark_repository_path": module_info["repository_root"],
        "benchmark_module_source": module_info["map_benchmark_file"],
        "telemetry": telemetry,
    }
    record_path = ROOT / "experiments" / "d2_baseline_record.json"
    with open(record_path, "w") as fh:
        json.dump(record, fh, indent=2)

    dim_dir = ROOT / "experiments" / "high_dimensional" / "D2"
    dim_dir.mkdir(parents=True, exist_ok=True)
    with open(dim_dir / "metrics.json", "w") as fh:
        json.dump({
            "D": DIM,
            **metrics,
            "final_f_loss": telemetry["final_f_loss"],
            "final_g_loss": telemetry["final_g_loss"],
            "f_gradient_norm": telemetry["final_f_gradient_norm"],
            "g_gradient_norm": telemetry["final_g_gradient_norm"],
            "parameter_norm": telemetry["final_f_parameter_norm"],
            "parameter_count": parameter_count,
            "training_time": training_time,
            "iterations_per_second": N_ITERS / training_time,
            "NaN/Inf": telemetry["NaN/Inf"],
            "completed_iterations": N_ITERS,
        }, fh, indent=2)
    with open(dim_dir / "loss_history.json", "w") as fh:
        json.dump(history, fh, indent=2)
    with open(dim_dir / "config.json", "w") as fh:
        json.dump(record, fh, indent=2)
    print(f"\nResults saved to: {out_path}")
    print(f"D=2 baseline record updated: {record_path}")
    print("=" * 65)


if __name__ == '__main__':
    main()
