"""
Generate exact telemetry trajectory plots and quantitatively analyze optimization dynamics.

Generates:
Set 1 (Across D=2, D=8, D=16, D=32):
  1. f_loss vs iteration
  2. g_loss vs iteration
  3. f_gradient_norm vs iteration
  4. g_gradient_norm vs iteration

Set 2 (D=16 Optimization comparison: Baseline, Inner_20, Inner_50):
  5. f_loss vs iteration
  6. g_loss vs iteration
  7. f_gradient_norm vs iteration
  8. g_gradient_norm vs iteration

Summary plots:
  9. L2-UVP vs Dimension
 10. Gradient norm vs Dimension
"""
import os
import sys
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

out_dir = ROOT / "experiments" / "telemetry_analysis"
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Load Dimension Histories (D=2, D=8, D=16, D=32)
dim_histories = {}
for d in [2, 8, 16, 32]:
    p = ROOT / "experiments" / "high_dimensional" / f"D{d}" / "loss_history.json"
    with open(p) as f:
        dim_histories[d] = json.load(f)

# 2. Load D=16 Optimization Histories (Baseline, Inner_20, Inner_50)
opt_histories = {}
opt_dir = ROOT / "experiments" / "ablation_optimization_d16"
for config in ["Baseline", "Inner_20", "Inner_50"]:
    p = opt_dir / config / "loss_history.json"
    with open(p) as f:
        opt_histories[config] = json.load(f)

# -----------------------------------------------------------------------
# SET 1: Across Dimensions (D=2, D=8, D=16, D=32)
# -----------------------------------------------------------------------

# 1.1 f_loss vs iteration
plt.figure(figsize=(7, 4.5))
for d, h in dim_histories.items():
    plt.plot(h["f_loss"], label=f"D={d}", linewidth=1.0)
plt.xlabel("Iteration")
plt.ylabel("f_loss")
plt.title("f_loss vs Iteration across Dimensions")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig(out_dir / "f_loss_vs_iteration_dims.png", dpi=200, bbox_inches="tight")
plt.close()

# 1.2 g_loss vs iteration
plt.figure(figsize=(7, 4.5))
for d, h in dim_histories.items():
    plt.plot(h["g_loss"], label=f"D={d}", linewidth=1.0)
plt.xlabel("Iteration")
plt.ylabel("g_loss")
plt.title("g_loss vs Iteration across Dimensions")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig(out_dir / "g_loss_vs_iteration_dims.png", dpi=200, bbox_inches="tight")
plt.close()

# 1.3 f_gradient_norm vs iteration
plt.figure(figsize=(7, 4.5))
for d, h in dim_histories.items():
    plt.plot(h["f_grad_norm"], label=f"D={d}", linewidth=1.0)
plt.xlabel("Iteration")
plt.ylabel(r"f Gradient Norm $\|\nabla_\theta L_f\|$")
plt.title("f Gradient Norm vs Iteration across Dimensions")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig(out_dir / "f_grad_norm_vs_iteration_dims.png", dpi=200, bbox_inches="tight")
plt.close()

# 1.4 g_gradient_norm vs iteration
plt.figure(figsize=(7, 4.5))
for d, h in dim_histories.items():
    plt.plot(h["g_grad_norm"], label=f"D={d}", linewidth=1.0)
plt.xlabel("Iteration")
plt.ylabel(r"g Gradient Norm $\|\nabla_\phi L_g\|$")
plt.title("g Gradient Norm vs Iteration across Dimensions")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig(out_dir / "g_grad_norm_vs_iteration_dims.png", dpi=200, bbox_inches="tight")
plt.close()


# -----------------------------------------------------------------------
# SET 2: D=16 Optimization Dynamics (Baseline vs Inner_20 vs Inner_50)
# -----------------------------------------------------------------------

labels = {
    "Baseline": "Baseline (inner=10, lr=1e-3)",
    "Inner_20": "Inner_20 (inner=20, lr=1e-3)",
    "Inner_50": "Inner_50 (inner=50, lr=1e-3)"
}

# 2.1 f_loss vs iteration (D=16 opt)
plt.figure(figsize=(7, 4.5))
for cfg, h in opt_histories.items():
    plt.plot(h["f_loss"], label=labels[cfg], linewidth=1.0)
plt.xlabel("Iteration")
plt.ylabel("f_loss")
plt.title("D=16 Optimization: f_loss vs Iteration")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig(out_dir / "f_loss_vs_iteration_d16_opt.png", dpi=200, bbox_inches="tight")
plt.close()

# 2.2 g_loss vs iteration (D=16 opt)
plt.figure(figsize=(7, 4.5))
for cfg, h in opt_histories.items():
    plt.plot(h["g_loss"], label=labels[cfg], linewidth=1.0)
plt.xlabel("Iteration")
plt.ylabel("g_loss")
plt.title("D=16 Optimization: g_loss vs Iteration")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig(out_dir / "g_loss_vs_iteration_d16_opt.png", dpi=200, bbox_inches="tight")
plt.close()

# 2.3 f_gradient_norm vs iteration (D=16 opt)
plt.figure(figsize=(7, 4.5))
for cfg, h in opt_histories.items():
    plt.plot(h["f_grad_norm"], label=labels[cfg], linewidth=1.0)
plt.xlabel("Iteration")
plt.ylabel(r"f Gradient Norm $\|\nabla_\theta L_f\|$")
plt.title(r"D=16 Optimization: f Gradient Norm $\|\nabla_\theta L_f\|$ vs Iteration")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig(out_dir / "f_grad_norm_vs_iteration_d16_opt.png", dpi=200, bbox_inches="tight")
plt.close()

# 2.4 g_gradient_norm vs iteration (D=16 opt)
plt.figure(figsize=(7, 4.5))
for cfg, h in opt_histories.items():
    plt.plot(h["g_grad_norm"], label=labels[cfg], linewidth=1.0)
plt.xlabel("Iteration")
plt.ylabel(r"g Gradient Norm $\|\nabla_\phi L_g\|$")
plt.title(r"D=16 Optimization: g Gradient Norm $\|\nabla_\phi L_g\|$ vs Iteration")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig(out_dir / "g_grad_norm_vs_iteration_d16_opt.png", dpi=200, bbox_inches="tight")
plt.close()


# -----------------------------------------------------------------------
# Quantitative Telemetry Calculations & Logging
# -----------------------------------------------------------------------
print("=" * 80)
print("QUANTITATIVE OPTIMIZATION TELEMETRY ANALYSIS")
print("=" * 80)

print("\n--- 1. Across Dimensions (D=2, D=8, D=16, D=32) ---")
print(f"{'D':<4} | {'f_loss (init -> final)':<24} | {'g_loss (init -> final)':<24} | {'f_gnorm (init -> max -> final)':<32} | {'g_gnorm (init -> max -> final)':<32}")
print("-" * 125)

for d in [2, 8, 16, 32]:
    h = dim_histories[d]
    f_init, f_final = h["f_loss"][0], h["f_loss"][-1]
    g_init, g_final = h["g_loss"][0], h["g_loss"][-1]
    
    fg_init, fg_max, fg_final = h["f_grad_norm"][0], max(h["f_grad_norm"]), h["f_grad_norm"][-1]
    gg_init, gg_max, gg_final = h["g_grad_norm"][0], max(h["g_grad_norm"]), h["g_grad_norm"][-1]
    
    f_str = f"{f_init:.2f} -> {f_final:.2f}"
    g_str = f"{g_init:.2f} -> {g_final:.2f}"
    fg_str = f"{fg_init:.2f} -> {fg_max:.2f} -> {fg_final:.2f}"
    gg_str = f"{gg_init:.2f} -> {gg_max:.2f} -> {gg_final:.2f}"
    
    print(f"{d:<4} | {f_str:<24} | {g_str:<24} | {fg_str:<32} | {gg_str:<32}")


print("\n--- 2. D=16 Optimization Sweep (Baseline vs Inner_20 vs Inner_50) ---")
print(f"{'Config':<12} | {'f_loss (final)':<14} | {'g_loss (final)':<14} | {'f_gnorm (mean +- std)':<25} | {'g_gnorm (mean +- std)':<25}")
print("-" * 100)

for cfg in ["Baseline", "Inner_20", "Inner_50"]:
    h = opt_histories[cfg]
    f_final = h["f_loss"][-1]
    g_final = h["g_loss"][-1]
    
    fg_mean, fg_std = np.mean(h["f_grad_norm"]), np.std(h["f_grad_norm"])
    gg_mean, gg_std = np.mean(h["g_grad_norm"]), np.std(h["g_grad_norm"])
    
    fg_str = f"{fg_mean:.2f} +- {fg_std:.2f}"
    gg_str = f"{gg_mean:.2f} +- {gg_std:.2f}"
    
    print(f"{cfg:<12} | {f_final:<14.2f} | {g_final:<14.2f} | {fg_str:<25} | {gg_str:<25}")

print("\nPlots successfully saved to:", out_dir)
print("=" * 80)
