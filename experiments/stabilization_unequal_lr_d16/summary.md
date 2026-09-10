# Experiment D — Unequal player learning rates at D=16

All runs use the fixed D=16 baseline ICNN, clipping, batch size, 2,000 iterations, Adam betas, seed, inner iterations, benchmark, and evaluation protocol. Only `g_lr` changes.

| Config | f LR | g LR | L2-UVP (%) | Cosine | L2 Error | Peak f-grad | Peak g-grad | NaN/Inf |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline | 1.0e-03 | 1.0e-03 | 37.6382 | 0.7683 | 6.0141 | 2685.7641 | 197.8104 | False |
| G-slower-2x | 1.0e-03 | 5.0e-04 | 37.2489 | 0.7707 | 5.9519 | 453.3212 | 73.2934 | False |
| G-slower-4x | 1.0e-03 | 2.5e-04 | 37.2305 | 0.7726 | 5.9489 | 142.3187 | 45.4255 | False |

Interpretation is intentionally deferred until the measured changes in official transport metrics and both gradient trajectories are compared. Lower `g_lr` is not treated as successful solely because it produces smaller gradients.
