# Experiment D — Unequal player learning rates at D=32

All runs use the fixed D=32 baseline ICNN, clipping, batch size, 2,000 iterations, Adam betas, seed, inner iterations, benchmark, and evaluation protocol. Only `g_lr` changes.

| Config | f LR | g LR | L2-UVP (%) | Cosine | L2 Error | Peak f-grad | Peak g-grad | NaN/Inf |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline | 1.0e-03 | 1.0e-03 | 51.1976 | 0.7702 | 16.4573 | 12999.4749 | 919.3289 | False |
| G-slower-2x | 1.0e-03 | 5.0e-04 | 50.5208 | 0.7738 | 16.2398 | 1663.6683 | 229.2651 | False |
| G-slower-4x | 1.0e-03 | 2.5e-04 | 52.5786 | 0.7669 | 16.9012 | 325.0894 | 110.2319 | False |

Interpretation is intentionally deferred until the measured changes in official transport metrics and both gradient trajectories are compared. Lower `g_lr` is not treated as successful solely because it produces smaller gradients.
