# Consolidated scientific results — high-dimensional ICNN OT diagnostics

## Scope and provenance

This evidence sheet is a read-only consolidation of the saved experiment outputs. It does not replace or modify the project report, source code, benchmark, or prior numerical artifacts.

All listed runs use the official Mix3ToMix10 benchmark, seed 0, 2,000 iterations, batch size 256, Softplus ICNNs, Adam betas `(0.5, 0.9)`, and non-negative recurrent-weight clipping unless the row explicitly identifies the oracle objective. Dashes mean the field is not applicable, not that it was measured as zero. `g` is the maximising player; its raw objective is not a duality gap.

`Peak f-grad` and `Peak g-grad` are maxima over the stored parameter-gradient-norm telemetry. For the minimax runs, one `f` norm is recorded on each outer update and one `g` norm on the final inner `g` update of each outer iteration. The oracle rows have no `g` player or inner loop.

## Consolidated experiment table

| Study | Configuration | D | Architecture / width | f LR | g LR | Inner | L2-UVP (%) | L2 error | Cosine | Final f-objective | Final g-objective | Peak f-grad | Peak g-grad | Time (s) | NaN/Inf |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline dimension sweep | Baseline | 2 | ICNN 3×128 | 1e-3 | 1e-3 | 10 | 5.3153 | 0.1057 | 0.8626 | 1.7751 | 8.9529 | 10.90 | 15.11 | 142.54 | False |
| Baseline dimension sweep | Baseline | 4 | ICNN 3×128 | 1e-3 | 1e-3 | 10 | 12.7019 | 0.5065 | 0.8010 | 3.3156 | 32.9244 | 29.18 | 12.87 | 142.83 | False |
| Baseline dimension sweep | Baseline | 8 | ICNN 3×128 | 1e-3 | 1e-3 | 10 | 19.7814 | 1.5810 | 0.8336 | 6.1124 | 68.9368 | 317.69 | 27.74 | 153.01 | False |
| Baseline dimension sweep | Baseline | 16 | ICNN 3×128 | 1e-3 | 1e-3 | 10 | 37.6382 | 6.0141 | 0.7683 | 10.7124 | 160.3642 | 2685.76 | 197.81 | 158.81 | False |
| Baseline dimension sweep | Baseline | 32 | ICNN 3×128 | 1e-3 | 1e-3 | 10 | 51.1976 | 16.4573 | 0.7702 | 16.5547 | 300.3968 | 12999.47 | 919.33 | 157.90 | False |
| Capacity ablation | Width 64 | 16 | ICNN 3×64 | 1e-3 | 1e-3 | 10 | 36.7563 | 5.8450 | 0.7745 | 10.5186 | 152.0855 | 6905.90 | 545.54 | 96.97 | False |
| Capacity ablation | Width 128 | 16 | ICNN 3×128 | 1e-3 | 1e-3 | 10 | 37.6382 | 6.0141 | 0.7683 | 10.7124 | 160.3642 | 2685.76 | 197.81 | 141.19 | False |
| Capacity ablation | Width 256 | 16 | ICNN 3×256 | 1e-3 | 1e-3 | 10 | 36.4857 | 5.8396 | 0.7769 | 10.9100 | 172.2786 | 231.69 | 57.85 | 203.91 | False |
| Capacity ablation | Width 512 | 16 | ICNN 3×512 | 1e-3 | 1e-3 | 10 | 41.0076 | 6.6094 | 0.7443 | 10.7852 | 165.3538 | 87.77 | 202.35 | 301.27 | False |
| Oracle control | MSE(∇f(x), T*(x)) | 16 | ICNN 3×128 | 1e-3 | — | — | 29.1391 | 4.6299 | 0.8249 | 0.2830 MSE | — | 1.08 | — | 10.11 | False |
| Oracle control | MSE(∇f(x), T*(x)) | 32 | ICNN 3×128 | 1e-3 | — | — | 42.5931 | 13.6303 | 0.8083 | 0.4349 MSE | — | 1.39 | — | 11.77 | False |
| Unequal learning rates | Baseline | 16 | ICNN 3×128 | 1e-3 | 1e-3 | 10 | 37.6382 | 6.0141 | 0.7683 | 10.7124 | 160.3642 | 2685.76 | 197.81 | 80.77 | False |
| Unequal learning rates | G-slower-2x | 16 | ICNN 3×128 | 1e-3 | 5e-4 | 10 | 37.2489 | 5.9519 | 0.7707 | 10.7487 | 160.1705 | 453.32 | 73.29 | 83.24 | False |
| Unequal learning rates | G-slower-4x | 16 | ICNN 3×128 | 1e-3 | 2.5e-4 | 10 | 37.2305 | 5.9489 | 0.7726 | 10.6973 | 166.5325 | 142.32 | 45.43 | 82.65 | False |
| Unequal learning rates | Baseline | 32 | ICNN 3×128 | 1e-3 | 1e-3 | 10 | 51.1976 | 16.4573 | 0.7702 | 16.5547 | 300.3968 | 12999.47 | 919.33 | 93.14 | False |
| Unequal learning rates | G-slower-2x | 32 | ICNN 3×128 | 1e-3 | 5e-4 | 10 | 50.5208 | 16.2398 | 0.7738 | 16.5176 | 296.9798 | 1663.67 | 229.27 | 94.09 | False |
| Unequal learning rates | G-slower-4x | 32 | ICNN 3×128 | 1e-3 | 2.5e-4 | 10 | 52.5786 | 16.9012 | 0.7669 | 16.2198 | 281.7536 | 325.09 | 110.23 | 95.58 | False |

The D=16/D=32 unequal-LR baseline rows reproduce the refreshed sweep scores exactly. All comparisons of unequal-LR configurations therefore use matched baseline rows; all reported studies use one seed.

## Interpretation by hypothesis

### A. Function-class expressivity

The D=16 width sweep scales the ICNN from 11,537 parameters (width 64) to 550,929 (width 512), yet L2-UVP remains between 36.49% and 41.01%. There is no monotone improvement with capacity; width 256 is modestly best and width 512 is worse than the 3×128 baseline. This is strong evidence that simply adding width is not the dominant remedy under the tested clipped ICNN formulation.

It does not prove that the convex function class or clipping is unimportant. The oracle controls remain materially inaccurate (29.14% L2-UVP at D=16 and 42.59% at D=32), despite stable optimisation, so some combination of parameterisation, clipping, conditioning, approximation, or finite training budget remains plausible.

### B. Minimax optimisation instability

The baseline sweep shows both worsening transport error and rapidly increasing transient parameter-gradient peaks with dimension: the stored peak `f` norm grows from 10.90 at D=2 to 2,685.76 at D=16 and 12,999.47 at D=32. Removing the two-player game in the oracle control reduces those peaks to 1.08 and 1.39 respectively, while improving each corresponding official transport metric.

Slowing the `g` player independently corroborates the stability effect. At D=16, reducing `g_lr` 4× lowers the peak `f` norm from 2,685.76 to 142.32; at D=32 it lowers the peak from 12,999.47 to 325.09. However, accuracy does not improve consistently: the D=32 4× slower configuration is the most stable but has the worst L2-UVP (52.58%) among the D=32 unequal-LR runs. Thus, the results strongly establish that minimax dynamics are a major source of the severe gradient instability, but do not establish that this instability is the sole source of high-dimensional transport error.

### C. Computational cost

The baseline minimax runs take approximately 143–159 seconds over D=2–32 for the fixed 3×128 architecture; the width sweep takes 97 seconds at width 64 and 301 seconds at width 512. The oracle diagnostic, which removes the second potential and ten inner player updates, takes only 10–12 seconds at D=16/D=32. This identifies minimax alternation and model width as meaningful cost drivers, although these CPU timings are environment-specific and are not hardware-independent complexity measurements.

## What the evidence establishes strongly

- Baseline high-dimensional transport quality degrades sharply from D=2 to D=32 under the fixed training configuration.
- Expanding the tested ICNN width by about 48× does not systematically improve D=16 transport quality.
- The minimax formulation exhibits extreme transient parameter-gradient spikes at D=16/D=32; those spikes are absent in the paired oracle objective.
- Lower `g` learning rates greatly reduce recorded gradient peaks at D=16 and D=32 without creating NaN/Inf values.

## What the evidence suggests, but does not prove

- Minimax dynamics are a major contributor to the observed optimisation instability.
- The remaining oracle error suggests that the clipped ICNN parameterisation or its optimisation may impose an additional limitation.
- Reducing raw gradient spikes alone is insufficient to reliably recover high-dimensional transport accuracy.

## What remains unresolved

- The relative contribution of the ICNN function class, non-negative clipping, conditioning, and finite training budget to the residual oracle error.
- Whether the small official-metric changes under unequal learning rates persist across independent seeds and evaluation draws.
- Whether a controlled gradient penalty or EMA can improve accuracy rather than merely suppress gradient norms.
- Whether an alternative convex-potential parameterisation is needed after optimisation stabilisation is exhausted.

## Logical next step

No additional experiment is run as part of this consolidation. The appropriate next decision is with the advisor: pursue a controlled minimax-stabilisation study (for example, a documented gradient-penalty or EMA experiment) or investigate an alternative convex parameterisation. If a new method is selected, repeated seeds for the matched baseline and selected intervention should accompany it so that the small observed L2-UVP differences can be separated from run and evaluation variability.

## Source artifacts

- `experiments/d2_baseline_record.json`
- `experiments/high_dimensional/{D2,D4,D8,D16,D32}/loss_history.json` and `summary.json`
- `experiments/ablation_expressivity_d16/summary.json` and per-width `loss_history.json`
- `experiments/oracle_regression/summary.json`
- `experiments/stabilization_unequal_lr_d16/summary.json`
- `experiments/stabilization_unequal_lr_d32/summary.json`
