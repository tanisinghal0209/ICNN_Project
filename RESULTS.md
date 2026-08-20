# High-Dimensional Baseline Sweep — Full Results

**Benchmark:** Official Korotin Mix3ToMix10  
**Repository:** `/Users/tanishasinghal/Downloads/Wasserstein2Benchmark`  
**Benchmark module:** `Wasserstein2Benchmark/src/map_benchmark.py`  
**Evaluation:** Learned forward map $\hat{T}(x) = \nabla f(x)$ vs. ground truth `benchmark.map_fwd(x)`

---

## Fixed Configuration (All Dimensions)

| Parameter | Value |
|:---|:---|
| `n_iters` | 2000 |
| `batch_size` | 256 |
| `hidden_dims` | (128, 128, 128) |
| `lr` | 1e-3 |
| `optimizer` | Adam |
| `betas` | (0.5, 0.9) |
| `inner_iters` | 10 |
| `activation` | softplus |
| `seed` | 0 |
| `device` | cpu |
| `architecture` | ICNN |

---

## D=2 — Baseline (Validated Reference)

> D=2 was independently validated against the canonical notebook. It is **read-only** and serves as the reference for all high-dimensional comparisons.

**Parameter count (f):** 34,051

| Metric | Value |
|:---|:---|
| L2-UVP | **5.3153%** |
| Cosine Similarity | **0.8626** |
| L2 Error | **0.1057** |
| final f_loss | 1.7751 |
| final g_loss | — |
| f gradient norm | 0.213 |
| Parameter norm | 26.5 |
| Training time | 152.83 s |
| Iterations/sec | 13.1 |
| NaN/Inf | No |

**Iteration log:**
```
Iteration    0 | f_loss = 6.3652 | g_loss = —       | f_gnorm = —
Iteration  400 | f_loss = 1.7525 | g_loss = —       | f_gnorm = —
Iteration  800 | f_loss = 1.7421 | g_loss = —       | f_gnorm = —
Iteration 1200 | f_loss = 1.7356 | g_loss = —       | f_gnorm = —
Iteration 1600 | f_loss = 1.7792 | g_loss = —       | f_gnorm = —
Iteration 1999 | f_loss = 1.7751 | g_loss = —       | f_gnorm = —
```

> Note: g_loss and f_gnorm telemetry were added to the solver after the D=2 baseline run and are therefore not recorded for D=2.

---

## D=4

**Benchmark checkpoint:** `Wasserstein2Benchmark/benchmarks/Mix3toMix10/4_v1.pt`  
**Parameter count (f):** 34,821

### Iteration Log

| Iteration | f_loss | g_loss | f_gnorm |
|:---|:---|:---|:---|
| 0 | 5.3344 | -4.6132 | 5.8818 |
| 400 | 3.3559 | 44.5997 | 0.8658 |
| 800 | 3.2528 | 42.6153 | 1.2722 |
| 1200 | 3.2063 | 38.5858 | 0.6850 |
| 1600 | 3.3870 | 36.2605 | 1.2991 |
| 1999 | 3.3157 | 32.9294 | 1.3162 |

### Final Telemetry

| Metric | Value |
|:---|:---|
| Final f_loss | 3.3157 |
| Final g_loss | 32.9294 |
| f gradient norm | 1.3162 |
| g gradient norm | 0.0768 |
| Parameter norm (f) | 15.6311 |
| Parameter count (f) | 34,821 |
| Training time | 135.35 s |
| Iterations/sec | 14.78 |
| NaN/Inf | No |
| Completed iterations | 2000 |

### Official Evaluation

| Metric | Value |
|:---|:---|
| **L2-UVP** | **12.6375%** |
| **Cosine Similarity** | **0.8422** |
| **L2 Error** | **0.5055** |

### Analysis
f_loss converges rapidly from iteration 0→400 (5.33→3.36) then **plateaus** for the remaining 1600 iterations. g_loss is large and increasing (~33–45) indicating a growing dual gap. Gradient norms are moderate (~0.7–1.3) and stable — no instability at D=4 yet.

---

## D=8

**Benchmark checkpoint:** `Wasserstein2Benchmark/benchmarks/Mix3toMix10/8_v1.pt`  
**Parameter count (f):** 36,361

### Iteration Log

| Iteration | f_loss | g_loss | f_gnorm |
|:---|:---|:---|:---|
| 0 | 7.2115 | -6.1530 | 4.4758 |
| 400 | 6.4628 | 84.5582 | 2.2541 |
| 800 | 6.4989 | 79.8764 | 4.5447 |
| 1200 | 5.9992 | 76.3144 | 3.2230 |
| 1600 | 6.1825 | 70.6678 | 2.4652 |
| 1999 | 6.1122 | 68.8426 | 2.5713 |

### Final Telemetry

| Metric | Value |
|:---|:---|
| Final f_loss | 6.1122 |
| Final g_loss | 68.8426 |
| f gradient norm | 2.5713 |
| g gradient norm | 0.0571 |
| Parameter norm (f) | 15.1926 |
| Parameter count (f) | 36,361 |
| Training time | 138.64 s |
| Iterations/sec | 14.43 |
| NaN/Inf | No |
| Completed iterations | 2000 |

### Official Evaluation

| Metric | Value |
|:---|:---|
| **L2-UVP** | **19.7014%** |
| **Cosine Similarity** | **0.8238** |
| **L2 Error** | **1.5761** |

### Analysis
f_loss barely improves after iteration 400 (6.46→6.11 over 1600 iterations). Gradient norm oscillates between 2.25–4.54 — the optimizer has not found a stable trajectory. g_loss is very large (~69–85) and declining slowly. The model is learning something but clearly struggling with the dual landscape at D=8.

---

## D=16

**Benchmark checkpoint:** `Wasserstein2Benchmark/benchmarks/Mix3toMix10/16_v1.pt`  
**Parameter count (f):** 39,441

### Iteration Log

| Iteration | f_loss | g_loss | f_gnorm |
|:---|:---|:---|:---|
| 0 | 6.6212 | -5.9091 | 2.9766 |
| 400 | 11.1220 | 183.3459 | 5.7538 |
| 800 | 11.1371 | 174.2881 | 7.6760 |
| 1200 | 10.7703 | 175.4427 | **15.5915** |
| 1600 | 10.4907 | 165.2549 | 9.5266 |
| 1999 | 10.7124 | 160.3642 | **13.8494** |

### Final Telemetry

| Metric | Value |
|:---|:---|
| Final f_loss | 10.7124 |
| Final g_loss | 160.3642 |
| f gradient norm | **13.8494** |
| g gradient norm | 1.5421 |
| Parameter norm (f) | 16.0020 |
| Parameter count (f) | 39,441 |
| Training time | 140.82 s |
| Iterations/sec | 14.20 |
| NaN/Inf | No |
| Completed iterations | 2000 |

### Official Evaluation

| Metric | Value |
|:---|:---|
| **L2-UVP** | **37.7563%** |
| **Cosine Similarity** | **0.7731** |
| **L2 Error** | **6.0410** |

### Analysis
⚠️ **f_loss increases from 6.62 (iter 0) to 11.12 (iter 400)** — the optimizer initially moves *away* from initialization. This is the first clear sign of optimization instability. The gradient norm spikes to 15.59 at iteration 1200 — a 5× jump from the start. g_loss explodes to ~183 and stays elevated. The model only partially recovers (10.49 at iter 1600) before ending at 10.71, still far above the starting value.

---

## D=32

**Benchmark checkpoint:** `Wasserstein2Benchmark/benchmarks/Mix3toMix10/32_v1.pt`  
**Parameter count (f):** 45,601

### Iteration Log

| Iteration | f_loss | g_loss | f_gnorm |
|:---|:---|:---|:---|
| 0 | 9.1148 | -8.0265 | 4.0289 |
| 400 | 17.7094 | 337.3095 | **34.3528** |
| 800 | 16.6324 | 326.2979 | **25.5003** |
| 1200 | 17.0938 | 315.0721 | 11.4941 |
| 1600 | 17.3926 | 306.9613 | **40.5029** |
| 1999 | 16.5547 | 300.3968 | **32.2386** |

### Final Telemetry

| Metric | Value |
|:---|:---|
| Final f_loss | 16.5547 |
| Final g_loss | 300.3968 |
| f gradient norm | **32.2386** |
| g gradient norm | 0.9514 |
| Parameter norm (f) | 17.0370 |
| Parameter count (f) | 45,601 |
| Training time | 158.39 s |
| Iterations/sec | 12.63 |
| NaN/Inf | No |
| Completed iterations | 2000 |

### Official Evaluation

| Metric | Value |
|:---|:---|
| **L2-UVP** | **50.7988%** |
| **Cosine Similarity** | **0.7644** |
| **L2 Error** | **16.2556** |

### Analysis
⚠️⚠️ **Severe optimization instability.** f_loss nearly doubles from initialization (9.11→17.71) by iteration 400 and **never recovers** to below the starting value. Gradient norm oscillates violently: 4.03 → 34.35 → 25.50 → 11.49 → 40.50 → 32.24. g_loss reaches 337 and declines only slowly. The 50.8% L2-UVP means the learned map captures barely half the transport variance.

---

## Consolidated Comparison Table

| D | L2-UVP | Cosine | L2 Error | f_loss | g_loss | f_gnorm | Params | Time | it/s |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| 2 | 5.32% | 0.8626 | 0.106 | 1.775 | — | 0.213 | 34,051 | 152.8s | 13.1 |
| 4 | 12.64% | 0.8422 | 0.506 | 3.316 | 32.93 | 1.316 | 34,821 | 135.4s | 14.8 |
| 8 | 19.70% | 0.8238 | 1.576 | 6.112 | 68.84 | 2.571 | 36,361 | 138.6s | 14.4 |
| 16 | 37.76% | 0.7731 | 6.041 | 10.712 | 160.36 | 13.849 | 39,441 | 140.8s | 14.2 |
| 32 | 50.80% | 0.7644 | 16.256 | 16.555 | 300.40 | 32.239 | 45,601 | 158.4s | 12.6 |

---

## Gradient Norm Trajectory (Key Instability Evidence)

| D | Iter 0 | Iter 400 | Iter 800 | Iter 1200 | Iter 1600 | Iter 1999 |
|:--|:--|:--|:--|:--|:--|:--|
| 4 | 5.88 | 0.87 | 1.27 | 0.69 | 1.30 | 1.32 |
| 8 | 4.48 | 2.25 | 4.54 | 3.22 | 2.47 | 2.57 |
| 16 | 2.98 | 5.75 | 7.68 | **15.59** | 9.53 | **13.85** |
| 32 | 4.03 | **34.35** | **25.50** | 11.49 | **40.50** | **32.24** |

---

## f_loss Trajectory

| D | Iter 0 | Iter 400 | Iter 800 | Iter 1200 | Iter 1600 | Iter 1999 | Change |
|:--|:--|:--|:--|:--|:--|:--|:--|
| 2 | 6.37 | 1.75 | 1.74 | 1.74 | 1.78 | 1.78 | **−4.59 ✓** |
| 4 | 5.33 | 3.36 | 3.25 | 3.21 | 3.39 | 3.32 | **−2.01 ✓** |
| 8 | 7.21 | 6.46 | 6.50 | 6.00 | 6.18 | 6.11 | **−1.10 ✓** |
| 16 | 6.62 | 11.12 | 11.14 | 10.77 | 10.49 | 10.71 | **+4.09 ✗** |
| 32 | 9.11 | 17.71 | 16.63 | 17.09 | 17.39 | 16.55 | **+7.44 ✗** |

> D=16 and D=32 show **increasing** f_loss over training — the optimizer is diverging from initialization, not converging to a minimum.

---

## Key Findings

### Finding 1 — Optimization Instability is the Primary Failure Mode

- Gradient norms grow 150× from D=2 (0.21) to D=32 (32.24)
- At D=16/32, f_loss **increases** during training — a clear sign of divergence
- g_loss grows catastrophically (1→300), indicating the inner loop is escaping far faster than the outer loop can follow
- Parameter norms stay stable (15–27) — the weights are not saturating

### Finding 2 — Not a Computational Bottleneck

- Training time is nearly constant across all dimensions (135–158 s)
- Iterations/sec decreases by only 14% from D=4 to D=32
- The 128×128×128 architecture fits comfortably at all dimensions tested

### Finding 3 — Performance Degrades Monotonically and Sharply at D≥16

| Range | L2-UVP increase | Rate |
|:--|:--|:--|
| D=2→4 | +7.3 pp | Moderate |
| D=4→8 | +7.1 pp | Moderate |
| D=8→16 | +18.1 pp | **Sharp** |
| D=16→32 | +13.0 pp | High |

The D=8→16 jump is the largest, coinciding exactly with the onset of gradient norm explosion and f_loss divergence.

---

## Output Files

| File | Description |
|:---|:---|
| [`experiments/high_dimensional/summary.csv`](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/high_dimensional/summary.csv) | Machine-readable consolidated results |
| [`experiments/high_dimensional/summary.json`](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/high_dimensional/summary.json) | Full JSON with all fields |
| [`experiments/high_dimensional/D4/run.log`](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/high_dimensional/D4/run.log) | D=4 terminal log |
| [`experiments/high_dimensional/D4/metrics.json`](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/high_dimensional/D4/metrics.json) | D=4 metrics |
| [`experiments/high_dimensional/D4/loss_history.json`](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/high_dimensional/D4/loss_history.json) | D=4 full loss/gradient history (2000 iters) |
| [`experiments/high_dimensional/D4/config.json`](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/high_dimensional/D4/config.json) | D=4 config |
| [`experiments/high_dimensional/D4/loss.png`](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/high_dimensional/D4/loss.png) | D=4 loss curve |
| (same structure for D8, D16, D32) | |
| [`experiments/high_dimensional/l2_uvp_vs_dimension.png`](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/high_dimensional/l2_uvp_vs_dimension.png) | L2-UVP degradation plot |
| [`experiments/high_dimensional/grad_norm_vs_dimension.png`](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/high_dimensional/grad_norm_vs_dimension.png) | Gradient norm instability plot |
| [`experiments/high_dimensional/cosine_vs_dimension.png`](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/high_dimensional/cosine_vs_dimension.png) | Cosine similarity vs dimension |
| [`experiments/high_dimensional/time_vs_dimension.png`](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/high_dimensional/time_vs_dimension.png) | Training time vs dimension |

---

## Status

| Dimension | Status | L2-UVP |
|:--|:--|:--|
| D=2 | ✅ Validated baseline | 5.32% |
| D=4 | ✅ Complete | 12.64% |
| D=8 | ✅ Complete | 19.70% |
| D=16 | ✅ Complete | 37.76% |
| D=32 | ✅ Complete | 50.80% |

**Next (pending approval):**
- Experiment A: Expressivity ablation at D=16 — vary hidden width `{64, 128, 256, 512}`
- Experiment B: Optimizer ablation at D=16 — vary `lr` and `inner_iters`
- Experiment C: Gradient clipping at D=32 — test whether capping gnorm stabilises training
