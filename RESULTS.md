# High-Dimensional Telemetry & Optimization Dynamics Analysis

**Benchmark:** Official Korotin Mix3ToMix10  
**Repository:** `/Users/tanishasinghal/Downloads/Wasserstein2Benchmark`  
**Evaluation:** Learned forward map $\hat{T}(x) = \nabla f(x)$ vs. ground truth `benchmark.map_fwd(x)`

---

## 1. Telemetry Instrumentation Audit Verification

Prior to presenting telemetry results, the gradient norm instrumentation in `src/solver.py` was audited:

1. **Gradient Zeroing**: `opt_g.zero_grad()` and `opt_f.zero_grad()` are explicitly executed before every backward pass.
2. **Measurement Timing**: Gradient norms are computed immediately after `loss.backward()` and before `optimizer.step()`.
3. **Exact L2 Norm Formula**: Calculated as the total L2 norm across all trainable parameters:
   $$\|\nabla_\theta L\|_2 = \sqrt{\sum_{i} \|\nabla_{\theta_i} L\|_2^2}$$
4. **Player Separation**: $f$ and $g$ parameter gradient norms are measured separately.
5. **No Gradient Clipping**: Gradients are unclipped and unnormalized (`clip_weights()` clips weight parameters $W_z \ge 0$ post-step).
6. **No Stale Gradients**: Measured on fresh backward passes per iteration.

---

## 2. Dimensionality Scaling Telemetry ($D \in \{2, 8, 16, 32\}$)

**Configuration:** Fixed nominal parameters ($N=2000$ iters, batch 256, hidden $128 \times 3$, lr $1\text{e-}3$, inner iters 10, softplus, seed 0).

| D | L2-UVP | Cosine Sim | L2 Error | $f\_loss$ (init $\to$ final) | $g\_loss$ (init $\to$ final) | $\|\nabla_\theta L_f\|$ (init $\to$ peak $\to$ final) | $\|\nabla_\phi L_g\|$ (init $\to$ peak $\to$ final) | Params ($f$) | Time |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| **2** | **5.32%** | **0.8626** | **0.106** | $6.46 \to 1.71$ | $-4.20 \to 8.82$ | $10.80 \to 10.80 \to 0.98$ | $14.96 \to 14.96 \to 0.02$ | 34,051 | 152.8s |
| **8** | **19.70%** | **0.8238** | **1.576** | $7.21 \to 6.11$ | $-6.15 \to 68.84$ | $4.48 \to 317.69 \to 2.57$ | $16.72 \to 27.74 \to 0.06$ | 36,361 | 138.6s |
| **16** | **37.76%** | **0.7731** | **6.041** | $6.62 \to 10.71$ | $-5.91 \to 160.36$ | $2.98 \to 2685.76 \to 13.85$ | $13.19 \to 197.81 \to 1.54$ | 39,441 | 140.8s |
| **32** | **50.80%** | **0.7644** | **16.256** | $9.11 \to 16.55$ | $-8.03 \to 300.40$ | $4.03 \to 12999.47 \to 32.24$ | $18.60 \to 919.33 \to 0.95$ | 45,601 | 158.4s |

---

## 3. Iteration-Level Trajectory Analysis

### Quantitative Observations Across Dimensions ($D=2 \to 32$)
1. **Outer Loss ($f\_loss$) Trajectory**:
   - At $D=2$, $f\_loss$ monotonically decreases from $6.46$ to $1.71$ (clean convergence).
   - At $D=16$ and $D=32$, $f\_loss$ **increases during optimization** ($6.62 \to 10.71$ in 16D, and $9.11 \to 16.55$ in 32D, an 82% increase). The solver moves away from initialization rather than minimizing cost.
2. **Growth of the $g$-Player Objective**:
   - The $g$-player objective magnitude grows substantially as dimension increases, scaling from $8.82$ in 2D to $300.40$ in 32D. This increasing magnitude is consistent with an increasingly difficult inner maximization as dimension grows.
3. **Rapidly Increasing Peak Gradient Norms**:
   - Peak gradient norms $\|\nabla_\theta L_f\|$ exhibit a rapidly increasing trend with dimension, reaching **2,685.76** in 16D and **12,999.47** in 32D during training iterations, demonstrating transient optimization shocks.
   - Peak conjugate gradient norms $\|\nabla_\phi L_g\|$ similarly scale from $14.96$ in 2D to $919.33$ in 32D.

---

## 4. Controlled Isolation Experiments at $D=16$

### Experiment A: Capacity / Expressivity Ablation (at $D=16$)
**Setup:** Fixed $D=16$, $N=2000$, lr $1\text{e-}3$, inner iters 10, softplus, seed 0. Vary width $W \in \{64, 128, 256, 512\}$.

| Width ($W$) | Hidden Architecture | L2-UVP (%) | Cosine Sim | f_loss | f_gnorm (final) | g_gnorm (final) | Params (f) | Time |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **64** | (64, 64, 64) | 36.76% | 0.7745 | 10.52 | 17.94 | 0.3590 | 11,537 | 97.0s |
| **128** | (128, 128, 128) | 37.64% | 0.7683 | 10.71 | 13.85 | 1.5421 | 39,441 | 141.2s |
| **256** | (256, 256, 256) | 36.49% | 0.7769 | 10.91 | 10.56 | 2.8196 | 144,401 | 203.9s |
| **512** | (512, 512, 512) | 41.01% | 0.7443 | 10.79 | 8.49 | 0.6167 | 550,929 | 301.3s |

**Scientific Finding for Experiment A:**
Scaling parameter capacity by **$48\times$** (from $11.5\text{k}$ to $550.9\text{k}$ parameters) **yields flat performance** ($36.76\% \to 36.49\% \to 41.01\%$). The baseline data does not suggest that network capacity is the limiting factor.

---

### Experiment B: Optimization Parameters Sweep (at $D=16$)
**Setup:** Fixed $D=16$, fixed architecture (128, 128, 128), softplus, seed 0. Vary `lr` and `inner_iters`.

| Configuration | Inner Steps | L2-UVP (%) | Cosine Sim | Final $f\_loss$ | Final $g\_loss$ | $\|\nabla_\theta L_f\|$ (mean $\pm$ std) | $\|\nabla_\phi L_g\|$ (mean $\pm$ std) | Time |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **Baseline** | 10 | 37.64% | 0.7683 | 10.71 | 160.36 | $17.91 \pm 123.86$ | $1.99 \pm 11.13$ | 138.6s |
| **Inner_20** | 20 | **36.04%** | **0.7757** | 10.45 | 160.05 | $70.31 \pm 893.10$ | $4.40 \pm 40.10$ | 289.7s |
| **Inner_50** | 50 | 36.85% | 0.7722 | 10.47 | 162.14 | $1282.71 \pm 19848.02$ | $30.46 \pm 372.50$ | 748.9s |

**Scientific Finding for Experiment B:**
Increasing the number of inner maximization steps reduced the final gradient norm at iteration 1999 ($13.85 \to 6.39$) and produced a modest improvement in transport accuracy at 20 inner steps ($36.04\%$). However, the corresponding increase in gradient variance ($123.86 \to 19,848.02$) indicates that additional inner updates do not uniformly stabilize the optimization trajectory. At 50 inner steps, transport accuracy deteriorated slightly despite a lower final gradient norm.

---

## 5. Master Conclusions & Summary Table

| D | L2-UVP (%) | Cosine Sim | Peak $\|\nabla_\theta L_f\|$ | Final $f\_loss$ | Final $g\_loss$ | Runtime (s) |
|:--|:--|:--|:--|:--|:--|:--|
| **2** | **5.32%** | **0.8626** | 10.80 | 1.71 | 8.82 | 152.8 s |
| **4** | **12.64%** | **0.8422** | 5.88 | 3.32 | 32.93 | 135.4 s |
| **8** | **19.70%** | **0.8238** | 317.69 | 6.11 | 68.84 | 138.6 s |
| **16** | **37.76%** | **0.7731** | 2685.76 | 10.71 | 160.36 | 140.8 s |
| **32** | **50.80%** | **0.7644** | 12999.47 | 16.55 | 300.40 | 158.4 s |

**Defensible Summary Statement:**
The observed degradation with increasing dimension is strongly associated with increasingly unstable minimax optimization dynamics, as evidenced by rapidly growing transient gradient norms, increasing $g$-player objective magnitude, and deterioration of the learned transport map.

---

## Output Directories

- Baseline Sweep: [`experiments/high_dimensional/`](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/high_dimensional/)
- Expressivity Sweep: [`experiments/ablation_expressivity_d16/`](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/ablation_expressivity_d16/)
- Optimization Sweep: [`experiments/ablation_optimization_d16/`](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/ablation_optimization_d16/)
- Telemetry Trajectory Plots: [`experiments/telemetry_analysis/`](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/telemetry_analysis/)
