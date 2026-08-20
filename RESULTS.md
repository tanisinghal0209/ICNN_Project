# High-Dimensional Telemetry & Optimization Dynamics Analysis

**Benchmark:** Official Korotin Mix3ToMix10  
**Repository:** `/Users/tanishasinghal/Downloads/Wasserstein2Benchmark`  
**Evaluation:** Learned forward map $\hat{T}(x) = \nabla f(x)$ vs. ground truth `benchmark.map_fwd(x)`

---

## 1. Dimensionality Scaling Telemetry ($D \in \{2, 8, 16, 32\}$)

**Configuration:** Fixed nominal parameters ($N=2000$ iters, batch 256, hidden $128 \times 3$, lr $1\text{e-}3$, inner iters 10, softplus, seed 0).

| D | L2-UVP | Cosine Sim | L2 Error | $f\_loss$ (init $\to$ final) | $g\_loss$ (init $\to$ final) | $\|\nabla_\theta L_f\|$ (init $\to$ peak $\to$ final) | $\|\nabla_\phi L_g\|$ (init $\to$ peak $\to$ final) | Params ($f$) | Time |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| **2** | **5.32%** | **0.8626** | **0.106** | $6.46 \to 1.71$ | $-4.20 \to 8.82$ | $10.80 \to 10.80 \to 0.98$ | $14.96 \to 14.96 \to 0.02$ | 34,051 | 152.8s |
| **8** | **19.70%** | **0.8238** | **1.576** | $7.21 \to 6.11$ | $-6.15 \to 68.84$ | $4.48 \to 317.69 \to 2.57$ | $16.72 \to 27.74 \to 0.06$ | 36,361 | 138.6s |
| **16** | **37.76%** | **0.7731** | **6.041** | $6.62 \to 10.71$ | $-5.91 \to 160.36$ | $2.98 \to 2685.76 \to 13.85$ | $13.19 \to 197.81 \to 1.54$ | 39,441 | 140.8s |
| **32** | **50.80%** | **0.7644** | **16.256** | $9.11 \to 16.55$ | $-8.03 \to 300.40$ | $4.03 \to 12999.47 \to 32.24$ | $18.60 \to 919.33 \to 0.95$ | 45,601 | 158.4s |

---

## 2. Iteration-Level Trajectory Analysis

### Quantitative Observations Across Dimensions ($D=2 \to 32$)
1. **Outer Loss ($f\_loss$) Trajectory**:
   - At $D=2$, $f\_loss$ monotonically decreases from $6.46$ to $1.71$ (clean convergence).
   - At $D=16$ and $D=32$, $f\_loss$ **increases during optimization** ($6.62 \to 10.71$ in 16D, and $9.11 \to 16.55$ in 32D, an 82% increase). The solver moves away from initialization rather than minimizing cost.
2. **Inner Conjugate Loss ($g\_loss$) Dual Gap**:
   - The dual gap $g\_loss$ expands dramatically as dimension increases, growing from $8.82$ in 2D to $300.40$ in 32D.
3. **Transient Gradient Norm Spikes**:
   - Peak gradient norm $\|\nabla_\theta L_f\|$ escalates exponentially with dimension, reaching **2,685.76** in 16D and **12,999.47** in 32D during training iterations, demonstrating transient optimization shocks.
   - Peak conjugate gradient norm $\|\nabla_\phi L_g\|$ similarly scales from $14.96$ in 2D to $919.33$ in 32D.

---

## 3. D=16 Optimization Dynamics (`Baseline` vs `Inner_20` vs `Inner_50`)

**Setup:** Fixed $D=16$, fixed architecture `hidden_dims=(128,128,128)`. Vary inner maximization steps per outer step.

| Configuration | Inner Steps | L2-UVP (%) | Cosine Sim | Final $f\_loss$ | Final $g\_loss$ | $\|\nabla_\theta L_f\|$ (mean $\pm$ std) | $\|\nabla_\phi L_g\|$ (mean $\pm$ std) | Time |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **Baseline** | 10 | 37.64% | 0.7683 | 10.71 | 160.36 | $17.91 \pm 123.86$ | $1.99 \pm 11.13$ | 138.6s |
| **Inner_20** | 20 | **36.04%** | **0.7757** | 10.45 | 160.05 | $70.31 \pm 893.10$ | $4.40 \pm 40.10$ | 289.7s |
| **Inner_50** | 50 | 36.85% | 0.7722 | 10.47 | 162.14 | $1282.71 \pm 19848.02$ | $30.46 \pm 372.50$ | 748.9s |

### Quantitative Assessment of Optimization Ablation:
- Increasing inner steps from 10 to 50 provides $g$ with more optimization steps per $f$ update, reducing the final step gradient norm of $f$ at iteration 1999 ($13.85 \to 6.39$).
- However, intermediate iterations experience massive transient gradient variance (standard deviation of $\|\nabla_\theta L_f\|$ reaching $\pm 19,848.02$ under `Inner_50`), and the final transport accuracy **remains constrained at $\sim 36.8\%$ L2-UVP**.
- This indicates that increasing inner iterations alone does not restore transport accuracy to the low-dimensional regime.

---

## 4. Generated Plot Artifacts

- [`experiments/telemetry_analysis/f_loss_vs_iteration_dims.png`](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/telemetry_analysis/f_loss_vs_iteration_dims.png)
- [`experiments/telemetry_analysis/g_loss_vs_iteration_dims.png`](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/telemetry_analysis/g_loss_vs_iteration_dims.png)
- [`experiments/telemetry_analysis/f_grad_norm_vs_iteration_dims.png`](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/telemetry_analysis/f_grad_norm_vs_iteration_dims.png)
- [`experiments/telemetry_analysis/g_grad_norm_vs_iteration_dims.png`](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/telemetry_analysis/g_grad_norm_vs_iteration_dims.png)
- [`experiments/telemetry_analysis/f_loss_vs_iteration_d16_opt.png`](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/telemetry_analysis/f_loss_vs_iteration_d16_opt.png)
- [`experiments/telemetry_analysis/g_loss_vs_iteration_d16_opt.png`](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/telemetry_analysis/g_loss_vs_iteration_d16_opt.png)
- [`experiments/telemetry_analysis/f_grad_norm_vs_iteration_d16_opt.png`](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/telemetry_analysis/f_grad_norm_vs_iteration_d16_opt.png)
- [`experiments/telemetry_analysis/g_grad_norm_vs_iteration_d16_opt.png`](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/telemetry_analysis/g_grad_norm_vs_iteration_d16_opt.png)
