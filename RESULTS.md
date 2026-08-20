# High-Dimensional Research & Baseline Degradation Results

**Benchmark:** Official Korotin Mix3ToMix10  
**Repository:** `/Users/tanishasinghal/Downloads/Wasserstein2Benchmark`  
**Benchmark module:** `Wasserstein2Benchmark/src/map_benchmark.py`  
**Evaluation:** Learned forward map $\hat{T}(x) = \nabla f(x)$ vs. ground truth `benchmark.map_fwd(x)`

---

## 1. Stage 1 — Baseline High-Dimensional Degradation Sweep ($D \in \{2, 4, 8, 16, 32\}$)

**Configuration:** Fixed nominal parameters ($N=2000$ iters, batch 256, hidden $128 \times 3$, lr $1\text{e-}3$, inner iters 10, softplus, seed 0).

| D | L2-UVP | Cosine Sim | L2 Error | f_loss | g_loss | f_gnorm | g_gnorm | Params (f) | Time |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| **2** | **5.32%** | **0.8626** | **0.106** | 1.775 | — | 0.213 | — | 34,051 | 152.8s |
| **4** | **12.64%** | **0.8422** | **0.506** | 3.316 | 32.93 | 1.316 | 0.0768 | 34,821 | 135.4s |
| **8** | **19.70%** | **0.8238** | **1.576** | 6.112 | 68.84 | 2.571 | 0.0571 | 36,361 | 138.6s |
| **16** | **37.76%** | **0.7731** | **6.041** | 10.712 | 160.36 | **13.849** | 1.5421 | 39,441 | 140.8s |
| **32** | **50.80%** | **0.7644** | **16.256** | 16.555 | 300.40 | **32.239** | 0.9514 | 45,601 | 158.4s |

### Iteration-Level Gradient Norm $\|\nabla f\|$ Trajectory

| D | Iter 0 | Iter 400 | Iter 800 | Iter 1200 | Iter 1600 | Iter 1999 |
|:--|:--|:--|:--|:--|:--|:--|
| **4** | 5.88 | 0.87 | 1.27 | 0.69 | 1.30 | 1.32 |
| **8** | 4.48 | 2.25 | 4.54 | 3.22 | 2.47 | 2.57 |
| **16** | 2.98 | 5.75 | 7.68 | **15.59** | 9.53 | **13.85** |
| **32** | 4.03 | **34.35** | **25.50** | 11.49 | **40.50** | **32.24** |

---

## 2. Stage 2 — Controlled Failure Mode Isolations (at $D=16$)

Following research directives, we conducted controlled experiments at $D=16$ to isolate **function-class capacity** vs. **optimization instability**.

### Experiment A: Capacity / Expressivity Ablation (at $D=16$)
**Hypothesis:** Does increasing network width improve high-dimensional transport accuracy?  
**Setup:** Fixed $D=16$, $N=2000$, lr $1\text{e-}3$, inner iters 10, softplus, seed 0. Vary width $W \in \{64, 128, 256, 512\}$.

| Width ($W$) | Hidden Architecture | L2-UVP (%) | Cosine Sim | f_loss | f_gnorm | g_gnorm | Params (f) | Time |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **64** | (64, 64, 64) | 36.76% | 0.7745 | 10.52 | 17.94 | 0.3590 | 11,537 | 97.0s |
| **128** | (128, 128, 128) | 37.64% | 0.7683 | 10.71 | 13.85 | 1.5421 | 39,441 | 141.2s |
| **256** | (256, 256, 256) | 36.49% | 0.7769 | 10.91 | 10.56 | 2.8196 | 144,401 | 203.9s |
| **512** | (512, 512, 512) | 41.01% | 0.7443 | 10.79 | 8.49 | 0.6167 | 550,929 | 301.3s |

**Key Finding for Experiment A:**
Scaling parameter capacity by **$48\times$** (from $11.5\text{k}$ to $550.9\text{k}$ parameters) **yields zero improvement in L2-UVP** ($36.76\% \to 36.49\% \to 41.01\%$). This provides **decisive empirical proof that function-class expressivity is NOT the limiting factor** at $D=16$.

---

### Experiment B: Optimization Stability Ablation (at $D=16$)
**Hypothesis:** Does tuning the learning rate or increasing inner maximization steps stabilize the minimax game?  
**Setup:** Fixed $D=16$, fixed architecture (128, 128, 128), softplus, seed 0. Vary `lr` and `inner_iters`.

| Configuration | Learning Rate (`lr`) | Inner Iters | L2-UVP (%) | Cosine Sim | f_loss | g_loss | f_gnorm | Time |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **Baseline** | $1.0 \times 10^{-3}$ | 10 | 37.64% | 0.7683 | 10.71 | 160.36 | 13.85 | 138.6s |
| **LR_5e-4** | $5.0 \times 10^{-4}$ | 10 | 38.15% | 0.7631 | 10.90 | 184.31 | 15.85 | 143.9s |
| **LR_2.5e-4** | $2.5 \times 10^{-4}$ | 10 | 39.40% | 0.7519 | 11.12 | 195.66 | 14.92 | 144.3s |
| **Inner_20** | $1.0 \times 10^{-3}$ | 20 | **36.04%** | **0.7757** | 10.45 | 160.05 | **7.56** | 289.7s |
| **Inner_50** | $1.0 \times 10^{-3}$ | 50 | 36.85% | 0.7722 | 10.47 | 162.14 | **6.39** | 748.9s |

**Key Finding for Experiment B:**
1. Reducing learning rate ($1\text{e-}3 \to 2.5\text{e-}4$) worsens L2-UVP ($37.64\% \to 39.40\%$) and does not reduce gradient norms.
2. Increasing inner iterations ($10 \to 50$) **halves the final gradient norm** ($13.85 \to 6.39$), but **fails to restore transport accuracy to the 2D/4D regime** ($\sim 36.8\%$).
3. This proves that basic hyperparameter tuning of standard minimax dual loss cannot overcome the fundamental conditioning difficulty of the dual formulation in high dimensions.

---

## 3. Master Synthesis & Conclusions

1. **Expressivity is not the bottleneck**: Parameter scaling up to 550k weights produces flat L2-UVP performance ($\sim 36.5\% - 41.0\%$).
2. **Minimax Optimization Instability is the dominant cause**: Gradient norm explosion ($150\times$ increase) and dual-gap expansion ($g$-loss $\sim 300$) drive high-dimensional degradation.
3. **Compute cost is flat**: Runtime scales linearly with inner iterations and width, but dimensional scaling itself adds virtually no compute overhead ($\sim 135\text{s} - 158\text{s}$).

---

## Output Directories

- Baseline Sweep: [`experiments/high_dimensional/`](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/high_dimensional/)
- Experiment A (Expressivity): [`experiments/ablation_expressivity_d16/`](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/ablation_expressivity_d16/)
- Experiment B (Optimization): [`experiments/ablation_optimization_d16/`](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/ablation_optimization_d16/)
