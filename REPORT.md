# Input-Convex Neural Networks for Optimal Transport
**A Comprehensive Evaluation on Parametric Ablations, Failure Modes, Scalability, and High-Dimensional Wasserstein-2 Benchmarks**

---

## Abstract
This report presents an implementation and empirical analysis of Input-Convex Neural Networks (ICNNs) applied to continuous Optimal Transport (OT) under quadratic cost ($W_2$), reproducing the minimax dual formulation by Makkuva et al. (2020). Using a modular PyTorch-based solver, I validate ICNNs against analytical Gaussian maps, study ablation families across model capacities, investigate failure modes, measure scalability trends, and evaluate mapping quality against the NeurIPS 2021 Korotin continuous Wasserstein-2 benchmark (`Mix3ToMix10`) across dimensions $D \in \{2, 4, 8, 16, 32\}$. Controlled $D=16$ width ablations scale capacity by $48\times$ without systematic improvement. The key diagnostic result is more nuanced: oracle paired-map regression removes the two-player game and reduces peak parameter-gradient norms from 2,685.76/12,999.47 to 1.08/1.39 at $D=16/32$, while unequal player learning rates also greatly suppress minimax spikes. However, neither intervention fully recovers high-dimensional transport accuracy. The evidence therefore supports minimax dynamics as a major source of severe instability, while leaving the clipped ICNN parameterization, conditioning, approximation, and training budget as possible additional contributors to the remaining error.

---

## 1. Introduction
Optimal Transport (OT) defines a powerful mathematical framework for comparing, aligning, and interpolating probability distributions. Historically limited to discrete datasets, OT has recently found applications in continuous domains, such as generative modeling (VAEs, GANs) and domain adaptation. 

Computing the Monge map—the map that minimizes the transport cost between continuous distributions—remains a major challenge. Standard deep neural networks fail to enforce the cyclical monotonicity required of optimal mappings. Recently, Input-Convex Neural Networks (ICNNs) have emerged as a mathematically grounded solution. By restricting network weights, ICNNs parameterize convex potentials whose gradients define valid optimal transport mappings. 

This report presents a systematic research workflow evaluating the performance, scalability limits, failure points, and benchmark accuracy of ICNN-based optimal transport solvers.

---

## 2. Theoretical Background

### 2.1 Optimal Transport & Wasserstein Distance
The Kantorovich formulation of optimal transport between two probability measures $\mu$ and $\nu$ on $\mathbb{R}^D$ is given by:
$$W_2^2(\mu, \nu) = \inf_{\pi \in \Pi(\mu, \nu)} \mathbb{E}_{(x, y) \sim \pi} \left[ \|x - y\|^2 \right]$$
where $\Pi(\mu, \nu)$ is the set of joint distributions (couplings) whose marginals are $\mu$ and $\nu$.

### 2.2 Brenier's Theorem
For a quadratic cost function, Brenier's Theorem states that if $\mu$ is absolutely continuous with respect to the Lebesgue measure, there exists a unique optimal transport map $T: \mathbb{R}^D \to \mathbb{R}^D$ pushing forward $\mu$ to $\nu$ ($T_\# \mu = \nu$). Crucially, this map is the gradient of a convex potential $\varphi$:
$$T(x) = \nabla \varphi(x)$$
where $\varphi: \mathbb{R}^D \to \mathbb{R}$ is a convex function.

### 2.3 Kantorovich Duality
Using Fenchel-Rockafellar duality, the Wasserstein-2 distance can be rewritten in terms of dual potentials:
$$\frac{1}{2} W_2^2(\mu, \nu) = \frac{1}{2} \mathbb{E}_{x \sim \mu}[\|x\|^2] + \frac{1}{2} \mathbb{E}_{y \sim \nu}[\|y\|^2] - \inf_{\varphi \in \text{CVX}} \left( \mathbb{E}_{x \sim \mu}[\varphi(x)] + \mathbb{E}_{y \sim \nu}[\varphi^*(y)] \right)$$
where $\varphi^*$ is the Fenchel conjugate of $\varphi$, defined as:
$$\varphi^*(y) = \sup_{z \in \mathbb{R}^D} \langle y, z \rangle - \varphi(z)$$

### 2.4 Input-Convex Neural Networks (ICNNs)
To parameterize the convex potential $\varphi$, Amos et al. (2017) introduced Input-Convex Neural Networks. A network $\varphi(x; \theta)$ is convex in its input $x$ if its hidden state updates follow:
$$z_{k+1} = \sigma \left( W_z^{(k)} z_k + W_x^{(k)} x + b_k \right)$$
under the constraints that:
1. The hidden weights are non-negative: $W_z^{(k)} \geq 0$ for all $k \geq 1$.
2. The activation function $\sigma$ is convex and non-decreasing (e.g., Softplus or ReLU).

---

## 3. Minimax Dual Formulation (Makkuva et al., 2020)
Makkuva et al. target the difficulty of computing the Fenchel conjugate $\varphi^*$ in dual OT. Rather than optimizing the conjugate potential analytically, they approximate it using a second, conjugate ICNN $g \approx \varphi^*$. 

Since $\nabla \varphi^* = (\nabla \varphi)^{-1}$, the optimal mapping from target to source is parameterized by $\nabla g(y)$. They formulate the Fenchel conjugate computation as a minimax game:
$$\min_{f \in \text{CVX}} \max_{g \in \text{CVX}} L(f, g) := \mathbb{E}_{x \sim \mu}[f(x)] + \mathbb{E}_{y \sim \nu} \left[ \langle y, \nabla g(y) \rangle - f(\nabla g(y)) \right]$$
where $f$ and $g$ are input-convex networks. The optimal potential $g^*$ maximizes the inner term, resolving the conjugate requirement, while the outer potential $f^*$ minimizes the total transportation cost.

---

## 4. Telemetry Instrumentation Audit Verification

Prior to drawing conclusions from gradient trajectory metrics, the solver's instrumentation in `src/solver.py` was verified:

1. **Gradient Zeroing**: `opt_g.zero_grad()` and `opt_f.zero_grad()` are explicitly executed before every backward pass.
2. **Measurement Timing**: Gradient norms are computed immediately after `loss.backward()` and before `optimizer.step()`.
3. **Exact L2 Norm Formula**: Calculated as the total L2 norm across all trainable parameters:
   $$\|\nabla_\theta L\|_2 = \sqrt{\sum_{i} \|\nabla_{\theta_i} L\|_2^2}$$
4. **Player Separation**: $f$ and $g$ parameter gradient norms are measured separately.
5. **No Gradient Clipping**: Gradients are unclipped and unnormalized (`clip_weights()` clips weight parameters $W_z \ge 0$ post-step).
6. **No Stale Gradients**: Measured on fresh backward passes per iteration.
7. **Recorded $g$ Update**: The stored $g$ norm is taken from the final inner $g$ update of each outer iteration; it is not an accumulation across inner updates.

---

## 5. High-Dimensional Korotin Benchmark Evaluation ($D \in \{2, 4, 8, 16, 32\}$)

### 5.1 Baseline Sweep Telemetry Summary

| Dimension ($D$) | L2-UVP (%) | Cosine Similarity | L2 Error | $f$-Loss (init $\to$ final) | $g$-Loss (init $\to$ final) | Peak $\|\nabla_\theta L_f\|$ | Peak $\|\nabla_\phi L_g\|$ | Time (s) |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **D = 2** | **5.3153%** | **0.8626** | **0.1057** | $6.37 \to 1.78$ | $-4.23 \to 8.95$ | 10.90 | 15.11 | 142.54 s |
| **D = 4** | **12.7019%** | **0.8010** | **0.5065** | $5.33 \to 3.32$ | $-4.61 \to 32.92$ | 29.18 | 12.87 | 142.83 s |
| **D = 8** | **19.7814%** | **0.8336** | **1.5810** | $7.21 \to 6.11$ | $-6.15 \to 68.94$ | 317.69 | 27.74 | 153.01 s |
| **D = 16** | **37.6382%** | **0.7683** | **6.0141** | $6.62 \to 10.71$ | $-5.91 \to 160.36$ | **2685.76** | **197.81** | 158.81 s |
| **D = 32** | **51.1976%** | **0.7702** | **16.4573** | $9.11 \to 16.55$ | $-8.03 \to 300.40$ | **12999.47** | **919.33** | 157.90 s |

---

## 6. Controlled Failure Mode Isolation Experiments at $D=16$

### 6.1 Experiment A: Capacity / Expressivity Ablation (at $D=16$)
To evaluate whether network capacity limits performance, I varied hidden width $W \in \{64, 128, 256, 512\}$ at fixed $D=16$.

| Width ($W$) | Parameter Count ($f$) | L2-UVP (%) | Cosine Similarity | Final $f$-Loss | Final $\|\nabla f\|$ | Training Time (s) |
|:---|:---|:---|:---|:---|:---|:---|
| **64** | 11,537 | **36.76%** | 0.7745 | 10.52 | 17.94 | 97.0 s |
| **128** | 39,441 | **37.64%** | 0.7683 | 10.71 | 13.85 | 141.2 s |
| **256** | 144,401 | **36.49%** | 0.7769 | 10.91 | 10.56 | 203.9 s |
| **512** | 550,929 | **41.01%** | 0.7443 | 10.79 | 8.49 | 301.3 s |

**Finding:** Scaling parameter capacity by $48\times$ (from 11.5k to 550k parameters) produces flat performance. The baseline data does not suggest that network capacity is the limiting factor.

### 6.2 Experiment B: Optimization Parameters Sweep (at $D=16$)
To evaluate optimization dynamics, I varied learning rate `lr` and inner maximization steps `inner_iters` at fixed $D=16$.

| Configuration | Inner Steps | L2-UVP (%) | Cosine Similarity | Final $f$-Loss | Final $g$-Loss | $\|\nabla_\theta L_f\|$ (mean $\pm$ std) | $\|\nabla_\phi L_g\|$ (mean $\pm$ std) |
|:---|:---|:---|:---|:---|:---|:---|:---|
| **Baseline** | 10 | 37.64% | 0.7683 | 10.71 | 160.36 | $17.91 \pm 123.86$ | $1.99 \pm 11.13$ |
| **Inner 20** | 20 | **36.04%** | **0.7757** | 10.45 | 160.05 | $70.31 \pm 893.10$ | $4.40 \pm 40.10$ |
| **Inner 50** | 50 | 36.85% | 0.7722 | 10.47 | 162.14 | $1282.71 \pm 19848.02$ | $30.46 \pm 372.50$ |

**Finding:** Increasing the number of inner maximization steps reduced the final gradient norm and produced a modest improvement in transport accuracy at 20 inner steps. However, the corresponding increase in gradient variance indicates that additional inner updates do not uniformly stabilize the optimization trajectory. At 50 inner steps, transport accuracy deteriorated slightly despite a lower final gradient norm.

---

### 6.3 Experiment C: Oracle Supervised Regression (at $D=16$ and $D=32$)

The official benchmark supplies the ground-truth forward map $T^*(x)=\texttt{benchmark.map\_fwd}(x)$. This diagnostic retains the clipped 3$\times$128 Softplus ICNN but removes $g$, alternating updates, and inner maximization; it trains $f$ directly with $\operatorname{MSE}(\nabla_x f(x),T^*(x))$.

| D | Training | L2-UVP | Cosine | L2 Error | Peak $f$ parameter-gradient norm | NaN/Inf |
|:---|:---|:---|:---|:---|:---|:---|
| 16 | Minimax baseline | 37.64% | 0.7683 | 6.0141 | 2685.76 | False |
| 16 | Oracle MSE | **29.14%** | **0.8249** | **4.6299** | **1.08** | False |
| 32 | Minimax baseline | 51.20% | 0.7702 | 16.4573 | 12999.47 | False |
| 32 | Oracle MSE | **42.59%** | **0.8083** | **13.6303** | **1.39** | False |

**Finding:** Direct supervision yields finite, stable trajectories and better official transport metrics at both dimensions. The remaining oracle error means that this experiment does not clear the ICNN parameterization or clipping of all responsibility for the high-dimensional gap.

### 6.4 Experiment D: Unequal Player Learning Rates (at $D=16$ and $D=32$)

All settings are fixed to the corresponding baseline except $g$'s Adam learning rate. Peak norms are maxima over recorded parameter-gradient norms; the reported $g$ quantity is the player objective, not a duality gap.

| D | $f$ LR | $g$ LR | L2-UVP | Cosine | Peak $f$-grad | Peak $g$-grad | NaN/Inf |
|:---|:---|:---|:---|:---|:---|:---|:---|
| 16 | 1e-3 | 1e-3 | 37.64% | 0.7683 | 2685.76 | 197.81 | False |
| 16 | 1e-3 | 5e-4 | 37.25% | 0.7707 | 453.32 | 73.29 | False |
| 16 | 1e-3 | 2.5e-4 | **37.23%** | **0.7726** | **142.32** | **45.43** | False |
| 32 | 1e-3 | 1e-3 | 51.20% | 0.7702 | 12999.47 | 919.33 | False |
| 32 | 1e-3 | 5e-4 | **50.52%** | **0.7738** | 1663.67 | 229.27 | False |
| 32 | 1e-3 | 2.5e-4 | 52.58% | 0.7669 | **325.09** | **110.23** | False |

**Finding:** Slowing $g$ suppresses gradient spikes at both dimensions, but accuracy is not monotonic in stability. In particular, the D=32 4$\times$ slower configuration is the most stable and the least accurate among the three controlled D=32 runs. The L2-UVP differences are single-seed results under stochastic official evaluation and therefore require replication before making a statistical claim.

---

## 7. Conclusions

Under the tested architecture, training budget, and minimax optimization settings, transport accuracy deteriorates substantially with dimension. The combined width, oracle, and unequal-learning-rate diagnostics provide strong empirical evidence that minimax dynamics are a major contributor to severe high-dimensional gradient instability. They do not prove that instability is the sole cause of degraded transport accuracy: the oracle experiment remains materially inaccurate despite stable gradients, and the most stable unequal-LR configuration at D=32 is not the most accurate.

The next research decision should be made with the advisor: either evaluate one carefully specified minimax stabilizer (such as a gradient penalty or EMA) with repeated seeds, or investigate an alternative convex-potential parameterization. The evidence does not support claiming that the minimax game alone explains the high-dimensional error.

---

## 8. References
1. Makkuva, A., Taghvaei, A., Oh, S., & Lee, J. (2020). *Optimal transport using input-convex neural networks*. ICML.
2. Amos, B., Xu, L., & Kolter, J. Z. (2017). *Input convex neural networks*. ICML.
3. Korotin, A., Li, L., Genevay, A., Solomon, J. M., Filippov, A., & Burnaev, E. (2021). *Do neural optimal transport solvers work? A continuous Wasserstein-2 benchmark*. NeurIPS.
