# Input-Convex Neural Networks for Optimal Transport
**A Comprehensive Evaluation on Parametric Ablations, Failure Modes, Scalability, and High-Dimensional Wasserstein-2 Benchmarks**

---

## Abstract
This report presents a implementation and empirical analysis of Input-Convex Neural Networks (ICNNs) applied to continuous Optimal Transport (OT) under quadratic cost ($W_2$), reproducing the minimax dual formulation by Makkuva et al. (2020). Using a modular PyTorch-based solver, I validate ICNNs against analytical Gaussian maps, study ablation families across model capacities, investigate failure modes, measure scalability trends, and evaluate mapping quality against the NeurIPS 2021 Korotin continuous Wasserstein-2 benchmark (`Mix3ToMix10`) across dimensions $D \in \{2, 4, 8, 16, 32\}$. Following research directives, I isolate three potential failure causes: (1) function-class expressivity, (2) optimization instability, and (3) computational cost. Controlled ablations at $D=16$ show that scaling model capacity by $48\times$ (from 11k to 550k parameters) yields zero performance improvement (L2-UVP remaining flat at $\sim 36.5\% - 41.0\%$), indicating that capacity is not the limiting factor. Trajectory telemetry demonstrates that **transient gradient spikes (peaking at 12,999 in 32D) and growth of the $g$-player objective** are strongly associated with high-dimensional mapping accuracy degradation.

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

---

## 5. High-Dimensional Korotin Benchmark Evaluation ($D \in \{2, 4, 8, 16, 32\}$)

### 5.1 Baseline Sweep Telemetry Summary

| Dimension ($D$) | L2-UVP (%) | Cosine Similarity | L2 Error | $f$-Loss (init $\to$ final) | $g$-Loss (init $\to$ final) | Peak $\|\nabla_\theta L_f\|$ | Peak $\|\nabla_\phi L_g\|$ | Time (s) |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **D = 2** | **5.3153%** | **0.8626** | **0.1057** | $6.46 \to 1.71$ | $-4.20 \to 8.82$ | 10.80 | 14.96 | 152.83 s |
| **D = 4** | **12.6375%** | **0.8422** | **0.5055** | $5.33 \to 3.32$ | $-4.61 \to 32.93$ | 5.88 | 0.08 | 135.35 s |
| **D = 8** | **19.7014%** | **0.8238** | **1.5761** | $7.21 \to 6.11$ | $-6.15 \to 68.84$ | 317.69 | 27.74 | 138.64 s |
| **D = 16** | **37.7563%** | **0.7731** | **6.0410** | $6.62 \to 10.71$ | $-5.91 \to 160.36$ | **2685.76** | **197.81** | 140.82 s |
| **D = 32** | **50.7988%** | **0.7644** | **16.2556** | $9.11 \to 16.55$ | $-8.03 \to 300.40$ | **12999.47** | **919.33** | 158.39 s |

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

## 7. Conclusions

Under the tested architecture, training budget, and minimax optimization settings, transport accuracy deteriorates substantially with dimension. The observed degradation with increasing dimension is strongly associated with increasingly unstable minimax optimization dynamics, as evidenced by rapidly growing transient gradient norms, increasing $g$-player objective magnitude, and deterioration of the learned transport map.

---

## 8. References
1. Makkuva, A., Taghvaei, A., Oh, S., & Lee, J. (2020). *Optimal transport using input-convex neural networks*. ICML.
2. Amos, B., Xu, L., & Kolter, J. Z. (2017). *Input convex neural networks*. ICML.
3. Korotin, A., Li, L., Genevay, A., Solomon, J. M., Filippov, A., & Burnaev, E. (2021). *Do neural optimal transport solvers work? A continuous Wasserstein-2 benchmark*. NeurIPS.
