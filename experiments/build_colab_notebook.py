import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "ICNN_Optimal_Transport_Colab.ipynb"

def create_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source]
    }

def create_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source]
    }

def main():
    cells = []
    
    # ----------------------------------------------------
    # Cell 1: Markdown Introduction
    # ----------------------------------------------------
    cells.append(create_markdown_cell([
        "# Optimal Transport using Input-Convex Neural Networks (ICNN)",
        "",
        "This notebook contains a self-contained, high-performance implementation of the Input-Convex Neural Network (ICNN) solver for optimal transport under quadratic costs ($W_2$), reproducing the minimax dual formulation by Makkuva et al. (2020).",
        "",
        "## Core Theoretical Concepts",
        "1. **Monge-Kantorovich Dual**: We find the optimal transport mapping by optimizing dual potentials.",
        "2. **Brenier's Theorem**: For quadratic costs, the optimal transport map is unique and given by the gradient of a convex potential $\\nabla \\varphi$.",
        "3. **ICNN**: Amos et al. (2017) enforce convexity of the network $\\varphi$ by restricting its hidden weights to be non-negative and using convex, non-decreasing activations.",
        "",
        "Run the cells below sequentially to load the models, define the objectives, and execute the paper reproductions and failure-mode analyses."
    ]))
    
    # ----------------------------------------------------
    # Cell 2: Code Imports
    # ----------------------------------------------------
    cells.append(create_code_cell([
        "# Imports",
        "import time",
        "import math",
        "import json",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "import torch",
        "import torch.nn as nn",
        "import torch.nn.functional as F",
        "import torch.autograd as autograd",
        "",
        "# Enable inline plotting in Jupyter/Colab",
        "%matplotlib inline"
    ]))
    
    # ----------------------------------------------------
    # Cell 3: Neural Networks (ICNN and StandardMLP)
    # ----------------------------------------------------
    cells.append(create_markdown_cell([
        "## 1. Network Architectures",
        "We define both the `ICNN` (with weight clipping to preserve convexity) and `StandardMLP` (used as a non-convex failure-mode comparison)."
    ]))
    
    cells.append(create_code_cell([
        "class StandardMLP(nn.Module):",
        "    def __init__(self, input_dim, hidden_dims=(64, 64, 64), activation='softplus'):",
        "        super().__init__()",
        "        self.input_dim = input_dim",
        "        self.hidden_dims = list(hidden_dims)",
        "        self.act = {",
        "            'softplus': nn.Softplus(),",
        "            'relu': nn.ReLU(),",
        "            'leaky_relu': nn.LeakyReLU(0.2),",
        "            'elu': nn.ELU(),",
        "        }[activation]",
        "",
        "        dims = [input_dim] + self.hidden_dims + [1]",
        "        self.layers = nn.ModuleList()",
        "        for i in range(len(dims) - 1):",
        "            self.layers.append(nn.Linear(dims[i], dims[i + 1]))",
        "",
        "    def forward(self, x):",
        "        h = x",
        "        for i, layer in enumerate(self.layers):",
        "            h = layer(h)",
        "            if i < len(self.layers) - 1:",
        "                h = self.act(h)",
        "        return h.squeeze(-1)",
        "",
        "    @torch.no_grad()",
        "    def clip_weights(self):",
        "        pass",
        "",
        "    def grad(self, y):",
        "        y = y.clone().requires_grad_(True)",
        "        f_val = self.forward(y)",
        "        (grad_y,) = torch.autograd.grad(f_val.sum(), y, create_graph=True)",
        "        return grad_y",
        "",
        "class ICNN(nn.Module):",
        "    def __init__(self, input_dim, hidden_dims=(64, 64, 64), activation='softplus'):",
        "        super().__init__()",
        "        self.input_dim = input_dim",
        "        self.hidden_dims = list(hidden_dims)",
        "        self.act = {",
        "            'softplus': nn.Softplus(),",
        "            'relu': nn.ReLU(),",
        "            'leaky_relu': nn.LeakyReLU(0.2),",
        "            'elu': nn.ELU(),",
        "        }[activation]",
        "",
        "        dims = [input_dim] + self.hidden_dims + [1]",
        "        self.Wy = nn.ModuleList()",
        "        self.Wz = nn.ModuleList()",
        "        for i in range(len(dims) - 1):",
        "            out_d = dims[i + 1]",
        "            self.Wy.append(nn.Linear(input_dim, out_d, bias=True))",
        "            self.Wz.append(None if i == 0 else nn.Linear(self.hidden_dims[i - 1], out_d, bias=False))",
        "",
        "    def forward(self, y):",
        "        z = self.act(self.Wy[0](y))",
        "        for i in range(1, len(self.Wy)):",
        "            pre_act = self.Wz[i](z) + self.Wy[i](y)",
        "            z = pre_act if i == len(self.Wy) - 1 else self.act(pre_act)",
        "        return z.squeeze(-1)",
        "",
        "    @torch.no_grad()",
        "    def clip_weights(self):",
        "        for layer in self.Wz:",
        "            if layer is not None:",
        "                layer.weight.data.clamp_(min=0)",
        "",
        "    def grad(self, y):",
        "        y = y.clone().requires_grad_(True)",
        "        f_val = self.forward(y)",
        "        (grad_y,) = torch.autograd.grad(f_val.sum(), y, create_graph=True)",
        "        return grad_y"
    ]))
    
    # ----------------------------------------------------
    # Cell 4: Loss formulation
    # ----------------------------------------------------
    cells.append(create_markdown_cell([
        "## 2. Minimax Loss Formulation",
        "We implement the minimax dual objective from Section 3 of Makkuva et al. (2020)."
    ]))
    
    cells.append(create_code_cell([
        "def minimax_loss(f, g, x_mu, y_nu):",
        "    grad_g_y = g.grad(y_nu)",
        "    term_mu = f(x_mu).mean()",
        "    term_nu = (y_nu * grad_g_y).sum(dim=1).mean() - f(grad_g_y).mean()",
        "    f_loss = term_mu + term_nu",
        "    g_loss = -term_nu",
        "    return f_loss, g_loss"
    ]))
    
    # ----------------------------------------------------
    # Cell 5: Training Engine
    # ----------------------------------------------------
    cells.append(create_markdown_cell([
        "## 3. Training Loop",
        "A dataset-agnostic training loop that updates $f$ and $g$ alternatingly using the Adam optimizer."
    ]))
    
    cells.append(create_code_cell([
        "def train_icnn_ot(mu_sampler, nu_sampler, input_dim, n_iters=1000, batch_size=256,",
        "                   hidden_dims=(64, 64, 64), lr=1e-3, inner_iters=10, device='cpu',",
        "                   activation='softplus', log_every=250, model_type='icnn'):",
        "    if model_type.lower() == 'icnn':",
        "        f = ICNN(input_dim, hidden_dims, activation=activation).to(device)",
        "        g = ICNN(input_dim, hidden_dims, activation=activation).to(device)",
        "    else:",
        "        f = StandardMLP(input_dim, hidden_dims, activation=activation).to(device)",
        "        g = StandardMLP(input_dim, hidden_dims, activation=activation).to(device)",
        "",
        "    opt_f = torch.optim.Adam(f.parameters(), lr=lr, betas=(0.5, 0.9))",
        "    opt_g = torch.optim.Adam(g.parameters(), lr=lr, betas=(0.5, 0.9))",
        "    history = {'f_loss': []}",
        "",
        "    for it in range(n_iters):",
        "        # 1. Update potential g (inner maximization loop)",
        "        for _ in range(inner_iters):",
        "            x_mu, y_nu = mu_sampler(batch_size).to(device), nu_sampler(batch_size).to(device)",
        "            _, g_loss = minimax_loss(f, g, x_mu, y_nu)",
        "            opt_g.zero_grad()",
        "            g_loss.backward()",
        "            opt_g.step()",
        "            g.clip_weights()",
        "",
        "        # 2. Update potential f (outer minimization loop)",
        "        x_mu, y_nu = mu_sampler(batch_size).to(device), nu_sampler(batch_size).to(device)",
        "        f_loss, _ = minimax_loss(f, g, x_mu, y_nu)",
        "        opt_f.zero_grad()",
        "        f_loss.backward()",
        "        opt_f.step()",
        "        f.clip_weights()",
        "",
        "        history['f_loss'].append(f_loss.item())",
        "        if it % log_every == 0:",
        "            print(f'Iteration {it:4d} | f_loss = {f_loss.item():.4f}')",
        "",
        "    return f, g, history"
    ]))
    
    # ----------------------------------------------------
    # Cell 6: Data Samplers
    # ----------------------------------------------------
    cells.append(create_markdown_cell([
        "## 4. Distribution Samplers",
        "Definitions of various source ($\mu$) and target ($\nu$) samplers, including Gaussians, mixtures on circles, and disconnected support."
    ]))
    
    cells.append(create_code_cell([
        "def make_gaussian_sampler(mu, sigma, device='cpu'):",
        "    def sampler(batch_size):",
        "        return torch.randn(batch_size, len(mu), device=device) * sigma + torch.tensor(mu, device=device)",
        "    return sampler",
        "",
        "def make_mixture_sampler(means, covariances, weights, device='cpu'):",
        "    means = [torch.tensor(m, device=device, dtype=torch.float32) for m in means]",
        "    covariances = [torch.tensor(c, device=device, dtype=torch.float32) for c in covariances]",
        "    weights = np.array(weights, dtype=np.float32) / sum(weights)",
        "",
        "    def sampler(batch_size):",
        "        idx = np.random.choice(len(weights), size=batch_size, p=weights)",
        "        samples = []",
        "        for k in range(len(weights)):",
        "            n = int((idx == k).sum())",
        "            if n == 0: continue",
        "            z = torch.randn(n, means[0].shape[0], device=device)",
        "            L = torch.linalg.cholesky(covariances[k])",
        "            samples.append((z @ L.T) + means[k])",
        "        return torch.cat(samples, dim=0)",
        "    return sampler",
        "",
        "def make_disconnected_sampler(means, sigma, device='cpu'):",
        "    means = [torch.tensor(m, device=device, dtype=torch.float32) for m in means]",
        "    def sampler(batch_size):",
        "        idx = np.random.choice(len(means), size=batch_size)",
        "        samples = []",
        "        for k in range(len(means)):",
        "            n = int((idx == k).sum())",
        "            if n == 0: continue",
        "            z = torch.randn(n, len(means[k]), device=device)",
        "            samples.append(z * sigma + means[k])",
        "        return torch.cat(samples, dim=0)",
        "    return sampler",
        "",
        "def make_finite_pool_sampler(pool):",
        "    def sampler(batch_size):",
        "        idx = torch.randint(0, len(pool), (batch_size,), device=pool.device)",
        "        return pool[idx]",
        "    return sampler"
    ]))
    
    # ----------------------------------------------------
    # Cell 7: Plotting functions
    # ----------------------------------------------------
    cells.append(create_markdown_cell([
        "## 5. Visualization Helpers",
        "Helper functions to render distribution scatters, learned transport maps, and displacement vector fields."
    ]))
    
    cells.append(create_code_cell([
        "def plot_reproduction_results(mu_sampler, nu_sampler, f, g, title_suffix=''):",
        "    n_plot = 300",
        "    x_plot = mu_sampler(n_plot).detach().cpu().numpy()",
        "    y_plot = nu_sampler(n_plot).detach().cpu().numpy()",
        "    with torch.no_grad():",
        "        transport = g.grad(torch.tensor(y_plot, dtype=torch.float32)).detach().cpu().numpy()",
        "",
        "    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))",
        "    ",
        "    # 1. Source vs Target Scatter",
        "    axes[0].scatter(x_plot[:, 0], x_plot[:, 1], s=15, alpha=0.6, label='Source x (mu)', color='C0')",
        "    axes[0].scatter(y_plot[:, 0], y_plot[:, 1], s=15, alpha=0.6, label='Target y (nu)', color='C1')",
        "    axes[0].set_title(f'Distributions {title_suffix}')",
        "    axes[0].legend()",
        "    axes[0].grid(True, linestyle='--', alpha=0.5)",
        "    ",
        "    # 2. Transport Mapping",
        "    axes[1].scatter(y_plot[:, 0], y_plot[:, 1], s=15, alpha=0.5, label='Target y', color='C1')",
        "    axes[1].scatter(transport[:, 0], transport[:, 1], s=15, alpha=0.5, label='Transported y', color='C2')",
        "    axes[1].set_title(f'Learned Transport Map {title_suffix}')",
        "    axes[1].legend()",
        "    axes[1].grid(True, linestyle='--', alpha=0.5)",
        "    ",
        "    # 3. Vector displacement field",
        "    step = max(1, n_plot // 80)",
        "    axes[2].quiver(",
        "        y_plot[::step, 0], y_plot[::step, 1],",
        "        transport[::step, 0] - y_plot[::step, 0], transport[::step, 1] - y_plot[::step, 1],",
        "        angles='xy', scale_units='xy', scale=1, alpha=0.7, color='purple'",
        "    )",
        "    axes[2].scatter(y_plot[:, 0], y_plot[:, 1], s=8, alpha=0.3, color='gray')",
        "    axes[2].set_title(f'Displacement Field {title_suffix}')",
        "    axes[2].grid(True, linestyle='--', alpha=0.5)",
        "    ",
        "    plt.tight_layout()",
        "    plt.show()"
    ]))
    
    # ----------------------------------------------------
    # Cell 8: Main Experiments Heading
    # ----------------------------------------------------
    cells.append(create_markdown_cell([
        "## 6. Running Core Experiments",
        "Now we will run the main paper configurations and failure analysis. (We use 500 iterations for fast inline verification, which is sufficient to see the convergence trends)."
    ]))
    
    # ----------------------------------------------------
    # Cell 9: Main Execution and Plotting
    # ----------------------------------------------------
    cells.append(create_code_cell([
        "# ==========================================",
        "# Experiment 1: Gaussian -> Gaussian (with Analytical Verification)",
        "# ==========================================",
        "print('=== Experiment 1: Gaussian -> Gaussian ===')",
        "mu = [0.0, 0.0]",
        "sigma_mu = 1.0",
        "nu = [2.0, -1.0]",
        "sigma_nu = 0.6",
        "",
        "mu_sampler = make_gaussian_sampler(mu, sigma_mu)",
        "nu_sampler = make_gaussian_sampler(nu, sigma_nu)",
        "",
        "f_g, g_g, history_g = train_icnn_ot(",
        "    mu_sampler, nu_sampler, input_dim=2, n_iters=800, log_every=200",
        ")",
        "",
        "# Plot results",
        "plot_reproduction_results(mu_sampler, nu_sampler, f_g, g_g, '(Gaussian to Gaussian)')",
        "",
        "# W2 Verification",
        "f_g.eval(); g_g.eval()",
        "with torch.no_grad():",
        "    x_eval = mu_sampler(4000)",
        "    y_eval = nu_sampler(4000)",
        "    term_mu = f_g(x_eval).mean().item()",
        "with torch.enable_grad():",
        "    y_eval_g = y_eval.clone().requires_grad_(True)",
        "    grad_g = g_g.grad(y_eval_g)",
        "    term_nu = (y_eval * grad_g).sum(dim=1).mean().item() - f_g(grad_g).mean().item()",
        "",
        "e_x2 = (x_eval**2).sum(dim=1).mean().item()",
        "e_y2 = (y_eval**2).sum(dim=1).mean().item()",
        "learned_w2 = math.sqrt(max(0.0, e_x2 + e_y2 - 2.0 * (term_mu + term_nu)))",
        "analytical_w2 = math.sqrt((nu[0]-mu[0])**2 + (nu[1]-mu[1])**2 + 2.0 * (sigma_mu - sigma_nu)**2)",
        "print(f'Analytical W2: {analytical_w2:.4f} | Learned W2: {learned_w2:.4f} | Error: {abs(learned_w2 - analytical_w2):.4f}')",
        "",
        "# ==========================================",
        "# Experiment 2: Multimodal Circle Mixture",
        "# ==========================================",
        "print('\\n=== Experiment 2: Multimodal Circle Mixture ===')",
        "def circle_means(k, radius=2.0):",
        "    angles = np.linspace(0, 2 * math.pi, k, endpoint=False)",
        "    return [[radius * math.cos(a), radius * math.sin(a)] for a in angles]",
        "",
        "means = circle_means(8, radius=2.0)",
        "covs = [np.eye(2) * 0.08 for _ in range(8)]",
        "weights = [1.0 / 8] * 8",
        "",
        "mu_mix = make_gaussian_sampler([0.0, 0.0], 0.8)",
        "nu_mix = make_mixture_sampler(means, covs, weights)",
        "",
        "f_mix, g_mix, _ = train_icnn_ot(mu_mix, nu_mix, input_dim=2, n_iters=1000, log_every=250)",
        "plot_reproduction_results(mu_mix, nu_mix, f_mix, g_mix, '(Multimodal Mixture)')",
        "",
        "# ==========================================",
        "# Experiment 3: Disconnected Support",
        "# ==========================================",
        "print('\\n=== Experiment 3: Disconnected Support ===')",
        "mu_disc = make_disconnected_sampler([[-2.0, 0.0], [2.0, 0.0]], 0.3)",
        "nu_disc = make_disconnected_sampler([[-1.5, -1.5], [1.5, 1.5]], 0.5)",
        "",
        "f_disc, g_disc, _ = train_icnn_ot(mu_disc, nu_disc, input_dim=2, n_iters=1000, log_every=250)",
        "plot_reproduction_results(mu_disc, nu_disc, f_disc, g_disc, '(Disconnected Support)')",
        "",
        "# ==========================================",
        "# Experiment 4: Failure Mode - Removing Convexity Constraint (MLP)",
        "# ==========================================",
        "print('\\n=== Experiment 4: Failure Mode - Standard MLP (No Convexity) ===')",
        "f_mlp, g_mlp, _ = train_icnn_ot(",
        "    mu_sampler, nu_sampler, input_dim=2, n_iters=800, log_every=200, model_type='mlp'",
        ")",
        "# Note how non-convex maps fail to map monotonic trajectories and cross each other",
        "plot_reproduction_results(mu_sampler, nu_sampler, f_mlp, g_mlp, '(Failure Mode: Non-Convex MLP)')"
    ]))

    # ----------------------------------------------------
    # Assemble notebook dictionary
    # ----------------------------------------------------
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(notebook, fh, indent=2)
        
    print(f"Successfully generated Google Colab Notebook at: {OUT_PATH}")

if __name__ == "__main__":
    main()
