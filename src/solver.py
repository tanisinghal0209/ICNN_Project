"""
Training loop for the ICNN-based Optimal Transport solver.

Extracted 1:1 from the canonical notebook (Cell 10).
Source of truth: notebooks/ICNN_Optimal_Transport_Colab_FINAL.ipynb

Canonical defaults (do NOT change during refactoring):
    n_iters=1500, batch_size=256, hidden_dims=(128,128,128),
    lr=1e-3, betas=(0.5,0.9), inner_iters=10, activation='softplus'

Return contract (matches notebook): f, g, history, final_f_loss, training_time
"""
import time
import torch

from src.icnn import ICNN, StandardMLP
from src.losses import minimax_loss


def train_icnn_ot(
    mu_sampler, nu_sampler, input_dim,
    n_iters=1500, batch_size=256,
    hidden_dims=(128, 128, 128), lr=1e-3, inner_iters=10,
    device=None, activation='softplus', log_every=300,
    model_type='icnn',
):
    """
    Train the ICNN optimal transport solver.

    Returns: (f, g, history, final_f_loss, training_time)
    This 5-element return contract matches the canonical notebook exactly.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    torch.manual_seed(0)

    if model_type.lower() == 'icnn':
        f = ICNN(input_dim, hidden_dims, activation=activation).to(device)
        g = ICNN(input_dim, hidden_dims, activation=activation).to(device)
    else:
        f = StandardMLP(input_dim, hidden_dims, activation=activation).to(device)
        g = StandardMLP(input_dim, hidden_dims, activation=activation).to(device)

    opt_f = torch.optim.Adam(f.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_g = torch.optim.Adam(g.parameters(), lr=lr, betas=(0.5, 0.9))
    history = {'f_loss': []}
    start_time = time.time()

    for it in range(n_iters):
        # 1. Update potential g (inner maximization loop)
        for _ in range(inner_iters):
            x_mu = mu_sampler(batch_size).to(device)
            y_nu = nu_sampler(batch_size).to(device)
            _, g_loss = minimax_loss(f, g, x_mu, y_nu)
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()
            g.clip_weights()

        # 2. Update potential f (outer minimization loop)
        x_mu = mu_sampler(batch_size).to(device)
        y_nu = nu_sampler(batch_size).to(device)
        f_loss, _ = minimax_loss(f, g, x_mu, y_nu)
        opt_f.zero_grad()
        f_loss.backward()
        opt_f.step()
        f.clip_weights()

        history['f_loss'].append(f_loss.item())

        # Enhanced logging — final iteration always printed
        if it % log_every == 0 or it == n_iters - 1:
            print(f'Iteration {it:4d} | f_loss = {f_loss.item():.4f}')

    end_time = time.time()
    training_time = end_time - start_time
    final_f_loss = history['f_loss'][-1]

    return f, g, history, final_f_loss, training_time
