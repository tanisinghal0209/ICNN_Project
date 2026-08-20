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


def compute_grad_norm(model):
    """Compute the L2 norm of the gradients of the model."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def compute_param_norm(model):
    """Compute the L2 norm of the parameters of the model."""
    total_norm = 0.0
    for p in model.parameters():
        param_norm = p.data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def check_nan_inf(model):
    """Check if any model parameters or gradients contain NaN or Inf."""
    for p in model.parameters():
        if not torch.isfinite(p).all():
            return True
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return True
    return False


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
    history = {
        'f_loss': [],
        'g_loss': [],
        'f_grad_norm': [],
        'g_grad_norm': [],
        'f_param_norm': [],
        'g_param_norm': [],
        'has_nan': [],
    }
    start_time = time.time()

    for it in range(n_iters):
        # 1. Update potential g (inner maximization loop)
        last_g_loss = 0.0
        g_grad_norm = 0.0
        for idx in range(inner_iters):
            x_mu = mu_sampler(batch_size).to(device)
            y_nu = nu_sampler(batch_size).to(device)
            _, g_loss = minimax_loss(f, g, x_mu, y_nu)
            opt_g.zero_grad()
            g_loss.backward()
            if idx == inner_iters - 1:
                g_grad_norm = compute_grad_norm(g)
                last_g_loss = g_loss.item()
            opt_g.step()
            g.clip_weights()

        # 2. Update potential f (outer minimization loop)
        x_mu = mu_sampler(batch_size).to(device)
        y_nu = nu_sampler(batch_size).to(device)
        f_loss, _ = minimax_loss(f, g, x_mu, y_nu)
        opt_f.zero_grad()
        f_loss.backward()
        f_grad_norm = compute_grad_norm(f)
        opt_f.step()
        f.clip_weights()

        history['f_loss'].append(f_loss.item())
        history['g_loss'].append(last_g_loss)
        history['f_grad_norm'].append(f_grad_norm)
        history['g_grad_norm'].append(g_grad_norm)
        history['f_param_norm'].append(compute_param_norm(f))
        history['g_param_norm'].append(compute_param_norm(g))
        history['has_nan'].append(check_nan_inf(f) or check_nan_inf(g))

        # Enhanced logging — final iteration always printed
        if it % log_every == 0 or it == n_iters - 1:
            print(f'Iteration {it:4d} | f_loss = {f_loss.item():.4f} | g_loss = {last_g_loss:.4f} | f_gnorm = {f_grad_norm:.4f}')

    end_time = time.time()
    training_time = end_time - start_time
    final_f_loss = history['f_loss'][-1]

    return f, g, history, final_f_loss, training_time
