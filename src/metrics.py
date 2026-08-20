"""
Official L2-UVP and Cosine Similarity evaluation metrics.

Extracted 1:1 from the canonical notebook (Cell 20).
Source of truth: notebooks/ICNN_Optimal_Transport_Colab_FINAL.ipynb

Computed against benchmark.map_fwd (ground-truth forward map),
exactly as in Korotin et al. (2021), Section 4.2.
"""
import numpy as np
import torch


def compute_official_l2_uvp(benchmark, f_model, device, n_samples=8192, var_samples=16384):
    """
    L2-UVP / cosine metrics computed exactly as in Korotin et al. (2021), §4.2,
    against the benchmark's own ground-truth forward map (benchmark.map_fwd), not a
    hand-derived analytical formula.
    """
    f_model.eval()

    # 1. Sample mu, compute OUR learned forward map T_hat = grad(f)(x)
    X = benchmark.input_sampler.sample(n_samples).to(device)
    X = X.clone().requires_grad_(True)
    with torch.enable_grad():
        T_hat = f_model.grad(X)

    # 2. Ground-truth forward map from the official benchmark object
    # Temporarily unfreeze the potential's parameters to allow internal gradient computation.
    for param in benchmark.potential.parameters():
        param.requires_grad = True

    # Remove `torch.no_grad()` context here as `map_fwd`'s `nograd=True` handles the final output.
    T_true = benchmark.map_fwd(X, nograd=True)

    # Refreeze the potential's parameters to restore its original state.
    for param in benchmark.potential.parameters():
        param.requires_grad = False

    T_hat_np = T_hat.detach().cpu().numpy()
    T_true_np = T_true.detach().cpu().numpy()

    # 3. Var(Q) estimated from a large sample of the official OUTPUT measure (nu),
    #    exactly as prescribed by the L2-UVP definition (not Var of our own map).
    Y_var_sample = benchmark.output_sampler.sample(var_samples).cpu().numpy()
    var_Q = np.sum((Y_var_sample - Y_var_sample.mean(axis=0)) ** 2, axis=1).mean()

    l2_error = np.sum((T_hat_np - T_true_np) ** 2, axis=1).mean()
    l2_uvp = 100.0 * l2_error / var_Q

    cos = torch.nn.CosineSimilarity(dim=1)
    cos_sim = cos(T_hat.detach().cpu() - X.detach().cpu(),
                  T_true.cpu() - X.detach().cpu()).mean().item()

    return {
        "L2-UVP": float(l2_uvp),
        "Cosine Similarity": float(cos_sim),
        "L2 Error": float(l2_error),
    }
