"""
Probability distribution samplers for toy datasets.

Extracted 1:1 from the canonical notebook (Cells 12 & 16).
Source of truth: notebooks/ICNN_Optimal_Transport_Colab_FINAL.ipynb

Only the 4 samplers used in the active D=2 pipeline are included.
MNIST/VAE samplers are excluded from this refactoring phase.
"""
import math
import numpy as np
import torch


def circle_means(k, radius=2.5):
    """Generate k points evenly spaced along a circle (used for mixture experiments)."""
    angles = np.linspace(0, 2 * math.pi, k, endpoint=False)
    return [[radius * math.cos(a), radius * math.sin(a)] for a in angles]


def make_gaussian_sampler(mu, sigma, device='cpu'):
    """Isotropic Gaussian sampler. Extracted from notebook Cell 12."""
    def sampler(batch_size):
        return torch.randn(batch_size, len(mu), device=device) * sigma + torch.tensor(mu, device=device)
    return sampler


def make_mixture_sampler(means, covariances, weights, device='cpu'):
    """Gaussian Mixture Model sampler. Extracted from notebook Cell 12."""
    means_t = [torch.tensor(m, device=device, dtype=torch.float32) for m in means]
    covariances_t = [torch.tensor(c, device=device, dtype=torch.float32) for c in covariances]
    weights = np.array(weights, dtype=np.float32) / sum(weights)

    def sampler(batch_size):
        idx = np.random.choice(len(weights), size=batch_size, p=weights)
        samples = []
        for k in range(len(weights)):
            n = int((idx == k).sum())
            if n == 0:
                continue
            z = torch.randn(n, means_t[0].shape[0], device=device)
            L = torch.linalg.cholesky(covariances_t[k])
            samples.append((z @ L.T) + means_t[k])
        return torch.cat(samples, dim=0)

    return sampler


def make_disconnected_sampler(means, sigma, device='cpu'):
    """Disconnected-support Gaussian sampler. Extracted from notebook Cell 12."""
    means_t = [torch.tensor(m, device=device, dtype=torch.float32) for m in means]

    def sampler(batch_size):
        idx = np.random.choice(len(means_t), size=batch_size)
        samples = []
        for k in range(len(means_t)):
            n = int((idx == k).sum())
            if n == 0:
                continue
            z = torch.randn(n, len(means_t[k]), device=device)
            samples.append(z * sigma + means_t[k])
        return torch.cat(samples, dim=0)

    return sampler


def make_finite_pool_sampler(pool):
    """Finite-pool sampler (sampling with replacement). Extracted from notebook Cell 12."""
    def sampler(batch_size):
        idx = torch.randint(0, len(pool), (batch_size,), device=pool.device)
        return pool[idx]
    return sampler
