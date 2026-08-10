"""MNIST latent-space samplers -- same n -> Tensor[n, dim] contract as the toy Gaussian samplers, so train_icnn_ot needs zero changes to use these."""
import numpy as np
import torch


def load_mnist_arrays(mnist_dataset):
    """Pull the whole dataset into memory once as tensors, for fast repeated sampling."""
    images = torch.stack([img for img, _ in mnist_dataset])
    labels = np.array([label for _, label in mnist_dataset])
    return images, labels


def make_mnist_latent_sampler(digit_set, vae, images, labels, device):
    """Returns a sampler function: n -> Tensor[n, latent_dim], drawing real MNIST images from the given digit classes and encoding them with vae.encode() (mean only)."""
    mask = np.isin(labels, list(digit_set))
    valid_idx = np.where(mask)[0]

    def sampler(n):
        idx = np.random.choice(valid_idx, size=n, replace=True)
        imgs = images[idx].to(device)
        with torch.no_grad():
            mu, _ = vae.encode(imgs)
        return mu.cpu()

    return sampler
