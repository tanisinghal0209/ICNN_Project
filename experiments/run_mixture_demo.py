from pathlib import Path
import json
import math
import torch
import numpy as np

from experiments.experiment_runner import make_gaussian_sampler, make_mixture_sampler, run_experiment, plot_transport_experiment


def load_config(path):
    with open(path, 'r') as fh:
        return json.load(fh)


def circle_means(k, radius=2.0):
    angles = np.linspace(0, 2 * math.pi, k, endpoint=False)
    return [[radius * math.cos(a), radius * math.sin(a)] for a in angles]


def main():
    root = Path(__file__).resolve().parents[1]
    cfg_path = root / 'configs' / 'mixture_2d.json'
    config = load_config(cfg_path)

    # source: single gaussian
    mu = [0.0, 0.0]
    sigma_mu = 0.8
    mu_sampler = make_gaussian_sampler(mu, sigma_mu, device=config.get('device','cpu'))

    # target: 8-gaussian mixture on a circle
    k = 8
    means = circle_means(k, radius=2.0)
    covs = [np.eye(2) * 0.08 for _ in range(k)]
    weights = [1.0 / k] * k
    nu_sampler = make_mixture_sampler(means, covs, weights, device=config.get('device','cpu'))

    out_dir, metrics = run_experiment('mixture_2d_demo', config, mu_sampler, nu_sampler, plotting_fn=plot_transport_experiment)
    print('Experiment saved to', out_dir)
    print('Metrics:', metrics)


if __name__ == '__main__':
    main()
