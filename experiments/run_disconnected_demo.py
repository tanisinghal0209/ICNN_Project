from pathlib import Path
import json
import numpy as np
import torch

from experiments.experiment_runner import run_experiment, plot_transport_experiment


def load_config(path):
    with open(path, 'r') as fh:
        return json.load(fh)


def make_disconnected_sampler(means, sigma, device='cpu'):
    means = [torch.tensor(m, device=device, dtype=torch.float32) for m in means]

    def sampler(batch_size):
        idx = np.random.choice(len(means), size=batch_size)
        samples = []
        for k in range(len(means)):
            mask = idx == k
            if mask.sum() == 0:
                continue
            n = int(mask.sum())
            z = torch.randn(n, len(means[k]), device=device, dtype=torch.float32)
            samples.append(z * sigma + means[k])
        return torch.cat(samples, dim=0)

    return sampler


def main():
    root = Path(__file__).resolve().parents[1]
    cfg_path = root / 'configs' / 'disconnected_2d.json'
    config = load_config(cfg_path)

    # source: two separated gaussians
    source_means = [[-2.0, 0.0], [2.0, 0.0]]
    source_sigma = 0.3
    mu_sampler = make_disconnected_sampler(source_means, source_sigma, device=config.get('device', 'cpu'))

    # target: two diagonal clusters with different shape and location
    target_means = [[-1.5, -1.5], [1.5, 1.5]]
    target_sigma = 0.5
    nu_sampler = make_disconnected_sampler(target_means, target_sigma, device=config.get('device', 'cpu'))

    out_dir, metrics = run_experiment('disconnected_2d_demo', config, mu_sampler, nu_sampler, plotting_fn=plot_transport_experiment)
    print('Experiment saved to', out_dir)
    print('Metrics:', metrics)


if __name__ == '__main__':
    main()
