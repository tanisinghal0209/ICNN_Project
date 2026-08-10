from pathlib import Path
import json
import torch

from experiments.experiment_runner import run_experiment


def load_config(path):
    with open(path, 'r') as fh:
        return json.load(fh)


def make_simple_sampler(mu, sigma, device='cpu'):
    def sampler(batch_size):
        return torch.randn(batch_size, len(mu), device=device) * sigma + torch.tensor(mu, device=device, dtype=torch.float32)
    return sampler


def main():
    root = Path(__file__).resolve().parents[1]
    base_cfg = load_config(root / 'configs' / 'gaussian_2d.json')
    config = base_cfg.copy()
    config['n_iters'] = 200
    config['batch_size'] = 256
    config['plot_n'] = 256

    mu = [0.0, 0.0]
    nu = [2.0, -1.0]
    sigma = 0.6
    mu_sampler = make_simple_sampler(mu, sigma, device=config.get('device', 'cpu'))
    nu_sampler = make_simple_sampler(nu, sigma, device=config.get('device', 'cpu'))

    runs = [
        {'name': 'ablation_width_64', 'hidden_dims': [64, 64, 64]},
        {'name': 'ablation_width_256', 'hidden_dims': [256, 256, 256]},
        {'name': 'ablation_depth_2', 'hidden_dims': [128, 128]},
    ]

    for run in runs:
        cfg = config.copy()
        cfg['hidden_dims'] = run['hidden_dims']
        out_dir, metrics = run_experiment(run['name'], cfg, mu_sampler, nu_sampler)
        print(run['name'], metrics)


if __name__ == '__main__':
    main()
