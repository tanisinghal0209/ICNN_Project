from pathlib import Path
import json
import torch

from experiments.experiment_runner import make_gaussian_sampler, run_experiment, plot_transport_experiment


def load_config(path):
    with open(path, 'r') as fh:
        return json.load(fh)


def main():
    root = Path(__file__).resolve().parents[1]
    cfg_path = root / 'configs' / 'gaussian_2d.json'
    config = load_config(cfg_path)

    # simple source and target: translated gaussian
    mu = [0.0, 0.0]
    sigma_mu = 0.6
    nu = [2.0, -1.0]
    sigma_nu = 0.6

    mu_sampler = make_gaussian_sampler(mu, sigma_mu, device=config.get('device','cpu'))
    nu_sampler = make_gaussian_sampler(nu, sigma_nu, device=config.get('device','cpu'))

    out_dir, metrics = run_experiment('gaussian_2d_demo', config, mu_sampler, nu_sampler, plotting_fn=plot_transport_experiment)
    print('Experiment saved to', out_dir)
    print('Metrics:', metrics)


if __name__ == '__main__':
    main()
