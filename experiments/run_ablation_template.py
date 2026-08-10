from pathlib import Path

from experiments.experiment_runner import make_gaussian_sampler, run_experiment


if __name__ == "__main__":
    config = {
        "input_dim": 2,
        "n_iters": 3000,
        "batch_size": 256,
        "hidden_dims": [64, 64, 64],
        "lr": 1e-3,
        "inner_iters": 10,
        "activation": "softplus",
        "device": "cpu",
        "log_every": 500,
        "checkpoint_every": 1000,
        "plot_n": 256,
        "exp_name": "ablation_template",
    }
    mu_sampler = make_gaussian_sampler([0.0, 0.0], 1.0, device=config["device"])
    nu_sampler = make_gaussian_sampler([0.0, 0.0], 1.0, device=config["device"])
    run_experiment("ablation_template", config, mu_sampler, nu_sampler)
