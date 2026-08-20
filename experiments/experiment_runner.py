import json
import time
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.solver import train_icnn_ot


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)


def save_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write(str(text))


def save_figure(path, fig):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def make_gaussian_sampler(mu, sigma, device="cpu"):
    def sampler(batch_size):
        return torch.randn(batch_size, len(mu), device=device) * sigma + torch.tensor(mu, device=device)
    return sampler


def make_mixture_sampler(means, covariances, weights, device="cpu"):
    means = [torch.tensor(m, device=device, dtype=torch.float32) for m in means]
    covariances = [torch.tensor(c, device=device, dtype=torch.float32) for c in covariances]
    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.sum()

    def sampler(batch_size):
        idx = np.random.choice(len(weights), size=batch_size, p=weights)
        samples = []
        for k in range(len(weights)):
            mask = idx == k
            if mask.sum() == 0:
                continue
            n = int(mask.sum())
            mean = means[k]
            cov = covariances[k]
            z = torch.randn(n, mean.shape[0], device=device, dtype=torch.float32)
            L = torch.linalg.cholesky(cov)
            samples.append((z @ L.T) + mean)
        return torch.cat(samples, dim=0)

    return sampler


def run_experiment(name, config, mu_sampler, nu_sampler, plotting_fn=None):
    root = ROOT / "experiments" / name
    root.mkdir(parents=True, exist_ok=True)

    config_path = root / "config.json"
    save_json(config_path, config)
    save_json(root / "config.yaml", config)

    start = time.time()
    f, g, history = train_icnn_ot(
        mu_sampler,
        nu_sampler,
        input_dim=config["input_dim"],
        n_iters=config.get("n_iters", 1000),
        batch_size=config.get("batch_size", 256),
        hidden_dims=tuple(config.get("hidden_dims", [64, 64, 64])),
        lr=config.get("lr", 1e-3),
        inner_iters=config.get("inner_iters", 10),
        device=config.get("device", "cpu"),
        activation=config.get("activation", "leaky_relu"),
        log_every=config.get("log_every", 100),
        log_dir=str(root),
        exp_name="run",
        checkpoint_every=config.get("checkpoint_every", 0),
    )
    runtime = time.time() - start

    metrics = {
        "runtime_sec": round(runtime, 4),
        "final_f_loss": history["f_loss"][-1] if history["f_loss"] else None,
        "n_iters": len(history["f_loss"]),
        "seed": config.get("seed", 0),
        "model_type": config.get("model_type", "icnn"),
    }
    save_json(root / "metrics.json", metrics)
    save_text(root / "runtime.txt", f"{runtime:.4f}\n")

    if plotting_fn is not None:
        plot_data = plotting_fn(mu_sampler, nu_sampler, f, g, config, history)
        for label, fig in plot_data.items():
            save_figure(root / f"{label}.png", fig)
    else:
        fig = plt.figure(figsize=(5, 4))
        plt.plot(history["f_loss"], label="f_loss")
        plt.xlabel("iteration")
        plt.ylabel("f_loss")
        plt.legend()
        save_figure(root / "loss.png", fig)

    return root, metrics


def simple_transport_plot(mu_sampler, nu_sampler, f, g, config, history):
    dim = config["input_dim"]
    assert dim == 2, "Plotting only supported for 2D experiments"
    n_plot = config.get("plot_n", 256)
    y = torch.randn(n_plot, dim)
    transport = g.grad(y).detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()

    fig1, ax1 = plt.subplots()
    ax1.scatter(y_np[:, 0], y_np[:, 1], s=10, alpha=0.4)
    ax1.scatter(transport[:, 0], transport[:, 1], s=10, alpha=0.4)
    ax1.set_title("transport_map")

    fig2, ax2 = plt.subplots()
    ax2.quiver(y_np[::5, 0], y_np[::5, 1], transport[::5, 0] - y_np[::5, 0], transport[::5, 1] - y_np[::5, 1], angles='xy', scale_units='xy', scale=1)
    ax2.set_title("vector_field")

    fig3, ax3 = plt.subplots(figsize=(5, 4))
    ax3.plot(history["f_loss"], label="f_loss")
    ax3.set_title("training_loss")
    ax3.set_xlabel("iteration")
    ax3.set_ylabel("f_loss")
    ax3.legend()

    return {"transport_map": fig1, "vector_field": fig2, "training_loss": fig3}


def plot_dataset_scatter(ax, x, y, title, labels=("x", "y")):
    ax.scatter(x[:, 0], x[:, 1], s=18, alpha=0.5, label=labels[0], color="C0")
    ax.scatter(y[:, 0], y[:, 1], s=18, alpha=0.5, label=labels[1], color="C1")
    ax.set_title(title)
    ax.legend()


def plot_transport_experiment(mu_sampler, nu_sampler, f, g, config, history):
    dim = config["input_dim"]
    assert dim == 2, "Plotting only supported for 2D experiments"

    n_plot = config.get("plot_n", 256)
    x_plot = mu_sampler(n_plot).detach().cpu().numpy()
    y_plot = nu_sampler(n_plot).detach().cpu().numpy()
    transport = g.grad(torch.tensor(y_plot, dtype=torch.float32)).detach().cpu().numpy()

    fig1, ax1 = plt.subplots(figsize=(5, 5))
    plot_dataset_scatter(ax1, x_plot, y_plot, "source_vs_target")

    fig2, ax2 = plt.subplots(figsize=(5, 5))
    ax2.scatter(y_plot[:, 0], y_plot[:, 1], s=12, alpha=0.4, label="y", color="C1")
    ax2.scatter(transport[:, 0], transport[:, 1], s=12, alpha=0.4, label="transported y", color="C2")
    ax2.set_title("transport_map")
    ax2.legend()

    fig3, ax3 = plt.subplots(figsize=(5, 4))
    ax3.plot(history["f_loss"], label="f_loss")
    ax3.set_title("training_loss")
    ax3.set_xlabel("iteration")
    ax3.set_ylabel("f_loss")
    ax3.legend()

    fig4, ax4 = plt.subplots(figsize=(5, 5))
    step = max(1, n_plot // 64)
    ax4.quiver(
        y_plot[::step, 0], y_plot[::step, 1],
        transport[::step, 0] - y_plot[::step, 0], transport[::step, 1] - y_plot[::step, 1],
        angles='xy', scale_units='xy', scale=1,
    )
    ax4.set_title("vector_field")

    return {
        "source_target_scatter": fig1,
        "transport_map": fig2,
        "training_loss": fig3,
        "vector_field": fig4,
    }
