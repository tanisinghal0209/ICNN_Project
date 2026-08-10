"""Dataset-agnostic training loop with logging + checkpointing."""
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from config import load_config, parse_args
from icnn import ICNN, StandardMLP
from losses import minimax_loss


def save_checkpoint(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path, map_location=None):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state_dict = torch.load(path, map_location=map_location)
    model.load_state_dict(state_dict)
    return model


def train_icnn_ot(mu_sampler, nu_sampler, input_dim, n_iters=3000, batch_size=256,
                   hidden_dims=(64, 64, 64), lr=1e-3, inner_iters=10, device="cpu",
                   activation="softplus", log_every=500, log_dir=None, exp_name="run", checkpoint_every=1000,
                   seed=0, optimizer="adam", convexity_coeff=0.0, model_type="icnn"):
    torch.manual_seed(seed)
    if model_type.lower() == "icnn":
        f = ICNN(input_dim, hidden_dims, activation=activation).to(device)
        g = ICNN(input_dim, hidden_dims, activation=activation).to(device)
    elif model_type.lower() == "mlp":
        f = StandardMLP(input_dim, hidden_dims, activation=activation).to(device)
        g = StandardMLP(input_dim, hidden_dims, activation=activation).to(device)
    else:
        raise ValueError(f"Unsupported model_type '{model_type}'. Use 'icnn' or 'mlp'.")
    if optimizer.lower() == "adam":
        opt_f = torch.optim.Adam(f.parameters(), lr=lr, betas=(0.5, 0.9))
        opt_g = torch.optim.Adam(g.parameters(), lr=lr, betas=(0.5, 0.9))
    elif optimizer.lower() == "sgd":
        opt_f = torch.optim.SGD(f.parameters(), lr=lr)
        opt_g = torch.optim.SGD(g.parameters(), lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer '{optimizer}'. Use 'adam' or 'sgd'.")
    history = {"f_loss": []}
    loss_history_path = None

    run_dir = csv_path = checkpoint_dir = figure_dir = None
    if log_dir is not None:
        run_dir = Path(log_dir) / exp_name
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        figure_dir = run_dir / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        config = dict(input_dim=input_dim, n_iters=n_iters, batch_size=batch_size,
                       hidden_dims=list(hidden_dims), lr=lr, inner_iters=inner_iters,
                       activation=activation, device=str(device), model_type=model_type,
                       optimizer=optimizer, convexity_coeff=convexity_coeff, seed=seed)
        with open(run_dir / "config.json", "w") as fh:
            json.dump(config, fh, indent=2)
        loss_history_path = run_dir / "loss_history.json"
        with open(loss_history_path, "w") as fh:
            json.dump({"f_loss": []}, fh, indent=2)
        csv_path = run_dir / "log.csv"
        with open(csv_path, "w", newline="") as fh:
            csv.writer(fh).writerow(["iter", "f_loss", "elapsed_sec"])
        print(f"Logging to: {run_dir}")

    start_time = time.time()
    for it in range(n_iters):
        for _ in range(inner_iters):
            x_mu, y_nu = mu_sampler(batch_size).to(device), nu_sampler(batch_size).to(device)
            _, g_loss = minimax_loss(f, g, x_mu, y_nu)
            if convexity_coeff:
                g_loss = g_loss + convexity_coeff * g.convexity_penalty()
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()
            g.clip_weights()

        x_mu, y_nu = mu_sampler(batch_size).to(device), nu_sampler(batch_size).to(device)
        f_loss, _ = minimax_loss(f, g, x_mu, y_nu)
        opt_f.zero_grad()
        f_loss.backward()
        opt_f.step()
        f.clip_weights()

        history["f_loss"].append(f_loss.item())
        if it % log_every == 0:
            print(f"iter {it:5d} | f_loss {f_loss.item(): .4f}")

        if run_dir is not None:
            with open(csv_path, "a", newline="") as fh:
                csv.writer(fh).writerow([it, f_loss.item(), round(time.time() - start_time, 2)])
            if loss_history_path is not None:
                with open(loss_history_path, "w") as fh:
                    json.dump({"f_loss": history["f_loss"]}, fh, indent=2)
            if checkpoint_every and it > 0 and it % checkpoint_every == 0:
                torch.save(f.state_dict(), checkpoint_dir / f"f_iter{it}.pt")
                torch.save(g.state_dict(), checkpoint_dir / f"g_iter{it}.pt")

    if run_dir is not None:
        torch.save(f.state_dict(), checkpoint_dir / "f_final.pt")
        torch.save(g.state_dict(), checkpoint_dir / "g_final.pt")
        metrics = {
            "final_f_loss": history["f_loss"][-1] if history["f_loss"] else None,
            "n_iters": n_iters,
            "batch_size": batch_size,
            "hidden_dims": list(hidden_dims),
            "lr": lr,
            "inner_iters": inner_iters,
            "activation": activation,
        }
        if loss_history_path is not None:
            with open(loss_history_path, "w") as fh:
                json.dump({"f_loss": history["f_loss"]}, fh, indent=2)
        with open(run_dir / "metrics.json", "w") as fh:
            json.dump(metrics, fh, indent=2)
        plt.figure()
        plt.plot(range(len(history["f_loss"])), history["f_loss"], label="f_loss")
        plt.xlabel("iter")
        plt.ylabel("f_loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figure_dir / "training_loss.png")
        print(f"Saved final checkpoints to: {checkpoint_dir}")
        print(f"Saved training loss figure to: {figure_dir / 'training_loss.png'}")

    return f, g, history


def main(argv=None):
    args = parse_args(argv)
    if hasattr(args, "config"):
        delattr(args, "config")

    if args.device not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device '{args.device}'. Use 'cpu' or 'cuda'.")

    def mu_sampler(batch_size):
        return torch.randn(batch_size, args.input_dim, device=args.device)

    def nu_sampler(batch_size):
        return torch.randn(batch_size, args.input_dim, device=args.device)

    train_icnn_ot(
        mu_sampler,
        nu_sampler,
        input_dim=args.input_dim,
        n_iters=args.n_iters,
        batch_size=args.batch_size,
        hidden_dims=tuple(args.hidden_dims),
        lr=args.lr,
        inner_iters=args.inner_iters,
        device=args.device,
        activation=args.activation,
        log_every=args.log_every,
        log_dir="logs",
        exp_name=args.exp_name,
        checkpoint_every=args.checkpoint_every,
        seed=args.seed,
        optimizer=args.optimizer,
        convexity_coeff=args.convexity_coeff,
        model_type=args.model_type,
    )


if __name__ == "__main__":
    main()
