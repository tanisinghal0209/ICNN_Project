"""CLI entry point for running ICNN optimal transport training."""
import sys
import torch
from src.config import parse_args
from src.solver import train_icnn_ot


def main(argv=None):
    args = parse_args(argv)
    if hasattr(args, "config"):
        delattr(args, "config")

    device = args.device
    if device not in {"cpu", "cuda"}:
        if device == "mps" or (device == "auto" and torch.backends.mps.is_available()):
            device = "mps"
        elif device == "cuda" or (device == "auto" and torch.cuda.is_available()):
            device = "cuda"
        else:
            device = "cpu"

    def mu_sampler(batch_size):
        return torch.randn(batch_size, args.input_dim, device=device)

    def nu_sampler(batch_size):
        return torch.randn(batch_size, args.input_dim, device=device)

    train_icnn_ot(
        mu_sampler,
        nu_sampler,
        input_dim=args.input_dim,
        n_iters=args.n_iters,
        batch_size=args.batch_size,
        hidden_dims=tuple(args.hidden_dims),
        lr=args.lr,
        inner_iters=args.inner_iters,
        device=device,
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
