"""Lightweight configuration helpers for ICNN experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "input_dim": 2,
    "n_iters": 3000,
    "batch_size": 256,
    "hidden_dims": [64, 64, 64],
    "lr": 1e-3,
    "inner_iters": 10,
    "activation": "softplus",
    "optimizer": "adam",
    "convexity_coeff": 0.0,
    "model_type": "icnn",
    "seed": 0,
    "device": "cpu",
    "log_every": 500,
    "checkpoint_every": 1000,
    "exp_name": "baseline",
}


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Load config from a JSON or YAML file, falling back to defaults."""
    config = dict(DEFAULT_CONFIG)
    if path is None:
        return config

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        if path.suffix.lower() in {".yaml", ".yml"}:
            loaded = yaml.safe_load(fh) or {}
        else:
            loaded = json.load(fh)

    if not isinstance(loaded, dict):
        raise TypeError("Config file must contain a mapping at the top level")

    config.update(loaded)
    return config


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an ICNN-based OT model")
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON/YAML config file")
    parser.add_argument("--input_dim", type=int, default=None)
    parser.add_argument("--n_iters", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--inner_iters", type=int, default=None)
    parser.add_argument("--activation", type=str, default=None)
    parser.add_argument("--optimizer", type=str, default=None)
    parser.add_argument("--convexity_coeff", type=float, default=None)
    parser.add_argument("--model_type", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log_every", type=int, default=None)
    parser.add_argument("--checkpoint_every", type=int, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)

    overrides = {
        key: value for key, value in vars(args).items() if value is not None and key != "config"
    }
    if "hidden_dims" in overrides:
        overrides["hidden_dims"] = list(overrides["hidden_dims"])

    config.update(overrides)
    return argparse.Namespace(**config)
