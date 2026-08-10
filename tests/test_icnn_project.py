import os
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure the project root is importable when running tests directly.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from icnn import ICNN
from losses import minimax_loss
from train import load_checkpoint, load_config, save_checkpoint, train_icnn_ot


class TestICNNProject(unittest.TestCase):

    def test_icnn_forward_pass(self):
        model = ICNN(input_dim=4, hidden_dims=(8, 8), activation="relu")
        x = torch.randn(2, 4)
        out = model(x)
        self.assertEqual(out.shape, (2,))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_icnn_gradient_computation(self):
        model = ICNN(input_dim=3, hidden_dims=(6, 6), activation="softplus")
        y = torch.randn(5, 3)
        grad_y = model.grad(y)
        self.assertEqual(grad_y.shape, y.shape)
        self.assertTrue(torch.isfinite(grad_y).all().item())

    def test_icnn_convexity_clipping(self):
        model = ICNN(input_dim=3, hidden_dims=(5, 5), activation="relu")
        for layer in model.Wz:
            if layer is not None:
                layer.weight.data.fill_(-0.5)
        model.clip_weights()
        for layer in model.Wz:
            if layer is not None:
                self.assertTrue((layer.weight.data >= 0).all().item())

    def test_minimax_loss_returns_scalars(self):
        f = ICNN(input_dim=2, hidden_dims=(4, 4), activation="softplus")
        g = ICNN(input_dim=2, hidden_dims=(4, 4), activation="softplus")
        x_mu = torch.randn(6, 2)
        y_nu = torch.randn(6, 2)
        f_loss, g_loss = minimax_loss(f, g, x_mu, y_nu)
        self.assertIsInstance(f_loss, torch.Tensor)
        self.assertIsInstance(g_loss, torch.Tensor)
        self.assertEqual(f_loss.shape, ())
        self.assertEqual(g_loss.shape, ())
        self.assertTrue(torch.isfinite(f_loss).item())
        self.assertTrue(torch.isfinite(g_loss).item())

    def test_checkpoint_save_and_load(self):
        model = ICNN(input_dim=2, hidden_dims=(4, 4), activation="relu")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            save_checkpoint(model, path)
            loaded = ICNN(input_dim=2, hidden_dims=(4, 4), activation="relu")
            loaded = load_checkpoint(loaded, path, map_location="cpu")
            for p_saved, p_loaded in zip(model.parameters(), loaded.parameters()):
                self.assertTrue(torch.allclose(p_saved, p_loaded))

    def test_config_loading_and_loss_history_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text('{"input_dim": 2, "n_iters": 1, "batch_size": 4, "hidden_dims": [4, 4], "lr": 1e-3, "inner_iters": 1, "activation": "relu", "device": "cpu", "log_every": 1, "checkpoint_every": 1, "exp_name": "cfg_test"}', encoding="utf-8")
            config = load_config(config_path)
            self.assertEqual(config["input_dim"], 2)
            self.assertEqual(config["exp_name"], "cfg_test")

    def test_tiny_training_run_creates_outputs(self):
        def mu_sampler(batch_size):
            return torch.randn(batch_size, 2)

        def nu_sampler(batch_size):
            return torch.randn(batch_size, 2)

        with tempfile.TemporaryDirectory() as tmpdir:
            f, g, history = train_icnn_ot(
                mu_sampler,
                nu_sampler,
                input_dim=2,
                n_iters=2,
                batch_size=8,
                hidden_dims=(8, 8),
                lr=1e-3,
                inner_iters=1,
                device="cpu",
                activation="leaky_relu",
                log_every=1,
                log_dir=tmpdir,
                exp_name="tiny_run",
                checkpoint_every=1,
            )

            run_dir = Path(tmpdir) / "tiny_run"
            self.assertTrue((run_dir / "config.json").exists())
            self.assertTrue((run_dir / "log.csv").exists())
            self.assertTrue((run_dir / "metrics.json").exists())
            self.assertTrue((run_dir / "figures" / "training_loss.png").exists())
            self.assertTrue((run_dir / "checkpoints" / "f_final.pt").exists())
            self.assertEqual(len(history["f_loss"]), 2)
            self.assertIsNotNone(f)
            self.assertIsNotNone(g)


if __name__ == "__main__":
    unittest.main()
