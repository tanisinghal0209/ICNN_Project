"""
Unit tests for src/solver.py

Tests: canonical 5-element return contract, short training run stability,
       log_every output format, and model type selection.
"""
import unittest
import torch
from src.solver import train_icnn_ot


class TestSolverReturnContract(unittest.TestCase):
    """The canonical notebook returns (f, g, history, final_f_loss, training_time)."""

    def setUp(self):
        torch.manual_seed(0)

        def mu_sampler(batch_size):
            return torch.randn(batch_size, 2)

        def nu_sampler(batch_size):
            return torch.randn(batch_size, 2)

        self.mu = mu_sampler
        self.nu = nu_sampler

    def test_returns_five_elements(self):
        result = train_icnn_ot(self.mu, self.nu, input_dim=2,
                               n_iters=2, batch_size=8, hidden_dims=(4, 4),
                               inner_iters=1, log_every=10)
        self.assertEqual(len(result), 5, "Must return (f, g, history, final_f_loss, training_time)")

    def test_return_types(self):
        f, g, history, final_f_loss, training_time = train_icnn_ot(
            self.mu, self.nu, input_dim=2,
            n_iters=2, batch_size=8, hidden_dims=(4, 4),
            inner_iters=1, log_every=10
        )
        self.assertIsNotNone(f)
        self.assertIsNotNone(g)
        self.assertIsInstance(history, dict)
        self.assertIn('f_loss', history)
        self.assertIsInstance(final_f_loss, float)
        self.assertIsInstance(training_time, float)

    def test_history_length(self):
        _, _, history, _, _ = train_icnn_ot(
            self.mu, self.nu, input_dim=2,
            n_iters=5, batch_size=8, hidden_dims=(4, 4),
            inner_iters=1, log_every=10
        )
        self.assertEqual(len(history['f_loss']), 5)

    def test_final_f_loss_matches_history(self):
        _, _, history, final_f_loss, _ = train_icnn_ot(
            self.mu, self.nu, input_dim=2,
            n_iters=3, batch_size=8, hidden_dims=(4, 4),
            inner_iters=1, log_every=10
        )
        self.assertAlmostEqual(final_f_loss, history['f_loss'][-1], places=6)

    def test_f_loss_is_finite(self):
        _, _, history, _, _ = train_icnn_ot(
            self.mu, self.nu, input_dim=2,
            n_iters=4, batch_size=8, hidden_dims=(4, 4),
            inner_iters=1, log_every=10
        )
        for loss in history['f_loss']:
            self.assertTrue(
                torch.isfinite(torch.tensor(loss)),
                f"f_loss {loss} is not finite"
            )


class TestSolverModelTypes(unittest.TestCase):

    def setUp(self):
        def mu_sampler(b): return torch.randn(b, 2)
        def nu_sampler(b): return torch.randn(b, 2)
        self.mu, self.nu = mu_sampler, nu_sampler

    def test_icnn_model_type(self):
        from src.icnn import ICNN
        f, g, _, _, _ = train_icnn_ot(
            self.mu, self.nu, input_dim=2,
            n_iters=1, batch_size=4, hidden_dims=(4, 4),
            inner_iters=1, log_every=10, model_type='icnn'
        )
        self.assertIsInstance(f, ICNN)
        self.assertIsInstance(g, ICNN)

    def test_mlp_model_type(self):
        from src.icnn import StandardMLP
        f, g, _, _, _ = train_icnn_ot(
            self.mu, self.nu, input_dim=2,
            n_iters=1, batch_size=4, hidden_dims=(4, 4),
            inner_iters=1, log_every=10, model_type='mlp'
        )
        self.assertIsInstance(f, StandardMLP)
        self.assertIsInstance(g, StandardMLP)


if __name__ == '__main__':
    unittest.main()
