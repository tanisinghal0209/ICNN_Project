"""
Unit tests for src/losses.py

Tests: minimax loss shapes, finiteness, and f/g relationship.
"""
import unittest
import torch
from src.icnn import ICNN
from src.losses import minimax_loss


class TestMinimaxLoss(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.f = ICNN(input_dim=2, hidden_dims=(8, 8), activation='softplus')
        self.g = ICNN(input_dim=2, hidden_dims=(8, 8), activation='softplus')

    def test_returns_scalars(self):
        x_mu = torch.randn(8, 2)
        y_nu = torch.randn(8, 2)
        f_loss, g_loss = minimax_loss(self.f, self.g, x_mu, y_nu)
        self.assertEqual(f_loss.shape, ())
        self.assertEqual(g_loss.shape, ())

    def test_finite_values(self):
        x_mu = torch.randn(8, 2)
        y_nu = torch.randn(8, 2)
        f_loss, g_loss = minimax_loss(self.f, self.g, x_mu, y_nu)
        self.assertTrue(torch.isfinite(f_loss))
        self.assertTrue(torch.isfinite(g_loss))

    def test_g_loss_is_negative_of_term_nu(self):
        """g_loss must be -term_nu, so f_loss + g_loss = term_mu."""
        x_mu = torch.randn(8, 2)
        y_nu = torch.randn(8, 2)
        f_loss, g_loss = minimax_loss(self.f, self.g, x_mu, y_nu)
        term_mu = self.f(x_mu).mean()
        # f_loss = term_mu + term_nu; g_loss = -term_nu
        # so f_loss + g_loss should ≈ term_mu
        self.assertAlmostEqual(
            (f_loss + g_loss).item(), term_mu.item(), places=5
        )

    def test_differentiable_wrt_f_params(self):
        x_mu = torch.randn(8, 2)
        y_nu = torch.randn(8, 2)
        f_loss, _ = minimax_loss(self.f, self.g, x_mu, y_nu)
        f_loss.backward()
        for p in self.f.parameters():
            self.assertIsNotNone(p.grad)

    def test_differentiable_wrt_g_params(self):
        # Use a fresh forward pass for g (create_graph allows double-backward through grad())
        x_mu = torch.randn(8, 2)
        y_nu = torch.randn(8, 2)
        _, g_loss = minimax_loss(self.f, self.g, x_mu, y_nu)
        g_loss.backward()
        # g.parameters include Wy and Wz layers; at least Wy should have grad
        wy_grads = [layer.weight.grad for layer in self.g.Wy if layer is not None]
        self.assertTrue(any(g is not None for g in wy_grads),
                        "At least one Wy parameter of g should have a gradient")


if __name__ == '__main__':
    unittest.main()
