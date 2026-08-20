"""
Unit tests for src/icnn.py

Tests: ICNN & StandardMLP forward pass, gradient computation,
       weight clipping, and convexity penalty.
"""
import unittest
import torch
from src.icnn import ICNN, StandardMLP


class TestICNNForward(unittest.TestCase):

    def test_icnn_forward_output_shape(self):
        model = ICNN(input_dim=2, hidden_dims=(8, 8), activation='softplus')
        x = torch.randn(4, 2)
        out = model(x)
        self.assertEqual(out.shape, (4,))

    def test_icnn_forward_finite(self):
        model = ICNN(input_dim=4, hidden_dims=(8, 8, 8), activation='relu')
        x = torch.randn(6, 4)
        out = model(x)
        self.assertTrue(torch.isfinite(out).all())

    def test_mlp_forward_output_shape(self):
        model = StandardMLP(input_dim=2, hidden_dims=(8, 8), activation='softplus')
        x = torch.randn(4, 2)
        out = model(x)
        self.assertEqual(out.shape, (4,))

    def test_mlp_forward_finite(self):
        model = StandardMLP(input_dim=4, hidden_dims=(8, 8), activation='leaky_relu')
        x = torch.randn(6, 4)
        out = model(x)
        self.assertTrue(torch.isfinite(out).all())


class TestICNNGradient(unittest.TestCase):

    def test_icnn_grad_shape(self):
        model = ICNN(input_dim=3, hidden_dims=(6, 6), activation='softplus')
        y = torch.randn(5, 3)
        grad = model.grad(y)
        self.assertEqual(grad.shape, (5, 3))

    def test_icnn_grad_finite(self):
        model = ICNN(input_dim=3, hidden_dims=(6, 6), activation='softplus')
        y = torch.randn(5, 3)
        grad = model.grad(y)
        self.assertTrue(torch.isfinite(grad).all())

    def test_icnn_grad_works_under_no_grad(self):
        """grad() must work even when called inside torch.no_grad() context."""
        model = ICNN(input_dim=2, hidden_dims=(4, 4), activation='softplus')
        y = torch.randn(4, 2)
        with torch.no_grad():
            grad = model.grad(y)
        self.assertEqual(grad.shape, (4, 2))
        self.assertTrue(torch.isfinite(grad).all())

    def test_mlp_grad_works_under_no_grad(self):
        """StandardMLP.grad() must also work inside torch.no_grad() context."""
        model = StandardMLP(input_dim=2, hidden_dims=(4, 4), activation='softplus')
        y = torch.randn(4, 2)
        with torch.no_grad():
            grad = model.grad(y)
        self.assertEqual(grad.shape, (4, 2))


class TestICNNWeightClipping(unittest.TestCase):

    def test_clip_weights_zeros_negatives(self):
        model = ICNN(input_dim=3, hidden_dims=(5, 5), activation='relu')
        for layer in model.Wz:
            if layer is not None:
                layer.weight.data.fill_(-1.0)
        model.clip_weights()
        for layer in model.Wz:
            if layer is not None:
                self.assertTrue((layer.weight.data >= 0).all())

    def test_clip_preserves_non_negatives(self):
        model = ICNN(input_dim=3, hidden_dims=(5, 5), activation='relu')
        for layer in model.Wz:
            if layer is not None:
                layer.weight.data.fill_(0.5)
        model.clip_weights()
        for layer in model.Wz:
            if layer is not None:
                self.assertTrue(torch.allclose(layer.weight.data, torch.full_like(layer.weight.data, 0.5)))

    def test_mlp_clip_is_noop(self):
        """StandardMLP.clip_weights() must be a no-op (returns None)."""
        model = StandardMLP(input_dim=2, hidden_dims=(4, 4), activation='relu')
        result = model.clip_weights()
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
