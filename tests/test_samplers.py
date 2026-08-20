"""
Unit tests for src/samplers.py

Tests: output shape, finiteness, and basic statistical properties.
"""
import unittest
import numpy as np
import torch
from src.samplers import (
    circle_means,
    make_gaussian_sampler,
    make_mixture_sampler,
    make_disconnected_sampler,
    make_finite_pool_sampler,
)


class TestCircleMeans(unittest.TestCase):

    def test_count(self):
        means = circle_means(8, radius=2.5)
        self.assertEqual(len(means), 8)

    def test_radius(self):
        for m in circle_means(6, radius=3.0):
            dist = np.sqrt(m[0]**2 + m[1]**2)
            self.assertAlmostEqual(dist, 3.0, places=5)

    def test_dimension(self):
        for m in circle_means(4):
            self.assertEqual(len(m), 2)


class TestGaussianSampler(unittest.TestCase):

    def test_output_shape(self):
        s = make_gaussian_sampler([0.0, 0.0], 1.0)
        out = s(50)
        self.assertEqual(out.shape, (50, 2))

    def test_finite(self):
        s = make_gaussian_sampler([1.0, -2.0], 0.5)
        self.assertTrue(torch.isfinite(s(100)).all())

    def test_mean_approximately_correct(self):
        torch.manual_seed(0)
        np.random.seed(0)
        s = make_gaussian_sampler([3.0, -3.0], 0.1)
        samples = s(5000)
        mean = samples.mean(dim=0)
        self.assertAlmostEqual(mean[0].item(), 3.0, delta=0.05)
        self.assertAlmostEqual(mean[1].item(), -3.0, delta=0.05)


class TestMixtureSampler(unittest.TestCase):

    def test_output_shape(self):
        means = [[0.0, 0.0], [5.0, 5.0]]
        covs = [np.eye(2), np.eye(2)]
        s = make_mixture_sampler(means, covs, [0.5, 0.5])
        out = s(64)
        self.assertEqual(out.shape, (64, 2))

    def test_finite(self):
        means = [[0.0, 0.0], [5.0, 5.0]]
        covs = [np.eye(2), np.eye(2)]
        s = make_mixture_sampler(means, covs, [0.5, 0.5])
        self.assertTrue(torch.isfinite(s(64)).all())


class TestDisconnectedSampler(unittest.TestCase):

    def test_output_shape(self):
        s = make_disconnected_sampler([[-2.0, 0.0], [2.0, 0.0]], 0.4)
        out = s(50)
        self.assertEqual(out.shape, (50, 2))

    def test_finite(self):
        s = make_disconnected_sampler([[-2.0, 0.0], [2.0, 0.0]], 0.4)
        self.assertTrue(torch.isfinite(s(50)).all())


class TestFinitePoolSampler(unittest.TestCase):

    def test_output_shape(self):
        pool = torch.randn(100, 4)
        s = make_finite_pool_sampler(pool)
        out = s(20)
        self.assertEqual(out.shape, (20, 4))

    def test_samples_come_from_pool(self):
        pool = torch.randn(30, 3)
        s = make_finite_pool_sampler(pool)
        samples = s(10)
        for sample in samples:
            found = any(torch.allclose(sample, p) for p in pool)
            self.assertTrue(found)


if __name__ == '__main__':
    unittest.main()
