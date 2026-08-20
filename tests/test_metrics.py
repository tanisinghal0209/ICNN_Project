"""
Unit tests for src/metrics.py

Tests: L2-UVP shape, type, and computation structure.
"""
import os
import unittest
from pathlib import Path
import torch
from src.icnn import ICNN
from src.benchmark import load_mix3to10_benchmark
from src.metrics import compute_official_l2_uvp


BENCH_PATH = os.environ.get(
    'WASSERSTEIN_BENCHMARK_PATH',
    '/Users/tanishasinghal/Downloads/Wasserstein2Benchmark'
)


def bench_available():
    return Path(BENCH_PATH).exists()


class TestOfficialL2UVP(unittest.TestCase):

    def test_returns_correct_keys(self):
        if not bench_available():
            self.skipTest("Wasserstein2Benchmark not available")
        benchmark, _ = load_mix3to10_benchmark(dim=2, benchmark_path=BENCH_PATH)
        model = ICNN(input_dim=2, hidden_dims=(8, 8), activation='softplus')
        metrics = compute_official_l2_uvp(benchmark, model, device='cpu',
                                           n_samples=32, var_samples=64)
        self.assertIn('L2-UVP', metrics)
        self.assertIn('Cosine Similarity', metrics)
        self.assertIn('L2 Error', metrics)

    def test_returns_python_floats(self):
        if not bench_available():
            self.skipTest("Wasserstein2Benchmark not available")
        benchmark, _ = load_mix3to10_benchmark(dim=2, benchmark_path=BENCH_PATH)
        model = ICNN(input_dim=2, hidden_dims=(8, 8), activation='softplus')
        metrics = compute_official_l2_uvp(benchmark, model, device='cpu',
                                           n_samples=32, var_samples=64)
        self.assertIsInstance(metrics['L2-UVP'], float)
        self.assertIsInstance(metrics['Cosine Similarity'], float)
        self.assertIsInstance(metrics['L2 Error'], float)

    def test_l2_uvp_is_non_negative(self):
        if not bench_available():
            self.skipTest("Wasserstein2Benchmark not available")
        benchmark, _ = load_mix3to10_benchmark(dim=2, benchmark_path=BENCH_PATH)
        model = ICNN(input_dim=2, hidden_dims=(8, 8), activation='softplus')
        metrics = compute_official_l2_uvp(benchmark, model, device='cpu',
                                           n_samples=32, var_samples=64)
        self.assertGreaterEqual(metrics['L2-UVP'], 0.0)
        self.assertGreaterEqual(metrics['L2 Error'], 0.0)

    def test_cosine_similarity_in_range(self):
        if not bench_available():
            self.skipTest("Wasserstein2Benchmark not available")
        benchmark, _ = load_mix3to10_benchmark(dim=2, benchmark_path=BENCH_PATH)
        model = ICNN(input_dim=2, hidden_dims=(8, 8), activation='softplus')
        metrics = compute_official_l2_uvp(benchmark, model, device='cpu',
                                           n_samples=32, var_samples=64)
        self.assertGreaterEqual(metrics['Cosine Similarity'], -1.0)
        self.assertLessEqual(metrics['Cosine Similarity'], 1.0)


if __name__ == '__main__':
    unittest.main()
