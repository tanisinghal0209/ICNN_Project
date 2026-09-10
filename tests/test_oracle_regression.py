"""Tests for the standalone oracle-supervised ICNN diagnostic."""
import os
import unittest
from pathlib import Path

import torch

from experiments.run_oracle_regression import (
    BENCHMARK_PATH,
    ICNN,
    sample_oracle_batch,
    train_oracle_icnn,
    transport_gradient,
)
from src.benchmark import load_mix3to10_benchmark


class TestOracleRegressionCore(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

        def mu_sampler(batch_size):
            return torch.randn(batch_size, 3)

        def target_map(x):
            # The actual benchmark's map is also evaluated from a grad-enabled x.
            return 2.0 * x

        self.mu_sampler = mu_sampler
        self.target_map = target_map

    def test_oracle_target_and_icnn_gradient_shapes(self):
        x, target = sample_oracle_batch(
            self.mu_sampler, self.target_map, batch_size=7, device="cpu"
        )
        model = ICNN(3, hidden_dims=(5, 5))
        prediction = transport_gradient(model, x)
        self.assertEqual(target.shape, (7, 3))
        self.assertEqual(prediction.shape, (7, 3))

    def test_oracle_mse_is_scalar_and_backpropagates_to_icnn_parameters(self):
        x, target = sample_oracle_batch(
            self.mu_sampler, self.target_map, batch_size=6, device="cpu"
        )
        model = ICNN(3, hidden_dims=(5, 5))
        prediction = transport_gradient(model, x)
        loss = torch.nn.functional.mse_loss(prediction, target)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(loss.requires_grad)
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))
        self.assertTrue(
            all(
                torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
                if parameter.grad is not None
            )
        )

    def test_weight_clipping_and_single_model_training_contract(self):
        model = ICNN(3, hidden_dims=(5, 5))
        for layer in model.Wz:
            if layer is not None:
                layer.weight.data.fill_(-1.0)
        model.clip_weights()
        self.assertTrue(all(
            (layer.weight >= 0).all() for layer in model.Wz if layer is not None
        ))

        result = train_oracle_icnn(
            self.mu_sampler,
            self.target_map,
            input_dim=3,
            n_iters=2,
            batch_size=5,
            hidden_dims=(5, 5),
            device="cpu",
            log_every=10,
        )
        self.assertEqual(len(result), 4)
        trained_model, history, _, _ = result
        self.assertIsInstance(trained_model, ICNN)
        self.assertEqual(len(history["oracle_mse"]), 2)
        self.assertNotIn("g_loss", history)


@unittest.skipUnless(Path(BENCHMARK_PATH).exists(), "Wasserstein2Benchmark is not available")
class TestOfficialOracleBenchmarkIntegration(unittest.TestCase):
    def test_d16_and_d32_oracle_pair_shapes(self):
        # Both dimensions are intentionally checked through the project's official loader.
        for dimension in (16, 32):
            benchmark, module_info = load_mix3to10_benchmark(
                dim=dimension, benchmark_path=BENCHMARK_PATH, device="cpu"
            )

            def mu_sampler(batch_size):
                return benchmark.input_sampler.sample(batch_size)

            def target_map(x):
                return benchmark.map_fwd(x, nograd=True)

            x, target = sample_oracle_batch(mu_sampler, target_map, batch_size=4, device="cpu")
            self.assertEqual(x.shape, (4, dimension))
            self.assertEqual(target.shape, (4, dimension))
            self.assertIn("Wasserstein2Benchmark", module_info["map_benchmark_file"])


if __name__ == "__main__":
    unittest.main()
