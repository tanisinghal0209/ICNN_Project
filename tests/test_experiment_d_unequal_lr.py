"""Tests for the D=16 unequal-player-learning-rate experiment module."""
import unittest

import torch

from experiments.run_experiment_d_unequal_lr import (
    CONFIGS,
    HIDDEN_DIMS,
    ICNN,
    make_optimizers,
    train_icnn_ot_unequal_lr,
)


class TestUnequalLearningRateConfiguration(unittest.TestCase):
    def test_controlled_configs_keep_f_rate_fixed_and_reduce_only_g_rate(self):
        self.assertEqual(
            CONFIGS,
            (
                ("Baseline", 1e-3, 1e-3),
                ("G-slower-2x", 1e-3, 5e-4),
                ("G-slower-4x", 1e-3, 2.5e-4),
            ),
        )
        self.assertEqual(HIDDEN_DIMS, (128, 128, 128))

    def test_optimizers_use_distinct_requested_rates(self):
        f = ICNN(2, hidden_dims=(4, 4))
        g = ICNN(2, hidden_dims=(4, 4))
        opt_f, opt_g = make_optimizers(f, g, f_lr=1e-3, g_lr=2.5e-4)
        self.assertEqual(opt_f.param_groups[0]["lr"], 1e-3)
        self.assertEqual(opt_g.param_groups[0]["lr"], 2.5e-4)

    def test_short_run_preserves_two_player_telemetry(self):
        torch.manual_seed(0)

        def mu_sampler(batch_size):
            return torch.randn(batch_size, 2)

        def nu_sampler(batch_size):
            return torch.randn(batch_size, 2)

        f, g, history, final_f_loss, training_time = train_icnn_ot_unequal_lr(
            mu_sampler=mu_sampler,
            nu_sampler=nu_sampler,
            input_dim=2,
            f_lr=1e-3,
            g_lr=5e-4,
            n_iters=3,
            batch_size=8,
            hidden_dims=(4, 4),
            inner_iters=1,
            device="cpu",
            log_every=10,
        )
        self.assertIsInstance(f, ICNN)
        self.assertIsInstance(g, ICNN)
        self.assertEqual(len(history["f_loss"]), 3)
        self.assertEqual(len(history["g_loss"]), 3)
        self.assertEqual(len(history["f_grad_norm"]), 3)
        self.assertEqual(len(history["g_grad_norm"]), 3)
        self.assertEqual(final_f_loss, history["f_loss"][-1])
        self.assertGreaterEqual(training_time, 0.0)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in history["f_loss"]))


if __name__ == "__main__":
    unittest.main()
