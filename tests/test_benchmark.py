"""
Unit tests for src/benchmark.py

Tests: benchmark loading from the official repository, sampler shapes,
       module source verification, and path configuration.
"""
import os
import unittest
from pathlib import Path
from src.benchmark import load_mix3to10_benchmark, get_benchmark_path, get_benchmark_samplers


BENCH_PATH = os.environ.get(
    'WASSERSTEIN_BENCHMARK_PATH',
    '/Users/tanishasinghal/Downloads/Wasserstein2Benchmark'
)


def bench_available():
    return Path(BENCH_PATH).exists()


class TestBenchmarkPathConfig(unittest.TestCase):

    def test_explicit_path_arg(self):
        if not bench_available():
            self.skipTest("Wasserstein2Benchmark not available")
        path = get_benchmark_path(BENCH_PATH)
        self.assertTrue(path.exists())

    def test_env_var_path(self):
        if not bench_available():
            self.skipTest("Wasserstein2Benchmark not available")
        os.environ['WASSERSTEIN_BENCHMARK_PATH'] = BENCH_PATH
        path = get_benchmark_path()
        self.assertTrue(path.exists())

    def test_missing_path_raises(self):
        orig = os.environ.pop('WASSERSTEIN_BENCHMARK_PATH', None)
        try:
            with self.assertRaises(EnvironmentError):
                get_benchmark_path(benchmark_path=None)
        finally:
            if orig:
                os.environ['WASSERSTEIN_BENCHMARK_PATH'] = orig


class TestBenchmarkLoading(unittest.TestCase):

    def test_d2_benchmark_loads(self):
        if not bench_available():
            self.skipTest("Wasserstein2Benchmark not available")
        benchmark, module_info = load_mix3to10_benchmark(dim=2, benchmark_path=BENCH_PATH)
        self.assertEqual(benchmark.dim, 2)
        self.assertIsNotNone(benchmark.input_sampler)
        self.assertIsNotNone(benchmark.output_sampler)
        self.assertIsNotNone(benchmark.potential)

    def test_module_comes_from_official_repo(self):
        """Verify that benchmark module file is inside the official repository."""
        if not bench_available():
            self.skipTest("Wasserstein2Benchmark not available")
        _, module_info = load_mix3to10_benchmark(dim=2, benchmark_path=BENCH_PATH)
        self.assertIn('Wasserstein2Benchmark', module_info['map_benchmark_file'])
        self.assertIn('map_benchmark', module_info['map_benchmark_file'])
        print(f"\n  Benchmark module source: {module_info['map_benchmark_file']}")
        print(f"  Repository root: {module_info['repository_root']}")
        print(f"  Checkpoints dir: {module_info['checkpoints_dir']}")

    def test_sampler_shapes(self):
        if not bench_available():
            self.skipTest("Wasserstein2Benchmark not available")
        benchmark, _ = load_mix3to10_benchmark(dim=2, benchmark_path=BENCH_PATH)
        mu_sampler, nu_sampler = get_benchmark_samplers(benchmark, device='cpu')
        x = mu_sampler(32)
        y = nu_sampler(32)
        self.assertEqual(x.shape, (32, 2))
        self.assertEqual(y.shape, (32, 2))

    def test_cwd_not_changed(self):
        """Benchmark loading must not permanently change the working directory."""
        if not bench_available():
            self.skipTest("Wasserstein2Benchmark not available")
        import os
        orig_cwd = os.getcwd()
        load_mix3to10_benchmark(dim=2, benchmark_path=BENCH_PATH)
        self.assertEqual(os.getcwd(), orig_cwd)


if __name__ == '__main__':
    unittest.main()
