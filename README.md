# ICNN Optimal Transport Solver

A PyTorch implementation and research suite for continuous optimal transport under quadratic costs ($W_2$), reproducing the minimax dual formulation proposed by Makkuva et al. (2020) and validating against the NeurIPS 2021 Korotin continuous Wasserstein-2 benchmark.

---

## 1. Project Directory Structure

```text
ICNN_Project/
│
├── src/                         # ICNN, minimax loss, solver, benchmark, metrics
├── experiments/                 # Saved outputs and reproducible experiment modules
│   ├── high_dimensional/        # D=2,4,8,16,32 baseline sweep
│   ├── ablation_expressivity_d16/
│   ├── oracle_regression/       # Paired-map oracle diagnostic
│   ├── stabilization_unequal_lr_d16/
│   ├── stabilization_unequal_lr_d32/
│   └── CONSOLIDATED_SCIENTIFIC_RESULTS.md
├── notebooks/
│   └── ICNN_Optimal_Transport_Colab_FINAL.ipynb
├── tests/                       # Unit tests suite
├── Report.tex                   # LaTeX report
├── REPORT.md                    # Markdown research report
├── RESULTS.md                   # Telemetry-focused results note
└── README.md                    # This document
```

---

## 2. Installation & Setup

1. **Clone the repository** (or navigate to the workspace).
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Required packages*: `torch`, `numpy`, `scipy`, `matplotlib`, `pyyaml`, `pytest`.

---

## 3. Running the Tests

To verify that the model forward passes, weight clipping, and training steps are functioning correctly, run the unit test suite:
```bash
python -m unittest discover -s tests
```

---

## 4. Reproducing the High-Dimensional Diagnostic Suite

The official `Wasserstein2Benchmark` clone must be available locally. Set its path once for the current shell:

```bash
export WASSERSTEIN_BENCHMARK_PATH=/path/to/Wasserstein2Benchmark
```

The checked-in experiment modules and outputs are:

```bash
# Check the official benchmark integration.
python experiments/verify_benchmarks.py

# Baseline Mix3ToMix10 sweep: D=4,8,16,32.
python experiments/run_high_d_sweep.py

# Controlled D=16 capacity and inner-iteration ablations.
python experiments/run_experiment_a_width.py
python experiments/run_experiment_b_optimization.py

# Oracle paired-map control: D=16 and D=32.
python experiments/run_oracle_regression.py

# Unequal player learning-rate control, one dimension at a time.
python experiments/run_experiment_d_unequal_lr.py --dimension 16
python experiments/run_experiment_d_unequal_lr.py --dimension 32
```

Each new diagnostic writes to its own output directory and preserves the existing baseline results. The official metrics are L2-UVP, L2 error, and cosine similarity of the learned forward map $\hat T(x)=\nabla f(x)$ against `benchmark.map_fwd(x)`.

---

## 5. Completed High-Dimensional Diagnostics

The current research results are consolidated in [the scientific evidence sheet](experiments/CONSOLIDATED_SCIENTIFIC_RESULTS.md). The controlled findings are:

| Diagnostic | Main result |
|---|---|
| Baseline sweep | L2-UVP rises from 5.32% at D=2 to 50.80% at D=32; recorded peak `f` parameter-gradient norm rises from 10.80 to 12,999.47. |
| D=16 width ablation | Increasing width from 64 to 512 does not systematically improve L2-UVP (36.49%--41.01%). |
| Oracle map regression | Removing the two-player game yields finite, small gradient peaks (1.08 at D=16; 1.39 at D=32) and better, though still imperfect, transport metrics. |
| Unequal player learning rates | Slowing `g` strongly suppresses gradient spikes at D=16/D=32, but accuracy changes are small and non-monotonic. |

These results provide strong empirical evidence that minimax dynamics are a major contributor to the observed gradient instability. They do not establish that the minimax game is the sole source of high-dimensional transport error: the remaining oracle-regression error leaves the clipped ICNN parameterization, conditioning, and finite training budget as open factors.

Reproduce the completed diagnostic modules with:

```bash
python experiments/run_oracle_regression.py
python experiments/run_experiment_d_unequal_lr.py --dimension 16
python experiments/run_experiment_d_unequal_lr.py --dimension 32
```

The diagnostic modules use the official benchmark path configured through `WASSERSTEIN_BENCHMARK_PATH`; they do not overwrite baseline results.

---

## 6. Google Colab Notebook

For interactive cloud training, use [notebooks/ICNN_Optimal_Transport_Colab_FINAL.ipynb](notebooks/ICNN_Optimal_Transport_Colab_FINAL.ipynb).
* Upload this notebook to your Google Drive.
* Open in Google Colab, choose GPU or CPU execution, and run the cells from top to bottom.
* It contains the validated benchmark workflow and a results-only section documenting the completed high-dimensional diagnostics. Local saved results remain the authoritative numerical record.
