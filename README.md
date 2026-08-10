# ICNN Optimal Transport Solver

A PyTorch implementation and research suite for continuous optimal transport under quadratic costs ($W_2$), reproducing the minimax dual formulation proposed by Makkuva et al. (2020) and validating against the NeurIPS 2021 Korotin continuous Wasserstein-2 benchmark.

---

## 1. Project Directory Structure

```text
ICNN_Project/
│
├── configs/                     # YAML/JSON configurations for experiments
│   ├── baseline.yaml
│   ├── gaussian_2d.json
│   ├── mixture_2d.json
│   └── disconnected_2d.json
│
├── experiments/                 # Sweeps, reproductions, and checklists
│   ├── run_experiment_suite.py  # Run parameter sweeps (depth, width, lr, etc.)
│   ├── run_paper_reproduction.py# Gaussian-to-Gaussian, multimodal, disconnected support
│   ├── run_failure_analysis.py  # 5 targeted failure modes
│   ├── run_scalability_study.py # Measure runtime vs dimension and size
│   ├── run_korotin_benchmark.py # Integrate and evaluate on Mix3ToMix10 benchmark
│   ├── generate_report_table.py # Script to compile summary.csv
│   ├── experiment_checklist.md  # Map requirements to checkpoints/metrics
│   └── summary.csv              # Main sweeps output table
│
├── figures/                     # Generated charts and scalability curves
├── checkpoints/                 # Saved model weights
│
├── icnn.py                      # ICNN and StandardMLP network architectures
├── losses.py                    # Minimax dual loss implementation
├── train.py                     # Training loop and CLI entrypoint
├── data.py                      # Latent VAE samplers
├── vae.py                       # Convolutional VAE module for MNIST
│
├── ICNN_Optimal_Transport_Colab.ipynb # Self-contained Google Colab notebook
│
├── tests/                       # Unit tests suite
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

## 4. Running the Experiment Suite

### 1. Parameter Sweeps & Ablation Studies
Trains baseline and sweeps over hidden dimensions, layers, learning rates, activation functions, sample sizes, and non-convex MLP comparisons:
```bash
python experiments/run_experiment_suite.py
```
This updates the consolidated sweep records:
* JSON: `experiments/summary.json`
* CSV: `experiments/summary.csv` (Run `python experiments/generate_report_table.py` to compile).

### 2. Paper Reproduction Distributions
Trains the three core continuous target mappings from the paper for 2000 iterations to ensure full convergence:
```bash
python experiments/run_paper_reproduction.py
```
* **Gaussian → Gaussian**: Dynamically validates the learned distance against the closed-form analytical $W_2$ metric.
* **Multimodal Mixture**: Maps a single Gaussian to 8 components on a circle.
* **Disconnected Support**: Maps separated clusters without intersecting transport paths.

### 3. Failure-Mode Analysis
Intentionally violates theoretical and optimization assumptions to study degradation:
```bash
python experiments/run_failure_analysis.py
```
Runs five cases: tiny datasets ($N=50$), removing the convexity constraint (non-convex standard MLP), large learning rate ($\text{lr}=0.1$ causing divergence), high-dimensional mapping ($D=128$), and poor spatial overlap (shift of 15 std deviations).

### 4. Scalability Sweeps
Measures computation time scaling against dimension and dataset size:
```bash
python experiments/run_scalability_study.py
```
Generates scalability charts under `figures/`:
* `figures/scalability_dim_vs_time.png`
* `figures/scalability_size_vs_time.png`

### 5. Korotin Benchmark Integration
Links dynamically with the `Wasserstein2Benchmark` codebase (found in Downloads) to evaluate our PyTorch potentials against the Mix3ToMix10 mixture problems:
```bash
python experiments/run_korotin_benchmark.py
```
Evaluates:
* **L2-UVP (L2 Unexplained Variance Percentage)** (Goal: $<10\%$ in 2D)
* **Cosine Similarity** of displacement vectors (Goal: $>0.90$ in 2D)

---

## 5. Google Colab Notebook

For interactive cloud training, use the self-contained [ICNN_Optimal_Transport_Colab.ipynb](file:///Users/tanishasinghal/Downloads/ICNN_Project/ICNN_Optimal_Transport_Colab.ipynb) notebook. 
* Upload this notebook to your Google Drive.
* Open in Google Colab, choose GPU or CPU execution, and run the cells from top to bottom.
* It trains the baseline, mixture, disconnected support, and MLP failure configurations, displaying all loss curves, scatter plots, and vector field diagrams inline.
