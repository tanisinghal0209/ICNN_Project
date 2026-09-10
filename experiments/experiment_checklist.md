# Experiment Output Checklist

This checklist records the current, reproducible experiment artifacts in this repository. All high-dimensional rows use the official Korotin `Mix3ToMix10` benchmark and preserve prior output directories.

| Study | Dimensions | Reproduction module | Saved result artifacts | Main conclusion |
|---|---|---|---|---|
| Baseline high-dimensional sweep | 2, 4, 8, 16, 32 | [run_high_d_sweep.py](run_high_d_sweep.py) | [summary.json](high_dimensional/summary.json), per-dimension [metrics](high_dimensional/D16/metrics.json), histories, and plots | Transport quality degrades and recorded minimax parameter-gradient spikes grow with dimension. |
| Capacity / expressivity ablation | 16 | [run_experiment_a_width.py](run_experiment_a_width.py) | [summary.json](ablation_expressivity_d16/summary.json), [width plots](ablation_expressivity_d16/) | Width 64--512 does not give systematic L2-UVP improvement. |
| Inner-iteration optimisation ablation | 16 | [run_experiment_b_optimization.py](run_experiment_b_optimization.py) | [summary.json](ablation_optimization_d16/summary.json), trajectory plots | More inner updates do not uniformly stabilise or recover accuracy. |
| Oracle paired-map regression | 16, 32 | [run_oracle_regression.py](run_oracle_regression.py) | [summary](oracle_regression/summary.md), [D16](oracle_regression/D16/), [D32](oracle_regression/D32/) | Removing the two-player game yields much smaller gradient peaks and better, but still imperfect, transport metrics. |
| Unequal player learning rates | 16 | [run_experiment_d_unequal_lr.py](run_experiment_d_unequal_lr.py) | [summary](stabilization_unequal_lr_d16/summary.md), [gradient trajectories](stabilization_unequal_lr_d16/f_grad_norm_comparison.png) | Slowing $g$ suppresses spikes; accuracy gain is small. |
| Unequal player learning rates | 32 | [run_experiment_d_unequal_lr.py](run_experiment_d_unequal_lr.py) with `--dimension 32` | [summary](stabilization_unequal_lr_d32/summary.md), [gradient trajectories](stabilization_unequal_lr_d32/f_grad_norm_comparison.png) | Gradient stability improves, but transport accuracy is non-monotonic. |

## Final evidence and deliverables

- [Consolidated scientific results](CONSOLIDATED_SCIENTIFIC_RESULTS.md): complete table, caveats, and next-decision framing.
- [Oracle tests](../tests/test_oracle_regression.py) and [unequal-LR tests](../tests/test_experiment_d_unequal_lr.py): validate new diagnostic behavior.
- [Markdown report](../REPORT.md), [telemetry results](../RESULTS.md), [LaTeX report](../Report.tex), and [canonical notebook](../notebooks/ICNN_Optimal_Transport_Colab_FINAL.ipynb): synchronized project documentation.

The current evidence supports minimax dynamics as a major source of severe gradient instability. It does not establish that minimax instability alone explains the remaining high-dimensional transport error.
