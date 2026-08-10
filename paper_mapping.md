# Paper to Code Traceability Map

## Model architecture
- Equation (5) / Input convex neural network
  - `ICNN_Project/icnn.py`, class `ICNN`
  - `OT-ICNN/High_dim_experiments/optimal_transport_modules/icnn_modules.py`, `ICNN` and `Simple_Feedforward_*` classes

## Loss formulation
- Section 3 / Minimax dual objective
  - `ICNN_Project/losses.py`, function `minimax_loss`
  - `OT-ICNN/High_dim_experiments/Gaussian_to_Gaussian.py`, function `train` (loss terms `g_OT_loss`, `remaining_f_loss`, `w_2_loss`)

## Convexity enforcement
- Positive weight constraints on hidden layers
  - `ICNN_Project/icnn.py`, method `clip_weights`
  - `OT-ICNN/High_dim_experiments/optimal_transport_modules/icnn_modules.py`, class `ConvexLinear` and model weight clipping via `torch.relu`

## Gradient map / Brenier map
- `\nabla g(y)` computation
  - `ICNN_Project/icnn.py`, method `grad`
  - `ICNN_Project/losses.py`, function `minimax_loss` uses `g.grad(y_nu)`
  - `OT-ICNN/High_dim_experiments/Gaussian_to_Gaussian.py`, function `compute_optimal_transport_map`

## Training loop and optimization
- Alternating minimax training
  - `ICNN_Project/train.py`, function `train_icnn_ot`
  - `OT-ICNN/High_dim_experiments/Gaussian_to_Gaussian.py`, the `train(epoch)` function and outer epoch loop

## Checkpointing and experiment logging
- Automatic experiment outputs and metrics
  - `ICNN_Project/train.py`, log directory creation, `config.json`, `log.csv`, `metrics.json`, checkpoint save / figure save

## Evaluation / visualization
- Experiment result files
  - `ICNN_Project/train.py`, saved figures and metrics
  - `OT-ICNN/High_dim_experiments/Gaussian_to_Gaussian.py`, plots saved to `results_save_path`

## Specific paper figure correspondence
- Gaussian → Gaussian baseline
  - `ICNN_Project/experiments/baseline/` (future folder)
- Transport map visualization
  - `ICNN_Project/experiments/*/transport_map.png` (future output)
- Vector field visualization
  - `ICNN_Project/experiments/*/vector_field.png` (future output)
