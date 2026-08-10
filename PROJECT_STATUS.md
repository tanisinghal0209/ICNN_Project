# Project Status

## Completed
- Inspected the core ICNN, loss, training, data, and VAE modules for internal consistency.
- Created the required directory structure for checkpoints, logs, figures, experiments, configs, and results.
- Added a lightweight config system with a default config and YAML-based entry point.
- Enabled the training pipeline to save checkpoints, loss history, config metadata, logs, and plots.
- Added a reusable CLI entry point so experiments can be launched with `python train.py --config configs/baseline.yaml`.

## Remaining
- Add experiment-specific config files for the paper ablation families.
- Add reusable experiment runner scripts for baseline, depth, width, learning-rate, activation, convexity, and toy-dataset sweeps.
- Prepare data samplers for the different toy datasets and connect them to the experiment runner.

## Known issues
- The current training loop uses simple Gaussian samplers by default; dataset-specific samplers still need to be added for the full paper setup.
- The ICNN implementation uses a simple activation mapping and weight clipping, which is adequate for the current experiments but should be reviewed if more complex settings are required.
- The CLI currently uses a minimal sampler; it is ready for extension rather than redesign.

## Next experiments
- Baseline run
- Layer-depth ablation
- Hidden-width ablation
- Learning-rate sweep
- Activation-function sweep
- Convexity-coefficient sweep
- Toy-dataset comparisons
