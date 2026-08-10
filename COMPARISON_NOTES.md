# Conceptual comparison with OT-ICNN

## Architecture
- The current ICNN implementation uses a compact, PyTorch-based input-convex network with a simple hidden-layer stack and activation mapping.
- The official OT-ICNN repository uses a more elaborate TensorFlow-based implementation with experiment-specific network construction and plotting utilities.
- Both share the same core idea of using input-convex networks for the dual potentials, but the official repository is more specialized to its paper experiments.

## Loss
- The current project uses the same minimax dual objective formulation based on the gradient of the second potential.
- The official implementation uses a TensorFlow expression of the same objective, with additional regularization terms in some experiment variants.
- The conceptual loss family is aligned; the main difference is the engineering scaffolding around it.

## Optimizer
- The current implementation uses Adam with the same $(\beta_1, \beta_2)$ values as the official setup.
- The official implementation also uses Adam and includes experiment-specific regularization choices.
- The current code is conceptually consistent with the paper’s optimizer choice while remaining lighter-weight.

## Training loop
- The current implementation uses a simple alternating-update loop over the two networks.
- The official implementation uses a more verbose training loop with experiment-specific logging, plotting, and figure creation.
- The current loop is functionally aligned with the paper’s minimax training logic, but it is less feature-rich.

## Interface/output needs for later Korotin benchmark integration
- A trained transport map in the form of a callable function or differentiable module that maps target samples to transported samples.
- A deterministic save/load interface for model weights and configuration.
- A standardized output directory containing checkpoints, loss history, and config metadata.
- A simple export path for sample transport outputs to support later benchmark evaluation.
