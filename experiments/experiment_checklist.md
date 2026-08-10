# Experiment Output Checklist

This checklist maps all project requirements to their corresponding folder structures, configuration files, checkpoint files, metrics, and visualization outputs in the repository.

---

## 1. Parameter sweeps & Ablations

| Sweep Family | Experiment Name | Config File | Metrics JSON | Checkpoint Folder | Final Loss Plots |
|---|---|---|---|---|---|
| **Baseline** | `baseline` | [config.yaml](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/baseline/config.yaml) | [metrics.json](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/baseline/metrics.json) | [checkpoints/](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/baseline/run/checkpoints/) | [loss.png](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/baseline/loss.png) |
| **Layer Depth** | `layers_2` | [config.yaml](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/layers_2/config.yaml) | [metrics.json](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/layers_2/metrics.json) | [checkpoints/](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/layers_2/run/checkpoints/) | [loss.png](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/layers_2/loss.png) |
| | `layers_3` | [config.yaml](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/layers_3/config.yaml) | [metrics.json](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/layers_3/metrics.json) | [checkpoints/](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/layers_3/run/checkpoints/) | [loss.png](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/layers_3/loss.png) |
| | `layers_4` | [config.yaml](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/layers_4/config.yaml) | [metrics.json](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/layers_4/metrics.json) | [checkpoints/](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/layers_4/run/checkpoints/) | [loss.png](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/layers_4/loss.png) |
| | `layers_5` | [config.yaml](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/layers_5/config.yaml) | [metrics.json](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/layers_5/metrics.json) | [checkpoints/](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/layers_5/run/checkpoints/) | [loss.png](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/layers_5/loss.png) |
| **Hidden Width** | `width_64` | [config.yaml](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/width_64/config.yaml) | [metrics.json](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/width_64/metrics.json) | [checkpoints/](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/width_64/run/checkpoints/) | [loss.png](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/width_64/loss.png) |
| | `width_128` | [config.yaml](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/width_128/config.yaml) | [metrics.json](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/width_128/metrics.json) | [checkpoints/](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/width_128/run/checkpoints/) | [loss.png](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/width_128/loss.png) |
| | `width_256` | [config.yaml](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/width_256/config.yaml) | [metrics.json](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/width_256/metrics.json) | [checkpoints/](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/width_256/run/checkpoints/) | [loss.png](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/width_256/loss.png) |
| | `width_512` | [config.yaml](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/width_512/config.yaml) | [metrics.json](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/width_512/metrics.json) | [checkpoints/](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/width_512/run/checkpoints/) | [loss.png](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/width_512/loss.png) |
| **Learning Rate**| `lr_0.01` | [config.yaml](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/lr_0.01/config.yaml) | [metrics.json](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/lr_0.01/metrics.json) | [checkpoints/](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/lr_0.01/run/checkpoints/) | [loss.png](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/lr_0.01/loss.png) |
| | `lr_0.001` | [config.yaml](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/lr_0.001/config.yaml) | [metrics.json](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/lr_0.001/metrics.json) | [checkpoints/](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/lr_0.001/run/checkpoints/) | [loss.png](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/lr_0.001/loss.png) |
| | `lr_0.0001` | [config.yaml](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/lr_0.0001/config.yaml) | [metrics.json](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/lr_0.0001/metrics.json) | [checkpoints/](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/lr_0.0001/run/checkpoints/) | [loss.png](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/lr_0.0001/loss.png) |
| **Activation** | `activation_relu` | [config.yaml](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/activation_relu/config.yaml) | [metrics.json](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/activation_relu/metrics.json) | [checkpoints/](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/activation_relu/run/checkpoints/) | [loss.png](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/activation_relu/loss.png) |
| | `activation_leaky_relu`| [config.yaml](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/activation_leaky_relu/config.yaml)| [metrics.json](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/activation_leaky_relu/metrics.json)| [checkpoints/](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/activation_leaky_relu/run/checkpoints/)| [loss.png](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/activation_leaky_relu/loss.png)|
| | `activation_elu` | [config.yaml](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/activation_elu/config.yaml) | [metrics.json](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/activation_elu/metrics.json) | [checkpoints/](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/activation_elu/run/checkpoints/) | [loss.png](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/activation_elu/loss.png) |
| **MLP Non-convex**| `mlp_baseline` | [config.yaml](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/mlp_baseline/config.yaml) | [metrics.json](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/mlp_baseline/metrics.json) | [checkpoints/](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/mlp_baseline/run/checkpoints/) | [loss.png](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/mlp_baseline/loss.png) |
| **Sample Size** | `samples_500` | [config.yaml](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/samples_500/config.yaml) | [metrics.json](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/samples_500/metrics.json) | [checkpoints/](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/samples_500/run/checkpoints/) | [loss.png](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/samples_500/loss.png) |
| | `samples_1000` | [config.yaml](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/samples_1000/config.yaml) | [metrics.json](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/samples_1000/metrics.json) | [checkpoints/](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/samples_1000/run/checkpoints/) | [loss.png](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/samples_1000/loss.png) |
| | `samples_5000` | [config.yaml](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/samples_5000/config.yaml) | [metrics.json](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/samples_5000/metrics.json) | [checkpoints/](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/samples_5000/run/checkpoints/) | [loss.png](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/samples_5000/loss.png) |
| | `samples_10000`| [config.yaml](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/samples_10000/config.yaml)| [metrics.json](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/samples_10000/metrics.json)| [checkpoints/](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/samples_10000/run/checkpoints/)| [loss.png](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/samples_10000/loss.png)|
| **Input Dim** | `dim_2` | [config.yaml](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/dim_2/config.yaml) | [metrics.json](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/dim_2/metrics.json) | [checkpoints/](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/dim_2/run/checkpoints/) | [loss.png](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/dim_2/loss.png) |
| | `dim_8` | [config.yaml](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/dim_8/config.yaml) | [metrics.json](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/dim_8/metrics.json) | [checkpoints/](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/dim_8/run/checkpoints/) | [loss.png](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/dim_8/loss.png) |
| | `dim_16` | [config.yaml](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/dim_16/config.yaml) | [metrics.json](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/dim_16/metrics.json) | [checkpoints/](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/dim_16/run/checkpoints/) | [loss.png](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/dim_16/loss.png) |
| | `dim_32` | [config.yaml](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/dim_32/config.yaml) | [metrics.json](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/dim_32/metrics.json) | [checkpoints/](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/dim_32/run/checkpoints/) | [loss.png](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/dim_32/loss.png) |
| **Mixture 2D** | `mixture` | [config.yaml](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/mixture/config.yaml) | [metrics.json](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/mixture/metrics.json) | [checkpoints/](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/mixture/run/checkpoints/) | [loss.png](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/mixture/loss.png) |

---

## 2. Paper Reproduction Distributions (Phase 2)

* **Gaussian → Gaussian**:
  * **Folder**: [gaussian_to_gaussian/](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/gaussian_to_gaussian/)
  * **Config**: [config.yaml](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/gaussian_to_gaussian/config.yaml)
  * **Checkpoints**: [f_final.pt & g_final.pt](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/gaussian_to_gaussian/run/checkpoints/)
  * **Metrics**: [metrics.json](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/gaussian_to_gaussian/metrics.json) *(contains analytical_w2, learned_w2, and w2_error)*
  * **Plots**:
    * [source_target_scatter.png](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/gaussian_to_gaussian/source_target_scatter.png)
    * [transport_map.png](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/gaussian_to_gaussian/transport_map.png)
    * [training_loss.png](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/gaussian_to_gaussian/training_loss.png)
    * [vector_field.png](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/gaussian_to_gaussian/vector_field.png)

* **Multimodal Mixture**:
  * **Folder**: [multimodal/](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/multimodal/)
  * **Config**: [config.yaml](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/multimodal/config.yaml)
  * **Checkpoints**: [f_final.pt & g_final.pt](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/multimodal/run/checkpoints/)
  * **Metrics**: [metrics.json](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/multimodal/metrics.json)
  * **Plots**:
    * [source_target_scatter.png](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/multimodal/source_target_scatter.png)
    * [transport_map.png](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/multimodal/transport_map.png)
    * [training_loss.png](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/multimodal/training_loss.png)
    * [vector_field.png](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/multimodal/vector_field.png)

* **Disconnected Support**:
  * **Folder**: [disconnected_support/](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/disconnected_support/)
  * **Config**: [config.yaml](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/disconnected_support/config.yaml)
  * **Checkpoints**: [f_final.pt & g_final.pt](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/disconnected_support/run/checkpoints/)
  * **Metrics**: [metrics.json](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/disconnected_support/metrics.json)
  * **Plots**:
    * [source_target_scatter.png](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/disconnected_support/source_target_scatter.png)
    * [transport_map.png](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/disconnected_support/transport_map.png)
    * [training_loss.png](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/disconnected_support/training_loss.png)
    * [vector_field.png](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/disconnected_support/vector_field.png)

---

## 3. Failure-Mode Analysis (Phase 4)

* **Very Small Dataset ($N = 50$)**:
  * **Folder**: [failure_small_data/](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/failure_small_data/)
  * **Config**: [config.yaml](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/failure_small_data/config.yaml)
  * **Metrics**: [metrics.json](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/failure_small_data/metrics.json)
  * **Loss Plot**: [training_loss.png](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/failure_small_data/run/figures/training_loss.png) *(reveals wild training instability/overfitting)*

* **Remove Convexity Constraint (MLP)**:
  * **Folder**: [failure_no_convexity/](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/failure_no_convexity/)
  * **Config**: [config.yaml](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/failure_no_convexity/config.yaml)
  * **Metrics**: [metrics.json](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/failure_no_convexity/metrics.json)
  * **Transport Map Plot**: [transport_map.png](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/failure_no_convexity/transport_map.png) *(illustrates invalid non-monotone transport mappings)*

* **Large Learning Rate ($\text{lr} = 0.1$)**:
  * **Folder**: [failure_large_lr/](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/failure_large_lr/)
  * **Config**: [config.yaml](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/failure_large_lr/config.yaml)
  * **Metrics**: [metrics.json](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/failure_large_lr/metrics.json)
  * **Loss Plot**: [training_loss.png](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/failure_large_lr/run/figures/training_loss.png) *(displays divergence with loss values starting at `721435.06`)*

* **Very High Dimension ($D = 128$)**:
  * **Folder**: [failure_high_dim/](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/failure_high_dim/)
  * **Config**: [config.yaml](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/failure_high_dim/config.yaml)
  * **Metrics**: [metrics.json](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/failure_high_dim/metrics.json)
  * **Loss Plot**: [training_loss.png](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/failure_high_dim/run/figures/training_loss.png) *(monitors training slower convergence)*

* **Poor Overlap (Shift = $15.0$)**:
  * **Folder**: [failure_poor_overlap/](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/failure_poor_overlap/)
  * **Config**: [config.yaml](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/failure_poor_overlap/config.yaml)
  * **Metrics**: [metrics.json](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/failure_poor_overlap/metrics.json)
  * **Plots**: [transport_map.png](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/failure_poor_overlap/transport_map.png) *(demonstrates struggle to bridge the spatial gap)*

---

## 4. Scalability Studies (Phase 5)

* **Dimension Scalability Sweep**:
  * **JSON Record**: [dim_scalability.json](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/dim_scalability.json)
  * **Scalability Plot**: [scalability_dim_vs_time.png](file:///Users/tanishasinghal/Downloads/ICNN_Project/figures/scalability_dim_vs_time.png)

* **Dataset Size Scalability Sweep**:
  * **JSON Record**: [size_scalability.json](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/size_scalability.json)
  * **Scalability Plot**: [scalability_size_vs_time.png](file:///Users/tanishasinghal/Downloads/ICNN_Project/figures/scalability_size_vs_time.png)

---

## 5. Consolidated Summaries

* **Main Sweep JSON**: [summary.json](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/summary.json)
* **Main Sweep CSV**: [summary.csv](file:///Users/tanishasinghal/Downloads/ICNN_Project/experiments/summary.csv)
