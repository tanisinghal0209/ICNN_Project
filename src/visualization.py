"""
Visualization helpers for transport maps, vector fields, and training curves.

Extracted 1:1 from the canonical notebook (Cell 14).
Source of truth: notebooks/ICNN_Optimal_Transport_Colab_FINAL.ipynb
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def plot_reproduction_results(mu_sampler, nu_sampler, f, g,
                               title_suffix='', save_path=None, show=False):
    """
    Renders distribution scatters, learned transport maps, and displacement vector fields.
    Extracted from notebook Cell 14.

    Args:
        save_path: If provided, saves the figure here instead of calling plt.show()
        show: If True, calls plt.show() (only use in interactive environments)
    """
    n_plot = 300
    x_plot_tensor = mu_sampler(n_plot).detach()
    y_plot_tensor = nu_sampler(n_plot).detach()

    model_device = next(g.parameters()).device

    with torch.enable_grad():
        y_plot_tensor_g = y_plot_tensor.clone().to(model_device).requires_grad_(True)
        transport_tensor = g.grad(y_plot_tensor_g)
        transport = transport_tensor.detach().cpu().numpy()

    x_plot = x_plot_tensor.cpu().numpy()
    y_plot = y_plot_tensor.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # 1. Source vs Target Scatter
    axes[0].scatter(x_plot[:, 0], x_plot[:, 1], s=15, alpha=0.6,
                    label=r'Source $\mu$', color='C0')
    axes[0].scatter(y_plot[:, 0], y_plot[:, 1], s=15, alpha=0.6,
                    label=r'Target $\nu$', color='C1')
    axes[0].set_title(f'Distributions {title_suffix}')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # 2. Transport Mapping
    axes[1].scatter(y_plot[:, 0], y_plot[:, 1], s=15, alpha=0.5,
                    label=r'Target $y$', color='C1')
    axes[1].scatter(transport[:, 0], transport[:, 1], s=15, alpha=0.5,
                    label=r'Transport $T(y)$', color='C2')
    axes[1].set_title(f'Learned Map {title_suffix}')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)

    # 3. Displacement vector field
    step = max(1, n_plot // 80)
    axes[2].quiver(
        y_plot[::step, 0], y_plot[::step, 1],
        transport[::step, 0] - y_plot[::step, 0],
        transport[::step, 1] - y_plot[::step, 1],
        angles='xy', scale_units='xy', scale=1, alpha=0.7, color='purple'
    )
    axes[2].scatter(y_plot[:, 0], y_plot[:, 1], s=8, alpha=0.3, color='gray')
    axes[2].set_title(f'Displacement Field {title_suffix}')
    axes[2].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    return fig
