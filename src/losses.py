"""
Minimax dual objective loss function.

Extracted 1:1 from the canonical notebook (Cell 08).
Source of truth: notebooks/ICNN_Optimal_Transport_Colab_FINAL.ipynb

Formula (Makkuva, Taghvaei, Oh & Lee, 2020, Sec. 3):
    f_loss = E_mu[f(x)] + E_nu[<y, grad_g(y)> - f(grad_g(y))]
    g_loss = -E_nu[<y, grad_g(y)> - f(grad_g(y))]
"""


def minimax_loss(f, g, x_mu, y_nu):
    grad_g_y = g.grad(y_nu)
    term_mu = f(x_mu).mean()
    term_nu = (y_nu * grad_g_y).sum(dim=1).mean() - f(grad_g_y).mean()
    f_loss = term_mu + term_nu
    g_loss = -term_nu
    return f_loss, g_loss
