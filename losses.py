"""Minimax dual objective, Makkuva, Taghvaei, Oh & Lee (2020), Sec. 3."""


def minimax_loss(f, g, x_mu, y_nu):
    grad_g_y = g.grad(y_nu)
    term_mu = f(x_mu).mean()
    term_nu = (y_nu * grad_g_y).sum(dim=1).mean() - f(grad_g_y).mean()
    f_loss = term_mu + term_nu
    g_loss = -term_nu
    return f_loss, g_loss
