"""
Input-Convex Neural Network (ICNN) and Standard MLP architectures.

Extracted 1:1 from the canonical notebook (Cell 06).
Source of truth: notebooks/ICNN_Optimal_Transport_Colab_FINAL.ipynb

DO NOT modify the mathematics here during the refactoring phase.
Any changes require explicit approval and a D=2 regression test.
"""
import torch
import torch.nn as nn


class StandardMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 128, 128), activation='softplus'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.act = {
            'softplus': nn.Softplus(),
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'elu': nn.ELU(),
        }[activation]

        dims = [input_dim] + self.hidden_dims + [1]
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

    def forward(self, x):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.act(h)
        return h.squeeze(-1)

    @torch.no_grad()
    def clip_weights(self):
        # StandardMLP has no convexity constraint — intentional no-op
        pass

    def grad(self, y):
        y = y.clone().requires_grad_(True)
        with torch.enable_grad():
            f_val = self.forward(y)
            (grad_y,) = torch.autograd.grad(f_val.sum(), y, create_graph=True)
        return grad_y


class ICNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 128, 128), activation='softplus'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.act = {
            'softplus': nn.Softplus(),
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'elu': nn.ELU(),
        }[activation]

        dims = [input_dim] + self.hidden_dims + [1]
        self.Wy = nn.ModuleList()
        self.Wz = nn.ModuleList()
        for i in range(len(dims) - 1):
            out_d = dims[i + 1]
            self.Wy.append(nn.Linear(input_dim, out_d, bias=True))
            self.Wz.append(None if i == 0 else nn.Linear(self.hidden_dims[i - 1], out_d, bias=False))

    def forward(self, y):
        z = self.act(self.Wy[0](y))
        for i in range(1, len(self.Wy)):
            pre_act = self.Wz[i](z) + self.Wy[i](y)
            z = pre_act if i == len(self.Wy) - 1 else self.act(pre_act)
        return z.squeeze(-1)

    @torch.no_grad()
    def clip_weights(self):
        for layer in self.Wz:
            if layer is not None:
                layer.weight.data.clamp_(min=0)

    def grad(self, y):
        y = y.clone().requires_grad_(True)
        with torch.enable_grad():
            f_val = self.forward(y)
            (grad_y,) = torch.autograd.grad(f_val.sum(), y, create_graph=True)
        return grad_y
