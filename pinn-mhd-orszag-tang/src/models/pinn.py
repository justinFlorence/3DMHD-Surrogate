import torch
import torch.nn as nn
import torch.nn.functional as F

class PINN(nn.Module):
    def __init__(self, input_dim=2, output_dim=8, hidden_dim=128, num_layers=6, activation='tanh'):
        super(PINN, self).__init__()
        self.activation = self._get_activation(activation)
        layers = [nn.Linear(input_dim, hidden_dim)]

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.ModuleList(layers)

    def _get_activation(self, name):
        return {
            'tanh': torch.tanh,
            'relu': F.relu,
            'sin': torch.sin,
            'silu': F.silu
        }.get(name, torch.tanh)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x

