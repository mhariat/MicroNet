import torch


class EmptyLayer(torch.nn.Module):
    r"""Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self, x):
        return x