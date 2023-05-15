import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import init


class L2Norm(nn.Module):
    def __init__(self, n_channels: int, scale=20):
        super().__init__()
        self.gamma = scale
        self.eps = 1e-10
        self.n_channels = n_channels
        self.weight = nn.Parameter(Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x: Tensor):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        y = x * self.weight[None, ..., None, None]
        return y
