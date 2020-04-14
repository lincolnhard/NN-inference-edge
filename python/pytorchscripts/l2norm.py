import torch
from torch import nn


class L2Norm(nn.Module):

    def __init__(self, in_channels, gamma=1.0, eps=1e-10):
        super(L2Norm, self).__init__()
        self.in_channels = in_channels
        self.gamma = gamma
        self.eps = eps
        self.weights = nn.Parameter(torch.Tensor(1, self.in_channels, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weights, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()
        out = self.weights * torch.div(x, norm + self.eps)
        return out
