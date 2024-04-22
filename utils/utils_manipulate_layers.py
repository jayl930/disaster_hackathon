import torch
import torch.nn as nn


class SwapDim(nn.Module):
    def __init__(self, dim0: int, dim1: int):
        super(SwapDim, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: torch.Tensor):
        return x.transpose(self.dim0, self.dim1)
