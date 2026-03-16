import torch
from torch import nn

class MyOBB(nn.Module):
    def __init__(self, model: torch.nn.Module):
        super(MyOBB, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)