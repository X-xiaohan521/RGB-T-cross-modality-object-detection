import torch
from torch import nn

class SM3Det(nn.Module):
    def __init__(self):
        super(SM3Det, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)