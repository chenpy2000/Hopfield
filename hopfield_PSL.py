import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class HopfieldPSL(nn.Module):
    def __init__(self):
        super(HopfieldPSL, self).__init__()
        self.skip = nn.Linear(1024, 1024)
        self.down = nn.Linear(1024, 100)
        self.up   = nn.Linear(100, 1024)
        self.activation = nn.Tanh()

    def forward(self, x):
        x_skip = self.skip(x)
        x_hidden = self.activation(self.down(x))
        x_hidden = self.activation(self.up(x_hidden))
        return x_skip + x_hidden