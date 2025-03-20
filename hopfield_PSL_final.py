import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class HopfieldPSL(nn.Module):
    def __init__(self, N=1):
        super(HopfieldPSL, self).__init__()
        
        self.skip = nn.Linear(1024, 1024)
        self.down = nn.Linear(1024, 100)
        self.up   = nn.Linear(100, 1024)
        self.activation = nn.Tanh()
        self.N = N
    
    def flow(self, x):
        x_skip = self.skip(x)
        x_hidden = self.up(self.down(x))
        x = self.activation(x_skip + x_hidden)
        return x
    
    def forward(self, x):
        for _ in range(self.N):
            x = self.flow(x)
        return x

class HopfieldPSL_Gray(nn.Module):
    def __init__(self, N=1):
        super(HopfieldPSL_Gray, self).__init__()
        
        self.skip = nn.Linear(8192, 8192)
        self.down = nn.Linear(8192, 100)
        self.up   = nn.Linear(100, 8192)
        self.activation = nn.Tanh()
        self.N = N
    
    def flow(self, x):
        x_skip = self.skip(x)
        x_hidden = self.up(self.down(x))
        x = self.activation(x_skip + x_hidden)
        return x
    
    def forward(self, x):
        for _ in range(self.N):
            x = self.flow(x)
        return x