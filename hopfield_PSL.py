import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)  # zero-initialize bias

class HopfieldPSL(nn.Module):
    def __init__(self):
        super(HopfieldPSL, self).__init__()
        
        self.skip = nn.Linear(1024, 1024)
        self.embed = nn.Linear(1024, 2048)
        self.down = nn.Linear(2048, 100)
        self.up   = nn.Linear(100, 1024)
        self.activation = nn.Tanh()

        self.apply(init_weights)

    def forward(self, x):
        states = []

        for _ in range(5):
            x_skip = self.skip(x)
            embedded = self.embed(x)
            x_hidden = self.activation(self.down(embedded))
            x_hidden = self.activation(self.up(x_hidden))
            x = x_skip + x_hidden  
            states.append(x)

        return states
    
    def retrieve_pattern(self, x):
        for _ in range(5):
            x_skip = self.skip(x)
            embedded = self.embed(x)
            x_hidden = self.activation(self.down(embedded))
            x_hidden = self.activation(self.up(x_hidden))
            x = x_skip + x_hidden  

        return x 
