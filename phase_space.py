import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

def init_weights_rnn(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)  # zero-initialize bias

class PhaseSpaceRNN(nn.Module):
    """
    Sets up the variables for the phase space RNN. 
    
    Args:
        image_size (int): number of pixels in image/number of input/output units 
        hidden_size (int): number of hidden units to use 
        embed_size (int): number of dimensions to embed the input image to
        learning_rate (float): learning rate for the AdamW optimizer 
    """
    def __init__(self, image_size, hidden_size, embed_size, learning_rate):
        self.embed = nn.Embedding(image_size, embed_size)
        self.fc_ih = nn.Linear(embed_size, hidden_size)
        self.fc_ho = nn.Linear(hidden_size, image_size)

        #initialize weights
        self.apply(init_weights)

        #initialize AdamW Optimizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
    """
    Trains the neural network to store representations of the image attractors.
    
    Args:
        patterns (torch.Tensor): Batch of binary patterns (shape: [num_patterns, num_units])
    """
    def forward(self, patterns, iterations):
        for pattern in patterns:
            flattened = torch.flatten(pattern)
            embedding = self.embed(flattened)
            fc1 = self.fc_ih(embedding)
            fc2 = self.fc_ho(fc1) + patterns #'+ patterns' is a skip connection
        
    """
    Args:
        patterns (torch.Tensor): batch of ground truth images (shape: [batch_size, image_size])
        corrupted_patterns (torch.Tensor): masked version of ground truth images (shape: [batch_size, num_units])
        max_iter (int): Max number of recurrent updates
        threshold (float): acceptable SSE to stop at (default 0) 
    """
    def retrieve_pattern(self, patterns, corrupted_patterns, max_iters=100, threshold=0):
        flattened = torch.flatten(corrupted_patterns)
        fc2 = flattened
        
        for _ in range(max_iters):
            embedding = self.embed(fc2)
            fc1 = self.fc_ih(embedding)
            fc2 = self.fc_ho(fc1) + fc2 #'+ fc2' is a skip connection

            #run over and over again until SSE is under a threshold, then return
            if torch.sum((patterns - fc2)**2) < threshold:
                break 
        
        return fc2
        