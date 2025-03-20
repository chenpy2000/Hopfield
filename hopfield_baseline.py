import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os


class HopfieldRNN(nn.Module):
    def __init__(self, num_units):

        super(HopfieldRNN, self).__init__()
        self.num_units = num_units
        self.weights = nn.Parameter(torch.zeros(num_units, num_units))

    def store_patterns(self, patterns):
        """
            Args: patterns (torch.Tensor): Batch of binary patterns (shape: [num_patterns, num_units])
        """
        
        # Hebbian learning rule (outer product sum)
        W = torch.zeros(self.num_units, self.num_units, dtype=torch.float32)
        for pattern in patterns:
            W += torch.outer(pattern, pattern)  # Outer product
        
        # no self connections
        W.fill_diagonal_(0)

        # normalize weights
        self.weights.data = W / patterns.shape[0]

    def forward(self, corrupted_patterns, max_iter=100):
        """
        Args:
            corrupted_patterns (torch.Tensor): Initial state (shape: [batch_size, num_units])
            max_iter (int): Max number of recurrent updates
        """

        patterns = corrupted_patterns.clone()

        for _ in range(max_iter):

            # Hopfield update rule
            reconstructed_patterns = torch.sign(torch.matmul(patterns, self.weights))

            # avoid 0 states
            reconstructed_patterns[reconstructed_patterns == 0] = 1

            # update for next iteration
            patterns = reconstructed_patterns

        return reconstructed_patterns
    
    def recall_accuracy(self, corrupted_patterns, original_patters, max_iter=100):

        recalled_patterns = self.forward(corrupted_patterns, max_iter)

        # calculate accuracy based on number of exact pattern matches
        correct_recalls = torch.sum(torch.all(recalled_patterns == original_patters, dim=1)).item()
        accuracy = correct_recalls / corrupted_patterns.shape[0]

        return accuracy
    
    def recall_loss(self, corrupted_patterns, original_patterns, max_iter=100):
        
        recalled_patterns = self.forward(corrupted_patterns, max_iter)

        # sse = torch.sum((original_patterns - recalled_patterns)**2)
        # return sse

        mse = F.mse_loss(recalled_patterns, original_patterns)
        return mse
        