import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import torch

from hopfield_baseline import HopfieldRNN

class HopfieldStorkey(HopfieldRNN):
    def __init__(self, num_units):

        super().__init__(num_units)

    def local_field(self, pattern, weights):
            """
            Compute local field h for Storkey rule.
            
            Args:
                pattern: [num_units, 1]
                weights: [num_units, num_units]
            
            Returns:
                Local field vector (shape: [num_units, 1])
            """
            return torch.matmul(weights, pattern) - torch.diag(weights).unsqueeze(1) * pattern

    def store_patterns(self, patterns):
        """
        Implements incremental Storkey learning rule.
    
        Args:
            patterns: Tensor of shape [num_patterns, num_units], values {-1, 1}
        """
        
        num_patterns = patterns.shape[0]
        self.weights.data.zero_()  # reset weights to zero
    
        for p in range(num_patterns):
            # Get the current pattern as a column vector
            pattern = patterns[p].unsqueeze(1)  # Shape: [num_units, 1]

            previous_weights = self.weights.data.clone()
    
            # Compute the local field
            h = self.local_field(pattern, previous_weights)  # Shape: [num_units, 1]
    
            # Compute delta_W according to Storkey rule
            delta_W = (torch.matmul(pattern, pattern.T) -
                       torch.matmul(pattern, h.T) -
                       torch.matmul(h, pattern.T)) / self.num_units
    
            # No self-connections
            delta_W.fill_diagonal_(0)
    
            # Incremental update of weights
            self.weights.data += delta_W


####################################################################

# class HopfieldRNNStorkey(HopfieldRNN):
#     def __init__(self, num_units):
#         super(HopfieldRNNStorkey, self).__init__(num_units)

#     def store_patterns(self, patterns):
#         """
#         Implements the incremental Storkey learning rule.
    
#         Args:
#             patterns (torch.Tensor): shape [num_patterns, num_units], values {-1, 1}
#         """
#         num_patterns, num_units = patterns.shape
#         self.weights.data.zero_()  # initialize weights to zero
    
#         for p in range(num_patterns):
#             pattern = patterns[p].unsqueeze(1)  # shape: [num_units, 1]
    
#             # Compute local fields h_ij for all neuron pairs
#             h = torch.matmul(self.weights.data, pattern) - self.weights.data * pattern  # exclude self-connections
    
#             # Update weights incrementally according to Storkey rule
#             delta_w = (1 / num_units) * (pattern @ pattern.T - pattern @ h(pattern, self.weights).T - h(pattern, self.weights) @ pattern.T)
            
#             # No self-connections
#             delta_W = delta_W.fill_diagonal_(0)
    
#             # Incremental update
#             self.weights.data += delta_W
    
#     def h(pattern, weights):
#         """
#         Compute local field h for Storkey rule.
        
#         Args:
#             pattern: [num_units, 1]
#             weights: [num_units, num_units]
        
#         Returns:
#             local field vector (shape: [num_units, 1])
#         """
#         return weights @ pattern - torch.diag(weights).unsqueeze(1) * pattern

###############################################

    # def store_patterns(self, patterns):
    #     """
    #     Store patterns in the Hopfield network using the Storkey learning rule.

    #     Args:
    #         patterns (torch.Tensor): Binary patterns (shape: [num_patterns, num_units])
    #     """
    #     # Initialize weight matrix
    #     W = torch.zeros(self.num_units, self.num_units, dtype=torch.float32)

    #  
    #     for pattern in patterns:
    #         outer_prod = torch.outer(pattern, pattern)
    #         W += outer_prod

    #     # Normalize weights by the number of patterns
    #     self.weights.data = W / patterns.shape[0]

    #     # Storkey adjustment: prevent unbounded weight growth by subtracting
    #     # the normalization term
    #     for i in range(self.num_units):
    #         for j in range(self.num_units):
    #             sum_term = torch.sum(self.weights[i] * self.weights[j])
    #             self.weights.data[i, j] = self.weights[i, j] - sum_term / self.num_units
