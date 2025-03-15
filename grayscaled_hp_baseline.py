import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

def encode(img):
    """
    Encodes a grayscale image (0-255) into a binary representation (-1,1).
    
    Args:
        img (torch.Tensor): Grayscale image of shape (H, W), values in [0, 255].
    
    Returns:
        torch.Tensor: Encoded binary vector of shape (H*W*8,), values in {-1,1}.
    """
    flat = img.flatten()
    flat_np = flat.numpy().astype(np.uint8)
    bits = np.unpackbits(flat_np.reshape(-1, 1), axis=1)
    encoded = 2 * bits - 1  # Convert {0,1} -> {-1,1}
    encoded_flat = encoded.flatten()
    return torch.tensor(encoded_flat, dtype=torch.int8)

def decode(encoded):
    """
    Decodes a binary vector (-1,1) back into a grayscale image (0-255).
    
    Args:
        encoded (torch.Tensor): Encoded binary vector of shape (H*W*8,), values in {-1,1}.
    
    Returns:
        torch.Tensor: Decoded grayscale image of shape (H, W), values in [0, 255].
    """
    if isinstance(encoded, torch.Tensor):
        encoded_reshaped = encoded.reshape(-1, 8)
        bits = ((encoded_reshaped + 1) // 2).numpy().astype(np.uint8)  # Convert {-1,1} -> {0,1}
    else:  # Assuming encoded is a numpy array
        encoded_reshaped = encoded.reshape(-1, 8)
        bits = ((encoded_reshaped + 1) // 2).astype(np.uint8)  # Convert {-1,1} -> {0,1}
    
    pixels = np.packbits(bits, axis=1)
    pixels = pixels.squeeze(1)
    
    img = torch.from_numpy(pixels).reshape(32, 32)  # Assuming a 32x32 image
    return img


class HopfieldRNNGrayscale(nn.Module):
    def __init__(self, num_units):
        """
        Hopfield-like network for grayscale images stored in binary form.

        Args:
            num_units (int): Number of neurons (should match flattened encoded image size).
        """
        super(HopfieldRNNGrayscale, self).__init__()
        self.num_units = num_units
        self.weights = nn.Parameter(torch.zeros(num_units, num_units), requires_grad=False)

    def store_patterns(self, patterns):
        """
        Stores patterns using Hebbian learning.

        Args:
            patterns (torch.Tensor): Batch of encoded binary patterns (-1,1), shape [num_patterns, num_units].
        """
        W = torch.zeros(self.num_units, self.num_units, dtype=torch.float32)
        for pattern in patterns:
            W += torch.outer(pattern, pattern)  # Hebbian learning rule
        
        # No self-connections
        W.fill_diagonal_(0)

        # Normalize by number of patterns
        self.weights.data = W / patterns.shape[0]

    def forward(self, corrupted_patterns, max_iter=10):
        """
        Runs the Hopfield update rule on binary-encoded grayscale patterns.

        Args:
            corrupted_patterns (torch.Tensor): Encoded corrupted input patterns, shape [batch_size, num_units].
            max_iter (int): Number of update iterations.

        Returns:
            torch.Tensor: Recalled binary-encoded patterns (-1,1).
        """
        patterns = corrupted_patterns.clone()

        for _ in range(max_iter):
            # Hopfield update rule
            reconstructed_patterns = torch.sign(torch.matmul(patterns, self.weights))

            # Avoid 0 states (set them to +1)
            reconstructed_patterns[reconstructed_patterns == 0] = 1

            patterns = reconstructed_patterns  # Update for next iteration

        return reconstructed_patterns

    def recall_accuracy(self, corrupted_patterns, original_patterns, max_iter=10):
        """
        Calculates recall accuracy based on exact binary matches.

        Args:
            corrupted_patterns (torch.Tensor): Noisy input patterns.
            original_patterns (torch.Tensor): Ground truth patterns.
            max_iter (int): Number of Hopfield iterations.

        Returns:
            float: Accuracy of pattern recall.
        """
        recalled_patterns = self.forward(corrupted_patterns, max_iter)
        correct_recalls = torch.sum(torch.all(recalled_patterns == original_patterns, dim=1)).item()
        accuracy = correct_recalls / corrupted_patterns.shape[0]
        return accuracy

    def recall_loss(self, corrupted_patterns, original_patterns, max_iter=10):
        """
        Computes squared error loss in pixel space after decoding.

        Args:
            corrupted_patterns (torch.Tensor): Noisy input patterns.
            original_patterns (torch.Tensor): Ground truth patterns.
            max_iter (int): Number of Hopfield iterations.

        Returns:
            float: Sum of squared pixel errors.
        """
        recalled_patterns = self.forward(corrupted_patterns, max_iter)

        # Convert back to grayscale
        decoded_recalled = torch.stack([decode(recalled) for recalled in recalled_patterns])
        decoded_original = torch.stack([decode(original) for original in original_patterns])

        # Compute sum of squared differences in pixel space
        sse = torch.sum((decoded_original - decoded_recalled) ** 2)
        return sse
