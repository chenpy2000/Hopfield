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

# Simple Grayscaled baseline Hopfield with 8192 num_units
# change the class name in visualize_grayscale.ipynb file to see the results - memorizes 30 patterns
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


# Grayscaled baseline Hopfield with tanh non-linearity and hidden units of 4096, num_units of 8192
# change the class name in visualize_grayscale.ipynb file to see the results - memorizes 50 patterns
class HopfieldRNNGrayscaleTANH(nn.Module):
    def __init__(self, num_units, hidden_size=4096):
        """
        Hopfield-like RNN with a non-linear hidden layer for grayscale images.

        Args:
            num_units (int): Number of binary units in the input (flattened image).
            hidden_size (int): Number of hidden tanh units.
        """
        super(HopfieldRNNGrayscale, self).__init__()
        self.input_size = num_units
        self.hidden_size = hidden_size

        # Input to hidden layer (non-linear)
        self.fc1 = nn.Linear(num_units, hidden_size)

        # Hidden to output layer (linear)
        self.fc2 = nn.Linear(hidden_size, num_units)

        # Hebbian-style memory storage (initialized as zero)
        self.weights = torch.zeros(hidden_size, num_units)

    def store_patterns(self, patterns):
        """
        Stores multiple binary patterns using Hebbian learning.

        Args:
            patterns (torch.Tensor): Encoded binary patterns (-1,1), shape [num_patterns, input_size].
        """
        patterns = patterns.to(dtype=torch.float32)  # Ensure float dtype

        hidden_activations = torch.tanh(self.fc1(patterns))  # Shape: [num_patterns, hidden_size]

        # Scale activations to [-0.8, 0.8]
        hidden_activations *= 0.8

        # Convert hidden activations to binary {-1,1}
        binary_hidden = torch.where(hidden_activations > 0, 1.0, -1.0)

        # Hebbian update rule (outer product)
        W = torch.zeros(self.hidden_size, self.input_size)
        for h, p in zip(binary_hidden, patterns):
            W += torch.outer(h, p)

        # Normalize by the number of patterns
        self.weights = W / patterns.shape[0]

    def forward(self, corrupted_patterns, max_iter=10):
        """
        Runs Hopfield-style recall using the stored patterns.

        Args:
            x (torch.Tensor): Corrupted input patterns (-1,1), shape [batch_size, input_size].
            max_iter (int): Number of recall update iterations.

        Returns:
            torch.Tensor: Reconstructed binary-encoded patterns (-1,1).
        """
        patterns = corrupted_patterns.clone()
        
        for _ in range(max_iter):
            hidden_activations = torch.tanh(self.fc1(patterns))  # Shape: [batch_size, hidden_size]
            hidden_activations *= 0.8  # Scale to [-0.8, 0.8]

            # Convert to binary {-1,1}
            binary_hidden = torch.where(hidden_activations > 0, 1.0, -1.0)

            # Reconstruct output using the stored weights
            reconstructed_patterns = torch.sign(torch.matmul(binary_hidden, self.weights))
            patterns = reconstructed_patterns

        return reconstructed_patterns  # Output remains in {-1,1} format


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
