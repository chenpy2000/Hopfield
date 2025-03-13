import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class HopfieldXavier(nn.Module):
    def __init__(self):
        super(HopfieldXavier, self).__init__()
        self.dropout = nn.Dropout(p=0.25)
        self.skip = nn.Linear(8192, 8192)
        self.down = nn.Linear(8192, 100)
        self.up   = nn.Linear(100, 8192)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        x_skip = self.skip(x)
        x_hidden = self.activation(self.down(x))
        x_hidden = self.activation(self.up(x_hidden))
        return x_skip + x_hidden
    
def train_model(model, epochs, optimizer, criterion, loader):
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for inputs in loader:
            optimizer.zero_grad()          # Clear gradients
            outputs = model(inputs)          # Forward pass
            loss = criterion(outputs, inputs)  # Compute reconstruction loss
            loss.backward()                  # Backpropagation
            optimizer.step()                 # Update weights
            
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

def encode(img):
    flat = img.flatten()
    flat_np = flat.numpy().astype(np.uint8)
    bits = np.unpackbits(flat_np.reshape(-1, 1), axis=1)
    encoded = 2 * bits - 1
    encoded_flat = encoded.flatten()
    return torch.tensor(encoded_flat, dtype=torch.int8)

def decode(encoded):
    encoded_reshaped = encoded.reshape(-1, 8)
    bits = ((encoded_reshaped + 1) // 2).numpy().astype(np.uint8)
    pixels = np.packbits(bits, axis=1)
    pixels = pixels.squeeze(1)
    
    img = torch.from_numpy(pixels).reshape(32, 32)
    return img

def prep_data(train_dataset, batch_size):
    data = []
    for i in range(2):
        fig = train_dataset[i][0]
        tensor_fig = (fig * 255).to(torch.uint8)
        fig_enc = encode(tensor_fig).to(torch.float)
        data.append(fig_enc)

    loader = DataLoader(data, batch_size=batch_size, shuffle=False)