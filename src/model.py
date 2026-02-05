"""
AeroNet: Physics-Informed Neural Network for Aeroacoustic Noise Prediction

This module implements a neural network architecture designed to predict
Sound Pressure Level (SPL) from airfoil parameters while respecting
Lighthill's aeroacoustic scaling laws.
"""

import torch
import torch.nn as nn


class AeroNet(nn.Module):
    """
    Physics-Informed Neural Network for Airfoil Self-Noise Prediction.
    
    Architecture:
        - Input: 5 features (Freq, Angle, Chord, Velocity, Thickness)
        - Hidden: 2 layers of 64 neurons with Tanh activation
        - Output: 1 neuron (SPL in dB, normalized)
    
    The Tanh activation is chosen over ReLU as it provides smoother
    gradients, which is beneficial for physics-informed learning where
    we need to capture continuous physical relationships.
    """
    
    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, output_dim: int = 1):
        """
        Initialize AeroNet.
        
        Args:
            input_dim: Number of input features (default: 5)
            hidden_dim: Number of neurons in hidden layers (default: 64)
            output_dim: Number of output features (default: 1)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Predicted SPL (normalized) of shape (batch_size, 1)
        """
        return self.net(x)


def create_model(device: str = 'cpu') -> AeroNet:
    """
    Factory function to create and initialize AeroNet.
    
    Args:
        device: Device to place the model on ('cpu' or 'cuda')
        
    Returns:
        Initialized AeroNet model
    """
    model = AeroNet()
    model = model.to(device)
    return model
