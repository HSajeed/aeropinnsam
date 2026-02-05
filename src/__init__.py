"""
Aero-PINN-SAM: Physics-Informed Neural Network for Aeroacoustic Noise Prediction

This package provides modules for:
- Neural network model (AeroNet)
- SAM optimizer for improved generalization
- Data loading from NASA dataset
- Auralization for audio output
"""

from .model import AeroNet, create_model
from .optimizer import SAM
from .data import load_airfoil_data, preprocess_data
from .audio import auralize_prediction, generate_airfoil_sound

__version__ = "1.0.0"
__author__ = "Sajeed"

__all__ = [
    "AeroNet",
    "create_model",
    "SAM",
    "load_airfoil_data",
    "preprocess_data",
    "auralize_prediction",
    "generate_airfoil_sound",
]
