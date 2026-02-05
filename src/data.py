"""
Data Loading and Preprocessing for Airfoil Self-Noise Dataset

This module handles loading the NASA Airfoil Self-Noise dataset from
the UCI Machine Learning Repository and preprocessing it for neural
network training.

Dataset: https://archive.ics.uci.edu/ml/datasets/airfoil+self-noise
"""

import io
import requests
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from typing import Tuple


# Dataset URL from UCI ML Repository
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat"

# Column names for the dataset
COLUMN_NAMES = ['Freq', 'Angle', 'Chord', 'Velocity', 'Thickness', 'SPL']


def load_airfoil_data(url: str = DATASET_URL) -> pd.DataFrame:
    """
    Download and load the airfoil self-noise dataset.
    
    Args:
        url: URL to the dataset file
        
    Returns:
        DataFrame containing the airfoil data
    """
    response = requests.get(url)
    response.raise_for_status()
    
    df = pd.read_csv(
        io.BytesIO(response.content), 
        sep='\t', 
        names=COLUMN_NAMES
    )
    return df


def preprocess_data(
    df: pd.DataFrame
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, StandardScaler, StandardScaler]:
    """
    Preprocess the airfoil data for neural network training.
    
    Applies StandardScaler normalization to features and target,
    which is crucial for neural network convergence.
    
    Args:
        df: Raw dataframe from load_airfoil_data
        
    Returns:
        Tuple containing:
            - X_tensor: Normalized input features
            - y_tensor: Normalized target values
            - velocity_raw: Raw velocity values (for physics loss)
            - scaler_x: Fitted feature scaler
            - scaler_y: Fitted target scaler
    """
    # Separate features and target
    X = df.drop('SPL', axis=1).values
    y = df['SPL'].values.reshape(-1, 1)
    
    # Keep raw velocity for physics loss computation
    velocity_raw = torch.tensor(
        df['Velocity'].values.reshape(-1, 1), 
        dtype=torch.float32
    )
    
    # Normalize features and target
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
    
    return X_tensor, y_tensor, velocity_raw, scaler_x, scaler_y


def inverse_transform_spl(
    y_scaled: np.ndarray, 
    scaler_y: StandardScaler
) -> np.ndarray:
    """
    Convert normalized SPL predictions back to dB scale.
    
    Args:
        y_scaled: Normalized SPL values
        scaler_y: Fitted target scaler
        
    Returns:
        SPL values in dB scale
    """
    return scaler_y.inverse_transform(y_scaled)


def get_data_loaders(
    X_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    batch_size: int = 32,
    shuffle: bool = True
) -> torch.utils.data.DataLoader:
    """
    Create PyTorch DataLoader for training.
    
    Args:
        X_tensor: Input features
        y_tensor: Target values
        batch_size: Batch size for training
        shuffle: Whether to shuffle data
        
    Returns:
        DataLoader for training
    """
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle
    )
    return loader
