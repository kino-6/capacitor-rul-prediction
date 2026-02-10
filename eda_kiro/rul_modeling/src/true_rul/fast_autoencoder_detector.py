"""
Fast Autoencoder Detector for testing purposes

This is a simplified version of the autoencoder detector optimized for speed
during testing and development.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from typing import Optional, Dict, Any
import logging
import time

logger = logging.getLogger(__name__)


class FastAutoencoderDetector(nn.Module):
    """
    Fast autoencoder detector optimized for testing
    
    This is a simplified version with:
    - Minimal architecture (single hidden layer)
    - Fast training (few epochs, early stopping)
    - Reduced complexity for testing purposes
    """
    
    def __init__(self, input_dim: int, encoding_dim: int = 8, device: Optional[str] = None):
        """Initialize fast autoencoder detector"""
        super(FastAutoencoderDetector, self).__init__()
        
        self.input_dim = input_dim
        self.encoding_dim = min(encoding_dim, input_dim // 4)  # Ensure reasonable size
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Minimal architecture for speed
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.encoding_dim),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.encoding_dim, input_dim)
        )
        
        self.to(self.device)
        
        # Initialize other attributes
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.reconstruction_threshold = None
        
        logger.info(f"Initialized FastAutoencoderDetector with input_dim={input_dim}, "
                   f"encoding_dim={self.encoding_dim}, device={self.device}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the autoencoder"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def fit(self, normal_data: np.ndarray, 
            epochs: int = 3,  # Very few epochs for speed
            batch_size: int = None,
            learning_rate: float = 0.01,
            verbose: bool = False) -> 'FastAutoencoderDetector':
        """
        Fast training on normal cycles data
        
        Args:
            normal_data: Feature vectors from normal cycles
            epochs: Number of training epochs (default: 3)
            batch_size: Batch size (default: all data)
            learning_rate: Learning rate
            verbose: Whether to print progress
        """
        if normal_data.size == 0:
            raise ValueError("normal_data cannot be empty")
        
        n_samples, n_features = normal_data.shape
        if n_features != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} features, got {n_features}")
        
        # Use all data as single batch for speed
        if batch_size is None:
            batch_size = n_samples
        
        logger.info(f"Fast training autoencoder on {n_samples} samples with {n_features} features")
        start_time = time.time()
        
        # Normalize the data
        normal_data_scaled = self.scaler.fit_transform(normal_data)
        
        # Create data loader
        dataset = TensorDataset(
            torch.FloatTensor(normal_data_scaled),
            torch.FloatTensor(normal_data_scaled)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize optimizer and loss
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Fast training loop
        self.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                reconstructed = self.forward(batch_x)
                loss = criterion(reconstructed, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if verbose:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.6f}")
        
        # Set reconstruction threshold
        self.eval()
        with torch.no_grad():
            train_tensor = torch.FloatTensor(normal_data_scaled).to(self.device)
            train_reconstructed = self.forward(train_tensor)
            train_errors = torch.mean((train_tensor - train_reconstructed) ** 2, dim=1)
            self.reconstruction_threshold = torch.quantile(train_errors, 0.95).item()
        
        training_time = time.time() - start_time
        self.is_fitted = True
        logger.info(f"Fast training completed in {training_time:.2f}s. Threshold: {self.reconstruction_threshold:.6f}")
        
        return self
    
    def get_reconstruction_error(self, x: np.ndarray) -> np.ndarray:
        """Return reconstruction error as anomaly score"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before computing reconstruction error")
        
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        # Normalize input
        x_scaled = self.scaler.transform(x)
        
        # Compute reconstruction error
        self.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x_scaled).to(self.device)
            reconstructed = self.forward(x_tensor)
            errors = torch.mean((x_tensor - reconstructed) ** 2, dim=1)
            errors_np = errors.cpu().numpy()
        
        return errors_np
    
    def predict_binary(self, x: np.ndarray) -> np.ndarray:
        """Return binary anomaly predictions"""
        if self.reconstruction_threshold is None:
            raise ValueError("Reconstruction threshold not set")
        
        errors = self.get_reconstruction_error(x)
        predictions = (errors > self.reconstruction_threshold).astype(int)
        return predictions
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'input_dim': self.input_dim,
            'encoding_dim': self.encoding_dim,
            'is_fitted': self.is_fitted,
            'device': str(self.device),
            'n_parameters': sum(p.numel() for p in self.parameters()),
            'reconstruction_threshold': self.reconstruction_threshold,
            'model_type': 'FastAutoencoder'
        }