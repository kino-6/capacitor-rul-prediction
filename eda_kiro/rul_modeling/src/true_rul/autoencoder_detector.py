"""
Autoencoder-based Anomaly Detector for RUL prediction system.

This module implements a neural network autoencoder for anomaly detection in
capacitor voltage data. The autoencoder learns to reconstruct normal patterns
and identifies anomalies based on reconstruction error.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class AutoencoderDetector(nn.Module):
    """
    Autoencoder-based anomaly detector for capacitor degradation detection.
    
    This detector uses a neural network autoencoder to learn normal patterns
    from voltage time-series features. Anomalies are detected based on
    reconstruction error - samples that cannot be well reconstructed are
    considered anomalous.
    
    Architecture:
    - Encoder: input_dim -> 128 -> 64 -> encoding_dim
    - Decoder: encoding_dim -> 64 -> 128 -> input_dim
    
    Attributes:
        input_dim (int): Dimension of input features
        encoding_dim (int): Dimension of the encoded representation
        encoder (nn.Sequential): Encoder network
        decoder (nn.Sequential): Decoder network
        scaler (StandardScaler): Feature scaler for normalization
        is_fitted (bool): Whether the model has been trained
        device (torch.device): Device for computation (CPU/GPU)
    """
    
    def __init__(self, input_dim: int, encoding_dim: int = 16, device: Optional[str] = None):
        """
        Initialize the autoencoder detector.
        
        Args:
            input_dim (int): Dimension of input features
            encoding_dim (int): Dimension of the encoded representation (bottleneck)
            device (Optional[str]): Device for computation ('cpu', 'cuda', or None for auto)
        """
        super(AutoencoderDetector, self).__init__()
        
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if encoding_dim <= 0 or encoding_dim >= input_dim:
            raise ValueError("encoding_dim must be positive and less than input_dim")
        
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, encoding_dim)
        )
        
        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, input_dim)
        )
        
        # Move model to device
        self.to(self.device)
        
        # Initialize other attributes
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.training_loss_history = []
        self.reconstruction_threshold = None
        
        logger.info(f"Initialized AutoencoderDetector with input_dim={input_dim}, "
                   f"encoding_dim={encoding_dim}, device={self.device}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the autoencoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Reconstructed output of shape (batch_size, input_dim)
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def fit(self, normal_data: np.ndarray, 
            epochs: int = 100, 
            batch_size: int = 32,
            learning_rate: float = 0.001,
            validation_split: float = 0.2,
            early_stopping_patience: int = 10,
            verbose: bool = True) -> 'AutoencoderDetector':
        """
        Train the autoencoder on normal cycles data.
        
        Args:
            normal_data (np.ndarray): Feature vectors from normal cycles.
                                    Shape: (n_samples, n_features)
            epochs (int): Maximum number of training epochs
            batch_size (int): Batch size for training
            learning_rate (float): Learning rate for optimizer
            validation_split (float): Fraction of data to use for validation
            early_stopping_patience (int): Epochs to wait before early stopping
            verbose (bool): Whether to print training progress
            
        Returns:
            AutoencoderDetector: Self for method chaining
            
        Raises:
            ValueError: If normal_data is empty or has invalid shape
        """
        if normal_data.size == 0:
            raise ValueError("normal_data cannot be empty")
        
        if len(normal_data.shape) != 2:
            raise ValueError("normal_data must be 2D array with shape (n_samples, n_features)")
        
        n_samples, n_features = normal_data.shape
        if n_features != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} features, got {n_features}")
        
        if n_samples < batch_size:
            logger.warning(f"Number of samples ({n_samples}) is less than batch_size ({batch_size}). "
                          f"Reducing batch_size to {n_samples}")
            batch_size = n_samples
        
        logger.info(f"Training autoencoder on {n_samples} normal samples with {n_features} features")
        
        # Normalize the data
        normal_data_scaled = self.scaler.fit_transform(normal_data)
        
        # Split into train and validation
        n_val = int(n_samples * validation_split)
        n_train = n_samples - n_val
        
        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        train_data = normal_data_scaled[train_indices]
        val_data = normal_data_scaled[val_indices] if n_val > 0 else None
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(train_data),
            torch.FloatTensor(train_data)  # Target is same as input for autoencoder
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if val_data is not None:
            val_dataset = TensorDataset(
                torch.FloatTensor(val_data),
                torch.FloatTensor(val_data)
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_loader = None
        
        # Initialize optimizer and loss function
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        self.train()
        best_val_loss = float('inf')
        patience_counter = 0
        self.training_loss_history = []
        
        for epoch in range(epochs):
            # Training phase
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                reconstructed = self.forward(batch_x)
                loss = criterion(reconstructed, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation phase
            val_loss = 0.0
            if val_loader is not None:
                self.eval()
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        reconstructed = self.forward(batch_x)
                        loss = criterion(reconstructed, batch_y)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                self.train()
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        if verbose:
                            logger.info(f"Early stopping at epoch {epoch+1}")
                        break
            
            self.training_loss_history.append({'epoch': epoch+1, 'train_loss': train_loss, 'val_loss': val_loss})
            
            if verbose and (epoch + 1) % 10 == 0:
                if val_loader is not None:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                else:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}")
        
        # Set reconstruction threshold based on training data
        self.eval()
        with torch.no_grad():
            train_tensor = torch.FloatTensor(normal_data_scaled).to(self.device)
            train_reconstructed = self.forward(train_tensor)
            train_errors = torch.mean((train_tensor - train_reconstructed) ** 2, dim=1)
            
            # Set threshold as 95th percentile of training reconstruction errors
            self.reconstruction_threshold = torch.quantile(train_errors, 0.95).item()
        
        self.is_fitted = True
        logger.info(f"Training completed. Reconstruction threshold: {self.reconstruction_threshold:.6f}")
        
        return self
    
    def get_reconstruction_error(self, x: np.ndarray) -> np.ndarray:
        """
        Return reconstruction error as anomaly score.
        
        Higher reconstruction errors indicate more anomalous behavior.
        
        Args:
            x (np.ndarray): Input feature vectors. Shape: (n_samples, n_features)
            
        Returns:
            np.ndarray: Reconstruction errors for each sample. Shape: (n_samples,)
                       Higher values = more anomalous
                       
        Raises:
            ValueError: If model is not fitted or input has wrong shape
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before computing reconstruction error. Call fit() first.")
        
        if x.size == 0:
            raise ValueError("Input x cannot be empty")
        
        # Handle single sample case
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        if len(x.shape) != 2:
            raise ValueError("Input x must be 1D or 2D array")
        
        n_samples, n_features = x.shape
        if n_features != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} features, got {n_features}")
        
        # Normalize input
        x_scaled = self.scaler.transform(x)
        
        # Compute reconstruction error
        self.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x_scaled).to(self.device)
            reconstructed = self.forward(x_tensor)
            
            # Mean squared error per sample
            errors = torch.mean((x_tensor - reconstructed) ** 2, dim=1)
            errors_np = errors.cpu().numpy()
        
        logger.debug(f"Computed reconstruction errors for {n_samples} samples. "
                    f"Error range: [{errors_np.min():.6f}, {errors_np.max():.6f}]")
        
        return errors_np
    
    def predict_binary(self, x: np.ndarray) -> np.ndarray:
        """
        Return binary anomaly predictions based on reconstruction threshold.
        
        Args:
            x (np.ndarray): Input feature vectors. Shape: (n_samples, n_features)
            
        Returns:
            np.ndarray: Binary predictions. Shape: (n_samples,)
                       0 = normal, 1 = anomaly
        """
        if self.reconstruction_threshold is None:
            raise ValueError("Reconstruction threshold not set. Model must be fitted first.")
        
        errors = self.get_reconstruction_error(x)
        predictions = (errors > self.reconstruction_threshold).astype(int)
        
        return predictions
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the fitted model.
        
        Returns:
            Dict[str, Any]: Model information including parameters and statistics
        """
        info = {
            'input_dim': self.input_dim,
            'encoding_dim': self.encoding_dim,
            'is_fitted': self.is_fitted,
            'device': str(self.device),
            'n_parameters': sum(p.numel() for p in self.parameters())
        }
        
        if self.is_fitted:
            info['reconstruction_threshold'] = self.reconstruction_threshold
            info['training_epochs'] = len(self.training_loss_history)
            if self.training_loss_history:
                info['final_train_loss'] = self.training_loss_history[-1]['train_loss']
                info['final_val_loss'] = self.training_loss_history[-1]['val_loss']
        
        return info
    
    def __repr__(self) -> str:
        """String representation of the detector."""
        status = "fitted" if self.is_fitted else "not fitted"
        return f"AutoencoderDetector(input_dim={self.input_dim}, encoding_dim={self.encoding_dim}, status={status})"