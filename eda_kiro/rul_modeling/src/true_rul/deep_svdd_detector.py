"""
Deep Support Vector Data Description (Deep SVDD) Anomaly Detector

This module implements Deep SVDD for anomaly detection in RUL prediction.
Deep SVDD learns a neural network representation that maps normal data
to a hypersphere with minimal volume.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DeepSVDDConfig:
    """Configuration for Deep SVDD detector."""
    hidden_dims: List[int]
    learning_rate: float = 0.001
    weight_decay: float = 1e-6
    epochs: int = 100
    batch_size: int = 32
    nu: float = 0.1  # Outlier fraction
    objective: str = "one-class"  # "one-class" or "soft-boundary"
    device: str = "cpu"
    random_seed: int = 42


class DeepSVDDNetwork(nn.Module):
    """Neural network for Deep SVDD."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Remove last dropout
        if layers:
            layers = layers[:-1]
        
        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1] if hidden_dims else input_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)


class DeepSVDDDetector:
    """
    Deep Support Vector Data Description anomaly detector.
    
    This detector learns a neural network representation that maps
    normal data to a hypersphere with minimal volume.
    """
    
    def __init__(self, config: DeepSVDDConfig):
        self.config = config
        self.network: Optional[DeepSVDDNetwork] = None
        self.center: Optional[torch.Tensor] = None
        self.radius: Optional[float] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.is_fitted = False
        
        # Set random seed for reproducibility
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)
        
    def _initialize_center(self, data_loader: torch.utils.data.DataLoader) -> torch.Tensor:
        """Initialize the center of the hypersphere."""
        logger.info("Initializing hypersphere center...")
        
        self.network.eval()
        centers = []
        
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                batch = batch.to(self.config.device)
                outputs = self.network(batch)
                centers.append(outputs)
        
        center = torch.cat(centers, dim=0).mean(dim=0)
        return center
    
    def _create_data_loader(self, X: np.ndarray) -> torch.utils.data.DataLoader:
        """Create PyTorch data loader from numpy array."""
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X)
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
    
    def fit(self, X: np.ndarray) -> 'DeepSVDDDetector':
        """
        Fit the Deep SVDD detector on normal data.
        
        Args:
            X: Training data (normal samples only)
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Training Deep SVDD detector on {X.shape[0]} samples...")
        
        # Initialize network
        input_dim = X.shape[1]
        self.network = DeepSVDDNetwork(input_dim, self.config.hidden_dims)
        self.network.to(self.config.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Create data loader
        data_loader = self._create_data_loader(X)
        
        # Initialize center
        self.center = self._initialize_center(data_loader)
        self.center = self.center.to(self.config.device)
        
        # Training loop
        self.network.train()
        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                batch = batch.to(self.config.device)
                
                # Forward pass
                outputs = self.network(batch)
                
                # Compute loss (distance to center)
                dist = torch.sum((outputs - self.center) ** 2, dim=1)
                
                if self.config.objective == "soft-boundary":
                    # Soft-boundary Deep SVDD
                    loss = self.radius ** 2 + (1 / self.config.nu) * torch.mean(
                        torch.max(torch.zeros_like(dist), dist - self.radius ** 2)
                    )
                else:
                    # One-class Deep SVDD
                    loss = torch.mean(dist)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            if epoch % 20 == 0:
                avg_loss = epoch_loss / n_batches
                logger.info(f"Epoch {epoch}/{self.config.epochs}, Loss: {avg_loss:.6f}")
        
        # Compute radius for soft-boundary objective
        if self.config.objective == "soft-boundary":
            self._compute_radius(data_loader)
        
        self.is_fitted = True
        logger.info("Deep SVDD training completed")
        return self
    
    def _compute_radius(self, data_loader: torch.utils.data.DataLoader) -> None:
        """Compute the radius of the hypersphere."""
        self.network.eval()
        distances = []
        
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                batch = batch.to(self.config.device)
                outputs = self.network(batch)
                dist = torch.sum((outputs - self.center) ** 2, dim=1)
                distances.append(dist)
        
        distances = torch.cat(distances, dim=0)
        self.radius = torch.quantile(distances, 1 - self.config.nu).item()
    
    def predict_score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores for input data.
        
        Args:
            X: Input data
            
        Returns:
            Anomaly scores (higher = more anomalous)
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")
        
        self.network.eval()
        scores = []
        
        # Create data loader for prediction
        data_loader = self._create_data_loader(X)
        
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                batch = batch.to(self.config.device)
                outputs = self.network(batch)
                
                # Compute distance to center
                dist = torch.sum((outputs - self.center) ** 2, dim=1)
                scores.append(dist.cpu().numpy())
        
        return np.concatenate(scores)
    
    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """
        Predict anomalies in input data.
        
        Args:
            X: Input data
            threshold: Decision threshold (if None, uses radius for soft-boundary)
            
        Returns:
            Binary predictions (1 = anomaly, 0 = normal)
        """
        scores = self.predict_score(X)
        
        if threshold is None:
            if self.config.objective == "soft-boundary" and self.radius is not None:
                threshold = self.radius ** 2
            else:
                # Use quantile-based threshold
                threshold = np.quantile(scores, 1 - self.config.nu)
        
        return (scores > threshold).astype(int)
    
    def get_feature_importance(self, X: np.ndarray) -> np.ndarray:
        """
        Compute feature importance based on gradient magnitudes.
        
        Args:
            X: Input data
            
        Returns:
            Feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before computing feature importance")
        
        self.network.eval()
        X_tensor = torch.FloatTensor(X).to(self.config.device)
        X_tensor.requires_grad_(True)
        
        # Forward pass
        outputs = self.network(X_tensor)
        
        # Compute distance to center
        dist = torch.sum((outputs - self.center) ** 2, dim=1)
        loss = torch.mean(dist)
        
        # Compute gradients
        loss.backward()
        
        # Feature importance as gradient magnitude
        importance = torch.abs(X_tensor.grad).mean(dim=0).cpu().numpy()
        
        return importance / np.sum(importance)  # Normalize
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'center': self.center,
            'radius': self.radius,
            'config': self.config,
            'is_fitted': self.is_fitted
        }, filepath)
        
        logger.info(f"Deep SVDD model saved to {filepath}")
    
    def load_model(self, filepath: str) -> 'DeepSVDDDetector':
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.config.device)
        
        # Reconstruct network
        input_dim = checkpoint['center'].shape[0]  # Infer from center
        self.network = DeepSVDDNetwork(input_dim, self.config.hidden_dims)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.network.to(self.config.device)
        
        # Load other attributes
        self.center = checkpoint['center'].to(self.config.device)
        self.radius = checkpoint['radius']
        self.is_fitted = checkpoint['is_fitted']
        
        logger.info(f"Deep SVDD model loaded from {filepath}")
        return self


def create_deep_svdd_detector(
    input_dim: int,
    hidden_dims: Optional[List[int]] = None,
    **kwargs
) -> DeepSVDDDetector:
    """
    Factory function to create a Deep SVDD detector with sensible defaults.
    
    Args:
        input_dim: Input feature dimension
        hidden_dims: Hidden layer dimensions
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured Deep SVDD detector
    """
    if hidden_dims is None:
        # Default architecture based on input dimension
        if input_dim <= 50:
            hidden_dims = [32, 16, 8]
        elif input_dim <= 100:
            hidden_dims = [64, 32, 16]
        else:
            hidden_dims = [128, 64, 32]
    
    config = DeepSVDDConfig(hidden_dims=hidden_dims, **kwargs)
    return DeepSVDDDetector(config)