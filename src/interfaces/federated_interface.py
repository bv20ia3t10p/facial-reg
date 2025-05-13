"""
Interfaces for federated learning following the Interface Segregation Principle.
"""

from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
from typing import List, Dict, Any, Tuple

class FederatedClientInterface(ABC):
    """Interface for federated learning clients."""
    
    @abstractmethod
    def initialize(self, model_path: str = None, data_dir: str = None) -> None:
        """
        Initialize the federated client.
        
        Args:
            model_path: Path to the initial model
            data_dir: Path to the client's data directory
        """
        pass
    
    @abstractmethod
    def train_on_local_data(self, epochs: int = 5, batch_size: int = 32) -> Dict[str, Any]:
        """
        Train the model on local data.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Dictionary with training metrics
        """
        pass
    
    @abstractmethod
    def get_model_update(self) -> Dict[str, Any]:
        """
        Get the model update after local training.
        
        Returns:
            Dictionary with model weights and metadata
        """
        pass
    
    @abstractmethod
    def apply_global_model(self, global_model_weights: Dict[str, np.ndarray]) -> None:
        """
        Apply the global model weights to the local model.
        
        Args:
            global_model_weights: Dictionary with global model weights
        """
        pass
    
    @abstractmethod
    def evaluate_model(self) -> Dict[str, float]:
        """
        Evaluate the model on local validation data.
        
        Returns:
            Dictionary with evaluation metrics
        """
        pass


class FederatedServerInterface(ABC):
    """Interface for federated learning server."""
    
    @abstractmethod
    def initialize(self, model_path: str = None, data_dir: str = None) -> None:
        """
        Initialize the federated server.
        
        Args:
            model_path: Path to the initial model
            data_dir: Path to the server's data directory
        """
        pass
    
    @abstractmethod
    def aggregate_model_updates(self, client_updates: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        Aggregate model updates from clients.
        
        Args:
            client_updates: List of client model updates
            
        Returns:
            Dictionary with aggregated model weights
        """
        pass
    
    @abstractmethod
    def update_global_model(self, aggregated_weights: Dict[str, np.ndarray]) -> None:
        """
        Update the global model with aggregated weights.
        
        Args:
            aggregated_weights: Dictionary with aggregated model weights
        """
        pass
    
    @abstractmethod
    def evaluate_global_model(self) -> Dict[str, float]:
        """
        Evaluate the global model on server data.
        
        Returns:
            Dictionary with evaluation metrics
        """
        pass
    
    @abstractmethod
    def save_global_model(self, path: str) -> str:
        """
        Save the global model to disk.
        
        Args:
            path: Path to save the model
            
        Returns:
            Path to the saved model
        """
        pass 