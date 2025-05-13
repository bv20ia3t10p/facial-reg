"""
Interface for federated learning server.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import tensorflow as tf

class FederatedServerInterface(ABC):
    """Interface for federated learning servers."""
    
    @abstractmethod
    def initialize_model(self, num_classes: int, use_dp: bool = False) -> tf.keras.Model:
        """
        Initialize the global model.
        
        Args:
            num_classes: Number of classes for the model
            use_dp: Whether to use differential privacy
            
        Returns:
            The initialized global model
        """
        pass
    
    @abstractmethod
    def register_client(self, client_id: str, client_info: Dict[str, Any]) -> bool:
        """
        Register a new client with the server.
        
        Args:
            client_id: Unique identifier for the client
            client_info: Client metadata
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    def process_client_update(self, client_id: str, update_data: Dict[str, Any]) -> bool:
        """
        Process a model update from a client.
        
        Args:
            client_id: Unique identifier for the client
            update_data: Client update data including model weights and metrics
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    def aggregate_models(self) -> Optional[tf.keras.Model]:
        """
        Aggregate client models using federated averaging.
        
        Returns:
            The aggregated global model, or None if aggregation failed
        """
        pass
    
    @abstractmethod
    def start_round(self) -> bool:
        """
        Start a new federated learning round.
        
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    def end_round(self) -> Dict[str, Any]:
        """
        End the current federated learning round and compute metrics.
        
        Returns:
            Dictionary with round metrics
        """
        pass 