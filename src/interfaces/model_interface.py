"""
Interface for face recognition models following the Interface Segregation Principle.
"""

from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
from typing import Tuple, List, Dict, Any

class FaceModelInterface(ABC):
    """Interface for face recognition models."""
    
    @abstractmethod
    def build_model(self, num_classes: int, input_shape: Tuple[int, int, int] = (112, 112, 3), 
                   training: bool = True) -> tf.keras.Model:
        """
        Build a face recognition model.
        
        Args:
            num_classes: Number of identity classes for the model to recognize
            input_shape: Shape of input images (height, width, channels)
            training: Whether the model will be used for training or inference
            
        Returns:
            A compiled TensorFlow model
        """
        pass
    
    @abstractmethod
    def get_embedding_model(self, model: tf.keras.Model = None) -> tf.keras.Model:
        """
        Get the embedding model for feature extraction.
        
        Args:
            model: Optional full model to extract the embedding portion
            
        Returns:
            A TensorFlow model that outputs face embeddings
        """
        pass
    
    @abstractmethod
    def extract_embeddings(self, images: np.ndarray) -> np.ndarray:
        """
        Extract embeddings from face images.
        
        Args:
            images: Batch of preprocessed face images
            
        Returns:
            Array of face embeddings
        """
        pass
    
    @abstractmethod
    def save_model(self, model: tf.keras.Model, path: str) -> str:
        """
        Save a model to disk.
        
        Args:
            model: Model to save
            path: Path to save the model
            
        Returns:
            Path to the saved model
        """
        pass
    
    @abstractmethod
    def load_model(self, path: str) -> tf.keras.Model:
        """
        Load a model from disk.
        
        Args:
            path: Path to the model
            
        Returns:
            Loaded model
        """
        pass 