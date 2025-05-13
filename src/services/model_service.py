"""
Service for managing face recognition models.
"""
import os
import numpy as np
import tensorflow as tf
from typing import Optional, Dict, List, Tuple, Any

from src.interfaces.model_interface import FaceModelInterface
from src.services.model_factory import ModelFactory
from src.config.model_config import MODEL_SAVE_PATH, LEARNING_RATE

class ModelService:
    """Service for managing face recognition models."""
    
    def __init__(self, model_type: str = 'standard'):
        """
        Initialize the model service.
        
        Args:
            model_type: Type of model to create ('standard' or 'dp')
        """
        self.factory = ModelFactory()
        self.model_type = model_type
        self.model_builder = self.factory.create_model(model_type)
        self.model = None
        self.embedding_model = None
    
    def build_model(self, num_classes: int, training: bool = True) -> tf.keras.Model:
        """
        Build a face recognition model.
        
        Args:
            num_classes: Number of identity classes
            training: Whether the model is for training
            
        Returns:
            Built model
        """
        model = self.model_builder.build(num_classes, training)
        
        # Compile the model for training
        if training:
            model.compile(
                optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=0.9),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
            )
        
        self.model = model
        return model
    
    def get_embedding_model(self) -> Optional[tf.keras.Model]:
        """
        Get a model that extracts face embeddings.
        
        Returns:
            Embedding model or None if main model is not built
        """
        if self.model is None:
            return None
        
        self.embedding_model = self.model_builder.get_embedding_model(self.model)
        return self.embedding_model
    
    def extract_embedding(self, face_img: np.ndarray) -> np.ndarray:
        """
        Extract embedding for a preprocessed face image.
        
        Args:
            face_img: Preprocessed face image
            
        Returns:
            Face embedding vector
        """
        if self.embedding_model is None:
            if self.model is not None:
                self.get_embedding_model()
            else:
                raise ValueError("Model not initialized. Cannot extract embedding.")
        
        # Ensure face_img has batch dimension
        if len(face_img.shape) == 3:
            face_img = np.expand_dims(face_img, axis=0)
        
        # Extract embedding
        embedding = self.embedding_model.predict(face_img)
        
        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def save_model(self, filepath: str = None) -> str:
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model, or None to use default path
            
        Returns:
            Path where the model was saved
        """
        if self.model is None:
            raise ValueError("Model not initialized. Cannot save.")
        
        if filepath is None:
            os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
            filepath = os.path.join(MODEL_SAVE_PATH, f'{self.model_type}_model.h5')
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
        
        return filepath
        
    def load_model(self, filepath: str = None, num_classes: int = None) -> tf.keras.Model:
        """
        Load a model from a file.
        
        Args:
            filepath: Path to load the model from, or None to use default path
            num_classes: Number of classes for the model, required if filepath is None
                         and the default file doesn't exist
            
        Returns:
            Loaded model
        """
        if filepath is None:
            filepath = os.path.join(MODEL_SAVE_PATH, f'{self.model_type}_model.h5')
            
            # If file doesn't exist, build a new model
            if not os.path.exists(filepath):
                if num_classes is None:
                    raise ValueError("num_classes must be provided when default model file doesn't exist")
                
                print(f"Model file {filepath} not found. Building new model.")
                return self.build_model(num_classes)
        
        # Load the model
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        
        return self.model 