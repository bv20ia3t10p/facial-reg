"""
Factory service for creating face recognition models.
"""
from typing import Dict, Type

from src.interfaces.model_interface import FaceModelInterface
from src.models.standard_face_model import StandardFaceModel
from src.models.dp_face_model import DPFaceModel

class ModelFactory:
    """Factory for creating face recognition models."""
    
    def __init__(self):
        self._models: Dict[str, Type[FaceModelInterface]] = {}
        
        # Register default models
        self.register_model('standard', StandardFaceModel)
        self.register_model('dp', DPFaceModel)
    
    def register_model(self, name: str, model_class: Type[FaceModelInterface]) -> None:
        """
        Register a new model type with the factory.
        
        Args:
            name: Name to register the model under
            model_class: Class implementing FaceModelInterface
        """
        self._models[name] = model_class
    
    def create_model(self, name: str) -> FaceModelInterface:
        """
        Create a new model instance by name.
        
        Args:
            name: Name of the registered model to create
            
        Returns:
            Instance of the requested model
            
        Raises:
            KeyError: If model name is not registered
        """
        if name not in self._models:
            raise KeyError(f"Model '{name}' is not registered")
        
        return self._models[name]() 