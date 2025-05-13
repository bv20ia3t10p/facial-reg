"""
Interface for face detection following the Interface Segregation Principle.
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Dict, Any

class BoundingBox:
    """Represents a face bounding box with confidence score."""
    
    def __init__(self, x: int, y: int, width: int, height: int, confidence: float):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.confidence = confidence
    
    def __repr__(self) -> str:
        return f"BoundingBox(x={self.x}, y={self.y}, width={self.width}, height={self.height}, confidence={self.confidence:.2f})"

class FaceDetectionInterface(ABC):
    """Interface for face detection operations."""
    
    @abstractmethod
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image as a NumPy array
            
        Returns:
            List of detected faces with bounding boxes and landmarks
        """
        pass
    
    @abstractmethod
    def extract_face(self, image: np.ndarray, face_info: Dict[str, Any], 
                    target_size: Tuple[int, int] = (112, 112)) -> np.ndarray:
        """
        Extract and align a face from an image.
        
        Args:
            image: Input image as a NumPy array
            face_info: Face information with bounding box and landmarks
            target_size: Target size for the extracted face
            
        Returns:
            Extracted and aligned face as a NumPy array
        """
        pass
    
    @abstractmethod
    def extract_all_faces(self, image: np.ndarray, 
                         target_size: Tuple[int, int] = (112, 112), 
                         min_confidence: float = 0.9) -> List[np.ndarray]:
        """
        Detect and extract all faces from an image.
        
        Args:
            image: Input image as a NumPy array
            target_size: Target size for the extracted faces
            min_confidence: Minimum confidence threshold for detection
            
        Returns:
            List of extracted and aligned faces
        """
        pass 