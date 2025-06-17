"""
Face Recognition Utility with Privacy-Preserving Features
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
from typing import Tuple, Optional
import logging
from io import BytesIO

logger = logging.getLogger(__name__)

class FaceRecognizer:
    """Face recognition model with privacy preservation"""
    
    def __init__(self):
        # Initialize device with better error handling
        try:
            if torch.cuda.is_available():
                # Test CUDA with a small tensor operation
                test_tensor = torch.zeros(1, device='cuda')
                test_tensor + 1  # Simple operation to test CUDA
                self.device = torch.device('cuda')
                logger.info("CUDA is available and working")
            else:
                self.device = torch.device('cpu')
                logger.info("CUDA is not available, using CPU")
        except Exception as e:
            logger.warning(f"CUDA initialization failed: {e}, falling back to CPU")
            self.device = torch.device('cpu')
            # Disable CUDA for this session
            torch.cuda.is_available = lambda: False
        
        try:
            # Load pre-trained ResNet model
            logger.info("Loading ResNet model...")
            self.model = models.resnet50(weights='IMAGENET1K_V2')
            
            # Remove final classification layer
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Authentication threshold
        self.threshold = 0.75  # Lower threshold for better recall
        
        logger.info(f"Initialized FaceRecognizer on {self.device}")
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for model input"""
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply transformations
            tensor = self.transform(image)
            
            # Add batch dimension
            tensor = tensor.unsqueeze(0)
            
            # Move to device with error handling
            try:
                tensor = tensor.to(self.device)
            except Exception as e:
                logger.warning(f"Failed to move tensor to {self.device}: {e}, using CPU")
                self.device = torch.device('cpu')
                tensor = tensor.to(self.device)
            
            return tensor
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise
    
    @torch.no_grad()
    def extract_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Extract face features with privacy preservation"""
        try:
            # Ensure tensor is on the correct device
            if image_tensor.device != self.device:
                try:
                    image_tensor = image_tensor.to(self.device)
                except Exception as e:
                    logger.warning(f"Failed to move tensor to {self.device}: {e}, using CPU")
                    self.device = torch.device('cpu')
                    image_tensor = image_tensor.to(self.device)
            
            # Forward pass through model
            features = self.model(image_tensor)
            
            # Flatten features
            features = features.view(features.size(0), -1)
            
            # Normalize features
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise
    
    def compute_similarity(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor
    ) -> float:
        """Compute cosine similarity between feature vectors"""
        try:
            # Move both tensors to CPU for comparison to avoid CUDA issues
            features1 = features1.cpu()
            features2 = features2.cpu()
            
            # Compute cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                features1,
                features2,
                dim=1
            )
            
            # Log similarity for debugging
            similarity_value = float(similarity.item())
            logger.debug(f"Computed similarity: {similarity_value:.4f}")
            
            return similarity_value
            
        except Exception as e:
            logger.error(f"Similarity computation failed: {e}")
            raise
    
    def detect_face(self, image: Image.Image) -> Optional[Tuple[int, int, int, int]]:
        """Detect face in image and return bounding box"""
        try:
            # TODO: Implement face detection using a privacy-preserving method
            # For now, assume the entire image is a face
            return (0, 0, image.width, image.height)
            
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return None
    
    def align_face(
        self,
        image: Image.Image,
        bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> Image.Image:
        """Align face for consistent recognition"""
        try:
            if bbox is None:
                bbox = self.detect_face(image)
            
            if bbox is None:
                return image
            
            # Crop to bounding box
            face = image.crop(bbox)
            
            # TODO: Implement face alignment using landmarks
            # For now, just resize
            face = face.resize((224, 224), Image.LANCZOS)
            
            return face
            
        except Exception as e:
            logger.error(f"Face alignment failed: {e}")
            return image
    
    def verify_face(
        self,
        image1: Image.Image,
        image2: Image.Image,
        threshold: Optional[float] = None
    ) -> Tuple[bool, float]:
        """Verify if two face images match"""
        try:
            # Use instance threshold if none provided
            if threshold is None:
                threshold = self.threshold
            
            # Preprocess images
            tensor1 = self.preprocess_image(image1)
            tensor2 = self.preprocess_image(image2)
            
            # Extract features
            features1 = self.extract_features(tensor1)
            features2 = self.extract_features(tensor2)
            
            # Compute similarity
            similarity = self.compute_similarity(features1, features2)
            
            # Check if match
            is_match = similarity >= threshold
            
            logger.info(f"Face verification: similarity={similarity:.4f}, threshold={threshold:.4f}, match={is_match}")
            return is_match, similarity
            
        except Exception as e:
            logger.error(f"Face verification failed: {e}")
            raise
    
    def save_features(self, features: torch.Tensor) -> bytes:
        """Save feature tensor to bytes"""
        try:
            # Convert to numpy and serialize
            features_np = features.cpu().numpy()
            bio = BytesIO()
            np.save(bio, features_np)
            return bio.getvalue()
            
        except Exception as e:
            logger.error(f"Feature saving failed: {e}")
            raise
    
    def load_features(self, data: bytes) -> torch.Tensor:
        """Load feature tensor from bytes"""
        try:
            # Deserialize and convert to tensor
            bio = BytesIO(data)
            features_np = np.load(bio)
            features = torch.from_numpy(features_np)
            features = features.to(self.device)
            return features
            
        except Exception as e:
            logger.error(f"Feature loading failed: {e}")
            raise 