"""
Biometric Service for facial recognition and verification
"""

import os
import io
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

from ..models.privacy_biometric_model import PrivacyBiometricModel

# Configure logging
logger = logging.getLogger(__name__)

class BiometricService:
    """Service for biometric recognition and verification"""
    
    def __init__(self, client_id: str = "client1"):
        """
        Initialize biometric service
        
        Args:
            client_id: Client identifier for model selection
        """
        self.client_id = client_id
        self.user_id_mapping: Dict[int, str] = {}
        
        # Get device configuration
        if torch.cuda.is_available() and os.getenv('CUDA_VISIBLE_DEVICES') is not None:
            self.device = torch.device('cuda')
            logger.info("Biometric service using GPU for computations")
        else:
            self.device = torch.device('cpu')
            logger.info("Biometric service using CPU for computations")
        
        # Load model
        self.model = self._load_model()
        
        # Load user ID mapping
        self._load_user_id_mapping()
        
        # Initialize transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _load_model(self) -> nn.Module:
        """Load the biometric model"""
        try:
            model_path = Path(f"/app/models/best_{self.client_id}_pretrained_model.pth")
            logger.info(f"Loading model from: {model_path}")
            
            if not model_path.exists():
                logger.error(f"Model file not found at: {model_path}")
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Load state dict
            state_dict = torch.load(model_path, map_location=self.device)
            
            # Extract model parameters
            num_identities = 300  # Default
            input_feature_dim = 512  # Default
            
            # Check if state_dict is a dictionary with metadata
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                # New format with metadata
                num_identities = state_dict.get('num_identities', 300)
                input_feature_dim = state_dict.get('feature_dim', 512)
                logger.info(f"Loaded model metadata: {num_identities} identities, feature_dim={input_feature_dim}")
                
                # Get the actual state dict
                model_state = state_dict['state_dict']
            elif isinstance(state_dict, dict) and any(k.startswith('feature_layer') for k in state_dict.keys()):
                # Old format without metadata
                model_state = state_dict
                
                # Try to determine feature dimension from the state dict
                for key, value in state_dict.items():
                    if 'feature_layer.0.weight' in key:
                        input_feature_dim = value.shape[1]
                        logger.info(f"Detected input feature dimension from state dict: {input_feature_dim}")
                        break
                        
                # Try to load mapping file for num_identities
                mapping_path = Path("/app/models/user_id_mapping.json")
                if mapping_path.exists():
                    try:
                        with open(mapping_path, 'r') as f:
                            mapping = json.load(f)
                            num_identities = len(mapping)
                            logger.info(f"Found {num_identities} identities in mapping file")
                    except Exception as e:
                        logger.warning(f"Failed to load user mapping: {e}, using default")
            else:
                # Unknown format
                logger.warning("Unknown model state format, using defaults")
                model_state = state_dict
            
            # Initialize model with correct parameters
            logger.info(f"Initializing model with {num_identities} identities and input_feature_dim={input_feature_dim}")
            model = PrivacyBiometricModel(
                num_identities=num_identities,
                privacy_enabled=True,
                embedding_dim=input_feature_dim
            ).to(self.device)
            
            # Load state dict
            try:
                model.load_state_dict(model_state)
            except Exception as e:
                logger.error(f"Error loading model state dict: {e}")
                logger.warning("Model state dict could not be loaded, using untrained model")
            
            model.eval()
            logger.info(f"Model loaded successfully on {self.device}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def reload_model(self) -> bool:
        """Reload the model from disk"""
        try:
            logger.info("Reloading biometric model...")
            self.model = self._load_model()
            # Reload user ID mapping as well
            self._load_user_id_mapping()
            logger.info("Biometric model reloaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to reload biometric model: {e}")
            return False
    
    def _load_user_id_mapping(self):
        """Load mapping between model indices and actual user IDs"""
        try:
            # First try to load from user_id_mapping.json if it exists
            mapping_path = Path("/app/models/user_id_mapping.json")
            if mapping_path.exists():
                logger.info(f"Loading user ID mapping from: {mapping_path}")
                with open(mapping_path, 'r') as f:
                    self.user_id_mapping = {int(k): v for k, v in json.load(f).items()}
                logger.info(f"Loaded {len(self.user_id_mapping)} user ID mappings from file")
                return
                
            # Fall back to directory-based mapping
            data_path = Path("/app/data")
            logger.info(f"Looking for user data in: {data_path}")
            
            if not data_path.exists():
                logger.error(f"Data directory not found at: {data_path}")
                raise FileNotFoundError(f"Data directory not found: {data_path}")
            
            user_folders = sorted([d for d in data_path.iterdir() 
                                if d.is_dir() and d.name.isdigit()])
            
            self.user_id_mapping = {i: folder.name for i, folder in enumerate(user_folders)}
            logger.info(f"Loaded {len(self.user_id_mapping)} user ID mappings from directories")
            
        except Exception as e:
            logger.error(f"Failed to load user ID mapping: {e}")
            raise
    
    def get_user_id_from_index(self, index: int) -> str:
        """Get user ID from model index"""
        if index not in self.user_id_mapping:
            logger.warning(f"Index {index} not found in user mapping with {len(self.user_id_mapping)} entries")
            # If we have a mapping, return the first user as fallback
            if self.user_id_mapping:
                fallback_id = list(self.user_id_mapping.values())[0]
                logger.warning(f"Using fallback user ID: {fallback_id}")
                return fallback_id
            # Otherwise generate a temporary ID
            return f"user_{int(datetime.utcnow().timestamp())}"
        return self.user_id_mapping[index]
    
    def preprocess_image(self, image_data: bytes) -> torch.Tensor:
        """Preprocess image for model input"""
        try:
            # Read image
            img = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Transform and add batch dimension
            img_tensor = self.transform(img).unsqueeze(0)
            return img_tensor.to(self.device)
        except Exception as e:
            logger.error(f"Failed to preprocess image: {str(e)}")
            raise ValueError(f"Failed to process image: {str(e)}")
    
    def identify_user(self, image_tensor: torch.Tensor) -> Tuple[str, float, Dict]:
        """
        Identify a user from an image
        
        Args:
            image_tensor: Preprocessed image tensor
            
        Returns:
            Tuple of (user_id, confidence, feature_embedding)
        """
        try:
            with torch.no_grad():
                # Forward pass through model
                identity_logits, embedding = self.model(image_tensor)
                
                # Get prediction
                probabilities = F.softmax(identity_logits, dim=1)[0]
                confidence, predicted_idx = torch.max(probabilities, dim=0)
                
                # Get user ID
                user_id = self.get_user_id_from_index(predicted_idx.item())
                
                # Feature embedding for verification
                features = embedding.cpu().numpy().tolist()
                
                return user_id, confidence.item(), features
                
        except Exception as e:
            logger.error(f"Error in identify_user: {e}")
            raise
    
    def verify_user(self, image_tensor: torch.Tensor, claimed_user_id: str) -> Tuple[bool, float]:
        """
        Verify if the person in the image matches the claimed identity
        
        Args:
            image_tensor: Preprocessed image tensor
            claimed_user_id: User ID to verify against
            
        Returns:
            Tuple of (is_match, confidence)
        """
        try:
            # Find the index for this user ID
            user_indices = [idx for idx, uid in self.user_id_mapping.items() if uid == claimed_user_id]
            
            if not user_indices:
                logger.warning(f"User ID {claimed_user_id} not found in mapping")
                return False, 0.0
            
            user_idx = user_indices[0]
            
            with torch.no_grad():
                # Forward pass
                identity_logits, _ = self.model(image_tensor)
                
                # Get probability for the claimed identity
                probabilities = F.softmax(identity_logits, dim=1)[0]
                confidence = probabilities[user_idx].item()
                
                # Verification threshold
                threshold = 0.7  # Could be configurable
                is_match = confidence >= threshold
                
                return is_match, confidence
                
        except Exception as e:
            logger.error(f"Error in verify_user: {e}")
            raise 