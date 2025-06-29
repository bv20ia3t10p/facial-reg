"""
Biometric Model Loader for loading trained models
"""

import os
import logging
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

# Import both model architectures
from ..models.privacy_biometric_model import PrivacyBiometricModel

# Add privacy_biometrics to path for ResNet50 models
import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from privacy_biometrics.models.resnet50 import ResNet50ModelPretrained
    RESNET50_AVAILABLE = True
except ImportError:
    RESNET50_AVAILABLE = False
    ResNet50ModelPretrained = None

logger = logging.getLogger(__name__)

class BiometricModelLoader:
    def __init__(self, client_id: str, device: torch.device):
        self.client_id = client_id
        self.device = device
        self.model_path = self._get_model_path()
        
    def _get_model_path(self) -> Path:
        """Get the model path based on client ID and environment"""
        # Check environment variable first
        model_path_env = os.getenv("MODEL_PATH")
        if model_path_env:
            return Path(model_path_env)
        
        # Default model paths
        if self.client_id == "coordinator" or os.getenv("NODE_TYPE") == "coordinator":
            return Path("/app/models/server_model.pth")
        else:
            return Path(f"/app/models/{self.client_id}_model.pth")
    
    def _detect_model_architecture(self, state_dict: dict) -> str:
        """Detect the model architecture from state dict keys"""
        keys = list(state_dict.keys())
        
        # Check for ResNet50 bottleneck patterns (conv3 is unique to bottleneck blocks)
        resnet50_patterns = [
            'layer1.0.conv3.weight',
            'layer1.0.bn3.weight', 
            'layer2.0.conv3.weight',
            'layer3.0.conv3.weight',
            'layer4.0.conv3.weight',
            'backbone.layer1.0.conv3.weight',  # If wrapped in backbone
        ]
        
        resnet50_score = sum(1 for pattern in resnet50_patterns if any(pattern in key for key in keys))
        
        # Check for torchvision ResNet patterns
        torchvision_patterns = [
            'conv1.weight',
            'bn1.weight',
            'layer1.0.conv1.weight',
            'layer1.0.conv2.weight',
            'layer1.0.conv3.weight',
        ]
        
        torchvision_score = sum(1 for pattern in torchvision_patterns if any(pattern in key for key in keys))
        
        # Check for custom PrivacyBiometricModel patterns
        custom_patterns = [
            'backbone.4.0.conv1.weight',
            'feature_layer.0.weight',
            'identity_head.0.weight',
        ]
        
        custom_score = sum(1 for pattern in custom_patterns if any(pattern in key for key in keys))
        
        logger.info(f"Architecture detection scores - ResNet50: {resnet50_score}, Torchvision: {torchvision_score}, Custom: {custom_score}")
        
        # Determine architecture based on scores
        if resnet50_score >= 2 or torchvision_score >= 3:
            return "resnet50"
        elif custom_score >= 2:
            return "privacy_biometric"
        else:
            # Default to ResNet50 since that's what our models are
            logger.warning("Could not clearly detect architecture, defaulting to resnet50")
            return "resnet50"
    
    def _get_num_identities_from_state_dict(self, state_dict: dict) -> int:
        """Extract number of identities from the state dict"""
        # Look for classifier layer - try different possible names
        classifier_keys = [
            'classifier.weight',
            'identity_head.4.weight', 
            'identity_head.3.weight',
            'fc.weight',
        ]
        
        for key_pattern in classifier_keys:
            for key in state_dict.keys():
                if key_pattern in key:
                    shape = state_dict[key].shape
                    logger.info(f"Found classifier layer {key} with shape {shape}")
                    return shape[0]
        
        # Default fallback
        logger.warning("Could not determine number of identities from state dict, using default 100")
        return 100  # Based on what we saw in the models

    def load_model(self) -> nn.Module:
        """Load the appropriate model based on the saved state dict"""
        try:
            if not self.model_path.exists():
                logger.error(f"Model file not found: {self.model_path}")
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            logger.info(f"Loading model from: {self.model_path}")
            
            # Load state dict
            try:
                state_dict = torch.load(self.model_path, map_location=self.device)
                logger.info(f"Successfully loaded state dict with keys: {list(state_dict.keys())[:10]}...")
            except Exception as e:
                logger.error(f"Failed to load state dict from {self.model_path}: {e}")
                raise
            
            # Handle wrapped state dict
            actual_state_dict = state_dict
            metadata = {}
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                actual_state_dict = state_dict['state_dict']
                metadata = {k: v for k, v in state_dict.items() if k != 'state_dict'}
                logger.info(f"Found wrapped state dict with metadata: {list(metadata.keys())}")
            
            # Detect architecture
            architecture = self._detect_model_architecture(actual_state_dict)
            logger.info(f"Detected model architecture: {architecture}")
            
            # Get number of identities
            num_identities = self._get_num_identities_from_state_dict(actual_state_dict)
            logger.info(f"Detected number of identities: {num_identities}")
            
            # Create model based on detected architecture
            if architecture == "resnet50" and RESNET50_AVAILABLE:
                logger.info("Creating ResNet50ModelPretrained")
                model = ResNet50ModelPretrained(
                    num_identities=num_identities,
                    embedding_dim=512,
                    privacy_enabled=True,
                    pretrained=False  # Don't load ImageNet weights since we're loading our own
                ).to(self.device)
            else:
                logger.info("Creating PrivacyBiometricModel")
                model = PrivacyBiometricModel(
                    num_identities=num_identities,
                    privacy_enabled=True,
                    embedding_dim=512
                ).to(self.device)
            
            # Load weights with more flexible approach
            try:
                # First try strict loading
                model.load_state_dict(actual_state_dict, strict=True)
                logger.info(f"Successfully loaded model weights with strict=True")
            except Exception as e:
                logger.warning(f"Strict loading failed: {e}")
                try:
                    # Try non-strict loading
                    missing_keys, unexpected_keys = model.load_state_dict(actual_state_dict, strict=False)
                    logger.info(f"Successfully loaded model weights with strict=False")
                    
                    if missing_keys:
                        logger.warning(f"Missing keys: {len(missing_keys)} (showing first 5): {missing_keys[:5]}")
                    if unexpected_keys:
                        logger.warning(f"Unexpected keys: {len(unexpected_keys)} (showing first 5): {unexpected_keys[:5]}")
                        
                except Exception as e2:
                    logger.error(f"Non-strict loading also failed: {e2}")
                    # Try to load compatible layers only
                    self._load_compatible_layers(model, actual_state_dict)
            
            # Set model to evaluation mode
            model.eval()
            
            logger.info(f"Model loaded successfully:")
            logger.info(f"  - Architecture: {architecture}")
            logger.info(f"  - Device: {self.device}")
            logger.info(f"  - Num identities: {num_identities}")
            logger.info(f"  - Embedding dim: {getattr(model, 'embedding_dim', 'unknown')}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _load_compatible_layers(self, model: nn.Module, state_dict: dict):
        """Load only compatible layers from state dict"""
        logger.info("Attempting to load compatible layers only")
        
        model_state = model.state_dict()
        compatible_state = {}
        
        for key in model_state.keys():
            if key in state_dict:
                if model_state[key].shape == state_dict[key].shape:
                    compatible_state[key] = state_dict[key]
                else:
                    logger.debug(f"Shape mismatch for {key}: model={model_state[key].shape}, saved={state_dict[key].shape}")
            else:
                logger.debug(f"Key {key} not found in saved state dict")
        
        logger.info(f"Loading {len(compatible_state)}/{len(model_state)} compatible layers")
        model.load_state_dict(compatible_state, strict=False) 