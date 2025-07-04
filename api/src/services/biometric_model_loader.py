"""
Enhanced Biometric Model Loader with backup support
"""

import os
import logging
import torch
import torch.nn as nn
from pathlib import Path

# Import both model architectures
from ..models.privacy_biometric_model import PrivacyBiometricModel

# ResNet50 model removed to make API independent of privacy_biometrics
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
        
        # Default model paths with backup fallbacks
        if self.client_id == "coordinator" or os.getenv("NODE_TYPE") == "coordinator":
            primary_path = Path("/app/models/server_model.pth")
            backup_paths = [
                Path("/app/models/backups/server_model_backup.pth"),
                Path("/app/models/best_pretrained_model.pth"),
                Path("/app/models/backups/best_pretrained_model_backup.pth")
            ]
        else:
            primary_path = Path(f"/app/models/{self.client_id}_model.pth")
            backup_paths = [
                Path(f"/app/models/backups/{self.client_id}_model_backup.pth"),
                Path("/app/models/server_model.pth"),  # Fallback to server model
                Path("/app/models/best_pretrained_model.pth"),
                Path("/app/models/backups/server_model_backup.pth"),
                Path("/app/models/backups/best_pretrained_model_backup.pth")
            ]
        
        # Return the first model that exists
        for path in [primary_path] + backup_paths:
            if path.exists():
                logger.info(f"Found model at: {path}")
                return path
        
        # If no models found, return the primary path (will cause error with helpful message)
        logger.error(f"No model files found! Tried: {[str(p) for p in [primary_path] + backup_paths]}")
        return primary_path
        
    def _create_backup_models(self):
        """Create backup models if they don't exist"""
        try:
            models_dir = Path("/app/models")
            backups_dir = models_dir / "backups"
            backups_dir.mkdir(exist_ok=True)
            
            # Define model files to backup
            model_files = [
                "server_model.pth",
                "client1_model.pth", 
                "client2_model.pth",
                "best_pretrained_model.pth"
            ]
            
            for model_file in model_files:
                source_path = models_dir / model_file
                backup_path = backups_dir / f"{model_file.replace('.pth', '_backup.pth')}"
                
                if source_path.exists() and not backup_path.exists():
                    import shutil
                    shutil.copy2(source_path, backup_path)
                    logger.info(f"Created backup: {backup_path}")
                    
        except Exception as e:
            logger.warning(f"Could not create model backups: {e}")
    
    def _detect_model_architecture(self, state_dict: dict) -> str:
        """Detect the model architecture from state dict keys"""
        keys = list(state_dict.keys())
        
        # Check for ResNet50ModelPretrained patterns (with backbone wrapper)
        resnet50_pretrained_patterns = [
            'backbone.4.0.conv3.weight',  # ResNet layer1 bottleneck
            'backbone.5.0.conv3.weight',  # ResNet layer2 bottleneck
            'backbone.6.0.conv3.weight',  # ResNet layer3 bottleneck
            'backbone.7.0.conv3.weight',  # ResNet layer4 bottleneck
            'embedding.0.weight',         # Embedding layer
            'classifier.weight',          # Final classifier
        ]
        
        resnet50_score = sum(1 for pattern in resnet50_pretrained_patterns if pattern in keys)
        
        # Check for standard torchvision ResNet patterns (direct, no backbone wrapper)
        torchvision_patterns = [
            'conv1.weight',
            'bn1.weight',
            'layer1.0.conv3.weight',
            'layer2.0.conv3.weight',
            'layer3.0.conv3.weight',
        ]
        
        torchvision_score = sum(1 for pattern in torchvision_patterns if pattern in keys)
        
        # Check for custom PrivacyBiometricModel patterns
        custom_patterns = [
            'feature_layer.0.weight',
            'identity_head.0.weight',
            'identity_head.4.weight',
        ]
        
        custom_score = sum(1 for pattern in custom_patterns if pattern in keys)
        
        logger.info(f"Architecture detection scores - ResNet50: {resnet50_score}, Torchvision: {torchvision_score}, Custom: {custom_score}")
        
        # Determine architecture based on scores
        if resnet50_score >= 3:  # Need at least 3 out of 6 ResNet50ModelPretrained patterns
            return "resnet50"
        elif torchvision_score >= 3:  # Standard torchvision ResNet
            return "resnet50"
        elif custom_score >= 2:  # PrivacyBiometricModel
            return "privacy_biometric"
        else:
            # Default to custom privacy biometric model
            logger.warning("Could not clearly detect architecture, defaulting to privacy_biometric")
            return "privacy_biometric"
    
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
        
        # Default fallback to 300 (from identity_mapping.json)
        logger.warning("Could not determine number of identities from state dict or mapping, using default 300")
        return 300

    def load_model(self) -> nn.Module:
        """Load the appropriate model based on the saved state dict with backup support"""
        try:
            # Create backups if needed
            self._create_backup_models()
            
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
            if architecture == "resnet50":
                logger.info("Creating ResNet50-compatible model")
                # Create a ResNet50-like model for compatibility
                model = self._create_resnet50_compatible_model(num_identities)
            else:
                logger.info("Creating PrivacyBiometricModel")
                model = PrivacyBiometricModel(
                    num_identities=num_identities,
                    privacy_enabled=True,
                    embedding_dim=512
                ).to(self.device)
            
            # Load weights with flexible approach
            try:
                # First try strict loading
                model.load_state_dict(actual_state_dict, strict=True)
                logger.info("Successfully loaded model weights with strict=True")
            except Exception as e:
                logger.warning(f"Strict loading failed: {e}")
                try:
                    # Try non-strict loading
                    missing_keys, unexpected_keys = model.load_state_dict(actual_state_dict, strict=False)
                    logger.info("Successfully loaded model weights with strict=False")
                    
                    if missing_keys:
                        logger.warning(f"Missing keys: {len(missing_keys)} (showing first 5): {missing_keys[:5]}")
                    if unexpected_keys:
                        logger.warning(f"Unexpected keys: {len(unexpected_keys)} (showing first 5): {unexpected_keys[:5]}")
                        
                except Exception as e2:
                    logger.error(f"Non-strict loading also failed: {e2}")
                    # Try to load compatible layers only
                    self._load_compatible_layers(model, actual_state_dict)
            
            # Attach metadata to the model object if it exists
            if metadata:
                for key, value in metadata.items():
                    setattr(model, key, value)
                logger.info(f"Attached metadata to model: {list(metadata.keys())}")

            # Set model to evaluation mode
            model.eval()
            
            logger.info("Model loaded successfully:")
            logger.info(f"  - Architecture: {architecture}")
            logger.info(f"  - Device: {self.device}")
            logger.info(f"  - Num identities: {num_identities}")
            logger.info(f"  - Embedding dim: {getattr(model, 'embedding_dim', 'unknown')}")
            logger.info(f"  - Model path: {self.model_path}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _create_resnet50_compatible_model(self, num_identities: int) -> nn.Module:
        """Create a ResNet50ModelPretrained-compatible model that matches the training architecture"""
        try:
            import torchvision.models as models
            
            class ResNet50ModelPretrainedCompatible(nn.Module):
                """Exact recreation of ResNet50ModelPretrained for weight compatibility"""
                
                def __init__(self, num_identities, embedding_dim=512):
                    super().__init__()
                    self.num_identities = num_identities
                    self.embedding_dim = embedding_dim
                    self.privacy_enabled = True
                    self.noise_scale = 0.1
                    
                    # Load pretrained ResNet50 backbone (same as training)
                    resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
                    
                    # Remove the final classification layer
                    self.backbone_output_dim = resnet50.fc.in_features  # 2048
                    self.backbone = nn.Sequential(*list(resnet50.children())[:-1])  # Remove fc layer
                    
                    # Add adaptive pooling to ensure consistent output size
                    self.backbone.add_module('avgpool', nn.AdaptiveAvgPool2d((1, 1)))
                    
                    # Embedding layers (exact match to training architecture)
                    self.embedding = nn.Sequential(
                        nn.Linear(self.backbone_output_dim, 1024),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.5),
                        nn.Linear(1024, embedding_dim)
                    )
                    
                    # Classification head
                    self.classifier = nn.Linear(embedding_dim, num_identities)
                    
                def forward(self, x, add_noise=False, return_embeddings=True):
                    """Forward pass matching the training architecture"""
                    # Extract features using pretrained backbone
                    x = self.backbone(x)
                    x = torch.flatten(x, 1)
                    
                    # Get embeddings
                    embeddings = self.embedding(x)
                    
                    # Add noise if requested and privacy is enabled
                    if add_noise and self.privacy_enabled:
                        noise = torch.randn_like(embeddings) * self.noise_scale
                        embeddings = embeddings + noise
                    
                    # Get logits
                    logits = self.classifier(embeddings)
                    
                    if return_embeddings:
                        return logits, embeddings
                    return logits, None
                    
                def get_feature_embeddings(self, x):
                    """Extract feature embeddings without classification"""
                    with torch.no_grad():
                        _, embeddings = self.forward(x, add_noise=False)
                    return embeddings
            
            model = ResNet50ModelPretrainedCompatible(num_identities).to(self.device)
            logger.info(f"Created ResNet50ModelPretrained-compatible model with {num_identities} classes")
            return model
            
        except ImportError:
            logger.error("torchvision not available, falling back to PrivacyBiometricModel")
            return PrivacyBiometricModel(
                num_identities=num_identities,
                privacy_enabled=True,
                embedding_dim=512
            ).to(self.device)

    def _load_compatible_layers(self, model: nn.Module, state_dict: dict):
        """Load only compatible layers from state dict"""
        logger.info("Attempting to load compatible layers only")
        
        model_state = model.state_dict()
        compatible_state = {}
        
        for key in model_state.keys():
            if key in state_dict:
                if model_state[key].shape == state_dict[key].shape:
                    compatible_state[key] = state_dict[key]
                    logger.debug(f"Compatible layer: {key} {model_state[key].shape}")
                else:
                    logger.warning(f"Shape mismatch for {key}: model={model_state[key].shape}, state_dict={state_dict[key].shape}")
            else:
                logger.warning(f"Missing key in state_dict: {key}")
        
        if compatible_state:
            model.load_state_dict(compatible_state, strict=False)
            logger.info(f"Loaded {len(compatible_state)} compatible layers out of {len(model_state)} total layers")
        else:
            logger.error("No compatible layers found!")
            raise ValueError("No compatible layers could be loaded from the state dict") 