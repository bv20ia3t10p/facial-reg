"""
Privacy-Enabled Biometric Model for Federated Learning
Designed for identity recognition in federated biometric authentication
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
from typing import Dict, Tuple, Any
from torchvision import models

logger = logging.getLogger(__name__)

def get_device():
    """Get the appropriate device for model execution"""
    if torch.cuda.is_available():
        try:
            # Test CUDA with a small tensor operation
            test_tensor = torch.zeros(1, device='cuda')
            test_tensor + 1  # Simple operation to test CUDA
            logger.info("CUDA is available and working")
            return torch.device('cuda')
        except Exception as e:
            logger.warning(f"CUDA test failed: {e}. Falling back to CPU")
            return torch.device('cpu')
    else:
        logger.info("CUDA is not available, using CPU")
        return torch.device('cpu')

class BasicBlock(nn.Module):
    """Basic ResNet block with two 3x3 convolutions"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Downsample if needed
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out

class PrivacyBiometricModel(nn.Module):
    """Privacy-Enabled Biometric Model"""
    def __init__(self, num_identities: int, embedding_dim: int = 512, privacy_enabled: bool = False):
        """
        Initialize the model
        
        Args:
            num_identities: Number of identities for classification
            embedding_dim: Dimension of the embedding vector
            privacy_enabled: Whether to enable privacy features
        """
        super(PrivacyBiometricModel, self).__init__()
        self.num_identities = num_identities
        self.embedding_dim = embedding_dim
        self.privacy_enabled = privacy_enabled
        
        self.backbone = self._create_backbone()
        self.backbone_output_dim = self._get_backbone_output_dim()
        
        # Determine the device
        self.device = self.get_default_device()
        
        # Initialize embedding and identity head
        self.embedding = nn.Sequential(
            nn.Linear(self.backbone_output_dim, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.identity_classifier = nn.Linear(self.embedding_dim, self.num_identities)

        # Move to device
        self.to(self.device)
        self._init_weights()
        
    def _create_backbone(self) -> nn.Module:
        """Creates the backbone model (e.g., MobileNetV3)"""
        # Load a pre-trained MobileNetV3 Large model
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        
        # Remove the original classifier by replacing it with an identity mapping
        model.classifier = nn.Sequential(
            nn.Identity()
        )
        
        return model
        
    def _get_backbone_output_dim(self) -> int:
        """Get the output dimension of the backbone"""
        # Create a dummy tensor and pass it through the backbone
        dummy_input = torch.randn(1, 3, 224, 224).to(self.get_default_device())
        
        # Ensure backbone is on the correct device
        backbone = self._create_backbone().to(self.get_default_device())
        
        with torch.no_grad():
            output = backbone(dummy_input)
            
        return output.shape[1]

    def _init_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def get_default_device(self):
        """Get default device, preferring CUDA if available"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        # Check for Apple Metal Performance Shaders (MPS)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pass input through the backbone
        
        Args:
            x: Input tensor
        
        Returns:
            identity_logits: Logits for identity classification
            embeddings: Extracted feature embeddings
        """
        # Pass input through the backbone
        backbone_features = self.backbone(x)
        
        # Adaptive pooling and flatten
        pooled_features = F.adaptive_avg_pool2d(backbone_features, (1, 1))
        flattened_features = pooled_features.view(pooled_features.size(0), -1)
        
        # Get embeddings
        embeddings = self.embedding(flattened_features)
        
        # Get identity logits
        identity_logits = self.identity_classifier(embeddings)
        
        return identity_logits, embeddings
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the model including architecture and parameters
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "architecture": "PrivacyBiometricModel (MobileNetV3 Large)",
            "num_identities": self.num_identities,
            "embedding_dim": self.embedding_dim,
            "privacy_enabled": self.privacy_enabled,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024)
        }

    def get_feature_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings without classification"""
        try:
            with torch.no_grad():
                # Ensure input is on the correct device
                if not x.is_cuda and self.device.type == 'cuda':
                    x = x.to(self.device)
                
                backbone_features = self.backbone(x)
                features = backbone_features.view(backbone_features.size(0), -1)
                features = self.embedding(features)
            return features
        except RuntimeError as e:
            if "CUDA" in str(e):
                logger.error(f"CUDA error during feature extraction: {e}")
                # Fallback to CPU if CUDA fails
                self.device = torch.device('cpu')
                self.to(self.device)
                logger.info("Model moved to CPU due to CUDA error")
                # Retry on CPU
                return self.get_feature_embeddings(x.to(self.device))
            raise
            
    def expand_for_new_identity(self, new_identity_count: int = 1):
        """Expand model to accommodate new identities"""
        old_num_identities = self.identity_classifier.out_features
        new_num_identities = old_num_identities + new_identity_count
        
        # Create new classifier with expanded output
        new_classifier = nn.Linear(self.embedding_dim, new_num_identities)
        
        # Copy existing weights
        new_classifier.weight.data[:old_num_identities] = self.identity_classifier.weight.data
        new_classifier.bias.data[:old_num_identities] = self.identity_classifier.bias.data
        
        # Initialize new identity weights
        nn.init.xavier_normal_(new_classifier.weight.data[old_num_identities:])
        nn.init.zeros_(new_classifier.bias.data[old_num_identities:])
        
        # Replace the classifier
        self.identity_classifier = new_classifier
        self.num_identities = new_num_identities
        
        print(f"Model expanded from {old_num_identities} to {new_num_identities} identities")
        return self
    
    def get_model_info(self) -> Dict:
        """Get model architecture information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "num_identities": self.num_identities,
            "backbone_output_dim": self.backbone_output_dim,
            "embedding_dim": self.embedding_dim,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "privacy_enabled": self.privacy_enabled,
            "model_size_mb": total_params * 4 / (1024 * 1024)
        } 