"""
Privacy-Enabled Biometric Model for Federated Learning
Designed for identity recognition in federated biometric authentication
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
from typing import Dict

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
    def __init__(self, num_identities, privacy_enabled=False, embedding_dim=512):
        super().__init__()
        self.device = get_device()
        logger.info(f"Initializing model on device: {self.device}")
        
        # Initial layers
        self.backbone = nn.Sequential(
            # Initial conv layer
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Layer 4 (64 -> 64)
            nn.Sequential(
                BasicBlock(64, 64),  # 4.0
                BasicBlock(64, 64)   # 4.1
            ),
            
            # Layer 5 (64 -> 128)
            nn.Sequential(
                BasicBlock(64, 128, stride=2),  # 5.0
                BasicBlock(128, 128)            # 5.1
            ),
            
            # Layer 6 (128 -> 256)
            nn.Sequential(
                BasicBlock(128, 256, stride=2),  # 6.0
                BasicBlock(256, 256)             # 6.1
            ),
            
            # Layer 7 (256 -> 512)
            nn.Sequential(
                BasicBlock(256, 512, stride=2),  # 7.0
                BasicBlock(512, 512)             # 7.1
            ),
            
            # Final pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Move backbone to device first
        self.backbone = self.backbone.to(self.device)
        
        # Freeze early layers to prevent overfitting
        layers_to_freeze = 6  # Freeze first 6 layers
        for i, (name, param) in enumerate(self.backbone.named_parameters()):
            if i < layers_to_freeze:
                param.requires_grad = False
        
        # Output dimension is 512 (matches ResNet18/34)
        self.backbone_output_dim = 512
        self.embedding_dim = embedding_dim
        
        # Add custom layers for biometric features
        self.feature_layer = nn.Sequential(
            nn.Linear(self.backbone_output_dim, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        ).to(self.device)
        
        # Identity classification head
        self.identity_head = nn.Sequential(
            nn.Linear(self.embedding_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_identities)
        ).to(self.device)
        
        self.privacy_enabled = privacy_enabled
        self.num_identities = num_identities
        
        # Initialize weights properly
        self._initialize_weights()
        
        # Set model to evaluation mode by default
        self.eval()
        
        logger.info(f"Model initialization complete with backbone output dim: {self.backbone_output_dim}, embedding dim: {self.embedding_dim}")
    
    def _initialize_weights(self):
        """Initialize model weights for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, add_noise=False, node_id=None):
        try:
            # Ensure input is on the correct device
            if not x.is_cuda and self.device.type == 'cuda':
                x = x.to(self.device)
                logger.debug(f"Moved input tensor to {self.device}")
            
            # Extract features through backbone
            features = self.backbone(x)
            features = features.view(features.size(0), -1)
            
            # Apply feature extraction
            features = self.feature_layer(features)
            
            # Add privacy noise if enabled
            if self.privacy_enabled and add_noise:
                noise_scale = 0.1
                noise = torch.randn_like(features, device=self.device) * noise_scale
                features = features + noise
            
            # Identity classification
            identity_logits = self.identity_head(features)
            
            return identity_logits, features
            
        except RuntimeError as e:
            if "CUDA" in str(e):
                logger.error(f"CUDA error during forward pass: {e}")
                # Fallback to CPU if CUDA fails
                self.device = torch.device('cpu')
                self.to(self.device)
                logger.info("Model moved to CPU due to CUDA error")
                # Retry forward pass on CPU
                return self.forward(x.to(self.device), add_noise, node_id)
            raise
    
    def get_feature_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings without classification"""
        try:
            with torch.no_grad():
                # Ensure input is on the correct device
                if not x.is_cuda and self.device.type == 'cuda':
                    x = x.to(self.device)
                
                backbone_features = self.backbone(x)
                features = backbone_features.view(backbone_features.size(0), -1)
                features = self.feature_layer(features)
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
        old_num_identities = self.identity_head[-1].out_features
        new_num_identities = old_num_identities + new_identity_count
        
        # Create new classifier with expanded output
        new_classifier = nn.Linear(256, new_num_identities)
        
        # Copy existing weights
        new_classifier.weight.data[:old_num_identities] = self.identity_head[-1].weight.data
        new_classifier.bias.data[:old_num_identities] = self.identity_head[-1].bias.data
        
        # Initialize new identity weights
        nn.init.xavier_normal_(new_classifier.weight.data[old_num_identities:])
        nn.init.zeros_(new_classifier.bias.data[old_num_identities:])
        
        # Replace the classifier
        self.identity_head[-1] = new_classifier
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