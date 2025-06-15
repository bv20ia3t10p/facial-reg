"""
Privacy-Enabled Biometric Model for Federated Learning
Designed for identity recognition in federated biometric authentication
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from typing import Tuple, Optional, Dict
import math

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: Global average pooling
        y = self.squeeze(x).view(b, c)
        # Excitation: FC layers
        y = self.excitation(y).view(b, c, 1, 1)
        # Scale: multiply with original features
        return x * y.expand_as(x)

class ChannelAttention(nn.Module):
    """Channel Attention Module from CBAM"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Average pooling branch
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        # Max pooling branch
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        # Combine and apply sigmoid
        out = avg_out + max_out
        attention = torch.sigmoid(out).view(b, c, 1, 1)
        return x * attention.expand_as(x)

class SpatialAttention(nn.Module):
    """Spatial Attention Module from CBAM"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        
    def forward(self, x):
        # Generate channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and apply convolution
        attention_map = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.conv(attention_map)
        attention_map = torch.sigmoid(attention_map)
        
        return x * attention_map

class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        # Apply channel attention first
        x = self.channel_attention(x)
        # Then apply spatial attention
        x = self.spatial_attention(x)
        return x

class FeatureAttention(nn.Module):
    """Feature-level attention for 1D vectors"""
    def __init__(self, in_dim, reduction=4):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim // reduction, in_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Generate attention weights
        attention_weights = self.attention(x)
        # Apply attention with residual connection
        return x * attention_weights + x

class PrivacyBiometricModel(nn.Module):
    """
    Privacy-compatible biometric model for federated learning
    Supports identity recognition for employee authentication
    """
    
    def __init__(self, 
                 num_identities: int = 50,
                 feature_dim: int = 512,
                 privacy_enabled: bool = True,
                 dropout_rate: float = 0.3):
        super().__init__()
        
        self.num_identities = num_identities
        self.feature_dim = feature_dim
        self.privacy_enabled = privacy_enabled
        
        # Build privacy-compatible backbone (excluding final avgpool and fc)
        backbone = resnet18(pretrained=True)
        backbone = self._manual_batchnorm_to_groupnorm(backbone)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # Remove avgpool and fc
        
        # Add SE block after backbone
        self.se_block = SEBlock(512, reduction=8)
        
        # Add CBAM after SE block
        self.cbam = CBAM(512, reduction=8, kernel_size=7)
        
        # Global average pooling and feature extraction
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # Add feature-level attention
        self.feature_attention = FeatureAttention(feature_dim, reduction=8)
        
        # Identity classification head with attention
        self.identity_classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            FeatureAttention(feature_dim, reduction=8),
            nn.Linear(feature_dim, num_identities)
        )
        
        # Privacy-specific components
        if privacy_enabled:
            self._setup_privacy_components()
        
        # Initialize weights
        self._initialize_weights()
    
    def _manual_batchnorm_to_groupnorm(self, model):
        """Manually convert BatchNorm layers to GroupNorm for privacy compatibility"""
        for name, module in model.named_children():
            if isinstance(module, nn.BatchNorm2d):
                # Replace BatchNorm2d with GroupNorm
                num_groups = min(32, module.num_features)
                group_norm = nn.GroupNorm(num_groups, module.num_features, 
                                        eps=module.eps, affine=module.affine)
                setattr(model, name, group_norm)
            else:
                # Recursively apply to child modules
                self._manual_batchnorm_to_groupnorm(module)
        return model
    
    def _setup_privacy_components(self):
        """Initialize privacy-related tracking variables"""
        self.privacy_budget_used = 0.0
        self.max_privacy_budget = 1.0
        self.noise_multiplier = 1.0
        self.max_grad_norm = 1.0
        self.privacy_accountant = PrivacyAccountant()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, 
                x: torch.Tensor, 
                add_noise: bool = False,
                node_id: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with optional privacy noise"""
        # Extract features using backbone (output is [B, 512, H, W])
        x = self.backbone(x)
        
        # Apply SE block for channel attention
        x = self.se_block(x)
        
        # Apply CBAM for comprehensive attention
        x = self.cbam(x)
        
        # Global average pooling
        x = self.avgpool(x)
        
        # Feature extraction and attention
        features = self.feature_extractor(x)
        features = self.feature_attention(features)
        
        # Add privacy noise during training if enabled
        if add_noise and self.training and self.privacy_enabled:
            noise_std = 0.01
            noise = torch.randn_like(features) * noise_std
            features = features + noise
        
        # Identity classification
        identity_logits = self.identity_classifier(features)
        
        return identity_logits, features
    
    def get_feature_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings without classification"""
        with torch.no_grad():
            backbone_features = self.backbone(x)
            features = self.feature_extractor(backbone_features)
        return features
    
    def expand_for_new_identity(self, new_identity_count: int = 1):
        """Expand model to accommodate new identities"""
        old_num_identities = self.num_identities
        new_num_identities = old_num_identities + new_identity_count
        
        # Get current classifier weights
        old_classifier = self.identity_classifier[-1]
        old_weight = old_classifier.weight.data
        old_bias = old_classifier.bias.data
        
        # Create new classifier with expanded output
        new_classifier = nn.Linear(old_weight.size(1), new_num_identities)
        
        # Copy existing weights
        new_classifier.weight.data[:old_num_identities] = old_weight
        new_classifier.bias.data[:old_num_identities] = old_bias
        
        # Initialize new identity weights
        nn.init.xavier_normal_(new_classifier.weight.data[old_num_identities:])
        nn.init.zeros_(new_classifier.bias.data[old_num_identities:])
        
        # Replace the classifier
        self.identity_classifier[-1] = new_classifier
        self.num_identities = new_num_identities
        
        print(f"Model expanded from {old_num_identities} to {new_num_identities} identities")
        return self
    
    def get_model_info(self) -> Dict:
        """Get model architecture information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "num_identities": self.num_identities,
            "feature_dim": self.feature_dim,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "privacy_enabled": self.privacy_enabled,
            "model_size_mb": total_params * 4 / (1024 * 1024)
        }

class PrivacyAccountant:
    """Track privacy budget consumption during training"""
    
    def __init__(self, max_epsilon: float = 1.0, delta: float = 1e-5):
        self.max_epsilon = max_epsilon
        self.delta = delta
        self.epsilon_used = 0.0
        self.training_steps = 0
    
    def consume_privacy_budget(self, epsilon_step: float):
        """Consume privacy budget for one training step"""
        self.epsilon_used += epsilon_step
        self.training_steps += 1
    
    def get_remaining_budget(self) -> float:
        """Get remaining privacy budget"""
        return max(0.0, self.max_epsilon - self.epsilon_used)
    
    def can_train(self) -> bool:
        """Check if we can continue training within privacy budget"""
        return self.epsilon_used < self.max_epsilon
    
    def get_privacy_stats(self) -> Dict:
        """Get privacy accounting statistics"""
        return {
            "epsilon_used": self.epsilon_used,
            "epsilon_remaining": self.get_remaining_budget(),
            "delta": self.delta,
            "training_steps": self.training_steps,
            "budget_exhausted": not self.can_train()
        }

class FederatedModelManager:
    """Manager for federated model operations"""
    
    def __init__(self, num_identities: int = 300):
        self.num_identities = num_identities
        self.models = {}
        self.global_model = None
    
    def create_node_model(self, node_id: str, privacy_enabled: bool = True):
        """Create model for a specific federated node"""
        model = PrivacyBiometricModel(
            num_identities=self.num_identities,
            privacy_enabled=privacy_enabled
        )
        self.models[node_id] = model
        return model
    
    def create_global_model(self):
        """Create global model for federated aggregation"""
        self.global_model = PrivacyBiometricModel(
            num_identities=self.num_identities,
            privacy_enabled=True
        )
        return self.global_model
    
    def synchronize_models(self, source_model, target_nodes: list):
        """Synchronize model weights across federated nodes"""
        source_state = source_model.state_dict()
        
        for node_id in target_nodes:
            if node_id in self.models:
                self.models[node_id].load_state_dict(source_state)
                print(f"Synchronized model weights to {node_id}")
    
    def get_federated_stats(self) -> Dict:
        """Get statistics about federated models"""
        stats = {
            "num_nodes": len(self.models),
            "global_identities": self.num_identities,
            "node_models": {}
        }
        
        for node_id, model in self.models.items():
            stats["node_models"][node_id] = model.get_model_info()
        
        if self.global_model:
            stats["global_model"] = self.global_model.get_model_info()
        
        return stats 