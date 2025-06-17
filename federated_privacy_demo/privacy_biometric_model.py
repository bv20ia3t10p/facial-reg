"""
Privacy-Enabled Biometric Model for Federated Learning
Designed for identity recognition in federated biometric authentication
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from typing import Tuple, Optional, Dict, List
import math
import logging

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
    """Privacy-Enabled Biometric Model"""
    def __init__(self, num_identities, privacy_enabled=False):
        super().__init__()
        self.device = get_device()
        logger.info(f"Initializing model on device: {self.device}")
        
        # Load pretrained ResNet18 without final layer
        self.backbone = resnet18(weights='IMAGENET1K_V1')
        
        # Move backbone to device first
        self.backbone = self.backbone.to(self.device)
        
        # Freeze early layers to prevent overfitting
        layers_to_freeze = 6  # Freeze first 6 layers
        for i, (name, param) in enumerate(self.backbone.named_parameters()):
            if i < layers_to_freeze:
                param.requires_grad = False
        
        # Remove the final fully connected layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Add custom layers for biometric features
        self.feature_layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)  # Add dropout for regularization
        ).to(self.device)
        
        # Identity classification head
        self.identity_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_identities)
        ).to(self.device)
        
        self.privacy_enabled = privacy_enabled
        
        # Initialize weights properly
        self._initialize_weights()
        
        # Set model to evaluation mode by default
        self.eval()
        
        logger.info("Model initialization complete")
    
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
                features = self.feature_layer(backbone_features.view(backbone_features.size(0), -1))
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
        new_classifier = nn.Linear(old_num_identities, new_num_identities)
        
        # Copy existing weights
        new_classifier.weight.data[:old_num_identities] = self.identity_head[-1].weight.data
        new_classifier.bias.data[:old_num_identities] = self.identity_head[-1].bias.data
        
        # Initialize new identity weights
        nn.init.xavier_normal_(new_classifier.weight.data[old_num_identities:])
        nn.init.zeros_(new_classifier.bias.data[old_num_identities:])
        
        # Replace the classifier
        self.identity_head[-1] = new_classifier
        self.identity_head[-2].out_features = new_num_identities
        
        print(f"Model expanded from {old_num_identities} to {new_num_identities} identities")
        return self
    
    def get_model_info(self) -> Dict:
        """Get model architecture information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "num_identities": self.identity_head[-1].out_features,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "privacy_enabled": self.privacy_enabled,
            "model_size_mb": total_params * 4 / (1024 * 1024)
        }

class PrivacyAccountant:
    """Tracks privacy budget consumption for differential privacy"""
    
    def __init__(self, max_epsilon: float = 1.0, delta: float = 1e-5, noise_multiplier: float = 1.0):
        """Initialize privacy accountant
        
        Args:
            max_epsilon: Maximum privacy budget (ε)
            delta: Privacy parameter (δ)
            noise_multiplier: Noise multiplier (σ) for DP-SGD
        """
        self.max_epsilon = max_epsilon
        self.delta = delta
        self.noise_multiplier = noise_multiplier
        self.epsilon_used = 0.0
        self.steps_taken = 0
        self.privacy_history = []
        logger.info(f"Initialized privacy accountant with ε={max_epsilon}, δ={delta}, σ={noise_multiplier}")
    
    def is_budget_exhausted(self) -> bool:
        """Check if privacy budget has been exhausted
        
        Returns:
            True if privacy budget is exhausted, False otherwise
        """
        return self.epsilon_used >= self.max_epsilon
    
    def apply_dp(self, loss: torch.Tensor, parameters: List[torch.Tensor]) -> torch.Tensor:
        """Apply differential privacy to gradients
        
        Args:
            loss: Loss tensor
            parameters: Model parameters
            
        Returns:
            Modified loss tensor with privacy guarantees
        """
        if self.is_budget_exhausted():
            logger.warning("Privacy budget exhausted, skipping DP application")
            return loss
        
        # Add noise to gradients
        for param in parameters:
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * self.noise_multiplier
                param.grad.add_(noise)
        
        return loss
    
    def consume_privacy_budget(self, epsilon_step: float):
        """Consume privacy budget
        
        Args:
            epsilon_step: Amount of privacy budget to consume
        """
        self.epsilon_used += epsilon_step
        self.steps_taken += 1
        self.privacy_history.append({
            'step': self.steps_taken,
            'epsilon_used': self.epsilon_used,
            'step_epsilon': epsilon_step
        })
    
    def get_remaining_budget(self) -> float:
        """Get remaining privacy budget
        
        Returns:
            Remaining privacy budget (ε)
        """
        return max(0, self.max_epsilon - self.epsilon_used)
    
    def can_train(self) -> bool:
        """Check if training can proceed
        
        Returns:
            True if there is remaining privacy budget, False otherwise
        """
        return not self.is_budget_exhausted()
    
    def get_privacy_stats(self) -> Dict:
        """Get current privacy statistics
        
        Returns:
            Dictionary containing privacy metrics
        """
        return {
            'epsilon_used': self.epsilon_used,
            'epsilon_remaining': self.get_remaining_budget(),
            'delta': self.delta,
            'noise_multiplier': self.noise_multiplier,
            'steps_taken': self.steps_taken,
            'budget_exhausted': self.is_budget_exhausted()
        }
    
    def reset(self):
        """Reset privacy accountant"""
        self.epsilon_used = 0.0
        self.steps_taken = 0
        self.privacy_history = []
        logger.info("Privacy accountant reset")
    
    def get_privacy_history(self) -> List[Dict[str, float]]:
        """Get history of privacy budget consumption
        
        Returns:
            List of dictionaries containing privacy metrics for each step
        """
        return self.privacy_history

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