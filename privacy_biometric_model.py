"""
Privacy-Enabled Biometric Model for Federated Learning
Designed for 300 identities (100 per node) with emotion detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from typing import Tuple, Optional, Dict
import math

class PrivacyBiometricModel(nn.Module):
    """
    Privacy-compatible biometric model for federated learning
    Supports both identity recognition and emotion detection
    """
    
    def __init__(self, 
                 num_identities: int = 300,
                 num_emotions: int = 7,
                 feature_dim: int = 512,
                 privacy_enabled: bool = True,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.num_identities = num_identities
        self.num_emotions = num_emotions
        self.feature_dim = feature_dim
        self.privacy_enabled = privacy_enabled
        
        # Build privacy-compatible backbone
        self.backbone = self._build_privacy_backbone()
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(2048, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # Identity classification head
        self.identity_classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, num_identities)
        )
        
        # Emotion classification head
        self.emotion_classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.LayerNorm(feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 4, num_emotions)
        )
        
        # Privacy-specific components
        if privacy_enabled:
            self._setup_privacy_components()
        
        # Initialize weights
        self._initialize_weights()
    
    def _build_privacy_backbone(self):
        """Build ResNet50 backbone compatible with differential privacy"""
        backbone = resnet50(pretrained=True)
        
        # Convert BatchNorm to GroupNorm for privacy compatibility
        # Note: Opacus 1.5.4 doesn't have convert_batchnorm_modules, so we use manual conversion
        backbone = self._manual_batchnorm_to_groupnorm(backbone)
        print("✅ Successfully converted BatchNorm to GroupNorm for privacy compatibility")
        
        # Remove the final classification layer
        backbone.fc = nn.Identity()
        
        return backbone
    
    def _manual_batchnorm_to_groupnorm(self, model):
        """Manually convert BatchNorm layers to GroupNorm for privacy compatibility"""
        for name, module in model.named_children():
            if isinstance(module, nn.BatchNorm2d):
                # Replace BatchNorm2d with GroupNorm
                num_groups = min(32, module.num_features)  # Use 32 groups or fewer if needed
                group_norm = nn.GroupNorm(num_groups, module.num_features, 
                                        eps=module.eps, affine=module.affine)
                setattr(model, name, group_norm)
            elif isinstance(module, nn.BatchNorm1d):
                # Replace BatchNorm1d with LayerNorm for 1D case
                layer_norm = nn.LayerNorm(module.num_features, eps=module.eps)
                setattr(model, name, layer_norm)
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
                node_id: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with optional privacy noise"""
        # Extract features using backbone
        backbone_features = self.backbone(x)
        
        # Feature extraction
        features = self.feature_extractor(backbone_features)
        
        # Add privacy noise during training if enabled
        if add_noise and self.training and self.privacy_enabled:
            noise_std = 0.01
            noise = torch.randn_like(features) * noise_std
            features = features + noise
        
        # Identity classification
        identity_logits = self.identity_classifier(features)
        
        # Emotion classification
        emotion_logits = self.emotion_classifier(features)
        
        return identity_logits, emotion_logits, features
    
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
            "num_emotions": self.num_emotions,
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