"""
Feature extraction for biometric models
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Optional

# Configure logging
logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Handles feature extraction from input images"""
    
    def __init__(self, model: nn.Module, device: torch.device):
        """
        Initialize feature extractor
        
        Args:
            model: The biometric model to use for feature extraction
            device: Torch device for computations
        """
        self.model = model
        self.device = device
        self._feature_extractor = None  # Lazy-loaded feature extractor
        
        # Get the feature dimension from the model
        self.feature_dim = getattr(model, 'embedding_dim', 512)  # Default to 512 for new architecture
        logger.info(f"Feature extractor initialized with feature_dim={self.feature_dim}")
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input tensor"""
        try:
            with torch.no_grad():
                # Ensure input is on the correct device
                if not x.is_cuda and self.device.type == 'cuda':
                    x = x.to(self.device)
                
                # Get model outputs - this will give us identity_logits and features
                _, features = self.model(x)
                
                # Log shapes for debugging
                logger.info(f"Extracted features shape: {features.shape}")
                
                return features
                
        except RuntimeError as e:
            if "CUDA" in str(e):
                logger.error(f"CUDA error during feature extraction: {e}")
                # Fallback to CPU if CUDA fails
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.model = self.model.to(self.device)
                logger.info("Model moved to CPU due to CUDA error")
                # Retry on CPU
                return self.extract_features(x.to(self.device))
            raise
    
    def get_feature_dim(self) -> int:
        """Get the feature dimension"""
        return self.feature_dim
    
    def visualize_features(self, features: torch.Tensor, save_path: Optional[str] = None):
        """Visualize feature embeddings (for debugging)"""
        try:
            # Convert features to numpy
            features_np = features.cpu().numpy()
            
            # Create heatmap
            plt.figure(figsize=(10, 5))
            plt.imshow(features_np.T, aspect='auto', cmap='viridis')
            plt.colorbar()
            plt.title('Feature Embeddings Heatmap')
            plt.xlabel('Sample')
            plt.ylabel('Feature Dimension')
            
            # Save or show
            if save_path:
                debug_dir = Path("/app/logs/debug_features")
                debug_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = debug_dir / f"features_{timestamp}.png"
                plt.savefig(output_path)
                logger.info(f"Feature visualization saved to {output_path}")
            else:
                plt.close()
            
        except Exception as e:
            logger.error(f"Error visualizing features: {e}")
            plt.close()

    def debug_feature_extraction(self, image_tensor):
        """
        Debug feature extraction process with visualizations and detailed logs
        
        Args:
            image_tensor: Input image tensor (B, C, H, W)
            
        Returns:
            Dictionary with debug information
        """
        try:
            debug_dir = Path("/app/logs/debug_features")
            debug_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Log input tensor shape
            logger.info(f"Input tensor shape: {image_tensor.shape}")
            
            # Extract features with gradient tracking for visualization
            self.model.eval()
            image_tensor.requires_grad_(True)
            
            # Get backbone features
            assert callable(self.model.backbone)
            backbone_features = self.model.backbone(image_tensor)
            logger.info(f"Backbone output shape: {backbone_features.shape}")
            
            # Get pooled features if applicable
            if hasattr(self.model, 'adaptive_pool'):
                assert callable(self.model.adaptive_pool)
                pooled_features = self.model.adaptive_pool(backbone_features)
                logger.info(f"Pooled features shape: {pooled_features.shape}")
            else:
                pooled_features = F.adaptive_avg_pool2d(backbone_features, (1, 1))
                logger.info(f"Default pooled features shape: {pooled_features.shape}")
            
            # Get flattened features
            flattened = pooled_features.view(pooled_features.size(0), -1)
            logger.info(f"Flattened features shape: {flattened.shape}")
            
            # Get final embedding if applicable
            if hasattr(self.model, 'embedding'):
                assert callable(self.model.embedding)
                embedding = self.model.embedding(flattened)
                logger.info(f"Final embedding shape: {embedding.shape}")
            else:
                embedding = flattened
            
            # Save feature statistics
            stats = {
                "backbone_shape": backbone_features.shape,
                "flattened_shape": flattened.shape,
                "embedding_shape": embedding.shape,
                "backbone_mean": backbone_features.mean().item(),
                "backbone_std": backbone_features.std().item(),
                "backbone_min": backbone_features.min().item(),
                "backbone_max": backbone_features.max().item(),
                "embedding_mean": embedding.mean().item(),
                "embedding_std": embedding.std().item(),
                "embedding_min": embedding.min().item(),
                "embedding_max": embedding.max().item(),
            }
            
            logger.info(f"Feature statistics: {stats}")
            
            # Visualize feature maps (first 16 channels)
            try:
                # Take first sample's feature maps
                feature_maps = backbone_features[0].detach().cpu()
                num_maps = min(16, feature_maps.size(0))
                
                plt.figure(figsize=(12, 12))
                for i in range(num_maps):
                    plt.subplot(4, 4, i+1)
                    plt.imshow(feature_maps[i].numpy(), cmap='viridis')
                    plt.axis('off')
                    plt.title(f'Map {i+1}')
                
                feature_map_path = debug_dir / f"feature_maps_{timestamp}.png"
                plt.tight_layout()
                plt.savefig(feature_map_path)
                plt.close()
                logger.info(f"Feature maps visualization saved to {feature_map_path}")
            except Exception as vis_err:
                logger.error(f"Error visualizing feature maps: {vis_err}")
            
            # Visualize embedding distribution
            try:
                plt.figure(figsize=(10, 6))
                plt.hist(embedding[0].detach().cpu().numpy(), bins=50)
                plt.title("Embedding Distribution")
                plt.xlabel("Value")
                plt.ylabel("Frequency")
                
                embedding_hist_path = debug_dir / f"embedding_hist_{timestamp}.png"
                plt.savefig(embedding_hist_path)
                plt.close()
                logger.info(f"Embedding histogram saved to {embedding_hist_path}")
            except Exception as hist_err:
                logger.error(f"Error creating embedding histogram: {hist_err}")
            
            # Compute gradients for feature importance
            try:
                # Get prediction
                assert callable(self.model.identity_classifier)
                identity_logits = self.model.identity_classifier(embedding)
                
                # Get the predicted class
                pred_class = torch.argmax(identity_logits, dim=1).item()
                
                # Compute gradients with respect to the predicted class
                self.model.zero_grad()
                identity_logits[0, pred_class].backward()
                
                # Get gradients at the input
                input_gradients = image_tensor.grad[0].detach().cpu()
                
                # Visualize input gradients (saliency map)
                saliency_map = torch.abs(input_gradients).sum(dim=0)
                saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
                
                plt.figure(figsize=(6, 6))
                plt.imshow(saliency_map.numpy(), cmap='hot')
                plt.colorbar()
                plt.title("Saliency Map")
                plt.axis('off')
                
                saliency_path = debug_dir / f"saliency_{timestamp}.png"
                plt.savefig(saliency_path)
                plt.close()
                logger.info(f"Saliency map saved to {saliency_path}")
            except Exception as grad_err:
                logger.error(f"Error computing gradients: {grad_err}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Feature extraction debug error: {e}", exc_info=True)
            return {"error": str(e)} 