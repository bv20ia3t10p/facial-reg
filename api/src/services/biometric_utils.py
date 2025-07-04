"""
Utility functions for biometric service
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Callable
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Configure logging
logger = logging.getLogger(__name__)

class BiometricUtilsMixin:
    """Utility methods for biometric service"""
    # Declare attributes from BiometricService to inform the linter
    client_id: str
    identity_to_index: Dict[str, int]
    index_to_identity: Dict[int, str]
    device: torch.device
    model: nn.Module
    initialized: bool
    preprocess_image: Callable[[bytes], torch.Tensor]

    def debug_mapping(self):
        """Print detailed mapping information for debugging"""
        logger.info("============ BIOMETRIC SERVICE MAPPING DEBUG ============")
        logger.info(f"Client ID: {self.client_id}")
        
        # Get current mapping
        mapping = self.index_to_identity
        logger.info(f"Total users in mapping: {len(mapping)}")
        
        # Print sample of mapping entries
        sample_count = min(10, len(mapping))
        logger.info(f"Sample mapping (first {sample_count} entries):")
        for class_idx, user_id in list(mapping.items())[:sample_count]:
            logger.info(f"  Class {class_idx} -> User {user_id}")
            
        # Check for folder structure
        partitioned_path = Path("/app/data/partitioned")
        client_dir = partitioned_path / self.client_id
        if client_dir.exists():
            logger.info(f"Checking folder structure at: {client_dir}")
            folders = sorted([d for d in client_dir.iterdir() 
                          if d.is_dir() and d.name.isdigit()], 
                          key=lambda x: int(x.name))
            
            logger.info(f"Found {len(folders)} folders in {client_dir}")
            
            # Check first few folders
            sample_folders = folders[:10]
            logger.info("Sample folders:")
            for folder in sample_folders:
                logger.info(f"  {folder.name}")
        else:
            logger.warning(f"Client directory not found: {client_dir}")
            
        logger.info("====================================================")
    
    def get_model_info(self):
        """Get information about the loaded model for debugging"""
        try:
            info = {
                "client_id": self.client_id,
                "device": str(self.device),
                "model_type": type(self.model).__name__,
                "initialized": self.initialized,
                "mapping_available": hasattr(self, 'index_to_identity') and bool(self.index_to_identity),
            }
            
            # Add mapping information if available
            if hasattr(self, 'index_to_identity'):
                mapping = self.index_to_identity
                info["mapping_count"] = len(mapping)
                info["mapping_sample"] = {k: mapping[k] for k in list(mapping.keys())[:5]} if mapping else {}
            
            # Try to get model path from environment
            model_path = os.environ.get("MODEL_PATH", "unknown")
            info["model_path"] = model_path
            
            # Check if model file exists
            if model_path != "unknown":
                file_exists = Path(model_path).exists()
                info["model_file_exists"] = file_exists
                if file_exists:
                    # Get file size
                    info["model_file_size"] = Path(model_path).stat().st_size
                    # Get last modified time
                    info["model_last_modified"] = datetime.fromtimestamp(
                        Path(model_path).stat().st_mtime).isoformat()
            
            # Add model attributes
            for attr in ['num_identities', 'embedding_dim', 'backbone_output_dim']:
                if hasattr(self.model, attr):
                    info[attr] = getattr(self.model, attr)
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {"error": str(e)}
    
    def get_top_predictions(self, image_tensor=None, image_data=None, top_k=5):
        """Get top-k predictions for an image"""
        try:
            # If image_tensor is not provided, preprocess image_data
            if image_tensor is None and image_data is not None:
                image_tensor = self.preprocess_image(image_data)
            elif image_tensor is None and image_data is None:
                return {"error": "No image data or tensor provided"}
                
            # Get predictions using identity predictor
            with torch.no_grad():
                # Forward pass through model to get identity logits and features
                identity_logits, features = self.model(image_tensor)
                
                # Get probabilities
                probabilities = F.softmax(identity_logits, dim=1)[0]
                
                # Get top k predictions
                top_k = min(top_k, len(probabilities))
                top_confidences, top_indices = torch.topk(probabilities, top_k)
                
                # Get current mapping
                mapping = self.index_to_identity
                
                # Format results
                top_predictions = []
                for i, (conf, idx) in enumerate(zip(top_confidences, top_indices)):
                    class_idx = idx.item()
                    user_id = mapping.get(int(class_idx), f"unknown_{class_idx}")
                    top_predictions.append({
                        "rank": i+1,
                        "class_index": class_idx,
                        "user_id": user_id,
                        "confidence": conf.item(),
                    })
                
                return top_predictions
        except Exception as e:
            logger.error(f"Error getting top predictions: {e}")
            return [] 