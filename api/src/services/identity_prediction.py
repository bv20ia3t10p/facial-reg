"""
Identity prediction functionality for biometric service
"""

import logging
import numpy as np
from typing import Dict, Tuple, List, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path

from ..models.privacy_biometric_model import PrivacyBiometricModel
from ..services.mapping_service import MappingService

# Configure logging
logger = logging.getLogger(__name__)

class IdentityPredictionService:
    def __init__(self, model_path: str, client_id: Optional[str] = None):
        self.model_path = Path(model_path)
        self.client_id = client_id or "client1"  # Default to client1
        self.model = None
        self.mapping_service = MappingService()
        self.mapping_service.initialize_mapping()
        # Initialize filtered mapping for the client
        self.mapping_service.get_filtered_mapping_for_client(self.client_id)
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize the model with proper configuration"""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Load state dict
            state_dict = torch.load(self.model_path)
            
            # Get number of identities from final layer
            num_identities = None
            for key in state_dict:
                if 'identity_classifier.weight' in key:
                    num_identities = state_dict[key].shape[0]
                    break
            
            if num_identities is None:
                raise ValueError("Could not determine number of identities from model")
            
            # Create model
            self.model = PrivacyBiometricModel(
                num_identities=num_identities,
                privacy_enabled=False
            )
            
            # Load weights
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
            logger.info(f"Successfully loaded model with {num_identities} identities")
            
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise
    
    def predict_identity(self, 
                        embeddings: torch.Tensor,
                        top_k: int = 5,
                        confidence_threshold: float = 0.5) -> List[Dict[str, float]]:
        """
        Predict identity from embeddings
        
        Args:
            embeddings: Tensor of shape (batch_size, embedding_dim)
            top_k: Number of top predictions to return
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            List of dictionaries containing predictions and confidences
        """
        try:
            if self.model is None:
                raise ValueError("Model not initialized")
            
            # Get predictions
            with torch.no_grad():
                logits = self.model.identity_classifier(embeddings)
                probs = torch.nn.functional.softmax(logits, dim=1)
            
            # Get top k predictions
            top_probs, top_indices = torch.topk(probs, k=min(top_k, probs.shape[1]))
            
            # Convert to list of predictions
            predictions = []
            for i in range(top_probs.shape[1]):
                prob = float(top_probs[0, i])
                if prob < confidence_threshold:
                    continue
                    
                idx = int(top_indices[0, i])
                try:
                    # Get original class ID using filtered mapping
                    class_id = self.mapping_service.get_identity_by_model_class(idx, use_filtered=True)
                    predictions.append({
                        'predicted_id': class_id,
                        'confidence': prob,
                        'model_index': idx
                    })
                except Exception as e:
                    logger.error(f"Error mapping index {idx}: {e}")
                    continue
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in predict_identity: {e}")
            raise
    
    def verify_identity(self,
                       embeddings: torch.Tensor,
                       claimed_id: str,
                       confidence_threshold: float = 0.5) -> Tuple[bool, float]:
        """
        Verify if embeddings match claimed identity
        
        Args:
            embeddings: Tensor of shape (batch_size, embedding_dim)
            claimed_id: Claimed identity to verify against
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            Tuple of (is_match, confidence)
        """
        try:
            if self.model is None:
                raise ValueError("Model not initialized")
            
            # Get model index for claimed ID using filtered mapping
            try:
                target_idx = self.mapping_service.get_model_class_by_identity(claimed_id, use_filtered=True)
            except Exception as e:
                logger.error(f"Error getting model index for ID {claimed_id}: {e}")
                return False, 0.0
            
            # Get predictions
            with torch.no_grad():
                logits = self.model.identity_classifier(embeddings)
                probs = torch.nn.functional.softmax(logits, dim=1)
            
            # Get probability for claimed identity
            confidence = float(probs[0, target_idx])
            is_match = confidence >= confidence_threshold
            
            # Log verification attempt
            logger.info(f"Verification attempt for ID {claimed_id}:")
            logger.info(f"  Target index: {target_idx}")
            logger.info(f"  Confidence: {confidence:.4f}")
            logger.info(f"  Result: {'✓' if is_match else '✗'}")
            
            # If not a match, log top prediction
            if not is_match:
                top_prob, top_idx = torch.max(probs, dim=1)
                try:
                    top_id = self.mapping_service.get_identity_by_model_class(int(top_idx), use_filtered=True)
                    logger.warning(f"  Top prediction was ID {top_id} with confidence {float(top_prob):.4f}")
                except Exception as e:
                    logger.error(f"Error getting class ID for index {top_idx}: {e}")
            
            return is_match, confidence
            
        except Exception as e:
            logger.error(f"Error in verify_identity: {e}")
            raise

class IdentityPredictor:
    """Handles user identity prediction and verification"""
    
    def __init__(self, model: nn.Module, mapping_service, feature_extractor, device: torch.device, client_id: str = "client1"):
        """
        Initialize identity predictor
        
        Args:
            model: The biometric model
            mapping_service: Service to map class indices to user IDs
            feature_extractor: Feature extraction component
            device: Torch device for computations
            client_id: Client identifier for filtered mapping
        """
        self.model = model
        self.mapping_service = mapping_service
        self.feature_extractor = feature_extractor
        self.device = device
        self.client_id = client_id
        
        # Get the feature dimension from the model
        self.feature_dim = getattr(model, 'embedding_dim', 512)
        self.backbone_output_dim = getattr(model, 'backbone_output_dim', 512)
        
        # Initialize mapping cache
        self.mapping_cache = None
        self.mapping_version = None
        
        # Load initial mapping
        self._refresh_mapping_cache()
        
        logger.info(f"Identity predictor initialized with feature_dim={self.feature_dim}, backbone_output_dim={self.backbone_output_dim}, client_id={self.client_id}")
    
    def _refresh_mapping_cache(self):
        """Refresh the mapping cache with filtered mapping for federated models"""
        try:
            self.mapping_service.initialize_mapping()
            # Create filtered mapping that matches training logic
            self.mapping_cache = self.mapping_service.get_filtered_mapping_for_client(self.client_id)
            self.mapping_version = datetime.now()
            logger.info(f"Filtered mapping cache refreshed with {len(self.mapping_cache)} entries for {self.client_id}")
        except Exception as e:
            logger.error(f"Failed to refresh filtered mapping cache: {e}")
    
    def identify_user(self, image_tensor: torch.Tensor) -> Tuple[str, float, List[float]]:
        """
        Identify user from image tensor using feature extraction
        
        Args:
            image_tensor: Preprocessed image tensor
            
        Returns:
            Tuple of (user_id, confidence, features)
        """
        try:
            # Extract features
            features = self.feature_extractor.extract_features(image_tensor)
            
            # Get identity prediction
            with torch.no_grad():
                assert callable(self.model.identity_classifier)
                identity_logits = self.model.identity_classifier(features)
                probabilities = F.softmax(identity_logits, dim=1)[0]
                
                # Get predicted class and confidence
                confidence, predicted_idx = torch.max(probabilities, 0)
                class_index = predicted_idx.item()
                
                # Map class index to user ID using filtered mapping
                user_id = self.mapping_service.get_identity_by_model_class(class_index, use_filtered=True)
                
                # Convert features to list
                feature_list = features[0].cpu().numpy().tolist()
                
                return user_id, confidence.item(), feature_list
                
        except Exception as e:
            logger.error(f"Error in identify_user: {e}")
            return "unknown", 0.0, []
    
    def identify_user_raw(self, image_tensor: torch.Tensor) -> Tuple[str, float, List[float]]:
        """
        Identify user from raw image tensor without feature extraction
        
        Args:
            image_tensor: Preprocessed image tensor
            
        Returns:
            Tuple of (user_id, confidence, features)
        """
        try:
            # Forward pass through model
            with torch.no_grad():
                identity_logits, features = self.model(image_tensor)
                probabilities = F.softmax(identity_logits, dim=1)[0]
                
                # Get predicted class and confidence
                confidence, predicted_idx = torch.max(probabilities, 0)
                class_index = predicted_idx.item()
                
                # Map class index to user ID using filtered mapping
                user_id = self.mapping_service.get_identity_by_model_class(class_index, use_filtered=True)
                
                # Convert features to list
                feature_list = features[0].cpu().numpy().tolist()
                
                return user_id, confidence.item(), feature_list
                
        except Exception as e:
            logger.error(f"Error in identify_user_raw: {e}")
            return "unknown", 0.0, []
    
    def verify_user(self, image_tensor: torch.Tensor, claimed_user_id: str) -> Tuple[bool, float]:
        """
        Verify if the image matches the claimed user ID
        
        Args:
            image_tensor: Input tensor of shape (batch_size, C, H, W)
            claimed_user_id: The claimed user ID to verify against
            
        Returns:
            Tuple of (is_match, confidence)
        """
        try:
            # Get prediction
            predicted_user_id, confidence, _ = self.identify_user(image_tensor)
            
            # Check if prediction matches claimed ID
            is_match = predicted_user_id == claimed_user_id
            
            return is_match, confidence
            
        except Exception as e:
            logger.error(f"Error during user verification: {e}")
            return False, 0.0
    
    def get_top_predictions(self, image_tensor: torch.Tensor, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Get top-k predictions for an image
        
        Args:
            image_tensor: Input tensor of shape (batch_size, C, H, W)
            top_k: Number of top predictions to return
            
        Returns:
            List of dictionaries containing prediction information
        """
        try:
            # Forward pass through model
            with torch.no_grad():
                identity_logits, features = self.model(image_tensor)
                probabilities = F.softmax(identity_logits, dim=1)[0]
                
                # Get top-k predictions
                top_confidences, top_indices = torch.topk(probabilities, min(top_k, len(probabilities)))
                
                # Format results
                results = []
                for i, (conf, idx) in enumerate(zip(top_confidences, top_indices)):
                    class_idx = idx.item()
                    user_id = self.mapping_service.get_identity_by_model_class(class_idx, use_filtered=True)
                    
                    results.append({
                        "rank": i+1,
                        "class_index": class_idx,
                        "user_id": user_id,
                        "confidence": conf.item()
                    })
                
                # Log prediction details with more information
                logger.info(f"Top {len(results)} predictions:")
                for pred in results:
                    logger.info(
                        f"  Rank {pred['rank']}: "
                        f"Model idx {pred['class_index']} -> "
                        f"User {pred['user_id']} "
                        f"(confidence: {pred['confidence']:.4f})"
                    )
                
                return results
                
        except Exception as e:
            logger.error(f"Error getting top predictions: {e}")
            return [] 