"""
Identity prediction functionality for biometric service
"""

import logging
from typing import Dict, Tuple, List, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Configure logging
logger = logging.getLogger(__name__)

class IdentityPredictor:
    """Handles user identity prediction and verification using direct mappings."""

    def __init__(
        self,
        model: nn.Module,
        identity_to_index: Dict[str, int],
        index_to_identity: Dict[int, str],
        device: torch.device,
    ):
        """
        Initialize identity predictor.

        Args:
            model: The biometric model.
            identity_to_index: Client-specific mapping from user ID to model index.
            index_to_identity: Client-specific reverse mapping.
            device: Torch device for computations.
        """
        self.model = model
        self.identity_to_index = identity_to_index
        self.index_to_identity = index_to_identity
        self.device = device
        self.feature_dim = getattr(model, "embedding_dim", 512)
        
        logger.info(
            f"IdentityPredictor initialized with {len(self.identity_to_index)} identities."
        )

    def identify_user_raw(self, image_tensor: torch.Tensor) -> Tuple[Optional[str], float, List[float]]:
        """
        Identifies a user from a raw image tensor.

        Returns:
            A tuple of (user_id, confidence, features). User ID is None if not found.
        """
        try:
            with torch.no_grad():
                identity_logits, features = self.model(image_tensor)
                probabilities = F.softmax(identity_logits, dim=1)[0]
                confidence, predicted_idx = torch.max(probabilities, 0)
                class_index = predicted_idx.item()
                
                # Map the predicted model index back to a user ID
                user_id = self.index_to_identity.get(int(class_index))
                
                if user_id is None:
                    logger.warning(
                        f"Predicted class_index {class_index} not found in client's mapping."
                    )

                return user_id, confidence.item(), features.cpu().numpy().flatten().tolist()
        except Exception as e:
            logger.error(f"Error during raw user identification: {e}", exc_info=True)
            return None, 0.0, []

    def verify_user(self, image_tensor: torch.Tensor, claimed_user_id: str) -> Tuple[bool, float]:
        """
        Verifies if the image tensor matches the claimed user ID.
        """
        target_idx = self.identity_to_index.get(claimed_user_id)
        if target_idx is None:
            logger.warning(f"Claimed user '{claimed_user_id}' not found in mapping.")
            return False, 0.0

        try:
            with torch.no_grad():
                identity_logits, _ = self.model(image_tensor)
                probabilities = F.softmax(identity_logits, dim=1)[0]
                confidence = probabilities[target_idx].item()
                is_match = confidence >= 0.5  # Using a default 0.5 threshold
                return is_match, confidence
        except Exception as e:
            logger.error(f"Error during user verification for '{claimed_user_id}': {e}")
            return False, 0.0

    def get_top_predictions(self, image_tensor: torch.Tensor, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Gets the top K predictions for a given image tensor.
        """
        predictions = []
        try:
            with torch.no_grad():
                identity_logits, _ = self.model(image_tensor)
                probabilities = F.softmax(identity_logits, dim=1)[0]
                top_probs, top_indices = torch.topk(probabilities, k=min(top_k, len(self.index_to_identity)))

                for i in range(top_probs.size(0)):
                    confidence = top_probs[i].item()
                    class_index = top_indices[i].item()
                    user_id = self.index_to_identity.get(int(class_index))
                    
                    if user_id:
                        predictions.append({
                            "user_id": user_id,
                            "confidence": confidence,
                            "rank": i + 1,
                        })
            return predictions
        except Exception as e:
            logger.error(f"Error getting top predictions: {e}")
            return [] 