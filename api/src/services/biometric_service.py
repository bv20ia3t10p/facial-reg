"""
Biometric Service for facial recognition and verification
"""

import os
import io
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime
import httpx
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy.orm import Session

from ..db.database import User
from .biometric_model_loader import BiometricModelLoader
from .identity_prediction import IdentityPredictor
from .biometric_utils import BiometricUtilsMixin
from ..utils.mapping_utils import create_client_specific_mapping

# Configure logging
logger = logging.getLogger(__name__)

class BiometricService(BiometricUtilsMixin):
    """Service for biometric recognition and verification"""
    
    def __init__(self, client_id: str = "client1"):
        """Initialize biometric service"""
        self.client_id = client_id
        self.initialized = False
        self.init_error = None
        self.identity_to_index = {}
        self.index_to_identity = {}

        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using {'GPU' if self.device.type == 'cuda' else 'CPU'}")
            
            # 1. Load the model
            self.model_loader = BiometricModelLoader(self.client_id, self.device)
            self.model = self.model_loader.load_model()
            
            # 2. Fetch full mapping and create client-specific map
            full_mapping_data = self._fetch_mapping_from_server()
            if not full_mapping_data:
                raise RuntimeError("Could not fetch mapping data from coordinator.")

            mapping_result = create_client_specific_mapping(full_mapping_data, self.client_id)
            if not mapping_result:
                raise RuntimeError(f"Could not create specific mapping for client {self.client_id}.")
            
            self.identity_to_index, self.index_to_identity = mapping_result

            # 3. Initialize identity predictor with the direct maps
            self.identity_predictor = IdentityPredictor(
                model=self.model, 
                identity_to_index=self.identity_to_index,
                index_to_identity=self.index_to_identity,
                device=self.device
            )
            
            # 4. Initialize image transforms
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            self.initialized = True
            logger.info("BiometricService initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize BiometricService: {e}", exc_info=True)
            self.init_error = str(e)

    def _fetch_mapping_from_server(self) -> Optional[Dict[str, Any]]:
        """Fetches the complete mapping data from the coordinator server."""
        server_url = os.getenv("SERVER_URL", "http://fl-coordinator:8080")
        try:
            with httpx.Client() as client:
                response = client.get(f"{server_url}/api/mapping", timeout=10)
            response.raise_for_status()
            logger.info("Successfully fetched full mapping data from coordinator.")
            return response.json()
        except httpx.RequestError as e:
            logger.error(f"Could not connect to coordinator at {server_url}: {e}")
        except Exception as e:
            logger.error(f"Error processing mapping response from coordinator: {e}")
        return None

    def reload_model(self) -> bool:
        """Reload the model from disk"""
        try:
            logger.info("Reloading biometric model...")
            self.model = self.model_loader.load_model()
            # Re-initialize the predictor as the model has changed
            self.identity_predictor = IdentityPredictor(
                model=self.model,
                identity_to_index=self.identity_to_index,
                index_to_identity=self.index_to_identity,
                device=self.device
            )
            logger.info("Biometric model reloaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to reload biometric model: {e}")
            return False
    
    def preprocess_image(self, image_data: bytes) -> torch.Tensor:
        """Preprocess image for model input"""
        try:
            img = Image.open(io.BytesIO(image_data))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0)
            return img_tensor.to(self.device)
        except Exception as e:
            logger.error(f"Failed to preprocess image: {e}")
            raise ValueError(f"Failed to process image: {e}")

    def predict_identity(self, image_data: bytes, db: Optional[Session] = None) -> Dict[str, Any]:
        """
        Predict identity from an image.
        """
        if not self.initialized:
            logger.error(f"BiometricService not initialized. Error: {self.init_error}")
            return {'success': False, 'error': 'BiometricService not initialized'}
            
        try:
            self.debug_input_image(image_data)
            image_tensor = self.preprocess_image(image_data)
            
            user_id, confidence, features = self.identity_predictor.identify_user_raw(image_tensor)
            logger.info(f"Model prediction: user_id={user_id}, confidence={confidence:.4f}")
            
            # Optional: Validate prediction against database if session is provided
            if db and user_id:
                user_exists = db.query(User).filter(User.id == user_id).first()
                if not user_exists:
                    logger.warning(f"Predicted user '{user_id}' not found in database.")
            
            return {
                'success': True,
                'user_id': user_id,
                'confidence': confidence,
                'features': features
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}", exc_info=True)
            return {'success': False, 'error': str(e), 'user_id': 'unknown'}
            
    def verify_user(self, image_data: bytes, claimed_user_id: str) -> Dict[str, Any]:
        """
        Verify if the image matches the claimed user ID.
        """
        if not self.initialized:
            return {'success': False, 'error': 'BiometricService not initialized'}

        try:
            image_tensor = self.preprocess_image(image_data)
            is_match, confidence = self.identity_predictor.verify_user(image_tensor, claimed_user_id)
            return {
                'success': True,
                'is_match': is_match,
                'confidence': confidence
            }
        except Exception as e:
            logger.error(f"Verification error for user '{claimed_user_id}': {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        if not self.initialized:
            return {"error": "Service not initialized."}
        
        # Extract info directly from model attributes, with fallbacks
        return {
            "model_path": str(self.model_loader.model_path),
            "num_identities": getattr(self.model, "num_identities", "N/A"),
            "embedding_dim": getattr(self.model, "embedding_dim", getattr(self.model, "feature_dim", "N/A")),
            "backbone_output_dim": getattr(self.model, "backbone_output_dim", "N/A"),
            "device": self.device.type,
            "initialization_error": self.init_error,
            "mapping_version": getattr(self.model, "mapping_version", "N/A"),
            "mapping_hash": getattr(self.model, "mapping_hash", "N/A"),
        }
    
    def debug_input_image(self, image_data: bytes) -> None:
        """
        Debug input image by saving it and logging its properties
        
        Args:
            image_data: Raw image data in bytes
        """
        try:
            # Create debug directory if it doesn't exist
            debug_dir = Path("/app/logs/debug_images")
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            # Read image
            img = Image.open(io.BytesIO(image_data))
            
            # Log image properties
            logger.info(f"Debug image info: format={img.format}, size={img.size}, mode={img.mode}")
            
            # Save original image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            original_path = debug_dir / f"original_{timestamp}.png"
            img.save(original_path)
            logger.info(f"Original image saved to {original_path}")
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
                logger.info("Converted image to RGB mode")
            
            # Apply transforms to see what the model will see
            img_tensor = self.transform(img).unsqueeze(0)
            logger.info(f"Transformed tensor shape: {img_tensor.shape}")
            
            # Convert tensor back to image for visualization
            # Denormalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_denorm = img_tensor[0] * std + mean
            
            # Convert to numpy and transpose for PIL
            img_np = img_denorm.numpy().transpose(1, 2, 0)
            img_np = np.clip(img_np, 0, 1)
            
            # Save processed image
            processed_img = Image.fromarray((img_np * 255).astype(np.uint8))
            processed_path = debug_dir / f"processed_{timestamp}.png"
            processed_img.save(processed_path)
            logger.info(f"Processed image saved to {processed_path}")
            
            # Save image statistics
            pixel_mean = img_tensor.mean().item()
            pixel_std = img_tensor.std().item()
            pixel_min = img_tensor.min().item()
            pixel_max = img_tensor.max().item()
            
            logger.info(f"Image statistics: mean={pixel_mean:.4f}, std={pixel_std:.4f}, min={pixel_min:.4f}, max={pixel_max:.4f}")
            
            # Save histogram data
            hist_path = debug_dir / f"histogram_{timestamp}.png"
            plt.figure(figsize=(10, 4))
            for i, color in enumerate(['r', 'g', 'b']):
                plt.hist(img_tensor[0, i].numpy().flatten(), bins=50, color=color, alpha=0.5)
            plt.title("RGB Histogram of Normalized Image")
            plt.savefig(hist_path)
            plt.close()
            logger.info(f"Histogram saved to {hist_path}")
            
            # Debug feature extraction process
            logger.info("Debugging feature extraction process...")
            try:
                # The FeatureExtractor is no longer used, so this part is removed.
                # The IdentityPredictor now handles feature extraction directly.
                # We can still get top predictions for the debug image.
                top_preds = self.get_top_predictions(image_tensor=img_tensor, top_k=5)
                logger.info(f"Top 5 predictions for debug image: {top_preds}")
            except Exception as feat_err:
                logger.error(f"Error in feature extraction debugging: {feat_err}", exc_info=True)
            
        except Exception as e:
            logger.error(f"Error debugging input image: {e}", exc_info=True) 