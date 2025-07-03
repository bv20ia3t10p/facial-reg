"""
Biometric Service for facial recognition and verification
"""

import os
import io
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
from datetime import datetime
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn

from .mapping_service import MappingService
from ..models.privacy_biometric_model import PrivacyBiometricModel
from ..db.database import User

# Import components from split files
from .biometric_model_loader import BiometricModelLoader
from .feature_extraction import FeatureExtractor
from .identity_prediction import IdentityPredictor
from .biometric_utils import BiometricUtilsMixin

# Configure logging
logger = logging.getLogger(__name__)

class BiometricService(BiometricUtilsMixin):
    """Service for biometric recognition and verification"""
    
    def __init__(self, client_id: str = "client1"):
        """
        Initialize biometric service
        
        Args:
            client_id: Client identifier for model selection
        """
        self.client_id = client_id
        self.initialized = False
        
        try:
            # Configure device
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using {'GPU' if self.device.type == 'cuda' else 'CPU'} for computations")
        
            # Load model
            self.model_loader = BiometricModelLoader(self.client_id, self.device)
            self.model = self.model_loader.load_model()
            
            # Initialize mapping service and set up filtered mapping for client
            self.mapping_service = MappingService()
            # Initialize the filtered mapping for this specific client
            self.mapping_service.get_filtered_mapping_for_client(self.client_id)
            
            # Initialize feature extractor
            self.feature_extractor = FeatureExtractor(self.model, self.device)
            
            # Initialize identity predictor with client_id for filtered mapping
            self.identity_predictor = IdentityPredictor(
                self.model, 
                self.mapping_service,
                self.feature_extractor,
                self.device,
                client_id=self.client_id
            )
            
            # Initialize transforms
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            # Mark initialization as successful
            self.initialized = True
            logger.info("BiometricService initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize BiometricService: {e}", exc_info=True)
            # Keep a reference to the error for diagnosis
            self.init_error = str(e)
    
    def reload_model(self) -> bool:
        """Reload the model from disk"""
        try:
            logger.info("Reloading biometric model...")
            self.model = self.model_loader.load_model()
            logger.info("Biometric model reloaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to reload biometric model: {e}")
            return False
    
    def update_model(self, new_model: nn.Module):
        """Update the service with a new model instance."""
        try:
            logger.info("Updating biometric service with a new model...")
            self.model = new_model
            # Re-initialize components that depend on the model
            self.feature_extractor = FeatureExtractor(self.model, self.device)
            self.identity_predictor = IdentityPredictor(
                self.model, 
                self.mapping_service,
                self.feature_extractor,
                self.device,
                client_id=self.client_id
            )
            logger.info("Biometric service updated successfully with new model.")
            return True
        except Exception as e:
            logger.error(f"Failed to update biometric model: {e}", exc_info=True)
            return False
    
    def preprocess_image(self, image_data: bytes) -> torch.Tensor:
        """Preprocess image for model input"""
        try:
            # Read image
            img = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Transform and add batch dimension
            img_tensor = self.transform(img).unsqueeze(0)
            return img_tensor.to(self.device)
        except Exception as e:
            logger.error(f"Failed to preprocess image: {str(e)}")
            raise ValueError(f"Failed to process image: {str(e)}")
    
    def initialize_mapping_from_directories(self) -> bool:
        """
        Initialize mapping by fetching from the server
        
        Note: This method no longer generates mappings locally.
        Clients now rely on the server for mapping information.
        """
        try:
            # Fetch the mapping from server
            success = self.mapping_service.initialize_mapping()
            
            if success:
                logger.info("Successfully fetched mapping from server")
                return True
            else:
                logger.warning("Failed to fetch mapping from server")
                return False
            
        except Exception as e:
            logger.error(f"Failed to initialize mapping: {e}")
            return False
            
    def predict_identity(self, image_data: bytes, db=None, email=None) -> Dict[str, Any]:
        """
        Predict identity from an image
        
        Args:
            image_data: Raw image data
            db: Database session (optional)
            email: User email for verification (optional)
            
        Returns:
            Dict containing prediction results
        """
        try:
            # Check if service is initialized
            if not getattr(self, 'initialized', False):
                logger.error("BiometricService not properly initialized")
                return {
                    'success': False,
                    'error': 'BiometricService not initialized',
                    'user_id': 'unknown',
                    'confidence': 0.0,
                    'features': []
                }
            
            # Debug input image
            self.debug_input_image(image_data)
            
            # Sync user mapping with database if provided
            if db is not None:
                self.sync_user_mapping_with_db(db)
            
            # Preprocess image
            image_tensor = self.preprocess_image(image_data)
            
            # Use raw image prediction
            logger.info("Using raw image for prediction")
            user_id, confidence, features = self.identity_predictor.identify_user_raw(image_tensor)
            
            # Log the original model prediction before any database validation
            logger.info(f"Model's original prediction: user_id={user_id}, confidence={confidence:.4f}")
            
            # Validate prediction against database if available
            if db is not None:
                validated_user_id = self._validate_against_database(db, user_id, email)
                if validated_user_id != user_id:
                    logger.info(f"User ID changed after database validation: {user_id} -> {validated_user_id}")
                    user_id = validated_user_id
            
            # Check if user is in any client mapping
            try:
                # Check if this user ID is in our current mapping
                self.mapping_service.initialize_mapping()  # Refresh mapping
                current_mapping = self.mapping_service.get_mapping()
                found_in_mapping = user_id in current_mapping.values()
                
                if found_in_mapping:
                    logger.info(f"User {user_id} found in current mapping")
                else:
                    logger.warning(f"User {user_id} not found in current mapping, might need synchronization")
            except Exception as mapping_err:
                logger.error(f"Error checking mappings: {mapping_err}")
            
            return {
                'success': True,
                'user_id': user_id,
                'confidence': confidence,
                'features': features
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'user_id': 'unknown',
                'confidence': 0.0,
                'features': []
            }
            
    def predict_identity_with_features(self, image_data: bytes, db=None, email=None) -> Dict[str, Any]:
        """
        Predict identity from an image using feature extraction
        
        Args:
            image_data: Raw image data
            db: Database session (optional)
            email: User email for verification (optional)
            
        Returns:
            Dict containing prediction results
        """
        try:
            # Check if service is initialized
            if not getattr(self, 'initialized', False):
                logger.error("BiometricService not properly initialized")
                return {
                    'success': False,
                    'error': 'BiometricService not initialized',
                    'user_id': 'unknown',
                    'confidence': 0.0,
                    'features': []
                }
            
            # Debug input image
            self.debug_input_image(image_data)
            
            # Sync user mapping with database if provided
            if db is not None:
                self.sync_user_mapping_with_db(db)
            
            # Preprocess image
            image_tensor = self.preprocess_image(image_data)
            
            # Get prediction using identity predictor with feature extraction
            user_id, confidence, features = self.identity_predictor.identify_user(image_tensor)
            
            # Log the original model prediction before any database validation
            logger.info(f"Model's original prediction (feature extraction): user_id={user_id}, confidence={confidence:.4f}")
            
            # Validate prediction against database if available
            if db is not None:
                validated_user_id = self._validate_against_database(db, user_id, email)
                if validated_user_id != user_id:
                    logger.info(f"User ID changed after database validation: {user_id} -> {validated_user_id}")
                    user_id = validated_user_id
            
            return {
                'success': True,
                'user_id': user_id,
                'confidence': confidence,
                'features': features
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'user_id': 'unknown',
                'confidence': 0.0,
                'features': []
            }
    
    def _validate_against_database(self, db, user_id, email=None):
        """Helper to validate predicted user against database"""
        try:
            # Initialize user to None
            user = None
            original_user_id = user_id  # Keep track of original prediction
            
            # If we have an email, prioritize matching by email first
            if email:
                logger.info(f"Prioritizing user lookup by email: {email}")
                user = db.query(User).filter(User.email == email).first()
                if user:
                    logger.info(f"Found user by email: {user.id}")
                    return user.id
            
            # If no user found by email, then try by predicted user ID
            if not user and user_id:
                logger.info(f"Attempting to find user by predicted ID: {user_id}")
                user = db.query(User).filter(User.id == user_id).first()
                if user:
                    logger.info(f"Verified user exists in database: {user.id}")
                    return user.id
            
            # IMPORTANT: If the user wasn't found in the database, but the model predicted a specific user,
            # we should trust the model's prediction rather than using a fallback
            if not user:
                logger.warning(f"User {original_user_id} not found in database, but keeping model prediction")
                # Check if the predicted user is in the mapping
                self.mapping_service.initialize_mapping()  # Refresh mapping
                if original_user_id in self.mapping_service.get_mapping().values():
                    logger.info(f"Predicted user {original_user_id} exists in mapping, using it despite database absence")
                    return original_user_id
                else:
                    logger.warning(f"Predicted user {original_user_id} not found in mapping either")
                    # Only use fallback as last resort if absolutely needed
                    if getattr(self.model, 'num_identities', 0) > 0:  # If model has identities, trust its prediction
                        logger.info(f"Using model's prediction: {original_user_id}")
                        return original_user_id
                    else:
                        # Last resort fallback - only if model has no identities
                        logger.warning("Model has no identities, looking for fallback user")
                        fallback_user = db.query(User).first()
                        if fallback_user:
                            logger.info(f"Using fallback user {fallback_user.id} from database")
                            return fallback_user.id
                        else:
                            logger.error("No users found in database")
                    
            return original_user_id  # Return original prediction if no changes
        except Exception as db_err:
            logger.error(f"Database validation error: {db_err}", exc_info=True)
            return user_id  # Return original if error
    
    def sync_user_mapping_with_db(self, db):
        """Synchronize user ID mapping with database records"""
        try:
            if db is None:
                logger.warning("Cannot sync user mapping: database session is None")
                return False
                
            logger.info("Synchronizing user mapping with database")
            
            # Always refresh mapping from directory structure
            self.mapping_service.initialize_mapping()
            
            # Log the mapping we're using
            logger.info(f"Using mapping with {len(self.mapping_service.get_mapping())} entries")
            sample_count = min(5, len(self.mapping_service.get_mapping()))
            if sample_count > 0:
                sample_items = list(self.mapping_service.get_mapping().items())[:sample_count]
                logger.info(f"Sample mapping (first {sample_count} entries): {sample_items}")
            
            # We don't modify the server mapping from the client anymore
            logger.info("Client is not allowed to modify server mapping")
            return True
                
        except Exception as e:
            logger.error(f"Failed to sync user mapping with database: {e}", exc_info=True)
            return False
    
    def debug_mapping(self):
        """Print detailed mapping information for debugging"""
        logger.info("============ BIOMETRIC SERVICE MAPPING DEBUG ============")
        logger.info(f"Client ID: {self.client_id}")
        
        # Get current mapping
        self.mapping_service.initialize_mapping()  # Refresh mapping
        mapping = self.mapping_service.get_mapping()
        logger.info(f"Total users in mapping: {len(mapping)}")
        
        # Print sample of mapping entries
        sample_count = min(10, len(mapping))
        logger.info(f"Sample mapping (first {sample_count} entries):")
        for i, (class_idx, user_id) in enumerate(mapping.items()):
            if i >= sample_count:
                break
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
            
        # Check server connectivity
        try:
            is_server_available = self.mapping_service.initialize_mapping()
            logger.info(f"Mapping server connectivity: {'OK' if is_server_available else 'UNAVAILABLE'}")
        except Exception as e:
            logger.error(f"Error checking server connectivity: {e}")
        
        logger.info("====================================================")
    
    def get_model_info(self):
        """Get information about the loaded model for debugging"""
        try:
            info = {
                "client_id": self.client_id,
                "device": str(self.device),
                "model_type": type(self.model).__name__,
                "initialized": self.initialized,
                "mapping_service_available": hasattr(self, 'mapping_service'),
            }
            
            # Add mapping info if available
            if hasattr(self, 'mapping_service'):
                self.mapping_service.initialize_mapping()  # Refresh mapping
                info["mapping_count"] = len(self.mapping_service.get_mapping())
                sample_mapping = {k: self.mapping_service.get_mapping()[k] for k in list(self.mapping_service.get_mapping().keys())[:5]} if self.mapping_service.get_mapping() else {}
                info["mapping_sample"] = sample_mapping
            
            # Add model details if available
            if hasattr(self, 'model'):
                model_path = getattr(self.model_loader, 'model_path', None)
                if model_path:
                    info["model_path"] = str(model_path)
                    info["model_file_exists"] = os.path.exists(model_path)
                    if info["model_file_exists"]:
                        info["model_file_size"] = os.path.getsize(model_path)
                        info["model_last_modified"] = datetime.fromtimestamp(
                            os.path.getmtime(model_path)
                        ).isoformat()
                
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
            if image_tensor is None and image_data is None:
                return []
                
            if image_tensor is None:
                if image_data is None:
                    logger.error("get_top_predictions called with no image data.")
                    return []
                image_tensor = self.preprocess_image(image_data)
                
            # Get predictions directly from model
            with torch.no_grad():
                identity_logits, features = self.model(image_tensor)
                probabilities = F.softmax(identity_logits, dim=1)[0]
                
                # Get top-k predictions
                top_confidences, top_indices = torch.topk(probabilities, min(top_k, len(probabilities)))
                
                # Convert to list of dictionaries
                results = []
                for i, (conf, idx) in enumerate(zip(top_confidences, top_indices)):
                    class_idx = int(idx.item())
                    user_id = self.mapping_service.get_identity_by_model_class(class_idx, use_filtered=True)
                    results.append({
                        "rank": i+1,
                        "class_index": class_idx,
                        "user_id": user_id,
                        "confidence": conf.item()
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"Error getting top predictions: {e}")
            return []
    
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
                logger.info(f"Converted image to RGB mode")
            
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
                feature_stats = self.feature_extractor.debug_feature_extraction(img_tensor.clone())
                logger.info(f"Feature extraction debug completed with stats: {feature_stats}")
                
                # Get top predictions for this image
                top_preds = self.get_top_predictions(image_tensor=img_tensor, top_k=5)
                logger.info(f"Top 5 predictions for debug image: {top_preds}")
            except Exception as feat_err:
                logger.error(f"Error in feature extraction debugging: {feat_err}", exc_info=True)
            
        except Exception as e:
            logger.error(f"Error debugging input image: {e}", exc_info=True) 