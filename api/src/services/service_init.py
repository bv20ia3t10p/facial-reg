"""
Service initialization and management
"""

import os
import logging
import torch
from pathlib import Path
import traceback
from datetime import datetime

from .emotion_service import EmotionService

logger = logging.getLogger(__name__)

# Global service instances
biometric_service = None
emotion_service = None
# Flag to track if services have been initialized
_services_initialized = False

def initialize_services():
    """Initialize all required services"""
    global _services_initialized, biometric_service, emotion_service
    
    try:
        # Initialize biometric service
        biometric_service = initialize_biometric_service()
        if not biometric_service:
            logger.error("Failed to initialize biometric service")
            return False
        
        # Initialize emotion service
        emotion_service = EmotionService()
        if not emotion_service:
            logger.error("Failed to initialize emotion service")
            return False
        
        # Mark services as initialized
        _services_initialized = True
        logger.info("Service initialization complete")
        return True
        
    except Exception as e:
        logger.error(f"Error during service initialization: {e}")
        _services_initialized = False
        return False

def initialize_biometric_service(client_id=None):
    """Initialize the biometric recognition service"""
    global biometric_service
    
    # Check if service is already initialized
    if biometric_service is not None:
        return biometric_service
    
    try:
        # Import here to avoid circular imports
        from ..services.biometric_service import BiometricService
        
        # Get client ID from environment or use default
        if client_id is None:
            client_id = os.getenv("CLIENT_ID", "client1")
        
        # First check if MODEL_PATH is already set in environment
        model_path_env = os.getenv("MODEL_PATH")
        if model_path_env:
            model_path = Path(model_path_env)
        else:
            # Check for model file existence
            model_path = Path(f"/app/models/best_{client_id}_pretrained_model.pth")
            os.environ["MODEL_PATH"] = str(model_path)
            
        # Check if the model file exists
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            # Try to find any model file as fallback
            model_files = list(Path("/app/models").glob("*.pth"))
            if model_files:
                logger.info(f"Found fallback model file: {model_files[0]}")
                # Set the fallback model path
                os.environ["MODEL_PATH"] = str(model_files[0])
            else:
                # Try to find model in the root directory
                root_model_path = Path(f"best_{client_id}_pretrained_model.pth")
                if root_model_path.exists():
                    logger.info(f"Found model in root directory: {root_model_path}")
                    # Create models directory and copy file
                    models_dir = Path("/app/models")
                    models_dir.mkdir(parents=True, exist_ok=True)
                    import shutil
                    shutil.copy2(root_model_path, models_dir / root_model_path.name)
                    os.environ["MODEL_PATH"] = str(models_dir / root_model_path.name)
                else:
                    logger.error("No model files found in /app/models directory or root")
                    return None
        
        # Create service instance
        biometric_service = BiometricService(client_id=client_id)
        
        # Verify model loaded correctly
        if not biometric_service.initialized:
            logger.error("BiometricService failed to initialize properly")
            if hasattr(biometric_service, 'init_error'):
                logger.error(f"Initialization error: {biometric_service.init_error}")
            return None
        
        # Log model info
        model_info = biometric_service.get_model_info()
        logger.info("Model loaded successfully:")
        logger.info(f"  - Num identities: {model_info.get('num_identities', 'unknown')}")
        logger.info(f"  - Backbone output dim: {model_info.get('backbone_output_dim', 'unknown')}")
        logger.info(f"  - Embedding dim: {model_info.get('embedding_dim', 'unknown')}")
        
        return biometric_service
        
    except Exception as e:
        logger.error(f"Error initializing biometric service: {e}", exc_info=True)
        return None

def initialize_emotion_service():
    """Initialize emotion service"""
    global emotion_service
    
    # Check if service is already initialized
    if emotion_service is not None:
        logger.info("Emotion service already initialized, returning existing instance")
        return emotion_service
    
    try:
        # Import here to avoid circular imports
        from ..services.emotion_service import EmotionService
        
        logger.info("Initializing emotion service")
        
        # Create service instance
        emotion_service = EmotionService()
        
        return emotion_service
        
    except Exception as e:
        logger.error(f"Failed to initialize emotion service: {e}")
        traceback.print_exc()
        return None

def reload_biometric_service():
    """Reload biometric service model"""
    global biometric_service
    if biometric_service:
        logger.info("Reloading biometric service model...")
        try:
            success = biometric_service.reload_model()
            logger.info("Biometric model reload " + ("successful" if success else "failed"))
            return success
        except Exception as e:
            logger.error(f"Error reloading model: {e}")
            return False
    else:
        logger.error("Cannot reload model: Biometric service not initialized")
        return False

def update_biometric_model(new_model):
    """Update biometric service with a new model from federated learning"""
    global biometric_service
    if not biometric_service:
        logger.error("Cannot update model: Biometric service not initialized")
        return False
    
    try:
        logger.info("Updating biometric service model from federated learning...")
        
        # Make sure model is in eval mode
        new_model.eval()
        
        # Save the model to a checkpoint file
        client_id = os.getenv("CLIENT_ID", "client1")
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = f"/app/models/checkpoint_{client_id}_{timestamp}.pth"
        
        # Save with metadata
        torch.save({
            'state_dict': new_model.state_dict(),
            'metadata': {
                'timestamp': timestamp,
                'source': 'federated_update',
                'client_id': client_id
            },
            'num_identities': getattr(new_model, 'num_identities', 100)
        }, checkpoint_path)
        
        logger.info(f"Saved updated model to checkpoint: {checkpoint_path}")
        
        # Update the model in the service
        biometric_service.model = new_model
        
        # Update the MODEL_PATH environment variable to point to the new checkpoint
        os.environ["MODEL_PATH"] = checkpoint_path
        logger.info(f"Updated MODEL_PATH to: {checkpoint_path}")
        
        return True
    except Exception as e:
        logger.error(f"Error updating biometric model: {e}")
        return False

# Initialize services on module import, but only if they haven't been initialized yet
try:
    if not _services_initialized:
        initialize_services()
except Exception as e:
    logger.error(f"Service initialization failed: {e}")
    traceback.print_exc() 