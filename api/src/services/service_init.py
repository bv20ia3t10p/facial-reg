"""
Service initialization and management
"""

import os
import logging
import importlib
import torch
from pathlib import Path

logger = logging.getLogger(__name__)

# Global service instances
biometric_service = None

def initialize_services():
    """Initialize all required services"""
    logger.info("Initializing services...")
    
    # Initialize biometric service
    global biometric_service
    biometric_service = initialize_biometric_service()
    
    logger.info("Service initialization complete")

def initialize_biometric_service():
    """Initialize the biometric recognition service"""
    try:
        client_id = os.getenv("CLIENT_ID", "client1")
        from ..services.biometric_service import BiometricService
        service = BiometricService(client_id=client_id)
        logger.info(f"Biometric service initialized for {client_id}")
        return service
    except Exception as e:
        logger.error(f"Failed to initialize biometric service: {e}")
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
        
        # Update the model in the service
        biometric_service.model = new_model
        
        # Save the updated model to disk
        client_id = os.getenv("CLIENT_ID", "client1")
        model_path = Path(f"/app/models/best_{client_id}_pretrained_model.pth")
        
        # Save with metadata
        metadata = {
            'source': 'federated_learning',
            'client_id': client_id
        }
        
        torch.save({
            'state_dict': new_model.state_dict(),
            'num_identities': getattr(new_model, 'num_identities', 100),
            'feature_dim': getattr(new_model, 'embedding_dim', 512),
            'metadata': metadata
        }, model_path)
        
        logger.info(f"Model saved to {model_path}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to update biometric model: {e}")
        return False 