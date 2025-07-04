#!/usr/bin/env python3
"""
Script to ensure the updated ResNet50-based model loads correctly
"""

import os
import sys
import logging
import torch
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.privacy_biometric_model import PrivacyBiometricModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    force=True
)
logger = logging.getLogger(__name__)

def test_model_loading():
    try:
        logger.info("Starting model loading test...")
        
        # Forcing CPU for compatibility
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Try to find model file
        model_paths = [
            Path("models"),
            Path("api/models"),
            Path("/app/models"),
            Path(os.path.expanduser("~/models"))
        ]
        
        model_file = None
        for path in model_paths:
            if path.exists():
                model_files = list(path.glob("*.pth"))
                if model_files:
                    model_file = model_files[0]
                    break
        
        if not model_file:
            logger.error("No model file found in standard locations")
            return False
            
        logger.info(f"Found model file: {model_file}")
        
        # Load state dict to determine number of identities
        logger.info("Loading state dict to determine model dimensions...")
        state_dict = torch.load(model_file, map_location=device)
        
        # Get number of identities from the classifier layer
        num_identities = state_dict['identity_head.4.weight'].shape[0]
        logger.info(f"Found {num_identities} identities in saved model")
        
        # Initialize model with correct number of identities
        logger.info("Creating model instance...")
        model = PrivacyBiometricModel(num_identities=num_identities)
        
        # Load the state dict
        logger.info("Applying state dict to model...")
        model.load_state_dict(state_dict, strict=True)
        logger.info("Model loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error testing model loading: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_model_loading()
    if not success:
        logger.error("Model loading test failed")
        sys.exit(1)
    sys.exit(0) 