#!/usr/bin/env python3
"""
Test script to verify model dimensions
"""

import sys
import os
import logging
import torch
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.privacy_biometric_model import PrivacyBiometricModel
from src.services.feature_extraction import FeatureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_dimensions():
    """Test model dimensions through the entire pipeline"""
    try:
        # Initialize model
        model = PrivacyBiometricModel(num_identities=100)
        device = torch.device('cpu')
        model = model.to(device)
        model.eval()
        
        # Create test input
        x = torch.randn(1, 3, 224, 224)
        logger.info(f"Input shape: {x.shape}")
        
        # Test direct model forward pass
        with torch.no_grad():
            identity_logits, features = model(x)
            logger.info(f"Features shape from model: {features.shape}")
            logger.info(f"Logits shape from model: {identity_logits.shape}")
        
        # Test feature extractor
        feature_extractor = FeatureExtractor(model, device)
        extracted_features = feature_extractor.extract_features(x)
        logger.info(f"Features shape from extractor: {extracted_features.shape}")
        
        # Verify dimensions match expected values
        assert features.shape[1] == 512, f"Expected feature dimension 512, got {features.shape[1]}"
        assert identity_logits.shape[1] == 100, f"Expected logits dimension 100, got {identity_logits.shape[1]}"
        
        logger.info("All dimension checks passed!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise

if __name__ == "__main__":
    test_model_dimensions() 