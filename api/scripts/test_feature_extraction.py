#!/usr/bin/env python3
"""
Consolidated script to test feature extraction and raw image prediction
"""

import os
import sys
import logging
import torch
import numpy as np
import time
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

# Add the parent directory to the path so we can import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.privacy_biometric_model import PrivacyBiometricModel
from src.services.feature_extraction import FeatureExtractor
from src.services.biometric_service import BiometricService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_test_image(image_path):
    """Load a test image from path"""
    try:
        if isinstance(image_path, (str, Path)):
            with open(image_path, 'rb') as f:
                return f.read()
        return None
    except Exception as e:
        logger.error(f"Failed to load test image: {e}")
        return None

def test_feature_extraction(model_path=None, image_path=None):
    """Test feature extraction with the updated model"""
    try:
        # Use CPU for testing
        device = torch.device("cpu")
        
        # Load or create model
        if model_path and Path(model_path).exists():
            logger.info(f"Loading model from {model_path}")
            model = PrivacyBiometricModel.load_state(model_path, device)
        else:
            logger.info("Creating new model")
            model = PrivacyBiometricModel(
                num_identities=100,
                embedding_dim=512,
                privacy_enabled=False
            )
        
        # Set model to evaluation mode
        model.eval()
        
        # Create feature extractor
        feature_extractor = FeatureExtractor(model, device)
        
        # Create test image tensor if no image path provided
        if image_path and Path(image_path).exists():
            logger.info(f"Loading image from {image_path}")
            # Load and preprocess image
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            image = Image.open(image_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        else:
            logger.info("Creating random image tensor")
            # Create random image tensor
            image_tensor = torch.rand(1, 3, 224, 224)
        
        logger.info(f"Image tensor shape: {image_tensor.shape}")
        
        # Test direct model feature extraction
        logger.info("Testing model.extract_features")
        try:
            features = model.extract_features(image_tensor)
            logger.info(f"Model extract_features successful, shape: {features.shape}")
        except Exception as e:
            logger.error(f"Model extract_features failed: {e}")
        
        # Test model.get_features
        logger.info("Testing model.get_features")
        try:
            features = model.get_features(image_tensor)
            logger.info(f"Model get_features successful, shape: {features.shape}")
        except Exception as e:
            logger.error(f"Model get_features failed: {e}")
        
        # Test feature extractor
        logger.info("Testing feature_extractor.extract_features")
        try:
            features = feature_extractor.extract_features(image_tensor)
            logger.info(f"Feature extractor successful, shape: {features.shape}")
        except Exception as e:
            logger.error(f"Feature extractor failed: {e}")
        
        # Test model forward pass
        logger.info("Testing model forward pass")
        try:
            with torch.no_grad():
                identity_logits, embedding = model(image_tensor)
                logger.info(f"Forward pass successful")
                logger.info(f"  identity_logits shape: {identity_logits.shape}")
                logger.info(f"  embedding shape: {embedding.shape}")
        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
        
        logger.info("Feature extraction tests completed")
        return True
        
    except Exception as e:
        logger.error(f"Error in feature extraction test: {e}")
        return False

def test_prediction_methods(image_path, client_id="client1"):
    """Test both prediction methods and compare results"""
    logger.info(f"Testing prediction methods with image: {image_path}")
    
    # Initialize biometric service
    service = BiometricService(client_id=client_id)
    
    if not service.initialized:
        logger.error(f"Failed to initialize biometric service: {getattr(service, 'init_error', 'Unknown error')}")
        return
    
    # Load test image
    image_data = load_test_image(image_path)
    if image_data is None:
        logger.error("No test image loaded, aborting test")
        return
    
    # Test raw image prediction
    logger.info("=== TESTING RAW IMAGE PREDICTION ===")
    start_time = time.time()
    raw_result = service.predict_identity(image_data)
    raw_time = time.time() - start_time
    
    # Test feature extraction prediction
    logger.info("=== TESTING FEATURE EXTRACTION PREDICTION ===")
    start_time = time.time()
    feature_result = service.predict_identity_with_features(image_data)
    feature_time = time.time() - start_time
    
    # Compare results
    logger.info("=== COMPARISON OF RESULTS ===")
    logger.info(f"Raw image prediction: User {raw_result['user_id']} with confidence {raw_result['confidence']:.4f} (took {raw_time:.4f}s)")
    logger.info(f"Feature extraction prediction: User {feature_result['user_id']} with confidence {feature_result['confidence']:.4f} (took {feature_time:.4f}s)")
    
    # Check if predictions match
    if raw_result['user_id'] == feature_result['user_id']:
        logger.info("✓ Both methods predicted the same user ID")
    else:
        logger.warning("✗ Methods predicted different user IDs")
    
    # Compare confidence
    conf_diff = abs(raw_result['confidence'] - feature_result['confidence'])
    logger.info(f"Confidence difference: {conf_diff:.4f}")
    
    # Compare feature vectors
    raw_features = np.array(raw_result['features'])
    feature_features = np.array(feature_result['features'])
    
    if raw_features.shape == feature_features.shape:
        cosine_sim = np.dot(raw_features, feature_features) / (np.linalg.norm(raw_features) * np.linalg.norm(feature_features))
        logger.info(f"Feature vector cosine similarity: {cosine_sim:.4f}")
    else:
        logger.warning(f"Feature vectors have different shapes: {raw_features.shape} vs {feature_features.shape}")
    
    return {
        "raw_result": raw_result,
        "feature_result": feature_result,
        "raw_time": raw_time,
        "feature_time": feature_time
    }

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Test feature extraction and prediction methods")
    parser.add_argument("--model-path", help="Path to the model file")
    parser.add_argument("--image-path", help="Path to the test image")
    parser.add_argument("--client-id", default="client1", help="Client ID for prediction testing")
    parser.add_argument("--test-type", choices=["features", "prediction", "both"], default="both",
                      help="Type of test to run")
    args = parser.parse_args()
    
    if args.test_type in ["features", "both"]:
        success = test_feature_extraction(args.model_path, args.image_path)
        if not success:
            sys.exit(1)
    
    if args.test_type in ["prediction", "both"] and args.image_path:
        test_prediction_methods(args.image_path, args.client_id)
    
    sys.exit(0) 