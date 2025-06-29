#!/usr/bin/env python3
"""
Script to test the federated client's model loading with the updated code
"""

import os
import sys
import logging
import asyncio
from pathlib import Path

# Add the parent directory to the path so we can import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.federated_client import FederatedClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_federated_client_model_loading():
    """Test the federated client's model loading with the updated code"""
    try:
        logger.info("Testing federated client model loading")
        
        # Set environment variables for testing
        os.environ["CLIENT_ID"] = "test_client"
        
        # Create a federated client
        client = FederatedClient()
        
        # If a model path is provided, use it
        if len(sys.argv) > 1:
            model_path = sys.argv[1]
            client.model_path = model_path
            logger.info(f"Using provided model path: {model_path}")
        
        # Load the model
        model = await client.load_or_download_model()
        
        if model is not None:
            logger.info("Model loaded successfully")
            logger.info(f"Model details:")
            logger.info(f"  num_identities: {model.num_identities}")
            logger.info(f"  embedding_dim: {getattr(model, 'embedding_dim', 'unknown')}")
            logger.info(f"  backbone_output_dim: {getattr(model, 'backbone_output_dim', 'unknown')}")
            logger.info(f"  privacy_enabled: {model.privacy_enabled}")
            return True
        else:
            logger.error("Failed to load model")
            return False
    
    except Exception as e:
        logger.error(f"Error testing federated client model loading: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_federated_client_model_loading())
    
    if success:
        logger.info("Federated client model loading test completed successfully")
        sys.exit(0)
    else:
        logger.error("Federated client model loading test failed")
        sys.exit(1) 