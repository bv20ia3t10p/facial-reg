"""
Alternative entry point for the API
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Fix the federated_integration.py file
    federated_integration_path = Path(__file__).parent / "federated_integration.py"
    logger.info(f"Checking federated_integration.py at {federated_integration_path}")
    
    if federated_integration_path.exists():
        with open(federated_integration_path, 'r') as f:
            content = f.read()
        
        # Check if the file contains 'Base'
        if 'class FederatedModelVersion(Base):' in content:
            logger.info("Found reference to 'Base' in federated_integration.py, fixing it...")
            
            # Replace it with FederatedBase
            content = content.replace('class FederatedModelVersion(Base):', 'class FederatedModelVersion(FederatedBase):')
            
            # Write the file back
            with open(federated_integration_path, 'w') as f:
                f.write(content)
                
            logger.info("Fixed reference to 'Base' in federated_integration.py")
    
    # Import the main app
    from .main import app
    logger.info("Successfully imported main app")
except Exception as e:
    logger.error(f"Error importing main app: {e}")
    raise 