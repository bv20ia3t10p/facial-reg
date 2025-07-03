"""
Mapping utilities for biometric service
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional
from sqlalchemy import create_engine, text

# Configure logging
logger = logging.getLogger(__name__)

# Get database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///database/facial_reg.db")

def get_client_databases():
    """Get all available client database files"""
    client_dbs = []
    
    # Check database directory
    db_path = Path("/app/database")
    if db_path.exists():
        # Look for client-specific database files
        client_db_files = list(db_path.glob("client*.db"))
        if client_db_files:
            logger.info(f"Found {len(client_db_files)} client database files")
            for db_file in client_db_files:
                client_id = db_file.stem  # Get filename without extension
                client_dbs.append({
                    "client_id": client_id,
                    "db_path": str(db_file),
                    "db_url": f"sqlite:///{db_file}"
                })
    
    # If no client databases, use the default
    if not client_dbs:
        client_dbs.append({
            "client_id": "default",
            "db_path": DATABASE_URL.replace("sqlite:///", ""),
            "db_url": DATABASE_URL
        })
        logger.info(f"Using default database: {DATABASE_URL}")
    
    return client_dbs

def generate_class_mapping(client_id: Optional[str] = None) -> Dict[str, str]:
    """
    Generate class-to-user mapping from partitioned data directories
    
    Args:
        client_id: Optional client ID to filter directories
        
    Returns:
        Dictionary mapping class indices (0 to N-1) to directory numbers
    """
    try:
        # Base path for partitioned data
        base_path = Path("/app/data/partitioned")
        if not base_path.exists():
            logger.error(f"Partitioned data directory not found at {base_path}")
            return {}
            
        # Get all client directories
        client_dirs = []
        if client_id:
            # If client_id specified, only look in that directory
            client_path = base_path / client_id
            if client_path.exists():
                client_dirs = [client_path]
            else:
                logger.error(f"Client directory not found: {client_path}")
                return {}
        else:
            # Get all client directories
            client_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("client")]
        
        # Collect all class directories across clients
        all_classes = set()
        for client_dir in client_dirs:
            # Get all numeric directories (class IDs)
            class_dirs = [d for d in client_dir.iterdir() if d.is_dir() and d.name.isdigit()]
            all_classes.update(d.name for d in class_dirs)
        
        if not all_classes:
            logger.error("No class directories found")
            return {}
            
        # Sort classes numerically for consistent ordering
        sorted_classes = sorted(list(all_classes), key=lambda x: int(x))
        
        # Create mapping from sequential indices (0 to N-1) to directory numbers
        mapping = {str(idx): class_id for idx, class_id in enumerate(sorted_classes)}
        
        # Log the mapping for debugging
        logger.info(f"Generated mapping with {len(mapping)} classes")
        logger.info("Sample of mapping (first 5 entries):")
        for idx, class_id in list(mapping.items())[:5]:
            logger.info(f"  Class index {idx} -> Directory {class_id}")
        
        return mapping
        
    except Exception as e:
        logger.error(f"Error generating class mapping: {e}")
        return {}

def get_and_validate_mapping(existing_mapping_service=None):
    """
    Validate server mapping against database
    
    Args:
        existing_mapping_service: Optional existing MappingService instance to reuse
    """
    try:
        # Get client ID from environment
        client_id = os.getenv("CLIENT_ID", "client1")
        
        # Use existing mapping service if provided, otherwise create new one
        if existing_mapping_service:
            mapping_service = existing_mapping_service
        else:
            from ..services.mapping_service import MappingService
            server_url = os.getenv("SERVER_URL", "http://fl-coordinator:8080")
            mapping_service = MappingService() # constructor doesn't take args anymore
        
        # Initialize mapping from directory structure
        mapping_service.initialize_mapping()
        mapping = mapping_service.get_mapping()
        
        if not mapping:
            logger.error("Failed to get mapping from server")
            return False
            
        logger.info(f"Successfully validated mapping with {len(mapping)} entries")
        return True
        
    except Exception as e:
        logger.error(f"Error validating mapping: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    logger.info("Validating and updating user ID mapping")
    get_and_validate_mapping() 