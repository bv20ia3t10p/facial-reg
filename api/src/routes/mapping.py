"""
Central mapping service endpoints for consistent class-to-user mapping across clients
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, Body, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Import the mapping service
from ..services.mapping_service import MappingService

# Configure logging
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Path for storing global mapping
GLOBAL_MAPPING_PATH = Path("/app/models/mappings/global_mapping.json")

# Memory cache for faster access
_global_mapping_cache = None

# Global mapping service instance
mapping_service = None

def get_mapping_service() -> MappingService:
    """Get or create mapping service instance"""
    global mapping_service
    if mapping_service is None:
        mapping_service = MappingService()
        # Initialize mapping
        mapping_service.initialize_mapping()
    return mapping_service

class MappingUpdateRequest(BaseModel):
    """Request model for updating global mapping"""
    mapping: Dict[str, str]
    force_update: bool = False
    client_id: str
    current_version: Optional[str] = None
    current_hash: Optional[str] = None
    
    model_config = {
        'protected_namespaces': ()
    }

def ensure_mapping_directory():
    """Ensure the mapping directory exists"""
    mapping_dir = GLOBAL_MAPPING_PATH.parent
    mapping_dir.mkdir(exist_ok=True, parents=True)

def get_current_mapping() -> Dict[str, str]:
    """
    Get the current global mapping from file or cache
    Falls back to generating from directories using the approach from improved_privacy_training.py
    """
    global _global_mapping_cache
    
    # Return from cache if available
    if _global_mapping_cache is not None:
        return _global_mapping_cache
    
    # Try to read from file
    try:
        if GLOBAL_MAPPING_PATH.exists():
            with open(GLOBAL_MAPPING_PATH, 'r') as f:
                mapping = json.load(f)
                _global_mapping_cache = mapping
                return mapping
    except Exception as e:
        logger.error(f"Error reading global mapping file: {e}")
    
    # Import here to avoid circular imports
    try:
        from ..utils.generate_mapping import generate_class_mapping
        
        # Generate new mapping using the improved approach
        logger.info("No existing mapping found, generating from partitioned data...")
        mapping = generate_class_mapping()  # Gets global mapping with no client_id filter
        
        if mapping:
            save_mapping(mapping)
            return mapping
    except Exception as e:
        logger.error(f"Error generating mapping: {e}")
    
    # Return empty mapping if not found
    return {}

def save_mapping(mapping: Dict[str, str]) -> bool:
    """Save the global mapping to file and update cache"""
    global _global_mapping_cache
    
    try:
        ensure_mapping_directory()
        with open(GLOBAL_MAPPING_PATH, 'w') as f:
            json.dump(mapping, f, indent=2)
        
        # Update cache
        _global_mapping_cache = mapping
        return True
    except Exception as e:
        logger.error(f"Error saving global mapping: {e}")
        return False

@router.get("/api/mapping")
def get_global_mapping():
    """
    Get the global class-to-user mapping
    
    Returns:
        Dict with class indices as keys and user IDs as values
    """
    mapping = get_current_mapping()
    
    return {
        "success": True,
        "mapping": mapping,
        "count": len(mapping)
    }

@router.post("/api/mapping/update")
def update_global_mapping(request: MappingUpdateRequest):
    """
    Update the global class-to-user mapping
    
    Args:
        request: Request containing new mapping and metadata
        
    Returns:
        Updated mapping
    """
    try:
        # Get current mapping
        current_mapping = get_current_mapping()
        
        # Check if update is needed
        if not request.force_update and current_mapping:
            if current_mapping == request.mapping:
                return {
                    "success": True,
                    "message": "Mapping unchanged - already matches requested mapping",
                    "mapping": current_mapping,
                    "count": len(current_mapping),
                    "version": request.current_version
                }
        
        # Save the new mapping
        if save_mapping(request.mapping):
            logger.info(f"Global mapping updated with {len(request.mapping)} entries from client {request.client_id}")
            
            # Return the new mapping
            return {
                "success": True,
                "message": "Global mapping updated successfully",
                "mapping": request.mapping,
                "count": len(request.mapping),
                "version": request.current_version
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to save global mapping")
            
    except Exception as e:
        logger.error(f"Error updating global mapping: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update global mapping: {str(e)}")

@router.get("/api/mapping/refresh")
def refresh_global_mapping():
    """
    Force refresh the global mapping from partitioned data directories
    Following the exact approach in improved_privacy_training.py
    
    Returns:
        Updated mapping
    """
    try:
        logger.info("Force refreshing global mapping from partitioned data directories")
        
        # Clear cache first
        global _global_mapping_cache
        _global_mapping_cache = None
        
        # Import to avoid circular imports
        from ..utils.generate_mapping import generate_class_mapping
        
        # Generate new mapping using the improved approach
        new_mapping = generate_class_mapping()  # Gets global mapping with no client_id filter
        
        # Save the new mapping
        if save_mapping(new_mapping):
            logger.info(f"Global mapping refreshed with {len(new_mapping)} classes")
            
            # Return the new mapping
            return {
                "success": True,
                "message": "Global mapping refreshed successfully",
                "mapping": new_mapping,
                "count": len(new_mapping)
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to save refreshed global mapping")
    except Exception as e:
        logger.error(f"Error refreshing global mapping: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to refresh global mapping: {str(e)}")

# Removed duplicate debug endpoint - using the one below with filtered mapping support

@router.get("/api/mapping/debug")
def debug_mapping_info():
    """Debug endpoint to show current mapping configuration"""
    try:
        service = get_mapping_service()
        client_id = os.getenv("CLIENT_ID", "client1")
        
        # Get global mapping
        global_mapping = service.get_mapping()
        
        # Get filtered mapping for client
        filtered_mapping = service.get_filtered_mapping_for_client(client_id)
        
        # Get some sample mappings
        sample_predictions = {}
        if filtered_mapping:
            # Show how model predictions map to identities
            for model_class in range(min(10, len(filtered_mapping))):
                identity = service.get_identity_by_model_class(model_class, use_filtered=True)
                sample_predictions[f"model_class_{model_class}"] = identity
        
        return {
            "success": True,
            "debug_info": {
                "client_id": client_id,
                "environment_vars": {
                    "CLIENT_ID": os.getenv("CLIENT_ID"),
                    "NODE_TYPE": os.getenv("NODE_TYPE"),
                    "MODEL_PATH": os.getenv("MODEL_PATH")
                },
                "global_mapping": {
                    "count": len(global_mapping) if global_mapping else 0,
                    "sample": dict(list(global_mapping.items())[:5]) if global_mapping else {}
                },
                "filtered_mapping": {
                    "count": len(filtered_mapping) if filtered_mapping else 0,
                    "sample": dict(list(filtered_mapping.items())[:5]) if filtered_mapping else {},
                    "client_data_dir": service.client_data_dir
                },
                "sample_model_predictions": sample_predictions,
                "mapping_metadata": service.get_mapping_info()
            }
        }
    except Exception as e:
        logger.error(f"Error in debug mapping: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Legacy function exports for compatibility - removed duplicates 