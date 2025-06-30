"""
Federated Learning Coordinator for Biometric API
Manages federated learning rounds and model aggregation
"""

import os
import logging
import json
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import hashlib
import uuid
import sys
import base64
import io

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn
from sqlalchemy.orm import Session

from .models.privacy_biometric_model import PrivacyBiometricModel
# Removed privacy_biometrics dependency to make API independent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Suppress Opacus validator info messages
logging.getLogger('opacus.validators.module_validator').setLevel(logging.WARNING)
logging.getLogger('opacus.validators.batch_norm').setLevel(logging.WARNING)

# Import privacy components
from .privacy.privacy_engine import PrivacyEngine
from .utils.security import generate_uuid
from .db.database import get_db
from .utils.datetime_utils import get_current_time, get_current_time_str, get_current_time_formatted
from .utils.generate_mapping import generate_class_mapping as generate_global_mapping_from_directories
from .routes.mapping import save_mapping  # reuse existing helper

# Force CPU usage regardless of CUDA availability
torch.cuda.is_available = lambda: False
device = torch.device('cpu')
logger.info("Coordinator using CPU (forced)")

# Initialize FastAPI app
app = FastAPI(
    title="Federated Learning Coordinator",
    description="Manages federated learning for biometric models",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class ClientInfo(BaseModel):
    client_id: str
    dataset_size: int
    model_version: Optional[str] = None
    privacy_budget_remaining: Optional[float] = None
    
    model_config = {
        'protected_namespaces': ()
    }

class FederatedRound(BaseModel):
    round_id: int
    start_time: datetime
    status: str
    active_clients: List[str] = []
    aggregation_completed: bool = False
    metrics: Optional[Dict[str, Any]] = None
    
    model_config = {
        'protected_namespaces': ()
    }

class ModelUpdateRequest(BaseModel):
    client_id: str
    round_id: int
    update_type: str = "weights"  # Renamed from model_update_type
    dataset_size: int
    privacy_metrics: Optional[Dict[str, float]] = None
    training_metrics: Optional[Dict[str, float]] = None
    mapping_size: int
    
    model_config = {
        'protected_namespaces': ()
    }

class MappingUpdateRequest(BaseModel):
    """Request model for updating global mapping"""
    classes: List[str]
    force_update: bool = False
    
    model_config = {
        'protected_namespaces': ()
    }

# Global state
global_model: Any = None  # will hold PrivacyBiometricModel once initialized
current_round: Any = None  # will hold FederatedRound instance after create_new_round
registered_clients: Dict[str, ClientInfo] = {}
client_updates: Dict[str, Dict] = {}
federated_config = {
    'federated_rounds': int(os.getenv('FEDERATED_ROUNDS', '10')),
    'max_epsilon': float(os.getenv('MAX_DP_EPSILON', '100.0')),
    'noise_multiplier': float(os.getenv('DP_NOISE_MULTIPLIER', '0.2')),
    'max_grad_norm': float(os.getenv('DP_MAX_GRAD_NORM', '5.0')),
    'delta': float(os.getenv('DP_DELTA', '1e-5')),
    'enable_dp': os.getenv('ENABLE_DP', 'true').lower() == 'true',
    'num_identities': int(os.getenv('NUM_IDENTITIES', '100')),
}

# Create directory for model storage
model_dir = Path('/app/models')
model_dir.mkdir(exist_ok=True)
model_history_dir = model_dir / "history"
model_history_dir.mkdir(exist_ok=True)

# Path for storing global mapping
GLOBAL_MAPPING_PATH = Path("/app/models/mappings/global_mapping.json")

# Memory cache for faster access
_global_mapping_cache = None

# Simplified mapping management (API-independent)
_global_mapping_cache = None

def get_default_mapping() -> Dict[str, int]:
    """Get default identity mapping"""
    global _global_mapping_cache
    if _global_mapping_cache is None:
        # Generate default mapping for 300 identities (as per config)
        num_identities = federated_config.get('num_identities', 100)
        _global_mapping_cache = {str(i): i for i in range(num_identities)}
        logger.info(f"Initialized default mapping with {num_identities} identities")
    return _global_mapping_cache

def get_current_mapping() -> Dict[str, Any]:
    """
    Get current mapping from identity_mapping.json file
    
    Returns:
        Dictionary with mapping information in coordinator format
    """
    try:
        # Try to load from identity_mapping.json file
        identity_mapping_file = Path("/app/data/identity_mapping.json")
        
        if identity_mapping_file.exists():
            logger.info(f"Loading mapping from {identity_mapping_file}")
            
            with open(identity_mapping_file, 'r') as f:
                mapping_data = json.load(f)
            
            # Extract the mapping from the file
            raw_mapping = mapping_data.get("mapping", {})
            
            # Convert to proper format - mapping should be {identity: global_label}
            identity_mapping = {str(k): int(v) for k, v in raw_mapping.items()}
            reverse_mapping = {int(v): str(k) for k, v in raw_mapping.items()}
            
            # Format for coordinator consumption
            coordinator_mapping = {
                "version": mapping_data.get("version", "1.0.0"),
                "total_identities": mapping_data.get("total_identities", len(identity_mapping)),
                "mapping_hash": mapping_data.get("hash", "loaded_from_file"),
                "source_file": str(identity_mapping_file),
                
                # Identity mappings
                "identity_to_index": identity_mapping,
                "index_to_identity": reverse_mapping,
                
                # Metadata
                "loaded_identities": len(identity_mapping),
                "identity_range": {"min": min(identity_mapping.values()), "max": max(identity_mapping.values())},
                
                # Legacy compatibility
                "mapping": identity_mapping,  # For backward compatibility
                "reverse_mapping": {v: k for k, v in identity_mapping.items()}
            }
            
            logger.info(f"Loaded mapping from file with {len(identity_mapping)} identities")
            return coordinator_mapping
            
        else:
            logger.warning(f"Identity mapping file not found at {identity_mapping_file}, using default")
            # Fallback to default mapping
            identity_mapping = get_default_mapping()
            reverse_mapping = {v: str(k) for k, v in identity_mapping.items()}
            
            coordinator_mapping = {
                "version": "1.0.0",
                "total_identities": len(identity_mapping),
                "mapping_hash": "default",
                "source_file": "default_fallback",
                
                # Identity mappings
                "identity_to_index": identity_mapping,
                "index_to_identity": reverse_mapping,
                
                # Metadata
                "loaded_identities": len(identity_mapping),
                "identity_range": {"min": 0, "max": len(identity_mapping) - 1},
                
                # Legacy compatibility
                "mapping": identity_mapping,
                "reverse_mapping": {v: k for k, v in identity_mapping.items()}
            }
            
            logger.info(f"Using default mapping with {len(identity_mapping)} identities")
            return coordinator_mapping
        
    except Exception as e:
        logger.error(f"Error getting mapping: {e}")
        return {"error": f"Failed to get mapping: {str(e)}"}

def validate_client_mapping(client_mapping: Dict[str, Any]) -> bool:
    """
    Validate client mapping against default mapping
    
    Args:
        client_mapping: Client's mapping to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        default_mapping = get_default_mapping()
        
        # Extract client's identity mapping
        if "identity_to_index" in client_mapping:
            client_identities = client_mapping["identity_to_index"]
        elif "mapping" in client_mapping:
            client_identities = client_mapping["mapping"]
        else:
            logger.error("Client mapping missing required fields")
            return False
        
        # Basic validation - check if client mapping is a subset of default mapping
        for identity, index in client_identities.items():
            if str(identity) not in default_mapping or default_mapping[str(identity)] != index:
                logger.warning(f"Identity {identity} with index {index} not in default mapping")
        
        is_valid = True  # Accept client mappings for now
        
        if is_valid:
            logger.info(f"Client mapping validated successfully ({len(client_identities)} identities)")
        else:
            logger.warning("Client mapping validation failed")
        
        return is_valid
        
    except Exception as e:
        logger.error(f"Error validating client mapping: {e}")
        return False

def get_identity_by_index(index: int) -> Optional[str]:
    """
    Get identity name by index using default mapping
    
    Args:
        index: Identity index
        
    Returns:
        Identity name if found, None otherwise
    """
    try:
        default_mapping = get_default_mapping()
        reverse_mapping = {v: k for k, v in default_mapping.items()}
        return reverse_mapping.get(index)
    except Exception as e:
        logger.error(f"Error getting identity for index {index}: {e}")
        return None

def get_index_by_identity(identity: str) -> Optional[int]:
    """
    Get index by identity name using default mapping
    
    Args:
        identity: Identity name
        
    Returns:
        Index if found, None otherwise
    """
    try:
        default_mapping = get_default_mapping()
        return default_mapping.get(str(identity))
    except Exception as e:
        logger.error(f"Error getting index for identity {identity}: {e}")
        return None

def get_all_identities() -> List[str]:
    """
    Get all identity names from default mapping
    
    Returns:
        List of identity names
    """
    try:
        default_mapping = get_default_mapping()
        return list(default_mapping.keys())
    except Exception as e:
        logger.error(f"Error getting all identities: {e}")
        return []

def get_mapping_statistics() -> Dict[str, Any]:
    """
    Get comprehensive mapping statistics
    
    Returns:
        Dictionary with mapping statistics
    """
    try:
        default_mapping = get_default_mapping()
        
        # Add coordinator-specific statistics
        statistics = {
            "total_identities": len(default_mapping),
            "version": "1.0.0",
            "source": "API-independent",
            "coordinator_version": "2.0.0",
            "centralized_mapping": False,
            "validation_status": "active"
        }
        
        return statistics
        
    except Exception as e:
        logger.error(f"Error getting mapping statistics: {e}")
        return {"error": f"Failed to get statistics: {str(e)}"}

def refresh_mapping() -> bool:
    """
    Refresh mapping (API-independent version)
    
    Returns:
        True if successfully refreshed, False otherwise
    """
    try:
        global _global_mapping_cache
        _global_mapping_cache = None  # Reset to force reload
        default_mapping = get_default_mapping()  # This will recreate the mapping
        
        if default_mapping:
            logger.info("Successfully refreshed default mapping")
            return True
        else:
            logger.error("Failed to refresh default mapping")
            return False
            
    except Exception as e:
        logger.error(f"Error refreshing mapping: {e}")
        return False

def create_client_mapping_subset(client_data_dir: str) -> Dict[str, Any]:
    """
    Create a mapping subset for a specific client (simplified version)
    
    Args:
        client_data_dir: Path to client's data directory
        
    Returns:
        Dictionary with client-specific mapping
    """
    try:
        # Use default mapping as client mapping
        default_mapping = get_default_mapping()
        
        if not default_mapping:
            logger.warning(f"No identities available for client data directory: {client_data_dir}")
            return {"error": "No identities found"}
        
        # Create reverse mapping
        client_reverse_mapping = {v: k for k, v in default_mapping.items()}
        
        # Format for client consumption
        client_subset = {
            "version": "1.0.0",
            "total_identities": len(default_mapping),
            "data_directory": client_data_dir,
            "mapping_hash": "default",
            "source_file": "API-independent",
            
            # Client-specific mappings
            "identity_to_index": default_mapping,
            "index_to_identity": {str(k): v for k, v in client_reverse_mapping.items()},
            
            # Legacy compatibility
            "mapping": default_mapping,
            "reverse_mapping": client_reverse_mapping
        }
        
        logger.info(f"Created client mapping subset with {len(default_mapping)} identities for {client_data_dir}")
        return client_subset
        
    except Exception as e:
        logger.error(f"Error creating client mapping subset: {e}")
        return {"error": f"Failed to create subset: {str(e)}"}

# Legacy compatibility functions
def load_from_identity_mapping() -> Optional[Dict[str, Any]]:
    """
    Legacy function for loading from identity_mapping.json
    Now uses centralized mapping manager
    
    Returns:
        Mapping dictionary or None if failed
    """
    try:
        current_mapping = get_current_mapping()
        if "error" in current_mapping:
            return None
        return current_mapping
    except Exception as e:
        logger.error(f"Error in legacy load_from_identity_mapping: {e}")
        return None

# Legacy functions removed - now using centralized mapping manager

def load_or_create_global_model():
    """Load or create the global model"""
    global global_model
    
    # Create model directory if it doesn't exist
    model_dir.mkdir(exist_ok=True, parents=True)
    model_history_dir.mkdir(exist_ok=True, parents=True)
    
    # Get current mapping using centralized manager
    mapping_data = get_current_mapping()
    if "error" in mapping_data:
        logger.error(f"Failed to get centralized mapping: {mapping_data['error']}")
        num_identities = federated_config['num_identities']
    else:
        num_identities = mapping_data.get('total_identities', federated_config['num_identities'])
    
    logger.info(f"Using {num_identities} identities from centralized mapping")
    
    # Define model path
    model_path = model_dir / "best_pretrained_model.pth"
    
    try:
        if model_path.exists():
            logger.info(f"Loading global model from {model_path}")
            # Load state dictionary
            state_dict = torch.load(model_path, map_location='cpu')
            
            # Handle metadata wrapper
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                stored_num_identities = state_dict.get('num_identities', num_identities)
                embedding_dim = state_dict.get('feature_dim', 512)
                backbone_output_dim = state_dict.get('backbone_output_dim', 2048)
                mapping_version = state_dict.get('mapping_version')
                mapping_hash = state_dict.get('mapping_hash')
                state_dict = state_dict['state_dict']
                
                # Verify mapping consistency
                if stored_num_identities != num_identities:
                    logger.warning(f"Stored model has different number of identities ({stored_num_identities}) than current mapping ({num_identities})")
                    # We'll create a new model with correct size
                    raise ValueError("Model size mismatch with current mapping")
            else:
                embedding_dim = 512
                backbone_output_dim = 2048
            
            # Create model with correct parameters
            model = PrivacyBiometricModel(
                num_identities=num_identities,
                embedding_dim=embedding_dim,
                privacy_enabled=federated_config['enable_dp']
            ).to(device)
            
            # Load weights
            try:
                model.load_state_dict(state_dict, strict=False)
                logger.info(f"Successfully loaded model with {num_identities} identities")
            except Exception as e:
                logger.error(f"Error loading state dict: {e}")
                raise ValueError("Failed to load model weights")
                
        else:
            logger.info("Global model not found, creating new one")
            # Create a new model with mapping-based parameters
            model = PrivacyBiometricModel(
                num_identities=num_identities,
                privacy_enabled=federated_config['enable_dp']
            ).to(device)
            
            # Save initial model with mapping metadata
            save_model_version(model, 0, {})
            
    except Exception as e:
        logger.error(f"Error loading/creating global model: {e}")
        # Create a new model as fallback
        model = PrivacyBiometricModel(
            num_identities=num_identities,
            privacy_enabled=federated_config['enable_dp']
        ).to(device)
        # Save initial model
        current_mapping = get_current_mapping()
        save_model_version(model, 0, {})
    
    model.eval()  # Set to evaluation mode
    global_model = model
    return global_model

def aggregate_model_updates(updates: Dict[str, Dict]) -> Dict:
    """Federated Averaging of model updates"""
    if not updates:
        raise ValueError("No updates to aggregate")
    
    # Extract model parameters and weights (dataset sizes)
    parameters = []
    weights = []
    
    for client_id, update in updates.items():
        parameters.append(update['parameters'])
        weights.append(update['dataset_size'])
    
    # Normalize weights
    total_samples = sum(weights)
    if total_samples == 0:
        # All clients reported zero samples – fall back to equal weighting
        normalized_weights = [1.0 / len(weights)] * len(weights)
        logger.warning("All client dataset_size values are 0; using equal weights for aggregation")
    else:
        normalized_weights = [w / total_samples for w in weights]
    
    # Perform weighted average
    avg_params = {}
    for key in parameters[0].keys():
        # Skip if not a tensor
        if not isinstance(parameters[0][key], torch.Tensor):
            continue
            
        # Weighted sum for this parameter
        avg_params[key] = sum(p[key] * w for p, w in zip(parameters, normalized_weights))
    
    return avg_params

# Typing: can return None if saving fails
def save_model_version(model: nn.Module, round_id: int, metrics: Dict) -> Optional[Path]:
    """Save a version of the model with metadata"""
    try:
        # Create filename with round number
        timestamp = get_current_time_str()
        filename = f"model_round_{round_id}_{timestamp}.pth"
        save_path = model_history_dir / filename
        
        # Get current mapping info
        mapping_wrapper = get_current_mapping()
        if not mapping_wrapper:
            raise ValueError("Global mapping not available")
        
        # Extract real mapping
        if "identity_to_index" in mapping_wrapper:
            server_mapping_dict = mapping_wrapper["identity_to_index"]
        elif "mapping" in mapping_wrapper:
            server_mapping_dict = mapping_wrapper["mapping"]
        else:
            server_mapping_dict = mapping_wrapper
        mapping_size = len(server_mapping_dict)
        mapping_hash = hashlib.sha256(json.dumps(sorted(server_mapping_dict.items())).encode()).hexdigest()[:8]
        
        # Save model with metadata
        save_dict = {
            'state_dict': model.state_dict(),
            'round_id': round_id,
            'timestamp': timestamp,
            'num_identities': model.num_identities,
            'feature_dim': model.embedding_dim,
            'backbone_output_dim': getattr(model, "backbone_output_dim", None),
            'privacy_enabled': model.privacy_enabled,
            'mapping_size': mapping_size,
            'mapping_hash': mapping_hash,
            'mapping_version': get_current_time_str(),
            'metrics': metrics
        }
        
        # Save to history directory
        torch.save(save_dict, save_path)
        
        # Also update the best model
        best_model_path = model_dir / "best_pretrained_model.pth"
        torch.save(save_dict, best_model_path)
        
        logger.info(f"Saved model version for round {round_id} to {save_path}")
        return save_path
        
    except Exception as e:
        logger.error(f"Error saving model version: {e}")
        return None

def create_new_round():
    """Initialize a new federated learning round"""
    global current_round, client_updates
    
    # Clear updates from previous round
    client_updates = {}
    
    # Increment round ID
    round_id = 1
    if current_round:
        round_id = current_round.round_id + 1
    
    # Create new round
    current_round = FederatedRound(
        round_id=round_id,
        start_time=get_current_time(),
        status="active"
    )
    
    logger.info(f"Created new federated round: {round_id}")
    return current_round

def evaluate_global_model(model):
    """Placeholder for global model evaluation (would use validation data)"""
    # In a real implementation, this would use a validation dataset
    # For now, return dummy metrics
    return {
        'accuracy': 0.85,
        'loss': 0.15,
        'timestamp': get_current_time_str()
    }

@app.on_event("startup")
async def startup_event():
    """Initialize coordinator on startup"""
    logger.info("Starting Federated Learning Coordinator...")
    
    # Ensure we're using the correct model path for the server
    os.environ["MODEL_PATH"] = "/app/models/best_pretrained_model.pth"
    logger.info(f"Server will load model from: {os.environ['MODEL_PATH']}")
    
    # Load global model
    load_or_create_global_model()
    
    # Create initial round
    create_new_round()
    
    logger.info("Federated Learning Coordinator initialized successfully")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": get_current_time_str()
    }

@app.get("/config")
async def get_config():
    """Get federated learning configuration"""
    return {
        "federated_rounds": federated_config['federated_rounds'],
        "privacy": {
            "enabled": federated_config['enable_dp'],
            "max_epsilon": federated_config['max_epsilon'],
            "noise_multiplier": federated_config['noise_multiplier'],
            "max_grad_norm": federated_config['max_grad_norm'],
            "delta": federated_config['delta']
        },
        "version": "1.0.0"
    }

@app.get("/status")
async def get_status():
    """Get current federated learning status"""
    global current_round, registered_clients, client_updates
    
    return {
        "active": True,
        "current_round": current_round.dict() if current_round else None,
        "registered_clients": len(registered_clients),
        "client_ids": list(registered_clients.keys()),
        "updates_received": len(client_updates),
        "config": federated_config
    }

@app.post("/clients/register")
async def register_client(client_info: ClientInfo):
    """Register a client for federated learning"""
    global registered_clients
    
    # Register or update client info
    registered_clients[client_info.client_id] = client_info
    
    logger.info(f"Client registered: {client_info.client_id}")
    return {"success": True, "client_id": client_info.client_id}

@app.get("/rounds/current")
async def get_current_round():
    """Get information about the current round"""
    global current_round
    if not current_round:
        raise HTTPException(status_code=404, detail="No active round")
    
    return current_round

@app.post("/rounds/new")
async def start_new_round(background_tasks: BackgroundTasks):
    """Start a new federated learning round"""
    round = create_new_round()
    return round

@app.get("/models/global")
async def get_global_model_info():
    """Get information about the current global model"""
    global global_model, current_round
    
    if global_model is None:
        try:
            global_model = load_or_create_global_model()
        except Exception as e:
            logger.error(f"Failed to load global model: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load global model: {str(e)}")
    
    # Get latest model file
    model_files = list(model_history_dir.glob("model_round_*.pth"))
    latest_model = None
    latest_round = 0
    
    if model_files:
        for model_file in model_files:
            try:
                # Extract round number from filename
                round_num = int(model_file.name.split("_")[2])
                if round_num > latest_round:
                    latest_round = round_num
                    latest_model = model_file
            except Exception:
                pass
    
    return {
        "current_round": current_round.round_id if current_round else 0,
        "model_version": f"1.0.{latest_round}",
        "num_identities": getattr(global_model, "num_identities", 100),
        "feature_dim": getattr(global_model, "feature_dim", 512),
        "latest_model_path": str(latest_model) if latest_model else None,
        "privacy_enabled": getattr(global_model, "privacy_enabled", False),
        "registered_clients": len(registered_clients)
    }

@app.get("/models/global/download")
async def download_global_model():
    """Download the current global model"""
    model_path = model_dir / "global_model.pth"
    
    if not model_path.exists():
        # If model doesn't exist, create it
        global_model = load_or_create_global_model()
        save_model_version(global_model, 0, {})
    
    return FileResponse(
        path=model_path,
        filename="global_model.pth",
        media_type="application/octet-stream"
    )

@app.post("/models/update")
async def notify_model_update(request: ModelUpdateRequest):
    """Notify coordinator of pending model update"""
    global client_updates, current_round
    
    logger.info(f"Model update notification from client {request.client_id}")
    
    try:
        # Verify client is registered
        if request.client_id not in registered_clients:
            raise HTTPException(status_code=400, detail="Client not registered")
        
        # Verify round is active
        if not current_round or current_round.status != "active":
            raise HTTPException(status_code=400, detail="No active round")
        
        # Verify round number matches
        if request.round_id != current_round.round_id:
            raise HTTPException(status_code=400, detail=f"Invalid round ID. Expected {current_round.round_id}, got {request.round_id}")
        
        # Get current mapping size
        mapping_wrapper = get_current_mapping()
        if not mapping_wrapper:
            raise HTTPException(status_code=500, detail="Global mapping not available")

        # Extract the actual mapping dict regardless of wrapper structure
        if "identity_to_index" in mapping_wrapper:
            server_mapping_dict = mapping_wrapper["identity_to_index"]
        elif "mapping" in mapping_wrapper:
            server_mapping_dict = mapping_wrapper["mapping"]
        else:
            server_mapping_dict = mapping_wrapper  # fallback – assume it *is* the mapping

        server_mapping_size = len(server_mapping_dict)

        # Verify mapping size in metadata
        if request.mapping_size != server_mapping_size:
            raise HTTPException(
                status_code=400,
                detail=f"Mapping size mismatch. Server has {server_mapping_size} classes, client has {request.mapping_size}"
            )
        
        return {"success": True, "round_id": current_round.round_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing model update notification: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/update/{client_id}/{round_id}/upload")
async def upload_model_update(
    client_id: str,
    round_id: int,
    background_tasks: BackgroundTasks,
    update: Dict = Body(...)
):
    """Upload model weights for federated learning"""
    global client_updates, current_round
    
    logger.info(f"Receiving model update from client {client_id} for round {round_id}")
    
    try:
        # Verify client is registered
        if client_id not in registered_clients:
            raise HTTPException(status_code=400, detail="Client not registered")
        
        # Verify round is active
        if not current_round or current_round.status != "active":
            raise HTTPException(status_code=400, detail="No active round")
        
        # Verify round number matches
        if round_id != current_round.round_id:
            raise HTTPException(status_code=400, detail=f"Invalid round ID. Expected {current_round.round_id}, got {round_id}")
        
        # Get current mapping size
        mapping_wrapper = get_current_mapping()
        if not mapping_wrapper:
            raise HTTPException(status_code=500, detail="Global mapping not available")

        # Extract the actual mapping dict regardless of wrapper structure
        if "identity_to_index" in mapping_wrapper:
            server_mapping_dict = mapping_wrapper["identity_to_index"]
        elif "mapping" in mapping_wrapper:
            server_mapping_dict = mapping_wrapper["mapping"]
        else:
            server_mapping_dict = mapping_wrapper  # fallback – assume it *is* the mapping

        server_mapping_size = len(server_mapping_dict)

        # Verify mapping size in metadata
        if update.get("mapping_size") != server_mapping_size:
            raise HTTPException(
                status_code=400,
                detail=f"Mapping size mismatch. Server has {server_mapping_size} classes, client has {update.get('mapping_size')}"
            )
        
        # Decode base64-encoded weights back into a state_dict
        try:
            decoded_bytes = base64.b64decode(update["weights"])
            buffer = io.BytesIO(decoded_bytes)
            state_dict = torch.load(buffer, map_location=torch.device('cpu'))
        except Exception as e:
            logger.error(f"Failed to decode model weights from client {client_id}: {e}")
            raise HTTPException(status_code=400, detail="Invalid model weights format")

        # Store update
        client_updates[client_id] = {
            "parameters": state_dict,
            "dataset_size": registered_clients[client_id].dataset_size,
            "timestamp": get_current_time(),
            "mapping_size": update["mapping_size"],
            "mapping_hash": update.get("mapping_hash")
        }
        
        # Add client to active clients list
        if client_id not in current_round.active_clients:
            current_round.active_clients.append(client_id)
        
        # Check if we have enough updates to aggregate
        if len(client_updates) >= federated_config.get("min_clients", 2):
            # Trigger aggregation in background
            background_tasks.add_task(aggregate_and_update_model)
        
        return {"success": True, "message": "Model update received"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing model update: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def aggregate_and_update_model():
    """Aggregate client updates and update the global model"""
    global global_model, client_updates, current_round
    
    # Runtime safety: ensure required globals are set (helps both mypy & runtime)
    assert global_model is not None, "Global model not initialized"
    assert current_round is not None, "Current round not initialized"

    logger.info(f"Starting model aggregation for round {current_round.round_id}")
    
    try:
        # Verify mapping consistency
        mapping_wrapper = get_current_mapping()
        if not mapping_wrapper:
            raise ValueError("Global mapping not available")
        
        # Extract real mapping
        if "identity_to_index" in mapping_wrapper:
            server_mapping_dict = mapping_wrapper["identity_to_index"]
        elif "mapping" in mapping_wrapper:
            server_mapping_dict = mapping_wrapper["mapping"]
        else:
            server_mapping_dict = mapping_wrapper
        mapping_size = len(server_mapping_dict)
        mapping_hash = hashlib.sha256(json.dumps(sorted(server_mapping_dict.items())).encode()).hexdigest()[:8]
        
        # Verify all clients have correct mapping size and version
        for client_id, update in client_updates.items():
            if update["mapping_size"] != mapping_size:
                raise ValueError(f"Mapping size mismatch for client {client_id}")
            if update.get("mapping_hash") != mapping_hash:
                raise ValueError(f"Mapping version mismatch for client {client_id}")
        
        # Aggregate updates
        aggregated_params = aggregate_model_updates(client_updates)
        
        # Verify aggregated model dimensions
        if 'classifier.weight' in aggregated_params:
            classifier_size = aggregated_params['classifier.weight'].shape[0]
            if classifier_size != mapping_size:
                raise ValueError(f"Aggregated model classifier size ({classifier_size}) doesn't match mapping size ({mapping_size})")
        
        # Update global model
        global_model.load_state_dict(aggregated_params)
        
        # Evaluate the aggregated model
        metrics = evaluate_global_model(global_model)
        
        # Add mapping metrics
        metrics.update({
            'mapping_size': mapping_size,
            'mapping_hash': mapping_hash,
            'mapping_version': get_current_time_str()
        })
        
        # Save the new model version with mapping metadata
        save_path = save_model_version(global_model, current_round.round_id, metrics)
        
        # Update round status
        current_round.aggregation_completed = True
        current_round.status = "completed"
        current_round.metrics = metrics
        
        logger.info(f"Model aggregation completed for round {current_round.round_id}")
        
        # If we've reached the max rounds, set status to "final"
        if current_round.round_id >= federated_config['federated_rounds']:
            current_round.status = "final"
            logger.info("Final federated round completed")
            
    except Exception as e:
        logger.error(f"Error during model aggregation: {e}")
        current_round.status = "failed"
        current_round.metrics = {'error': str(e)}
        
        # Create new round to continue training
        create_new_round()

@app.get("/api/mapping")
async def get_global_mapping(request: Request):
    """
    Get the global class-to-user mapping using centralized identity_mapping.json
    
    Returns:
        Dict with class indices as keys and user IDs as values
    """
    client_id = request.headers.get("X-Client-ID")
    logger.info(f"Mapping request from client: {client_id}")
    
    mapping_data = get_current_mapping()
    
    if "error" in mapping_data:
        return {
            "success": False,
            "error": mapping_data["error"],
            "mapping": {},
            "count": 0
        }
    
    # Convert to legacy format for API compatibility
    legacy_mapping = mapping_data.get("index_to_identity", {})
    
    return {
        "success": True,
        "mapping": legacy_mapping,
        "count": len(legacy_mapping),
        "version": mapping_data.get("version", "unknown"),
        "source": "centralized_identity_mapping"
    }

@app.post("/api/mapping/update")
async def update_global_mapping(request: MappingUpdateRequest):
    """
    Update the global class-to-user mapping
    
    Args:
        request: Request containing array of classes in order
        
    Returns:
        Updated mapping
    """
    logger.info("POST /api/mapping/update endpoint called")
    try:
        # Get current mapping - will be replaced
        current_mapping = get_current_mapping()
        
        # Check if update is needed
        if not request.force_update and current_mapping:
            current_classes = list(current_mapping.values())
            if current_classes == request.classes:
                return {
                    "success": True,
                    "message": "Mapping unchanged - already matches requested classes",
                    "mapping": current_mapping,
                    "count": len(current_mapping)
                }
        
        # Create new mapping from provided classes
        new_mapping = {str(idx): class_id for idx, class_id in enumerate(request.classes)}
        
        # Save the new mapping
        if save_mapping(new_mapping):
            logger.info(f"Global mapping updated with {len(new_mapping)} classes")
            
            # Return the new mapping
            return {
                "success": True,
                "message": "Global mapping updated successfully",
                "mapping": new_mapping,
                "count": len(new_mapping)
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to save global mapping")
            
    except Exception as e:
        logger.error(f"Error updating global mapping: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update global mapping: {str(e)}")

@app.get("/api/mapping/generate")
async def generate_mapping():
    """
    Generate a new global mapping from partitioned data directories
    This follows the same approach as in improved_privacy_training.py
    """
    logger.info("GET /api/mapping/generate endpoint called")
    try:
        # Check if partitioned data directory exists
        partitioned_path = Path("/app/data/partitioned")
        directory_exists = partitioned_path.exists()
        
        # Generate new mapping from directories
        new_mapping = generate_global_mapping_from_directories()
        
        if not new_mapping:
            return {
                "success": False,
                "message": "Failed to generate mapping - no classes found",
                "count": 0,
                "directory_exists": directory_exists
            }
        
        # Check if this is a default mapping
        is_default = len(new_mapping) == 100 and all(new_mapping[str(i)] == str(i) for i in range(100))
        
        # Save the new mapping
        if save_mapping(new_mapping):
            logger.info(f"Generated and saved new global mapping with {len(new_mapping)} classes")
            
            return {
                "success": True,
                "message": "Global mapping generated successfully" + (" (using default mapping)" if is_default else ""),
                "mapping": new_mapping,
                "count": len(new_mapping),
                "directory_exists": directory_exists,
                "is_default_mapping": is_default,
                "data_path": str(partitioned_path)
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to save generated mapping")
            
    except Exception as e:
        logger.error(f"Error generating mapping: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate mapping: {str(e)}")

@app.get("/api/mapping/refresh")
async def refresh_global_mapping(request: Request):
    """
    Force refresh the global mapping from centralized identity_mapping.json
    
    Returns:
        Updated mapping
    """
    try:
        logger.info("Force refreshing global mapping from centralized source")
        
        # Get client ID from headers
        client_id = request.headers.get("X-Client-ID")
        logger.info(f"Mapping refresh request from client: {client_id}")
        
        # Refresh centralized mapping
        success = refresh_mapping()
        
        if success:
            mapping_data = get_current_mapping()
            
            if "error" not in mapping_data:
                logger.info(f"Global mapping refreshed with {mapping_data.get('total_identities', 0)} identities")
                
                # Return the new mapping in legacy format
                legacy_mapping = mapping_data.get("index_to_identity", {})
                
                return {
                    "success": True,
                    "message": "Global mapping refreshed successfully from centralized source",
                    "mapping": legacy_mapping,
                    "count": len(legacy_mapping),
                    "version": mapping_data.get("version", "unknown"),
                    "source": mapping_data.get("source_file", "unknown"),
                    "timestamp": get_current_time_str()
                }
            else:
                raise HTTPException(status_code=500, detail=f"Failed to get refreshed mapping: {mapping_data['error']}")
        else:
            raise HTTPException(status_code=500, detail="Failed to refresh centralized mapping")
            
    except Exception as e:
        logger.error(f"Error refreshing global mapping: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to refresh global mapping: {str(e)}")

@app.post("/api/mapping/add-user")
async def add_user_to_mapping(request: Dict = Body(...)):
    """Add a new user to the global mapping"""
    try:
        user_id = request.get("user_id")
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        
        # Load current identity mapping
        identity_mapping_file = Path("/app/data/identity_mapping.json")
        
        if identity_mapping_file.exists():
            with open(identity_mapping_file, 'r') as f:
                mapping_data = json.load(f)
        else:
            # Create new mapping structure
            mapping_data = {
                "version": "1.0.0",
                "mapping": {},
                "total_identities": 0,
                "hash": ""
            }
        
        current_mapping = mapping_data.get("mapping", {})
        
        # Check if user already exists
        if user_id in current_mapping:
            logger.info(f"User {user_id} already exists in mapping")
            return {"success": True, "message": f"User {user_id} already exists in mapping", "user_id": user_id}
        
        # Find the next available index
        existing_indices = set(current_mapping.values())
        next_index = 0
        while next_index in existing_indices:
            next_index += 1
        
        # Add the new user
        current_mapping[user_id] = next_index
        mapping_data["mapping"] = current_mapping
        mapping_data["total_identities"] = len(current_mapping)
        
        # Update hash
        mapping_str = json.dumps(current_mapping, sort_keys=True)
        mapping_data["hash"] = hashlib.sha256(mapping_str.encode()).hexdigest()[:16]
        
        # Save updated mapping
        identity_mapping_file.parent.mkdir(parents=True, exist_ok=True)
        with open(identity_mapping_file, 'w') as f:
            json.dump(mapping_data, f, indent=2)
        
        # Clear cache to force reload
        global _global_mapping_cache
        _global_mapping_cache = None
        
        logger.info(f"Added user {user_id} to global mapping with index {next_index}")
        
        return {
            "success": True,
            "message": f"User {user_id} added to mapping",
            "user_id": user_id,
            "mapping_index": next_index,
            "total_identities": len(current_mapping)
        }
        
    except Exception as e:
        logger.error(f"Error adding user to mapping: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add user to mapping: {str(e)}")

@app.get("/api/mapping/debug")
async def debug_mapping(request: Request):
    """Debug endpoint to view the current mapping status"""
    try:
        # Get client ID from headers
        client_id = request.headers.get("X-Client-ID")
        logger.info(f"Mapping debug request from client: {client_id}")
        
        # Check if mapping file exists
        file_exists = GLOBAL_MAPPING_PATH.exists()
        file_size = GLOBAL_MAPPING_PATH.stat().st_size if file_exists else 0
        
        # Get current mapping
        mapping = get_current_mapping()
        
        # Get sample entries
        sample = {k: mapping[k] for k in list(mapping.keys())[:5]} if mapping else {}
        
        return {
            "success": True,
            "file_exists": file_exists,
            "file_path": str(GLOBAL_MAPPING_PATH),
            "file_size": file_size,
            "mapping_count": len(mapping),
            "cache_active": _global_mapping_cache is not None,
            "sample_entries": sample,
            "timestamp": get_current_time_str()
        }
    except Exception as e:
        logger.error(f"Error in debug mapping: {e}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    uvicorn.run("coordinator:app", host="0.0.0.0", port=8000, reload=True) 