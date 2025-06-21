"""
Federated Learning Coordinator for Biometric API
Manages federated learning rounds and model aggregation
"""

import os
import logging
import json
import time
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn

# Import privacy components
from .privacy.privacy_engine import PrivacyEngine
from .utils.security import generate_uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class FederatedRound(BaseModel):
    round_id: int
    start_time: datetime
    status: str
    active_clients: List[str] = []
    aggregation_completed: bool = False
    metrics: Optional[Dict[str, float]] = None

class ModelUpdateRequest(BaseModel):
    client_id: str
    round_id: int
    model_update_type: str = "weights"  # weights or gradients
    dataset_size: int
    privacy_metrics: Optional[Dict[str, float]] = None
    training_metrics: Optional[Dict[str, float]] = None

# Global state
global_model = None
current_round: FederatedRound = None
registered_clients: Dict[str, ClientInfo] = {}
client_updates: Dict[str, Dict] = {}
federated_config = {
    'federated_rounds': int(os.getenv('FEDERATED_ROUNDS', '10')),
    'max_epsilon': float(os.getenv('MAX_DP_EPSILON', '100.0')),
    'noise_multiplier': float(os.getenv('DP_NOISE_MULTIPLIER', '0.2')),
    'max_grad_norm': float(os.getenv('DP_MAX_GRAD_NORM', '5.0')),
    'delta': float(os.getenv('DP_DELTA', '1e-5')),
    'enable_dp': os.getenv('ENABLE_DP', 'true').lower() == 'true'
}

# Create directory for model storage
model_dir = Path('/app/models')
model_dir.mkdir(exist_ok=True)
model_history_dir = model_dir / "history"
model_history_dir.mkdir(exist_ok=True)

def load_or_create_global_model():
    """Load or create the global federated model"""
    global global_model
    
    try:
        # Import dynamically to avoid circular import
        from .models.privacy_biometric_model import PrivacyBiometricModel
    except ImportError:
        logger.error("Failed to import PrivacyBiometricModel")
        raise ImportError("Missing privacy_biometric_model module")
    
    # Set model path
    model_path = Path(os.getenv('MODEL_PATH', '/app/models/best_pretrained_model.pth'))
    
    if not model_path.exists():
        logger.warning(f"Model file not found at {model_path}, creating new model")
        # Load number of identities from config or use default 
        num_identities = 100  # Default
        global_model = PrivacyBiometricModel(
            num_identities=num_identities,
            privacy_enabled=federated_config['enable_dp']
        )
        # Save initial model
        torch.save(global_model.state_dict(), model_path)
        logger.info(f"Created new global model with {num_identities} identities")
    else:
        logger.info(f"Loading global model from {model_path}")
        # Load state dictionary
        state_dict = torch.load(model_path, map_location='cpu')
        
        # Determine params from state_dict
        num_identities = 100  # Default
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            num_identities = state_dict.get('num_identities', 100)
        
        # Create model with correct parameters
        global_model = PrivacyBiometricModel(
            num_identities=num_identities, 
            privacy_enabled=federated_config['enable_dp']
        )
        
        # Load weights
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            global_model.load_state_dict(state_dict['state_dict'])
        else:
            global_model.load_state_dict(state_dict)
            
        logger.info(f"Global model loaded with {num_identities} identities")
    
    global_model.eval()  # Set to evaluation mode
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

def save_model_version(model, round_id: int, metrics: Dict[str, float] = None):
    """Save a version of the model with metadata"""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"model_round_{round_id}_{timestamp}.pth"
    save_path = model_history_dir / filename
    
    # Create metadata
    metadata = {
        'round_id': round_id,
        'timestamp': timestamp,
        'metrics': metrics or {},
        'active_clients': len(client_updates),
        'client_ids': list(client_updates.keys()),
    }
    
    # Save model with metadata
    torch.save({
        'state_dict': model.state_dict(),
        'metadata': metadata,
        'num_identities': getattr(model, 'num_identities', 100)
    }, save_path)
    
    # Update current global model
    torch.save({
        'state_dict': model.state_dict(),
        'metadata': metadata,
        'num_identities': getattr(model, 'num_identities', 100)
    }, model_dir / "global_model.pth")
    
    logger.info(f"Saved model version for round {round_id} at {save_path}")
    return str(save_path)

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
        start_time=datetime.utcnow(),
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
        'timestamp': datetime.utcnow().isoformat()
    }

@app.on_event("startup")
async def startup_event():
    """Initialize coordinator on startup"""
    logger.info("Starting Federated Learning Coordinator...")
    
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
        "timestamp": datetime.utcnow().isoformat()
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
        save_model_version(global_model, 0)
    
    return FileResponse(
        path=model_path,
        filename="global_model.pth",
        media_type="application/octet-stream"
    )

@app.post("/models/update")
async def submit_model_update(update_info: ModelUpdateRequest, background_tasks: BackgroundTasks):
    """Submit a model update for the current round"""
    global client_updates, current_round
    
    # Validate client
    if update_info.client_id not in registered_clients:
        raise HTTPException(status_code=403, detail="Client not registered")
    
    # Validate round
    if not current_round or update_info.round_id != current_round.round_id:
        raise HTTPException(status_code=400, detail="Invalid round ID")
    
    # Store client metadata
    client_updates[update_info.client_id] = {
        "dataset_size": update_info.dataset_size,
        "waiting_for_parameters": True,
        "received_at": datetime.utcnow().isoformat(),
        "training_metrics": update_info.training_metrics,
        "privacy_metrics": update_info.privacy_metrics
    }
    
    # Update round status
    current_round.active_clients = list(client_updates.keys())
    
    logger.info(f"Received update info from {update_info.client_id} for round {update_info.round_id}")
    
    return {
        "success": True, 
        "upload_url": f"/models/update/{update_info.client_id}/{update_info.round_id}/upload",
        "client_count": len(client_updates)
    }

@app.post("/models/update/{client_id}/{round_id}/upload")
async def upload_model_parameters(
    client_id: str, 
    round_id: int, 
    model_file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """Upload model parameters for a round"""
    global client_updates, current_round, global_model
    
    if client_id not in client_updates:
        raise HTTPException(status_code=403, detail="Update not initiated")
    
    if int(round_id) != current_round.round_id:
        raise HTTPException(status_code=400, detail="Invalid round ID")
    
    try:
        # Read uploaded model file
        contents = await model_file.read()
        
        # Save temporarily
        temp_path = f"/app/cache/temp_model_{client_id}_{round_id}.pth"
        with open(temp_path, "wb") as f:
            f.write(contents)
        
        # Load parameters
        client_state_dict = torch.load(temp_path, map_location='cpu')
        
        # Store client parameters
        client_updates[client_id]["parameters"] = client_state_dict
        client_updates[client_id]["waiting_for_parameters"] = False
        
        logger.info(f"Received parameters from {client_id} for round {round_id}")
        
        # Check if all expected clients have submitted updates
        ready_for_aggregation = all(
            not update.get("waiting_for_parameters", True) 
            for update in client_updates.values()
        )
        
        # If all expected clients submitted, aggregate in background
        if ready_for_aggregation and len(client_updates) > 0:
            if background_tasks:
                background_tasks.add_task(aggregate_and_update_model)
            else:
                asyncio.create_task(aggregate_and_update_model())
        
        return {"success": True, "ready_for_aggregation": ready_for_aggregation}
    
    except Exception as e:
        logger.error(f"Error processing model upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process model: {str(e)}")

async def aggregate_and_update_model():
    """Aggregate client updates and update the global model"""
    global global_model, client_updates, current_round
    
    logger.info(f"Starting model aggregation for round {current_round.round_id}")
    
    try:
        # Aggregate updates
        aggregated_params = aggregate_model_updates(client_updates)
        
        # Update global model
        global_model.load_state_dict(aggregated_params)
        
        # Evaluate the aggregated model
        metrics = evaluate_global_model(global_model)
        
        # Save the new model version
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
        logger.error(f"Error during model aggregation: {str(e)}")
        current_round.status = "failed"
        current_round.metrics = {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("coordinator:app", host="0.0.0.0", port=8000, reload=True) 