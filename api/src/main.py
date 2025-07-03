"""
Privacy-Preserving Biometric Authentication API
with Federated Learning, Homomorphic Encryption, and Differential Privacy
"""

import logging
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer
import uvicorn
from datetime import datetime
import os
from pathlib import Path
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Suppress Opacus validator info messages
logging.getLogger('opacus.validators.module_validator').setLevel(logging.WARNING)
logging.getLogger('opacus.validators.batch_norm').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Import routes
from .routes import api_router
from .utils.rate_limiter import RateLimiter
from .privacy.privacy_engine import PrivacyEngine
from .privacy.federated_manager import FederatedManager
from .federated_integration import get_federated_integration, initialize_federated_integration
from .db.database import init_db
from .db.migrate import migrate_database
from .services.service_init import initialize_services
from .routes.mapping import MappingUpdateRequest, get_current_mapping, save_mapping

# Import federated learning components
from .federated_client import FederatedClient

# Initialize FastAPI app
app = FastAPI(
    title="BioEmo API",
    description="Privacy-Preserving Biometric Authentication API with FL, HE, and DP",
    version="1.0.0"
)

# Get node type from environment
node_type = os.getenv('NODE_TYPE', 'coordinator').lower()
logger.info(f"Starting as {node_type} node")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OAuth2
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Initialize rate limiter
rate_limiter = RateLimiter()

# Initialize privacy components
privacy_engine = PrivacyEngine()
federated_manager = FederatedManager()

# Federated client instance
federated_client = None

@app.on_event("startup")
async def startup():
    """Initialize API components on startup"""
    global federated_client
    logger.info("Starting API initialization...")
    
    # Initialize database and privacy components first
    init_db()
    await privacy_engine.initialize()
    await federated_manager.initialize()
    await migrate_database()
    
    # Check if services are already initialized
    from .services.service_init import _services_initialized, biometric_service, initialize_services
    if not _services_initialized:
        try:
            logger.info("Initializing services...")
            if not initialize_services():
                raise RuntimeError("Service initialization failed")
            
            # Check if biometric service is available
            if biometric_service is None:
                logger.error("CRITICAL ERROR: Biometric service failed to initialize")
                raise RuntimeError("Biometric service initialization failed")
        except Exception as e:
            logger.error(f"Failed to initialize services: {e}", exc_info=True)
            raise
    else:
        logger.info("Services already initialized, skipping initialization")
    
    # Initialize federated client if this is a client node
    if node_type == "client":
        logger.info("Initializing federated learning client...")
        federated_client = FederatedClient()
        await federated_client.initialize()
        logger.info("Federated learning client initialized")
    
    logger.info("API initialization complete")

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses"""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Apply rate limiting (currently disabled)"""
    # Rate limiting is disabled
    return await call_next(request)

# Include API router
app.include_router(api_router, prefix="/api")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    federated_status = None
    if federated_client:
        try:
            federated_status = await federated_client.get_status()
        except Exception as e:
            logger.error(f"Failed to get federated status: {e}")
            federated_status = {"error": str(e)}
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "privacy_engine": await privacy_engine.get_status(),
        "federated_learning": await federated_manager.get_status(),
        "federated_client": federated_status
    }

# Direct mapping endpoint to handle /api/mapping requests
@app.get("/api/mapping")
async def direct_mapping():
    """
    Direct mapping endpoint to handle /api/mapping requests
    This is a workaround for routing issues
    """
    logger.info("Direct mapping endpoint called")
    try:
        # Import the get_current_mapping function from mapping.py
        from .routes.mapping import get_current_mapping
        
        # Get the mapping
        mapping = get_current_mapping()
        
        return {
            "success": True,
            "mapping": mapping,
            "count": len(mapping)
        }
    except Exception as e:
        logger.error(f"Error in direct mapping endpoint: {e}")
        return {"success": False, "error": str(e)}

# Direct mapping update endpoint
@app.post("/api/mapping/update")
async def direct_mapping_update(request: MappingUpdateRequest):
    """
    Direct mapping update endpoint to handle /api/mapping/update requests
    This is a workaround for routing issues
    """
    logger.info("Direct mapping update endpoint called")
    try:
        # Get current mapping
        current_mapping = get_current_mapping()
        
        # Check if update is needed
        if not request.force_update and current_mapping:
            if current_mapping == request.mapping:
                return {
                    "success": True,
                    "message": "Mapping unchanged - already matches requested classes",
                    "mapping": current_mapping,
                    "count": len(current_mapping)
                }
        
        # Create new mapping from provided classes
        new_mapping = request.mapping
        
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
            return {
                "success": False, 
                "message": "Failed to save global mapping"
            }
            
    except Exception as e:
        logger.error(f"Error updating global mapping: {e}")
        return {"success": False, "message": f"Failed to update global mapping: {str(e)}"}

# Direct mapping debug endpoint
@app.get("/api/mapping/debug")
async def direct_mapping_debug():
    """
    Direct mapping debug endpoint to handle /api/mapping/debug requests
    This is a workaround for routing issues
    """
    logger.info("Direct mapping debug endpoint called")
    try:
        # Import the necessary functions from mapping.py
        from .routes.mapping import get_current_mapping, GLOBAL_MAPPING_PATH, _global_mapping_cache
        
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
            "sample_entries": sample
        }
    except Exception as e:
        logger.error(f"Error in debug mapping: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/model/reload")
async def reload_model():
    """Reload the biometric model after training"""
    try:
        logger.info("Reloading biometric model...")
        from .services.service_init import reload_biometric_service
        success = reload_biometric_service()
        if success:
            logger.info("Biometric model reloaded successfully")
            return {"success": True, "message": "Model reloaded successfully"}
        else:
            logger.warning("Failed to reload biometric model")
            return {"success": False, "message": "Failed to reload model"}
    except Exception as e:
        logger.error(f"Failed to reload biometric model: {e}")
        return {"success": False, "message": f"Failed to reload model: {str(e)}"}

# Federated learning endpoints
@app.get("/api/federated/status")
async def get_federated_status():
    """Get status of federated learning client"""
    if not federated_client:
        return {"active": False, "message": "Federated learning not enabled on this node"}
    
    try:
        status = await federated_client.get_status()
        return status
    except Exception as e:
        logger.error(f"Error getting federated status: {e}")
        return {"error": str(e)}

@app.post("/api/federated/trigger-round")
async def trigger_federated_round(background_tasks: BackgroundTasks):
    """Manually trigger participation in a federated round"""
    if not federated_client:
        return {"active": False, "message": "Federated learning not enabled on this node"}
    
    try:
        # Get current round
        current_round = await federated_client.get_current_round()
        if not current_round:
            return {"success": False, "message": "No active federated round available"}
        
        round_id = current_round["round_id"]
        
        # Start training in background
        async def train_and_submit():
            if not federated_client:
                logger.error("Federated client not available for background task.")
                return
            
            training_metrics, privacy_metrics = await federated_client.train_on_local_data(round_id)
            # The model is part of the client state now, so not passed directly
            success = await federated_client.submit_model_update(round_id)
            if success:
                federated_client.current_round = round_id
        
        background_tasks.add_task(train_and_submit)
        
        return {
            "success": True,
            "message": f"Started participation in federated round {round_id}",
            "round_id": round_id
        }
    
    except Exception as e:
        logger.error(f"Error triggering federated round: {e}")
        return {"success": False, "message": f"Failed to trigger federated round: {str(e)}"}

@app.post("/api/federated/sync-model")
async def sync_federated_model():
    """Download latest global model from coordinator"""
    if not federated_client:
        return {"active": False, "message": "Federated learning not enabled on this node"}
    
    try:
        # Download latest global model
        model = await federated_client.load_or_download_model()
        
        if not model:
            return {"success": False, "message": "Failed to download global model"}

        # Update model in biometric service
        from .services.service_init import biometric_service
        if not biometric_service:
            raise HTTPException(status_code=500, detail="Biometric service not initialized")
        
        success = biometric_service.update_model(model)
        
        return {
            "success": success,
            "message": "Successfully downloaded and applied latest global model" if success else "Failed to apply new model"
        }
        
    except Exception as e:
        logger.error(f"Error syncing model: {e}")
        return {"success": False, "message": f"Failed to sync model: {str(e)}"}

@app.on_event("shutdown")
def shutdown_event():
    logger.info("Shutting down API...")

if __name__ == "__main__":
    # Get configuration from environment
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8080"))
    reload = os.getenv("API_RELOAD", "true").lower() == "true"
    
    # Run the application
    uvicorn.run(
        "api.src.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    ) 