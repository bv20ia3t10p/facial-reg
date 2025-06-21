"""
Privacy-Preserving Biometric Authentication API
with Federated Learning, Homomorphic Encryption, and Differential Privacy
"""

import logging
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer
import uvicorn
from datetime import datetime
import os
from pathlib import Path

# Import routes
from .routes import api_router
from .utils.rate_limiter import RateLimiter
from .privacy.privacy_engine import PrivacyEngine
from .privacy.federated_manager import FederatedManager
from .federated_integration import get_federated_integration, initialize_federated_integration
from .db.database import init_db
from .db.migrate import migrate_database
from .services.service_init import initialize_services

# Import federated learning components
from .federated_client import FederatedClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="BioEmo API",
    description="Privacy-Preserving Biometric Authentication API with FL, HE, and DP",
    version="1.0.0"
)

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
    """Initialize components on startup"""
    global federated_client
    
    logger.info("Initializing API components...")
    
    # Create necessary directories
    for path in ["data", "models", "logs"]:
        Path(path).mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    init_db()
    await privacy_engine.initialize()
    await federated_manager.initialize()
    await migrate_database()
    
    # Initialize services
    initialize_services()
    
    # Initialize federated client if this is a client node
    node_type = os.getenv("NODE_TYPE", "").lower()
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
    global federated_client
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
    global federated_client
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
            training_metrics, privacy_metrics = await federated_client.train_on_local_data(round_id)
            success = await federated_client.submit_model_update(round_id, training_metrics, privacy_metrics)
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
    global federated_client
    if not federated_client:
        return {"active": False, "message": "Federated learning not enabled on this node"}
    
    try:
        # Download latest global model
        model = await federated_client.download_global_model()
        
        # Update model in biometric service
        from .services.service_init import update_biometric_model
        success = update_biometric_model(model)
        
        return {
            "success": success,
            "message": "Successfully downloaded and applied latest global model" if success else "Failed to apply global model"
        }
        
    except Exception as e:
        logger.error(f"Error syncing model: {e}")
        return {"success": False, "message": f"Failed to sync model: {str(e)}"}

if __name__ == "__main__":
    # Get configuration from environment
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "true").lower() == "true"
    
    # Run the application
    uvicorn.run(
        "api.src.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    ) 