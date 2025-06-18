"""
Privacy-Preserving Biometric Authentication API
with Federated Learning, Homomorphic Encryption, and Differential Privacy
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer
import uvicorn
import logging
from datetime import datetime
import os
from pathlib import Path

# Import routes
from .routes import auth, users, analytics, verification
from .utils.rate_limiter import RateLimiter
from .privacy.privacy_engine import PrivacyEngine
from .privacy.federated_manager import FederatedManager
from .db.database import init_db

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

@app.on_event("startup")
async def startup():
    """Initialize components on startup"""
    logger.info("Initializing API components...")
    
    # Create necessary directories
    for path in ["data", "models", "logs"]:
        Path(path).mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    init_db()
    await privacy_engine.initialize()
    await federated_manager.initialize()
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
    """Apply rate limiting"""
    client_ip = request.client.host
    path = request.url.path
    
    # Different rate limits for auth vs other endpoints
    if path.startswith("/api/auth"):
        if not rate_limiter.check_rate_limit(client_ip, "auth", max_requests=5):
            return JSONResponse(
                status_code=429,
                content={"error": "rate_limit_exceeded", "message": "Too many authentication attempts"}
            )
    else:
        if not rate_limiter.check_rate_limit(client_ip, "general", max_requests=10):
            return JSONResponse(
                status_code=429,
                content={"error": "rate_limit_exceeded", "message": "Too many requests"}
            )
    
    return await call_next(request)

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["authentication"])
app.include_router(users.router, prefix="/api/users", tags=["users"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["analytics"])
app.include_router(verification.router, prefix="/api/verification", tags=["verification"])

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "privacy_engine": await privacy_engine.get_status(),
        "federated_learning": await federated_manager.get_status()
    }

def main():
    """Main entry point for the application"""
    # Get configuration from environment
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "true").lower() == "true"
    
    # Run the application
    uvicorn.run(
        "src.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    main() 