"""
Phone-Based Biometric Authentication API
Lightweight FastAPI implementation with memory optimization
"""

import os
import sys
import logging
import asyncio
import gc
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path

# Third-party imports
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import torch
import sqlite3
import psutil

# Import federated integration
from .federated_integration import get_federated_integration, initialize_federated_integration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('D:/logs/api.log', maxBytes=10*1024*1024, backupCount=3),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Global variables
MODEL_CACHE = {}
MEMORY_THRESHOLD = 0.8
CLEANUP_INTERVAL = 300

class MemoryManager:
    """Lightweight memory manager"""
    
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.system_memory = psutil.virtual_memory().total / (1024**3)
        self.gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3) if self.gpu_available else 0
        logger.info(f"System Memory: {self.system_memory:.1f}GB, GPU Memory: {self.gpu_memory:.1f}GB")
    
    def get_memory_status(self) -> Dict[str, float]:
        """Get current memory usage"""
        sys_mem = psutil.virtual_memory()
        gpu_mem = {}
        
        if self.gpu_available:
            gpu_mem = {
                'gpu_allocated': torch.cuda.memory_allocated() / (1024**3),
                'gpu_cached': torch.cuda.memory_reserved() / (1024**3),
                'gpu_total': self.gpu_memory
            }
        
        return {
            'system_used': sys_mem.percent / 100,
            'system_available': sys_mem.available / (1024**3),
            'system_total': self.system_memory,
            **gpu_mem
        }

memory_manager = MemoryManager()

class BiometricDatabase:
    """Lightweight SQLite database"""
    
    def __init__(self, db_path: str = "D:/data/biometric.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_db()
    
    def init_db(self):
        """Initialize database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT UNIQUE NOT NULL,
                    enrolled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    active BOOLEAN DEFAULT 1
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS biometric_templates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    template_data BLOB NOT NULL,
                    template_type TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS auth_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    success BOOLEAN NOT NULL,
                    confidence_score REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ip_address TEXT,
                    device_info TEXT
                )
            """)
            
            conn.commit()
    
    def get_connection(self):
        """Get optimized database connection"""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=10000")
        return conn
    
    def enroll_user(self, user_id: str, template_data: bytes, template_type: str = "face"):
        """Enroll user"""
        with self.get_connection() as conn:
            conn.execute("INSERT OR REPLACE INTO users (user_id, enrolled_at) VALUES (?, CURRENT_TIMESTAMP)", (user_id,))
            conn.execute("INSERT INTO biometric_templates (user_id, template_data, template_type) VALUES (?, ?, ?)", 
                        (user_id, template_data, template_type))
            conn.commit()
    
    def get_user_templates(self, user_id: str) -> list:
        """Get user templates"""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT template_data, template_type, created_at FROM biometric_templates WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
            return cursor.fetchall()
    
    def log_auth_attempt(self, user_id: str, success: bool, confidence: float, ip_address: str = None, device_info: str = None):
        """Log authentication attempt"""
        with self.get_connection() as conn:
            conn.execute("INSERT INTO auth_attempts (user_id, success, confidence_score, ip_address, device_info) VALUES (?, ?, ?, ?, ?)",
                        (user_id, success, confidence, ip_address, device_info))
            if success:
                conn.execute("UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE user_id = ?", (user_id,))
            conn.commit()

db = BiometricDatabase()

class BiometricProcessor:
    """Lightweight biometric processor"""
    
    def __init__(self):
        self.model_path = "D:/models/best_pretrained_model.pth"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
    
    def load_model(self):
        """Load model with caching"""
        if "main_model" in MODEL_CACHE:
            MODEL_CACHE["main_model"]["last_used"] = datetime.now()
            return MODEL_CACHE["main_model"]["model"]
        
        try:
            if os.path.exists(self.model_path):
                logger.info(f"Loading model from {self.model_path}")
                model = torch.load(self.model_path, map_location=self.device, weights_only=True)
                
                if hasattr(model, 'eval'):
                    model.eval()
                
                MODEL_CACHE["main_model"] = {
                    "model": model,
                    "last_used": datetime.now(),
                    "size_mb": os.path.getsize(self.model_path) / (1024*1024)
                }
                
                logger.info(f"Model loaded successfully ({MODEL_CACHE['main_model']['size_mb']:.1f}MB)")
                return model
            else:
                logger.error(f"Model file not found: {self.model_path}")
                return None
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return None
    
    async def process_image(self, image_data: bytes) -> Dict[str, Any]:
        """Process biometric image"""
        try:
            temp_path = f"D:/cache/temp_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            
            with open(temp_path, 'wb') as f:
                f.write(image_data)
            
            result = await asyncio.get_event_loop().run_in_executor(None, self._process_image_sync, temp_path)
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return result
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise HTTPException(status_code=500, detail="Image processing failed")
    
    def _process_image_sync(self, image_path: str) -> Dict[str, Any]:
        """Synchronous image processing"""
        import cv2
        import numpy as np
        
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not load image")
            
            # Resize to conserve memory
            image = cv2.resize(image, (224, 224))
            
            # Dummy processing - replace with actual inference
            features = np.random.randn(512).astype(np.float32).tobytes()
            confidence = np.random.rand()
            
            return {
                "success": True,
                "confidence": confidence,
                "features": features,
                "processing_time": 0.1
            }
        except Exception as e:
            logger.error(f"Image processing error: {str(e)}")
            return {"error": "Processing failed", "confidence": 0.0}

processor = BiometricProcessor()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting biometric API...")
    
    # Initialize federated integration
    federated_integration = initialize_federated_integration()
    if federated_integration:
        logger.info("Federated learning integration initialized")
    else:
        logger.warning("Federated learning integration failed to initialize")
    
    # Start background cleanup
    asyncio.create_task(periodic_cleanup())
    
    yield
    
    logger.info("Shutting down biometric API...")
    cleanup_resources()

# FastAPI app
app = FastAPI(
    title="Phone-Based Biometric Authentication API",
    description="Lightweight biometric authentication system",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Routes
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    memory_status = memory_manager.get_memory_status()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "memory": memory_status,
        "gpu_available": torch.cuda.is_available(),
        "model_cached": "main_model" in MODEL_CACHE
    }

@app.get("/metrics")
async def get_metrics():
    """Metrics endpoint"""
    memory_status = memory_manager.get_memory_status()
    
    with db.get_connection() as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        
        cursor = conn.execute("SELECT COUNT(*) FROM auth_attempts WHERE timestamp > datetime('now', '-1 hour')")
        recent_attempts = cursor.fetchone()[0]
    
    return {
        "memory": memory_status,
        "database": {"total_users": user_count, "recent_attempts": recent_attempts},
        "cache": {"models_cached": len(MODEL_CACHE), "cache_size_mb": sum(v.get("size_mb", 0) for v in MODEL_CACHE.values())}
    }

@app.post("/enroll")
async def enroll_user(user_id: str, image: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    """Enroll a new user"""
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image format")
    
    if image.size > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large")
    
    try:
        image_data = await image.read()
        result = await processor.process_image(image_data)
        
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])
        
        db.enroll_user(user_id, result["features"], "face")
        
        background_tasks.add_task(db.log_auth_attempt, user_id, True, result["confidence"], None, "enrollment")
        
        return {
            "success": True,
            "user_id": user_id,
            "confidence": result["confidence"],
            "enrolled_at": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Enrollment error for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Enrollment failed")

@app.post("/authenticate")
async def authenticate_user(user_id: str, image: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    """Authenticate user"""
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image format")
    
    try:
        image_data = await image.read()
        result = await processor.process_image(image_data)
        
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])
        
        templates = db.get_user_templates(user_id)
        
        if not templates:
            background_tasks.add_task(db.log_auth_attempt, user_id, False, 0.0, None, "user_not_found")
            raise HTTPException(status_code=404, detail="User not enrolled")
        
        confidence = result["confidence"]
        threshold = 0.7
        success = confidence >= threshold
        
        background_tasks.add_task(db.log_auth_attempt, user_id, success, confidence, None, "authentication")
        
        return {
            "success": success,
            "user_id": user_id,
            "confidence": confidence,
            "threshold": threshold,
            "authenticated_at": datetime.now().isoformat() if success else None
        }
    except Exception as e:
        logger.error(f"Authentication error for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Authentication failed")

@app.get("/users/{user_id}/history")
async def get_user_history(user_id: str, limit: int = 50):
    """Get user authentication history"""
    with db.get_connection() as conn:
        cursor = conn.execute("SELECT success, confidence_score, timestamp, device_info FROM auth_attempts WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?", (user_id, limit))
        results = cursor.fetchall()
        
        return {
            "user_id": user_id,
            "history": [{"success": bool(row[0]), "confidence": row[1], "timestamp": row[2], "device_info": row[3]} for row in results]
        }

# Federated Learning Endpoints
@app.get("/federated/status")
async def get_federated_status():
    """Get federated learning status"""
    try:
        integration = get_federated_integration()
        status = integration.get_federated_status()
        return status
    except Exception as e:
        logger.error(f"Error getting federated status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get federated status")

@app.post("/federated/trigger-round")
async def trigger_federated_round():
    """Manually trigger a federated learning round"""
    try:
        integration = get_federated_integration()
        success = integration.trigger_federated_round()
        
        if success:
            return {"success": True, "message": "Federated round triggered successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to trigger federated round")
    except Exception as e:
        logger.error(f"Error triggering federated round: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to trigger federated round")

@app.get("/federated/model-history")
async def get_model_history():
    """Get federated model update history"""
    try:
        integration = get_federated_integration()
        history = integration.get_model_history()
        return {"model_history": history}
    except Exception as e:
        logger.error(f"Error getting model history: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get model history")

@app.post("/admin/reload-model")
async def reload_model(request_data: dict):
    """Admin endpoint to reload the biometric model"""
    try:
        reason = request_data.get("reason", "manual")
        model_hash = request_data.get("model_hash", "unknown")
        
        # Clear model cache to force reload
        global MODEL_CACHE
        MODEL_CACHE.clear()
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"Model cache cleared due to {reason} (hash: {model_hash})")
        
        return {
            "success": True,
            "message": "Model cache cleared, will reload on next request",
            "reason": reason,
            "model_hash": model_hash,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error reloading model: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to reload model")

# Background tasks
async def periodic_cleanup():
    """Periodic cleanup task"""
    while True:
        try:
            await asyncio.sleep(CLEANUP_INTERVAL)
            cleanup_resources()
        except Exception as e:
            logger.error(f"Cleanup error: {str(e)}")

def cleanup_resources():
    """Clean up resources"""
    try:
        # Clean old cache entries
        current_time = datetime.now()
        expired_keys = []
        
        for key, value in MODEL_CACHE.items():
            if current_time - value["last_used"] > timedelta(hours=1):
                expired_keys.append(key)
        
        for key in expired_keys:
            del MODEL_CACHE[key]
            logger.info(f"Removed expired cache entry: {key}")
        
        # Force garbage collection
        gc.collect()
        
        # GPU cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Memory status
        memory_status = memory_manager.get_memory_status()
        if memory_status["system_used"] > MEMORY_THRESHOLD:
            logger.warning(f"High memory usage: {memory_status['system_used']:.1%}")
        
    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=1, log_level="info", reload=False) 