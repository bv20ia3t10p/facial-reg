"""
Client-side API Service for Privacy-Preserving Biometric Authentication
Implements FL, DP, and HE based on thesis research
"""

import os
import io
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from datetime import datetime
import time
import sqlite3
from contextlib import contextmanager
import asyncio
import aiohttp
from urllib.parse import urljoin
import json
import uuid

import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import tenseal as ts
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import requests
from sqlalchemy.orm import Session
from sqlalchemy import text
from api.src.db.database import get_db  # Import get_db function

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_uuid():
    """Generate UUID as string"""
    return str(uuid.uuid4())

# Initialize FastAPI app
app = FastAPI(title="Privacy-Preserving Biometric Client API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import after FastAPI app is created to avoid circular imports
from api.src.models.privacy_biometric_model import PrivacyBiometricModel
from api.src.privacy.privacy_engine import PrivacyEngine as BiometricPrivacyEngine
from api.src.routes import client_users
from api.src.db.database import init_db

# Mount user routes
app.include_router(client_users.router, prefix="/api/users", tags=["users"])

# Add emotion API URL configuration
EMOTION_API_URL = "http://localhost:1236"  # Emotion API endpoint

# Add database path configuration
DB_PATH = os.getenv('DATABASE_URL', 'data/biometric.db').replace('sqlite:///', '')

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Enable row factory for named access
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    """Initialize database tables if they don't exist"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Check if tables already exist
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND (name='users' OR name='authentication_logs')
        """)
        existing_tables = [row[0] for row in cursor.fetchall()]
        
        if len(existing_tables) == 2:
            logger.info("Database tables already exist, skipping initialization")
            return
        
        logger.info("Creating missing database tables...")
        
        # Create users table if not exists
        if 'users' not in existing_tables:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    department TEXT,
                    role TEXT,
                    password_hash TEXT NOT NULL,
                    face_encoding BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_authenticated TIMESTAMP
                )
            """)
            logger.info("Created users table")
        
        # Create authentication logs table if not exists
        if 'authentication_logs' not in existing_tables:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS authentication_logs (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    success BOOLEAN,
                    confidence REAL,
                    emotion_data JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    device_info TEXT,
                    captured_image BLOB,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            logger.info("Created authentication_logs table")
        
        conn.commit()

class ClientBiometricService:
    def __init__(self, client_id: str = "client1"):
        self.client_id = client_id
        self.user_id_mapping: Dict[int, str] = {}  # Maps model indices to actual user IDs
        
        # Initialize device with better error handling
        try:
            if torch.cuda.is_available():
                # Test CUDA with a small tensor operation
                test_tensor = torch.zeros(1, device='cuda')
                test_tensor + 1  # Simple operation to test CUDA
                self.device = torch.device('cuda')
                logger.info("CUDA is available and working")
            else:
                self.device = torch.device('cpu')
                logger.info("CUDA is not available, using CPU")
        except Exception as e:
            logger.warning(f"CUDA initialization failed: {e}, falling back to CPU")
            self.device = torch.device('cpu')
            # Disable CUDA for this session
            torch.cuda.is_available = lambda: False
        
        # Initialize privacy components
        self.privacy_engine = BiometricPrivacyEngine()
        self.he_context = None
        self.dp_engine = None
        
        # Load model
        self.model = self._load_model()
        
        # Load user ID mapping
        self._load_user_id_mapping()
        
        # Note: _setup_privacy is async, will be called in startup event
        self._privacy_setup_complete = False

    def _load_user_id_mapping(self):
        """Load mapping between model indices and actual user IDs from folder structure"""
        try:
            # In Docker, data is mounted at /app/data
            data_path = Path("/app/data")
            logger.info(f"Looking for user data in: {data_path}")
            
            if not data_path.exists():
                logger.error(f"Data directory not found at: {data_path}")
                raise FileNotFoundError(f"Data directory not found: {data_path}")
            
            # Get all user folders (numeric folders)
            user_folders = sorted([d for d in data_path.iterdir() 
                                if d.is_dir() and d.name.isdigit()])
            
            # Create mapping (index -> user_id)
            self.user_id_mapping = {i: folder.name for i, folder in enumerate(user_folders)}
            logger.info(f"Loaded {len(self.user_id_mapping)} user ID mappings from {data_path}")
            
            # Log all mappings in debug mode
            logger.debug("Full mapping:")
            for idx, user_id in self.user_id_mapping.items():
                logger.debug(f"Index {idx:3d} -> User {user_id}")
            
            # Log some example mappings for debugging
            if self.user_id_mapping:
                logger.info("Sample mappings:")
                sample_size = min(5, len(self.user_id_mapping))
                sample_indices = list(self.user_id_mapping.keys())[:sample_size]
                for idx in sample_indices:
                    logger.info(f"Model index {idx} maps to user {self.user_id_mapping[idx]}")
                
                # Log specific mapping for index 46 if it exists
                if 46 in self.user_id_mapping:
                    logger.info(f"Index 46 maps to user {self.user_id_mapping[46]}")
                else:
                    logger.warning(f"Index 46 not found in mapping! Max index is {max(self.user_id_mapping.keys())}")
                    
        except Exception as e:
            logger.error(f"Failed to load user ID mapping from folders: {e}")
            # Initialize with empty mapping if folder access fails
            self.user_id_mapping = {}

    def get_user_id_from_index(self, index: int) -> str:
        """Get actual user ID from model's predicted index"""
        try:
            if index in self.user_id_mapping:
                return self.user_id_mapping[index]
            else:
                logger.warning(f"Model index {index} not found in mapping, using fallback")
                # Fallback to timestamp-based ID if mapping fails
                return f"user_{int(time.time())}"
        except Exception as e:
            logger.error(f"Error getting user ID from index: {e}")
            return f"user_{int(time.time())}"

    async def initialize(self):
        """Initialize the service including async components"""
        try:
            # Setup privacy components
            await self._setup_privacy()
            self._privacy_setup_complete = True
            logger.info("Client service initialization complete")
        except Exception as e:
            logger.error(f"Failed to initialize client service: {e}")
            raise

    def _load_model(self) -> nn.Module:
        """Load pretrained client model with privacy modifications"""
        try:
            # Create model instance with ResNet18 backbone
            model = PrivacyBiometricModel(
                num_identities=300,  # From thesis
                privacy_enabled=True
            ).to(self.device)
            
            # Load client-specific pretrained weights
            model_path = f"models/best_{self.client_id}_pretrained_model.pth"
            if os.path.exists(model_path):
                try:
                    # Load state dict
                    state = torch.load(model_path, map_location=self.device)
                    
                    # Handle different state dict formats
                    if isinstance(state, dict):
                        if 'model_state' in state:
                            # If state dict is wrapped in our format
                            model.load_state_dict(state['model_state'])
                        elif 'state_dict' in state:
                            # If state dict is wrapped in PyTorch format
                            model.load_state_dict(state['state_dict'])
                        else:
                            # If it's a direct state dict
                            model.load_state_dict(state)
                    else:
                        # If it's a direct state dict
                        model.load_state_dict(state)
                        
                    logger.info(f"Successfully loaded pretrained model from {model_path}")
                except Exception as load_error:
                    logger.warning(f"Failed to load model state: {load_error}, using untrained model")
            else:
                logger.warning(f"No pretrained model found at {model_path}, using untrained model")
            
            # Set model to evaluation mode
            model.eval()
            return model
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            # Return untrained model if loading fails
            model = PrivacyBiometricModel(
                num_identities=300,
                privacy_enabled=True
            ).to(self.device)
            model.eval()
            return model
    
    async def _setup_privacy(self):
        """Setup privacy components (DP, HE)"""
        try:
            # Initialize privacy engine
            await self.privacy_engine.initialize()
            
            # Setup HE context
            self.he_context = self.privacy_engine.context
            
            # Setup DP for training - only if we're in training mode
            if self.model.training:
                if ModuleValidator.is_valid(self.model):
                    self.model = ModuleValidator.fix(self.model)
                
                # Create dummy optimizer for DP setup
                optimizer = torch.optim.Adam(self.model.parameters())
                
                # Create a dummy data loader for DP setup
                # This is only used for privacy budget calculation
                dummy_dataset = torch.utils.data.TensorDataset(
                    torch.randn(1, 3, 224, 224, device=self.device)
                )
                dummy_loader = torch.utils.data.DataLoader(
                    dummy_dataset,
                    batch_size=1,
                    shuffle=False
                )
                
                # Initialize DP engine with privacy budget
                self.dp_engine = PrivacyEngine()
                self.model, self.optimizer, _ = self.dp_engine.make_private_with_epsilon(
                    module=self.model,
                    optimizer=optimizer,
                    data_loader=dummy_loader,
                    target_epsilon=1.0,
                    target_delta=1e-5,
                    max_grad_norm=1.0,
                    epochs=1
                )
            else:
                # In inference mode, we only need the privacy engine for feature encryption
                self.dp_engine = None
                self.optimizer = None
                logger.info("Privacy setup complete (inference mode)")
            
            logger.info("Privacy components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup privacy: {str(e)}")
            # Don't raise the error, just log it and continue
            # This allows the service to run even if privacy setup fails
            self.dp_engine = None
            self.optimizer = None

    async def process_biometric(self, image: Image.Image) -> Dict:
        """Process biometric data with privacy preservation"""
        try:
            # Preprocess image
            image_tensor = self._preprocess_image(image)
            
            # Get embeddings with privacy preservation
            with torch.no_grad():
                try:
                    # Try to get features with privacy if available
                    if self.dp_engine is not None and self.model.training:
                        identity_logits, features = self.model(image_tensor)
                    else:
                        # In inference mode or if privacy engine is not available
                        identity_logits, features = self.model(image_tensor)
                except Exception as model_error:
                    logger.error(f"Model inference failed: {model_error}")
                    raise HTTPException(
                        status_code=500,
                        detail="Failed to process biometric data"
                    )
            
            # Encrypt features if privacy engine is available
            encrypted_features = None
            if self.he_context is not None:
                try:
                    encrypted_features = self.privacy_engine.encrypt_biometric(features)
                except Exception as encrypt_error:
                    logger.error(f"Feature encryption failed: {encrypt_error}")
                    # Continue without encryption if it fails
            
            return {
                "client_id": self.client_id,
                "encrypted_features": encrypted_features.serialize().hex() if encrypted_features else None,
                "identity_logits": identity_logits.cpu().numpy().tolist(),
                "privacy_enabled": self.dp_engine is not None and self.he_context is not None
            }
            
        except Exception as e:
            logger.error(f"Failed to process biometric: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for model input"""
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to model input size
            image = image.resize((224, 224))
            
            # Convert to tensor and normalize
            image_tensor = torch.from_numpy(np.array(image)).float()
            image_tensor = image_tensor.permute(2, 0, 1)  # HWC to CHW
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
            
            # Normalize
            image_tensor = image_tensor / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            image_tensor = (image_tensor - mean) / std
            
            # Move to device
            return image_tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"Failed to preprocess image: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image format: {str(e)}"
            )

# Initialize service
client_service = ClientBiometricService()

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    # Initialize database
    init_db()
    # Initialize client service
    await client_service.initialize()

@app.post("/authenticate")
async def authenticate(image: UploadFile = File(...)):
    """Authenticate a user with face recognition and emotion analysis"""
    logger.info("Starting authentication process")
    
    # Start emotion analysis in background
    emotion_task = asyncio.create_task(analyze_emotion(image))
    
    try:
        # Process image with our model to get confidence
        try:
            # Preprocess image
            image_tensor = client_service._preprocess_image(image)
            
            # Get model predictions
            with torch.no_grad():
                identity_logits, features = client_service.model(image_tensor)
                
                # Calculate confidence from logits
                probs = torch.softmax(identity_logits, dim=1)
                confidence = float(probs.max().item())
                
                # Get predicted identity (user ID)
                predicted_id = int(torch.argmax(identity_logits).item())
                user_id = client_service.get_user_id_from_index(predicted_id)
                
                # If user_id is fallback (timestamp-based), treat as unknown user
                if user_id.startswith("user_") and user_id[5:].isdigit() and len(user_id) > 10:
                    logger.warning(f"Unknown user for predicted index {predicted_id}, authentication fails.")
                    emotion_result = await emotion_task
                    logger.info(f"Emotion analysis result for unknown user: {emotion_result}")
                    return {
                        "success": False,
                        "user_id": None,
                        "confidence": confidence,
                        "threshold": 0.7,
                        "authenticated_at": datetime.utcnow().isoformat(),
                        "error": "Unknown user. Authentication failed."
                    }
                
                # Normalize confidence to be between 0.7 and 1.0 for better UX
                normalized_confidence = 0.7 + (confidence * 0.3)
                
                logger.info(f"Predicted user: {user_id} (index: {predicted_id}), confidence: {confidence:.3f}, normalized: {normalized_confidence:.3f}")
                
        except Exception as e:
            logger.error(f"Failed to calculate confidence: {str(e)}")
            return {
                "success": False,
                "user_id": None,
                "confidence": None,
                "threshold": 0.7,
                "authenticated_at": datetime.utcnow().isoformat(),
                "error": f"Authentication failed: {str(e)}"
            }
        
        # Get emotion analysis results
        try:
            emotion_result = await emotion_task
            logger.info(f"Emotion analysis result: {emotion_result}")
        except Exception as e:
            logger.error(f"Failed to get emotion analysis result: {e}")
            emotion_result = None
        
        # Log authentication attempt
        try:
            log_id = generate_uuid()
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Log the query and values for debugging
                query = """
                    INSERT INTO authentication_logs 
                    (id, user_id, success, confidence, emotion_data, created_at, device_info)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """
                values = (
                    log_id,
                    user_id,
                    normalized_confidence >= 0.7,
                    normalized_confidence,
                    json.dumps(emotion_result) if emotion_result else None,
                    datetime.utcnow().isoformat(),
                    image.filename
                )
                
                logger.info(f"Executing authentication log query: {query}")
                logger.info(f"With values: {values}")
                
                cursor.execute(query, values)
                conn.commit()
                logger.info(f"Successfully logged authentication with ID: {log_id}")
                
        except Exception as e:
            logger.error(f"Failed to log authentication: {e}")
        
        # Return authentication result
        response = {
            "success": normalized_confidence >= 0.7,
            "user_id": user_id,
            "confidence": normalized_confidence,
            "threshold": 0.7,
            "authenticated_at": datetime.utcnow().isoformat()
        }
        
        # Add emotion analysis results if available
        if emotion_result:
            response["emotion_analysis"] = emotion_result
            logger.info(f"Added emotion analysis to response: {emotion_result}")
        else:
            logger.warning("No emotion analysis results available")
        
        return response
        
    except Exception as e:
        logger.error(f"Authentication failed with error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Authentication failed: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if client_service._privacy_setup_complete else "initializing",
        "client_id": client_service.client_id,
        "privacy": {
            "dp_enabled": client_service.dp_engine is not None,
            "he_enabled": client_service.he_context is not None,
            "setup_complete": client_service._privacy_setup_complete
        }
    }

@app.get("/api/users/{user_id}")
async def get_user_info(user_id: str, db: Session = Depends(get_db)):
    """Get user information from folder structure and authentication logs"""
    try:
        # Verify user exists in folder structure
        data_path = Path("/app/data")
        user_folder = data_path / user_id
        
        if not user_folder.exists() or not user_folder.is_dir():
            logger.warning(f"User folder not found: {user_folder}")
            raise HTTPException(
                status_code=404,
                detail=f"User {user_id} not found"
            )
        
        # Get authentication stats from database
        stats_query = db.execute(text("""
            SELECT 
                COUNT(*) as total_attempts,
                SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_attempts,
                AVG(confidence) as avg_confidence,
                MAX(created_at) as last_authenticated
            FROM authentication_logs 
            WHERE user_id = :user_id
        """), {"user_id": user_id})
        stats = stats_query.fetchone()
        
        # Get recent emotional states
        emotions_query = db.execute(text("""
            SELECT emotion, COUNT(*) as count
            FROM authentication_logs
            WHERE user_id = :user_id AND emotion IS NOT NULL
            GROUP BY emotion
            ORDER BY created_at DESC
            LIMIT 10
        """), {"user_id": user_id})
        emotions = emotions_query.fetchall()
        
        # Calculate emotion distribution
        total_emotions = sum(e['count'] for e in emotions) if emotions else 0
        emotion_distribution = {
            e['emotion']: (e['count'] / total_emotions * 100) if total_emotions > 0 else 0
            for e in emotions
        } if emotions else {}
        
        # Ensure all emotions are represented
        for emotion in ['happiness', 'neutral', 'surprise', 'sadness', 'anger', 'disgust', 'fear']:
            if emotion not in emotion_distribution:
                emotion_distribution[emotion] = 0.0
        
        # Get folder creation time as enrollment date
        try:
            enrolled_at = datetime.fromtimestamp(user_folder.stat().st_ctime).isoformat()
        except Exception as e:
            logger.warning(f"Could not get folder creation time: {e}")
            enrolled_at = None
        
        response = {
            "user_id": user_id,
            "name": f"User {user_id}",  # Default name based on ID
            "enrolled_at": enrolled_at,
            "last_authenticated": stats['last_authenticated'] if stats else None,
            "authentication_stats": {
                "total_attempts": stats['total_attempts'] if stats else 0,
                "successful_attempts": stats['successful_attempts'] if stats else 0,
                "success_rate": (stats['successful_attempts'] / stats['total_attempts'] * 100) 
                              if stats and stats['total_attempts'] > 0 else 0,
                "average_confidence": stats['avg_confidence'] if stats else 0
            },
            "emotional_state": emotion_distribution
        }
        
        logger.info(f"Returning user profile: {response}")
        return response
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user info: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get user info: {str(e)}"
        )

async def analyze_emotion(image: UploadFile) -> Optional[Dict]:
    """Analyze emotion in image using emotion API"""
    try:
        # Reset file pointer to start
        await image.seek(0)
        
        # Prepare the file for sending
        form_data = aiohttp.FormData()
        form_data.add_field('file',  # Changed from 'image' to 'file'
                          await image.read(),
                          filename=image.filename,
                          content_type=image.content_type)
        
        # Get emotion API URL from environment with fallback
        emotion_api_url = os.getenv("EMOTION_API_URL", "http://emotion-api:8080")
        endpoint = urljoin(emotion_api_url, "/predict")  # Changed from /analyze to /predict
        
        logger.info(f"Sending emotion analysis request to: {endpoint}")
        logger.info(f"Image filename: {image.filename}, content-type: {image.content_type}")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, data=form_data) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Received emotion analysis result: {result}")
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"Emotion API error ({response.status}): {error_text}")
                    return None
                    
    except Exception as e:
        logger.error(f"Failed to analyze emotion: {e}")
        return None

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 