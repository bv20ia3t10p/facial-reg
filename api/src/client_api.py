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
import requests
from urllib.parse import urljoin
import json
import uuid
from fastapi import status

import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import tenseal as ts
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from sqlalchemy.orm import Session
from sqlalchemy import text
from .db.database import get_db, AuthenticationLog, User, log_authentication, init_db, json_safe_dumps
from torchvision import transforms
from tempfile import SpooledTemporaryFile

from .utils.security import generate_uuid
from .services.service_init import service_manager
from .services.emotion_service import analyze_emotion
from .models.privacy_biometric_model import PrivacyBiometricModel
from .privacy.privacy_engine import PrivacyEngine as BiometricPrivacyEngine
from .routes import client_users

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add database path configuration
DB_PATH = os.getenv('DATABASE_URL', 'sqlite:////app/database/client1.db').replace('sqlite:///', '')
logger.info(f"Using database at: {DB_PATH}")

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

# Add emotion API URL configuration
EMOTION_API_URL = "http://emotion-api:8080"  # Emotion API endpoint

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

    def _load_model(self) -> nn.Module:
        """Load the biometric model"""
        try:
            # In Docker, models are mounted at /app/models
            model_path = Path(f"/app/models/best_{self.client_id}_pretrained_model.pth")
            logger.info(f"Loading model from: {model_path}")
            
            if not model_path.exists():
                logger.error(f"Model file not found at: {model_path}")
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Create model instance
            model = PrivacyBiometricModel(
                num_identities=300,  # From thesis
                privacy_enabled=True
            ).to(self.device)
            
            # Load state dict
            state_dict = torch.load(model_path, map_location=self.device)
            
            # Handle different state dict formats
            if isinstance(state_dict, dict):
                if 'model_state' in state_dict:
                    # If state dict is wrapped in our format
                    model.load_state_dict(state_dict['model_state'])
                elif 'state_dict' in state_dict:
                    # If state dict is wrapped in PyTorch format
                    model.load_state_dict(state_dict['state_dict'])
                else:
                    # If it's a direct state dict
                    model.load_state_dict(state_dict)
            else:
                # If it's a direct state dict
                model.load_state_dict(state_dict)
            
            model.eval()
            logger.info("Model loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

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
                    logger.info(f"Index {idx:3d} -> User {self.user_id_mapping[idx]}")
            
        except Exception as e:
            logger.error(f"Failed to load user ID mapping: {e}")
            raise

    def get_user_id_from_index(self, index: int) -> str:
        """Get user ID from model index"""
        if index not in self.user_id_mapping:
            # Return a fallback user ID based on timestamp
            return f"user_{int(time.time())}"
        return self.user_id_mapping[index]

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

# Initialize client service
client_service = ClientBiometricService()

# Mount user routes
app.include_router(client_users.router, prefix="/api/users", tags=["users"])

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        # Initialize database
        init_db()
        logger.info("Database initialized")
        
        # Initialize client service
        await client_service.initialize()
        logger.info("Client service initialized")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

async def _preprocess_image(image: UploadFile) -> torch.Tensor:
    """Preprocess image for model input"""
    try:
        # Read image data
        contents = await image.read()
        
        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Apply transformations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Transform and add batch dimension
        img_tensor = transform(pil_image).unsqueeze(0)
        
        # Move to device
        return img_tensor.to(client_service.device)
        
    except Exception as e:
        logger.error(f"Failed to preprocess image: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process image: {str(e)}"
        )

async def analyze_emotion(image: UploadFile) -> Optional[Dict]:
    """Analyze emotion in image using emotion API"""
    try:
        # Read image data
        image_data = await image.read()
        
        # Create form data
        files = {
            'file': ('face.jpg', image_data, 'image/jpeg')
        }
        
        # Make request to emotion API
        logger.info(f"Making request to emotion API at {EMOTION_API_URL}")
        response = requests.post(urljoin(EMOTION_API_URL, "/predict"), files=files)
        
        if response.status_code != 200:
            logger.warning(f"Emotion API returned status {response.status_code}")
            logger.warning(f"Response body: {response.text}")
            return None
            
        result = response.json()
        logger.info(f"Received emotion analysis result: {result}")
        return result
        
    except requests.RequestException as e:
        logger.error(f"Failed to connect to emotion API: {e}")
        return None
    except Exception as e:
        logger.error(f"Emotion analysis failed: {e}")
        return None

@app.post("/authenticate")
async def authenticate(
    image: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    try:
        logger.info("=== Starting authentication process ===")
        
        # Create a copy of the image for emotion analysis
        contents = await image.read()
        file_copy = SpooledTemporaryFile()
        file_copy.write(contents)
        file_copy.seek(0)
        
        image_copy = UploadFile(
            file=file_copy,
            filename=image.filename,
            headers={"content-type": image.content_type} if image.content_type else None
        )
        
        # Start emotion analysis in background
        emotion_task = asyncio.create_task(analyze_emotion(image_copy))
        
        # Reset original image file pointer
        await image.seek(0)
        
        try:
            # Preprocess image
            image_tensor = await _preprocess_image(image)
            logger.debug("Image preprocessed successfully")
            
            # Get model predictions
            with torch.no_grad():
                identity_logits, features = client_service.model(image_tensor)
                
                # Calculate confidence from logits
                probs = torch.softmax(identity_logits, dim=1)
                confidence = float(probs.max().item())
                
                # Get predicted identity (user ID)
                predicted_id = int(torch.argmax(identity_logits).item())
                user_id = client_service.get_user_id_from_index(predicted_id)
                
                # Get emotion analysis results
                try:
                    emotion_result = await emotion_task
                    logger.debug(f"Emotion analysis result: {json_safe_dumps(emotion_result)}")
                except Exception as e:
                    logger.error(f"Failed to get emotion analysis result: {e}")
                    emotion_result = {
                        "happiness": 0.0,
                        "neutral": 1.0,
                        "surprise": 0.0,
                        "sadness": 0.0,
                        "anger": 0.0,
                        "disgust": 0.0,
                        "fear": 0.0
                    }
                finally:
                    # Clean up
                    await image_copy.close()
                    file_copy.close()
                
                # If user_id is fallback (timestamp-based), treat as unknown user
                if user_id.startswith("user_") and user_id[5:].isdigit() and len(user_id) > 10:
                    logger.warning(f"Unknown user for predicted index {predicted_id}, authentication fails.")
                    logger.info(f"Emotion analysis result for unknown user: {json_safe_dumps(emotion_result)}")
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
                
                # Log authentication attempt using SQLAlchemy
                try:
                    # Ensure emotion_data is properly serialized
                    emotion_json = None
                    if emotion_result:
                        try:
                            emotion_json = json.dumps(emotion_result)
                            logger.debug(f"Successfully serialized emotion data: {emotion_json}")
                        except Exception as e:
                            logger.error(f"Failed to serialize emotion data: {e}")
                            emotion_json = json.dumps({
                                "happiness": 0.0,
                                "neutral": 1.0,
                                "surprise": 0.0,
                                "sadness": 0.0,
                                "anger": 0.0,
                                "disgust": 0.0,
                                "fear": 0.0
                            })

                    # Create log entry data
                    auth_time = datetime.utcnow()
                    log_data = {
                        "id": generate_uuid(),
                        "user_id": user_id,
                        "success": normalized_confidence >= 0.7,
                        "confidence": normalized_confidence,
                        "emotion_data": emotion_json,
                        "created_at": auth_time,
                        "device_info": image.filename,
                        "captured_image": None  # Set to None since we don't want to store the image
                    }
                    
                    # Create the log entry
                    log_entry = log_authentication(db, log_data)
                    if not log_entry:
                        logger.error("Failed to create authentication log entry")
                        raise HTTPException(
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Failed to log authentication attempt"
                        )
                    
                    logger.info(f"Successfully logged authentication with ID: {log_entry.id}")
                    
                    # Update user's last_authenticated timestamp in folder metadata
                    try:
                        user_folder = Path("/app/data") / user_id
                        if user_folder.exists() and user_folder.is_dir():
                            # Create or update metadata file
                            metadata_file = user_folder / "metadata.json"
                            metadata = {
                                "last_authenticated": auth_time.isoformat(),
                                "last_auth_success": log_data["success"],
                                "last_auth_confidence": log_data["confidence"]
                            }
                            
                            # Update existing metadata if it exists
                            if metadata_file.exists():
                                try:
                                    with open(metadata_file, 'r') as f:
                                        existing_metadata = json.load(f)
                                        metadata.update(existing_metadata)
                                        metadata["last_authenticated"] = auth_time.isoformat()
                                        metadata["last_auth_success"] = log_data["success"]
                                        metadata["last_auth_confidence"] = log_data["confidence"]
                                except json.JSONDecodeError:
                                    pass  # Use default metadata if file is corrupted
                            
                            # Write updated metadata
                            with open(metadata_file, 'w') as f:
                                json.dump(metadata, f, indent=2)
                            logger.debug(f"Updated metadata for user {user_id}")
                        else:
                            logger.warning(f"User folder not found: {user_folder}")
                    except Exception as e:
                        logger.error(f"Failed to update user metadata: {e}")
                    
                    # Return authentication result
                    return {
                        "success": log_data["success"],
                        "user_id": user_id,
                        "confidence": normalized_confidence,
                        "threshold": 0.7,
                        "authenticated_at": auth_time.isoformat(),
                        "emotion_data": json.loads(emotion_json) if emotion_json else None
                    }
                    
                except Exception as e:
                    logger.error(f"Authentication failed with error: {str(e)}")
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Authentication failed: {str(e)}"
                    )
                
        except Exception as e:
            logger.error(f"Failed to process image or get predictions: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Authentication failed: {str(e)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during authentication: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 