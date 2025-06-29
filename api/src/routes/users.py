"""
User management routes for the API
"""

import io
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from zoneinfo import ZoneInfo

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy.sql import text

from ..db.database import User, create_user, get_db
from ..services.service_init import biometric_service
from ..utils.security import get_current_user, get_password_hash
from ..utils.common import generate_uuid
from ..models.privacy_biometric_model import PrivacyBiometricModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Detect system timezone
try:
    import platform
    if platform.system() == 'Windows':
        import tzlocal
        system_timezone = tzlocal.get_localzone_name()
    else:
        import subprocess
        system_timezone = subprocess.check_output(['timedatectl', 'show', '-p', 'Timezone', '--value']).decode('utf-8').strip()
    logger.info(f"System timezone detected: {system_timezone}")
except Exception as e:
    logger.warning(f"Could not detect system timezone, defaulting to UTC: {e}")
    system_timezone = 'UTC'

# Initialize router
router = APIRouter()

# Global variable to track if model is being trained
model_training_in_progress = False

# Function to train the model in the background
def train_model_task(db: Session = None):
    global model_training_in_progress
    
    try:
        # Prevent multiple training sessions
        if model_training_in_progress:
            logger.warning("Model training already in progress, skipping this request")
            return
        
        model_training_in_progress = True
        logger.info("Starting model training in background...")
        
        # Get a new database session if none was provided
        close_db = False
        if db is None:
            from ..db.database import SessionLocal
            db = SessionLocal()
            close_db = True
        
        try:
            # Initialize mapping service
            from ..services.mapping_service import MappingService
            mapping_service = MappingService()
            
            # Get current global mapping
            global_mapping = mapping_service.fetch_mapping(force_refresh=True)
            if not global_mapping:
                logger.warning("No global mapping available, will create new mapping")
            
            # 1. Load all user face encodings from the database
            logger.info("Loading user face encodings from database...")
            users = db.execute(text("""
                SELECT id, face_encoding FROM users
                WHERE face_encoding IS NOT NULL
                ORDER BY id
            """)).fetchall()
            
            if not users or len(users) == 0:
                logger.warning("No users with face encodings found in database")
                return
                
            logger.info(f"Loaded face encodings for {len(users)} users")
            
            # 2. Train or update the model
            import torch
            import torch.nn as nn
            import torch.optim as optim
            import numpy as np
            import os
            
            # Create models directory if it doesn't exist
            models_dir = Path("/app/models")
            models_dir.mkdir(exist_ok=True)
            
            # Prepare data - we need to extract features from raw face encodings
            user_features = []
            user_ids = []
            
            # Create a simple feature extractor to convert stored face encodings to feature vectors
            feature_extractor = face_recognizer.model
            feature_extractor.eval()
            
            # Determine the feature dimension from the first encoding
            feature_dim = None
            
            # Track user ID to class index mapping
            user_to_class_idx = {}
            next_class_idx = 0
            
            # If we have a global mapping, use it to maintain consistent indices
            if global_mapping:
                # Invert the mapping to get user_id -> class_idx
                for class_idx, user_id in global_mapping.items():
                    user_to_class_idx[user_id] = int(class_idx)
                next_class_idx = max(int(idx) for idx in global_mapping.keys()) + 1
            
            for i, user in enumerate(users):
                try:
                    # Load face encoding from database (this is already preprocessed)
                    face_encoding_bytes = user.face_encoding
                    
                    # Convert bytes to numpy array
                    face_encoding = np.frombuffer(face_encoding_bytes, dtype=np.float32)
                    
                    # If this is the first encoding, determine the feature dimension
                    if feature_dim is None:
                        feature_dim = face_encoding.shape[0]
                        logger.info(f"Detected feature dimension: {feature_dim}")
                    
                    # Reshape to match expected dimensions
                    face_encoding = face_encoding.reshape(1, -1)
                    
                    # Convert to torch tensor
                    face_tensor = torch.from_numpy(face_encoding).float()
                    
                    # Get or assign class index for this user
                    if user.id not in user_to_class_idx:
                        user_to_class_idx[user.id] = next_class_idx
                        next_class_idx += 1
                    
                    # Add to features list
                    user_features.append(face_tensor)
                    user_ids.append(user.id)
                    
                    logger.debug(f"Processed face encoding for user {user.id}, class_idx={user_to_class_idx[user.id]}")
                except Exception as e:
                    logger.error(f"Error processing face encoding for user {user.id}: {e}")
            
            if not user_features:
                logger.error("No valid face encodings found")
                return
                
            # Stack features into a single tensor
            features_tensor = torch.cat(user_features, dim=0)
            
            # Create labels tensor using the mapping
            labels = [user_to_class_idx[uid] for uid in user_ids]
            labels_tensor = torch.tensor(labels, dtype=torch.long)
            
            logger.info(f"Prepared training data: {features_tensor.shape} features, {labels_tensor.shape} labels")
            
            # Create a simple dataset and dataloader
            from torch.utils.data import TensorDataset, DataLoader
            dataset = TensorDataset(features_tensor, labels_tensor)
            dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
            
            # Initialize model with the correct number of identities
            num_identities = next_class_idx  # Use the next available index as the total count
            
            # Create a model with the correct feature dimension from the start
            logger.info(f"Creating model with {num_identities} identities and feature_dim={feature_dim}")
            model = PrivacyBiometricModel(
                num_identities=num_identities, 
                embedding_dim=feature_dim,
                privacy_enabled=True
            )
            
            # Setup optimizer and loss function
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = torch.nn.CrossEntropyLoss()
            
            # Training for 20 epochs
            logger.info("Beginning training for 20 epochs...")
            for epoch in range(1, 21):
                epoch_loss = 0.0
                correct = 0
                total = 0
                
                for batch_features, batch_labels in dataloader:
                    optimizer.zero_grad()
                    
                    # Forward pass with the biometric model
                    identity_logits, _ = model(batch_features)
                    
                    # Calculate loss
                    loss = criterion(identity_logits, batch_labels)
                    
                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()
                    
                    # Track statistics
                    epoch_loss += loss.item()
                    _, predicted = torch.max(identity_logits, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
                
                # Log epoch progress
                logger.info(f"Epoch {epoch}: Loss={epoch_loss/len(dataloader):.4f}, Accuracy={correct/total*100:.2f}%")
            
            logger.info("Training completed, saving model...")
            
            # Create mapping from class indices to user IDs
            idx_to_user = {str(class_idx): user_id for user_id, class_idx in user_to_class_idx.items()}
            
            # Update global mapping
            if mapping_service.update_mapping(idx_to_user):
                logger.info("Successfully updated global mapping")
            else:
                logger.warning("Failed to update global mapping")
            
            # Save the model with metadata
            client_id = os.getenv("CLIENT_ID", "client1")
            model_metadata = {
                'state_dict': model.state_dict(),
                'num_identities': num_identities,
                'feature_dim': feature_dim,
                'privacy_enabled': True,
                'client_id': client_id,
                'timestamp': datetime.utcnow().isoformat(),
                'mapping_version': mapping_service.mapping_version,
                'mapping_hash': mapping_service.mapping_hash
            }
            torch.save(model_metadata, models_dir / f"best_{client_id}_pretrained_model.pth")
            
            logger.info(f"Model saved as best_{client_id}_pretrained_model.pth")
            
            # Request model reload
            asyncio.create_task(reload_model())
            
            logger.info("Model training task completed successfully")
            
        except Exception as e:
            logger.error(f"Error during model training: {e}")
        finally:
            if close_db:
                db.close()
            
    except Exception as e:
        logger.error(f"Failed to train model: {e}")
    finally:
        model_training_in_progress = False

async def reload_model():
    """Reload the model after training"""
    try:
        logger.info("Reloading model...")
        
        # Reload in BiometricService through service_init
        from ..services.service_init import reload_biometric_service
        if reload_biometric_service:
            success = reload_biometric_service()
            if success:
                logger.info("Reloaded model through service_init")
            else:
                logger.warning("Model reload through service_init returned False")
        
        logger.info("Model reload completed")
    except Exception as e:
        logger.error(f"Failed to reload model: {e}")

@router.get("/{user_id}")
async def get_user_info(user_id: str, db: Session = Depends(get_db)):
    """Get user information and authentication history"""
    try:
        # Get authentication stats from authentication_logs with explicit transaction
        stats_query = db.execute(text("""
            SELECT 
                COUNT(*) as total_attempts,
                SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_attempts,
                AVG(confidence) as avg_confidence,
                MAX(created_at) as last_authenticated
            FROM authentication_logs 
            WHERE user_id = :user_id;
        """), {"user_id": user_id})
        stats = stats_query.fetchone()
        
        # Get recent authentications with fresh query
        recent_query = db.execute(text("""
            SELECT id, user_id, success, confidence, emotion_data, created_at, device_info
            FROM authentication_logs
            WHERE user_id = :user_id
            ORDER BY datetime(created_at) DESC
            LIMIT 20
        """), {"user_id": user_id})
        recent_auths = list(recent_query)  # Materialize the results
        
        # Get user data - all fields except face_encoding (BLOB)
        user_query = db.execute(text("""
            SELECT id, name, email, department, role, password_hash, created_at, last_authenticated
            FROM users 
            WHERE id = :user_id
        """), {"user_id": user_id})
        user = user_query.fetchone()
        
        # If user not found in database, check folder structure
        if not user:
            data_path = Path("/app/data")
            user_folder = data_path / user_id
            
            if not user_folder.exists() or not user_folder.is_dir():
                raise HTTPException(
                    status_code=404,
                    detail=f"User {user_id} not found"
                )
        
        # Log what we found
        logger.info(f"Found {len(recent_auths)} recent authentications for user {user_id}")
        if recent_auths:
            latest_auth = recent_auths[0]
            logger.info(f"Latest auth: ID={latest_auth.id}, timestamp={latest_auth.created_at}")
        
        # Convert datetime objects to ISO format strings
        def format_datetime(dt):
            if dt:
                try:
                    if isinstance(dt, str):
                        # Try to parse string to datetime first
                        dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
                    # Ensure datetime is UTC
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=ZoneInfo('UTC'))
                    # Convert to local timezone
                    local_dt = dt.astimezone(ZoneInfo(system_timezone))
                    return local_dt.isoformat()
                except Exception as e:
                    logger.error(f"Error formatting datetime {dt}: {e}")
                    return dt
            return None

        # Get the latest authentication with emotion data
        latest_emotion_data = None
        if recent_auths:
            for auth in recent_auths:
                if auth.emotion_data:
                    try:
                        latest_emotion_data = json.loads(auth.emotion_data)
                        break
                    except Exception as e:
                        logger.error(f"Error parsing emotion data: {e}")

        # Format authentication attempts
        formatted_attempts = []
        for auth in recent_auths:
            try:
                emotion_data = json.loads(auth.emotion_data) if auth.emotion_data else None
                formatted_attempts.append({
                    "id": auth.id,
                    "user_id": auth.user_id,
                    "success": bool(auth.success),
                    "confidence": float(auth.confidence),
                    "timestamp": format_datetime(auth.created_at),
                    "emotion_data": emotion_data,
                    "device_info": auth.device_info
                })
            except Exception as e:
                logger.error(f"Error formatting authentication attempt: {e}")
                continue

        # Calculate success rate
        total_attempts = stats.total_attempts if stats else 0
        successful_attempts = stats.successful_attempts if stats else 0
        success_rate = (successful_attempts / total_attempts * 100) if total_attempts > 0 else 0
        avg_confidence = float(stats.avg_confidence) if stats and stats.avg_confidence else 0

        return {
            "user_id": user.id if user else user_id,
            "name": user.name if user else f"User {user_id}",
            "email": user.email if user else None,
            "department": user.department if user else None,
            "role": user.role if user else None,
            "enrolled_at": format_datetime(user.created_at) if user and user.created_at else None,
            "last_authenticated": format_datetime(user.last_authenticated) if user and user.last_authenticated else None,
            "authentication_stats": {
                "total_attempts": total_attempts,
                "successful_attempts": successful_attempts,
                "success_rate": success_rate,
                "average_confidence": avg_confidence
            },
            "recent_attempts": formatted_attempts,
            "latest_auth": formatted_attempts[0] if formatted_attempts else None,
            "emotional_state": latest_emotion_data or {
                "neutral": 1.0,
                "happy": 0.0,
                "sad": 0.0,
                "angry": 0.0,
                "surprised": 0.0,
                "fearful": 0.0,
                "disgusted": 0.0
            }
        }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user info: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get user information: {str(e)}"
        ) 

# User registration model
class UserRegistration(BaseModel):
    name: str
    email: str
    department: str
    role: str

@router.post("/register")
async def register_user(
    background_tasks: BackgroundTasks,
    name: str = Form(...),
    email: str = Form(...),
    department: str = Form(...),
    role: str = Form(...),
    images: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Register a new user with face images for biometric authentication"""
    try:
        # Check if the current user has HR permissions
        if current_user.department.lower() != "hr" and current_user.role != "admin":
            raise HTTPException(
                status_code=403,
                detail="Only HR department or admin users can register new users"
            )
        
        # Check if user with this email already exists
        existing_user = db.execute(
            text("SELECT id FROM users WHERE email = :email"),
            {"email": email}
        ).fetchone()
        
        if existing_user:
            raise HTTPException(
                status_code=400,
                detail=f"User with email {email} already exists"
            )
        
        # Process face images
        if not images or len(images) == 0:
            raise HTTPException(
                status_code=400,
                detail="At least one face image is required"
            )
        
        # Generate a user ID greater than any existing ID
        user_id = generate_uuid()
        logger.info(f"Generated user ID '{user_id}' for new user registration")
        user_data_dir = Path(f"/app/data/{user_id}")
        user_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Process and save face images
        face_features = None
        for i, image_file in enumerate(images):
            try:
                # Read image data
                image_data = await image_file.read()
                
                # Save image to user directory
                image_path = user_data_dir / f"face_{i}.jpg"
                with open(image_path, "wb") as f:
                    f.write(image_data)
                
                # Process image for face recognition
                image = Image.open(io.BytesIO(image_data))
                aligned_face = face_recognizer.align_face(image)
                
                # Extract face features
                tensor = face_recognizer.preprocess_image(aligned_face)
                features = face_recognizer.extract_features(tensor)
                
                # Use the first image's features for the user record
                if face_features is None:
                    face_features = face_recognizer.save_features(features)
            
            except Exception as e:
                logger.error(f"Error processing image {i}: {e}")
                continue
        
        if face_features is None:
            raise HTTPException(
                status_code=400,
                detail="Failed to process face images. Please ensure images contain clear faces."
            )
        
        # Always use "demo" as password
        user_password = "demo"
        
        # Create user record
        password_hash = get_password_hash(user_password)
        user_data = {
            "id": user_id,
            "name": name,
            "email": email,
            "department": department,
            "role": role,
            "password_hash": password_hash,
            "face_encoding": face_features,
            "created_at": datetime.utcnow()
        }
        
        new_user = create_user(db, user_data)
        if not new_user:
            raise HTTPException(
                status_code=500,
                detail="Failed to create user record"
            )
        
        # Automatically trigger model training in the background
        # This won't block the API response
        background_tasks.add_task(train_model_task)
        
        return {
            "success": True,
            "message": f"User registered successfully with default password: {user_password}. Model training started in the background.",
            "user_id": user_id,
            "name": name,
            "email": email,
            "department": department,
            "role": role
        }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to register user: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to register user: {str(e)}"
        ) 

@router.post("/reload-model")
async def reload_model_endpoint(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Reload the biometric model without retraining"""
    try:
        # Check if global biometric service is initialized
        if not biometric_service:
            logger.error("Global biometric service not initialized")
            raise HTTPException(
                status_code=500, 
                detail="Biometric service not initialized"
            )
        
        # Reload the model
        biometric_service.reload_model()
        
        return {
            "success": True,
            "message": "Model reloaded successfully"
        }
    except Exception as e:
        logger.error(f"Failed to reload model: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reload model: {str(e)}"
        )

@router.post("/train-model")
async def train_model_endpoint(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Endpoint to train the model on all users"""
    try:
        # Only allow HR or admin to access this endpoint
        if current_user.department.lower() != "hr" and current_user.role != "admin":
            raise HTTPException(
                status_code=403,
                detail="Only HR department or admin users can trigger model training"
            )
        
        # Start training in background
        background_tasks.add_task(train_model_task, db)
        
        return {
            "message": "Model training started in background",
            "success": True
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start model training: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start model training: {str(e)}"
        ) 