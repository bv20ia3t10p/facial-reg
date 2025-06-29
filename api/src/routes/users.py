"""
User management routes for the API
"""

import io
import json
import logging
import time
import uuid
import hashlib
import numpy as np
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

def add_user_to_identity_mapping(user_id: str) -> bool:
    """
    Add a new user to the global identity mapping
    
    Args:
        user_id: The new user's ID to add
        
    Returns:
        True if successfully added, False otherwise
    """
    try:
        # Load current identity mapping
        mapping_path = Path("/app/data/identity_mapping.json")
        if not mapping_path.exists():
            # Try alternative paths
            alternative_paths = [
                Path("./data/identity_mapping.json"),
                Path("../data/identity_mapping.json"),
                Path("/app/models/identity_mapping.json")
            ]
            for alt_path in alternative_paths:
                if alt_path.exists():
                    mapping_path = alt_path
                    break
            else:
                logger.error("Could not find identity_mapping.json")
                return False
        
        with open(mapping_path, 'r') as f:
            mapping_data = json.load(f)
        
        current_mapping = mapping_data.get("mapping", {})
        
        # Check if user already exists
        if user_id in current_mapping:
            logger.info(f"User {user_id} already exists in mapping")
            return True
        
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
        with open(mapping_path, 'w') as f:
            json.dump(mapping_data, f, indent=2)
        
        logger.info(f"Added user {user_id} to identity mapping with index {next_index}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to add user to identity mapping: {e}")
        return False

def create_user_data_directory(user_id: str, images: List[UploadFile]) -> bool:
    """
    Create data directory for new user and save images
    
    Args:
        user_id: The user's ID
        images: List of uploaded images
        
    Returns:
        True if successfully created, False otherwise
    """
    try:
        # Create user directory in client1 partition
        user_data_dir = Path(f"/app/data/partitioned/client1/{user_id}")
        user_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Save images with proper naming
        for i, image_file in enumerate(images):
            try:
                # Read image data
                image_data = image_file.file.read()
                image_file.file.seek(0)  # Reset file pointer
                
                # Validate image
                img = Image.open(io.BytesIO(image_data))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Save image with incrementing filename
                image_path = user_data_dir / f"user_image_{i:03d}.jpg"
                img.save(image_path, 'JPEG', quality=95)
                
                logger.info(f"Saved image {i} for user {user_id} at {image_path}")
                
            except Exception as e:
                logger.error(f"Error processing image {i} for user {user_id}: {e}")
                continue
        
        logger.info(f"Created data directory for user {user_id} with {len(images)} images")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create user data directory: {e}")
        return False

def process_user_images_for_database(images: List[UploadFile]) -> Optional[bytes]:
    """
    Process uploaded images and create face encoding for database storage
    
    Args:
        images: List of uploaded images
        
    Returns:
        Face encoding as bytes or None if processing failed
    """
    try:
        if not biometric_service or not biometric_service.initialized:
            logger.error("Biometric service not initialized")
            return None
        
        # Process the first valid image for face encoding
        for i, image_file in enumerate(images):
            try:
                # Read image data
                image_data = image_file.file.read()
                image_file.file.seek(0)  # Reset file pointer
                
                # Validate image
                img = Image.open(io.BytesIO(image_data))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Convert back to bytes for biometric service
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='JPEG')
                img_bytes = img_bytes.getvalue()
                
                # Use biometric service to extract features
                image_tensor = biometric_service.preprocess_image(img_bytes)
                features = biometric_service.feature_extractor.extract_features(image_tensor)
                
                # Convert features to bytes for database storage
                features_np = features.cpu().numpy().astype(np.float32)
                face_encoding = features_np.tobytes()
                
                logger.info(f"Successfully processed image {i} for face encoding")
                return face_encoding
                
            except Exception as e:
                logger.error(f"Error processing image {i} for face encoding: {e}")
                continue
        
        logger.error("Failed to process any images for face encoding")
        return None
        
    except Exception as e:
        logger.error(f"Failed to process user images: {e}")
        return None

def notify_coordinator_new_user(user_id: str) -> bool:
    """
    Notify the coordinator about a new user for mapping updates
    
    Args:
        user_id: The new user's ID
        
    Returns:
        True if successfully notified, False otherwise
    """
    try:
        import requests
        import os
        
        coordinator_url = os.getenv("SERVER_URL", "http://fl-coordinator:8000")
        
        # Try to notify coordinator about new user
        response = requests.post(
            f"{coordinator_url}/api/mapping/add-user",
            json={"user_id": user_id},
            timeout=10
        )
        
        if response.status_code == 200:
            logger.info(f"Successfully notified coordinator about new user {user_id}")
            return True
        else:
            logger.warning(f"Coordinator returned status {response.status_code} for new user {user_id}")
            return False
            
    except Exception as e:
        logger.warning(f"Could not notify coordinator about new user {user_id}: {e}")
        return False

# Function to train the model in the background
def train_model_task(db: Session = None):
    global model_training_in_progress
    
    try:
        # Prevent multiple training sessions
        if model_training_in_progress:
            logger.warning("Model training already in progress, skipping this request")
            return
        
        model_training_in_progress = True
        logger.info("Starting federated model update in background...")
        
        # Import federated training components
        try:
            import subprocess
            import os
            
            # Run federated training script
            script_path = "/app/train_federated.py"
            if not Path(script_path).exists():
                script_path = "/app/api/train_federated.py"
            if not Path(script_path).exists():
                script_path = "./train_federated.py"
            
            if Path(script_path).exists():
                logger.info(f"Running federated training script: {script_path}")
                result = subprocess.run(
                    ["python", script_path],
                    capture_output=True,
                    text=True,
                    timeout=1800  # 30 minutes timeout
                )
                
                if result.returncode == 0:
                    logger.info("Federated training completed successfully")
                else:
                    logger.error(f"Federated training failed: {result.stderr}")
            else:
                logger.warning("Federated training script not found, skipping training")
                
        except Exception as training_error:
            logger.error(f"Error during federated training: {training_error}")
        
        # Reload models after training
        try:
            if biometric_service:
                biometric_service.reload_model()
                logger.info("Biometric service model reloaded after training")
        except Exception as reload_error:
            logger.error(f"Error reloading model after training: {reload_error}")
            
    except Exception as e:
        logger.error(f"Error in training task: {e}")
    finally:
        model_training_in_progress = False
        logger.info("Training task completed")

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
        
        # Validate face images
        if not images or len(images) == 0:
            raise HTTPException(
                status_code=400,
                detail="At least one face image is required"
            )
        
        # Generate a unique user ID
        user_id = str(uuid.uuid4())[:8]  # Use shorter UUID for user ID
        
        # Ensure the user ID is unique
        while True:
            existing_id = db.execute(
                text("SELECT id FROM users WHERE id = :id"),
                {"id": user_id}
            ).fetchone()
            if not existing_id:
                break
            user_id = str(uuid.uuid4())[:8]
        
        logger.info(f"Generated unique user ID '{user_id}' for new user registration")
        
        # Process face images for database storage
        face_features = process_user_images_for_database(images)
        if face_features is None:
            raise HTTPException(
                status_code=400,
                detail="Failed to process face images. Please ensure images contain clear faces."
            )
        
        # Create user data directory and save images
        if not create_user_data_directory(user_id, images):
            raise HTTPException(
                status_code=500,
                detail="Failed to create user data directory"
            )
        
        # Add user to identity mapping
        if not add_user_to_identity_mapping(user_id):
            logger.warning(f"Failed to add user {user_id} to identity mapping, but continuing...")
        
        # Use "demo" as default password
        user_password = "demo"
        
        # Create user record in database
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
        
        # Notify coordinator about new user
        notify_coordinator_new_user(user_id)
        
        # Trigger federated training in the background
        background_tasks.add_task(train_model_task)
        
        logger.info(f"Successfully registered new user {user_id} ({name})")
        
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