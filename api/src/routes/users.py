"""
User management routes for the API
"""

import io
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy.sql import text

from ..db.database import User, create_user, get_db, AuthenticationLog
from ..services.service_init import biometric_service
from ..utils.security import get_current_user, get_password_hash
import pytz

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
                
                # As requested, save the processed image bytes directly
                face_encoding = img_bytes
                
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

def train_model_task(db: Optional[Session] = None):
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
        # Import here to avoid circular dependency issues at startup
        from .analytics import safe_parse_emotion_data
        
        # Get user data
        user = db.query(User).filter(User.id == user_id).first()
        
        # Get authentication logs for the user
        auth_logs = db.query(AuthenticationLog).filter(AuthenticationLog.user_id == user_id).order_by(
            AuthenticationLog.created_at.desc()
        ).limit(20).all()
        
        # If user not found in DB, check data directory as a fallback
        if not user:
            data_path = Path("/app/data")
            if not (data_path / user_id).exists():
                raise HTTPException(status_code=404, detail=f"User {user_id} not found")

        # Get stats from logs
        total_attempts = len(auth_logs)
        successful_attempts = sum(1 for log in auth_logs if log.success is True)
        success_rate = (successful_attempts / total_attempts * 100) if total_attempts > 0 else 0
        
        # Calculate average confidence from successful attempts
        confidences = [log.confidence for log in auth_logs if log.success is True and log.confidence is not None]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Format datetime objects to local timezone
        def format_datetime(dt):
            if not dt:
                return None
            try:
                # Ensure datetime is timezone-aware (UTC)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=pytz.UTC)
                # Convert to local timezone
                local_tz = pytz.timezone(system_timezone)
                return dt.astimezone(local_tz).isoformat()
            except Exception as e:
                logger.warning(f"Could not format datetime {dt}: {e}")
                return dt.isoformat() if hasattr(dt, 'isoformat') else str(dt)

        # Get latest emotion data
        latest_emotion_data = {}
        for log in auth_logs:
            if log.emotion_data is not None:
                try:
                    # Use the safe_parse_emotion_data function
                    parsed_data = safe_parse_emotion_data(log.emotion_data)
                    if parsed_data:
                        latest_emotion_data = parsed_data
                        break
                except Exception as e:
                    logger.error(f"Error parsing emotion data: {e}")

        # Format authentication attempts
        formatted_attempts = [{
            "id": log.id,
            "user_id": log.user_id,
            "success": log.success,
            "confidence": log.confidence,
            "timestamp": format_datetime(log.created_at),
            "emotion_data": safe_parse_emotion_data(log.emotion_data) if log.emotion_data is not None else None,
            "device_info": log.device_info
        } for log in auth_logs]

        # Get last authenticated timestamp from logs
        last_authenticated_log = auth_logs[0].created_at if auth_logs else None

        user_details = {
            "name": f"User {user_id}",
            "email": None,
            "department": None,
            "role": None,
            "enrolled_at": None,
        }
        if user is not None:
            user_details["name"] = user.name
            user_details["email"] = user.email
            user_details["department"] = user.department
            user_details["role"] = user.role
            user_details["enrolled_at"] = format_datetime(user.created_at)

        return {
            "user_id": user_id,
            **user_details,
            "last_authenticated": format_datetime(last_authenticated_log),
            "authentication_stats": {
                "total_attempts": total_attempts,
                "successful_attempts": successful_attempts,
                "success_rate": success_rate,
                "average_confidence": avg_confidence
            },
            "recent_attempts": formatted_attempts,
            "latest_auth": formatted_attempts[0] if formatted_attempts else None,
            "emotional_state": latest_emotion_data or {
                "neutral": 1.0, "happiness": 0.0, "sadness": 0.0, "anger": 0.0,
                "surprise": 0.0, "fear": 0.0, "disgust": 0.0
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
        user_department = str(current_user.department).lower() if current_user.department is not None else ""
        user_role = str(current_user.role) if current_user.role is not None else ""
        if user_department != "hr" and user_role != "admin":
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
        user_department = str(current_user.department).lower() if current_user.department is not None else ""
        user_role = str(current_user.role) if current_user.role is not None else ""
        if user_department != "hr" and user_role != "admin":
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