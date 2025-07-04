"""
Authentication routes for facial recognition
"""

import json
import logging
import traceback
from datetime import datetime, timedelta
from functools import wraps
import asyncio
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from ..db.database import get_db, User, AuthenticationLog
from ..services.service_init import initialize_biometric_service
from ..services.biometric_service import BiometricService
from ..services.emotion_service import EmotionService
from ..utils.common import generate_auth_log_id
from ..utils.rate_limiter import RateLimiter
from ..utils.security import (create_access_token)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Initialize services
auth_rate_limiter = RateLimiter(window_size=60)  # 60 seconds window

# Global service instances
biometric_service: Optional[BiometricService] = None
emotion_service: Optional[EmotionService] = None

def get_biometric_service() -> Optional[BiometricService]:
    """Get or create a biometric service instance"""
    global biometric_service
    if biometric_service is None:
        logger.info("Initializing BiometricService for the first time.")
        biometric_service = initialize_biometric_service()
    return biometric_service

def get_emotion_service() -> EmotionService:
    """Get or create an emotion service instance"""
    global emotion_service
    if emotion_service is None:
        logger.info("Initializing EmotionService for the first time.")
        emotion_service = EmotionService()
    return emotion_service

def with_db_retry(max_retries=3, retry_delay=1):
    """Decorator to retry database operations on connection errors"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except SQLAlchemyError as e:
                    last_error = e
                    logger.warning(f"Database error on attempt {attempt+1}/{max_retries}: {e}")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay)
                        # Try to refresh the database session if possible
                        if 'db' in kwargs and kwargs['db'] is not None:
                            try:
                                kwargs['db'].close()
                                kwargs['db'] = next(get_db())
                            except Exception as refresh_error:
                                logger.error(f"Failed to refresh database session: {refresh_error}")
                    else:
                        logger.error(f"All {max_retries} attempts failed: {e}")
                        traceback.print_exc()
            # If we get here, all retries failed
            if last_error:
                raise last_error
            raise SQLAlchemyError("All database retries failed, but no specific error was captured.")
        return wrapper
    return decorator

def format_emotion_data(emotion_data):
    """
    Convert emotion data to the format expected by the frontend
    """
    if not emotion_data:
        return {
            "happiness": 0.0,
            "neutral": 1.0,  # Default to neutral if no emotion data
            "surprise": 0.0,
            "sadness": 0.0,
            "anger": 0.0,
            "disgust": 0.0,
            "fear": 0.0
        }
    
    # Handle different emotion data formats
    if isinstance(emotion_data, str):
        try:
            emotion_data = json.loads(emotion_data)
        except json.JSONDecodeError:
            logger.warning("Failed to parse emotion data JSON")
            emotion_data = {}
    
    # Normalize emotion keys and ensure all required emotions are present
    formatted_emotions = {
        "happiness": 0.0,
        "neutral": 0.0,
        "surprise": 0.0,
        "sadness": 0.0,
        "anger": 0.0,
        "disgust": 0.0,
        "fear": 0.0
    }
    
    # Map common emotion field variations to standardized names
    emotion_mappings = {
        'happy': 'happiness',
        'neutral': 'neutral',
        'surprised': 'surprise',
        'surprise': 'surprise',
        'sad': 'sadness',
        'sadness': 'sadness',
        'angry': 'anger',
        'anger': 'anger',
        'disgusted': 'disgust',
        'disgust': 'disgust',
        'fearful': 'fear',
        'fear': 'fear'
    }
    
    # Extract emotion values from the input data
    for key, value in emotion_data.items():
        # Normalize key to lowercase
        normalized_key = key.lower()
        
        # Map to standard emotion name
        if normalized_key in emotion_mappings:
            standard_emotion = emotion_mappings[normalized_key]
            if isinstance(value, (int, float)):
                formatted_emotions[standard_emotion] = float(value)
        elif normalized_key in formatted_emotions:
            if isinstance(value, (int, float)):
                formatted_emotions[normalized_key] = float(value)
    
    # Ensure values are between 0 and 1
    for emotion, value in formatted_emotions.items():
        formatted_emotions[emotion] = max(0.0, min(1.0, value))
    
    return formatted_emotions

@router.post("/authenticate", status_code=status.HTTP_200_OK)
@with_db_retry(max_retries=3, retry_delay=1)
async def authenticate_user(
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    image: UploadFile = File(...),
    email: str = Form(None),
    password: str = Form(None),
    device_info: str = Form(""),
    biometric_svc: BiometricService = Depends(get_biometric_service),
    emotion_svc: EmotionService = Depends(get_emotion_service)
):
    """
    Authenticate a user using facial recognition and/or password
    """
    try:
        # Rate limit check
        client_ip = "127.0.0.1"
        if request.client:
            client_ip = request.client.host
        if auth_rate_limiter.is_blocked(client_ip, limit_type="auth", max_requests=5):
            logger.warning(f"Rate limit exceeded for client {client_ip}")
            raise HTTPException(status_code=429, detail="Too many authentication attempts. Please try again later.")

        if not biometric_svc:
            raise HTTPException(status_code=503, detail="Biometric service is not available.")

        image_data = await image.read()

        # Get predicted user ID and confidence
        prediction = biometric_svc.predict_identity(image_data)
        predicted_user_id = prediction.get('user_id')
        confidence = prediction.get('confidence', 0.0)
        threshold = 0.7

        logger.info(f"Using model's predicted user ID: {predicted_user_id}")

        user = db.query(User).filter(User.id == predicted_user_id).first()
        if not user:
            logger.warning(f"Predicted user {predicted_user_id} not found in database, creating temporary user object.")
            user = User(
                id=predicted_user_id,
                name=f"User {predicted_user_id}",
                email=f"user{predicted_user_id}@example.com",
                department="Unknown",
                role="User",
                password_hash="",
                created_at=datetime.utcnow(),
                last_authenticated=None
            )
            db.add(user)
        else:
            user.last_authenticated = datetime.utcnow()  # type: ignore
        
        auth_time = user.last_authenticated or datetime.utcnow()
        auth_log_id = generate_auth_log_id()
        
        background_tasks.add_task(process_emotion_analysis, image_data, auth_log_id, db, emotion_svc)

        auth_log = AuthenticationLog(
            id=auth_log_id,
            user_id=user.id,
            success=confidence >= threshold,
            confidence=confidence,
            threshold=threshold,
            device_info=device_info,
            created_at=auth_time
        )
        db.add(auth_log)
        
        db.commit()

        token_data = { "sub": user.id }
        access_token = create_access_token(data=token_data, expires_delta=timedelta(hours=24))

        return {
            "success": True,
            "message": "Authentication successful",
            "user_id": user.id,
            "confidence": confidence,
            "threshold": threshold,
            "authenticated_at": auth_time.isoformat() + "Z",
            "token": access_token,
            "token_type": "bearer",
            "user": {
                "id": user.id,
                "name": user.name,
                "email": user.email,
                "department": user.department,
                "role": user.role,
                "created_at": user.created_at.isoformat() + "Z" if user.created_at is not None else None,
                "last_authenticated": user.last_authenticated.isoformat() + "Z" if user.last_authenticated is not None else None
            },
            "emotions": None,
        }
    except Exception as e:
        logger.error(f"Authentication processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

async def process_emotion_analysis(image_data: bytes, auth_id: str, db: Session, emotion_service: EmotionService):
    """Process emotion analysis in the background and update the auth log"""
    try:
        logger.info(f"Starting emotion analysis for auth_id: {auth_id}")
        emotion_result = await emotion_service.detect_emotion(image_data)
        
        if emotion_result:
            logger.info(f"Emotion analysis result for auth_id {auth_id}: {emotion_result}")
            db.query(AuthenticationLog).filter(AuthenticationLog.id == auth_id).update({"emotion_data": emotion_result})
            db.commit()
            logger.info(f"Updated auth log {auth_id} with emotion data")
        else:
            logger.warning(f"No emotion data returned for auth_id: {auth_id}")
            
    except Exception as e:
        logger.error(f"Error in emotion analysis background task: {str(e)}") 