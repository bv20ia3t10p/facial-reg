"""
Authentication routes for facial recognition
"""

import base64
import io
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db.database import get_db, User, AuthenticationLog
from ..services.service_init import biometric_service as biometric_svc
from ..services.emotion_service import EmotionAnalyzer
from ..utils.common import generate_auth_log_id
from ..utils.face_recognition import FaceRecognizer
from ..utils.rate_limiter import RateLimiter
from ..utils.security import (authenticate_user, create_access_token,
                              get_current_user, get_password_hash)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Initialize face recognizer
face_recognizer = FaceRecognizer()

# Rate limiter for authentication attempts
auth_rate_limiter = RateLimiter(window_size=60)  # 60 seconds window

# Initialize emotion analyzer
emotion_analyzer = EmotionAnalyzer()

async def process_authentication(db, image_data, email=None, password=None, device_info=""):
    """
    Process an authentication request using facial recognition and/or password
    """
    try:
        # Predict identity using facial recognition
        prediction = biometric_svc.predict_identity(image_data, db)
        user_id = prediction['user_id']
        confidence = prediction['confidence']
        features = prediction.get('features', None)
        
        # Get threshold from config
        threshold = 0.7  # Default threshold
        
        # Get user from database
        user = db.query(User).filter(User.id == user_id).first()
        
        if not user:
            logger.error(f"User with ID {user_id} not found in database")
            return {
                "success": False, 
                "error": f"User with ID {user_id} not found",
                "authenticated_at": datetime.utcnow().isoformat()
            }
        
        # Create authentication log
        auth_time = datetime.utcnow()
        auth_log = AuthenticationLog(
            id=generate_auth_log_id(),
            user_id=user_id,
            success=confidence >= threshold,  # Set success based on threshold
            confidence=confidence,
            threshold=threshold,
            device_info=device_info,
            created_at=auth_time
        )
        
        # Analyze emotion if available
        try:
            emotion_data = await emotion_analyzer.analyze_emotion(image_data)
            if emotion_data:
                auth_log.emotion_data = json.dumps(emotion_data)
        except Exception as e:
            logger.warning(f"Emotion analysis failed: {e}")
        
        # Save the authentication log
        db.add(auth_log)
        db.commit()
        
        # Generate a token that includes user information
        token_data = {
            "sub": user_id,  # Subject (user id)
            "name": user.name,
            "email": user.email,
            "department": user.department,
            "role": user.role
        }
        
        # Create access token valid for 24 hours
        access_token = create_access_token(
            data=token_data,
            expires_delta=timedelta(hours=24)
        )
        
        # Update user's last_authenticated timestamp
        user.last_authenticated = auth_time
        db.commit()
        
        # Return the authentication result
        return {
            "success": True,
            "user_id": user_id,
            "name": user.name,
            "email": user.email,
            "department": user.department,
            "role": user.role,
            "confidence": confidence,
            "threshold": threshold,
            "meets_threshold": confidence >= threshold,
            "authenticated_at": auth_log.created_at.isoformat(),
            "token": access_token,
            "emotion_data": json.loads(auth_log.emotion_data) if auth_log.emotion_data else None
        }
    except Exception as e:
        logger.error(f"Authentication processing failed: {e}")
        raise

@router.post("/authenticate")
async def authenticate(
    request: Request,
    db: Session = Depends(get_db),
    image: UploadFile = File(...),
    email: str = Form(None),
    password: str = Form(None),
    device_info: str = Form(""),
):
    """
    Authenticate a user using facial recognition and/or password
    """
    try:
        # Rate limit check
        client_ip = request.client.host
        if auth_rate_limiter.is_blocked(client_ip, limit_type="auth", max_requests=5):
            raise HTTPException(
                status_code=429,
                detail="Too many authentication attempts. Please try again later."
            )
        
        # Read image
        image_data = await image.read()
        
        # Check if biometric service is initialized
        if biometric_svc is None:
            raise HTTPException(
                status_code=500,
                detail="Biometric service not available"
            )
        
        # Process the image for face recognition
        result = await process_authentication(db, image_data, email, password, device_info)
        
        # Return the authentication result
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Authentication error: {str(e)}"}
        )

async def process_emotion_analysis(image_data: bytes, auth_id: str, db: Session):
    """Process emotion analysis in the background and update the auth log"""
    try:
        logger.info(f"Starting emotion analysis for auth_id: {auth_id}")
        
        # Analyze emotion
        emotion_result = await emotion_analyzer.analyze_emotion(image_data)
        
        if emotion_result:
            logger.info(f"Emotion analysis result for auth_id {auth_id}: {emotion_result}")
            
            # Update auth log with emotion data
            auth_log = db.query(AuthenticationLog).filter(AuthenticationLog.id == auth_id).first()
            if auth_log:
                auth_log.emotion_data = emotion_result
                db.commit()
                logger.info(f"Updated auth log {auth_id} with emotion data")
            else:
                logger.error(f"Auth log {auth_id} not found for emotion update")
        else:
            logger.warning(f"No emotion data returned for auth_id: {auth_id}")
            
    except Exception as e:
        logger.error(f"Error in emotion analysis background task: {str(e)}") 