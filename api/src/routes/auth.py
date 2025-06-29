"""
Authentication routes for facial recognition
"""

import base64
import io
import json
import logging
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from functools import wraps
import asyncio

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError, OperationalError

from ..db.database import get_db, User, AuthenticationLog
from ..services.service_init import biometric_service, initialize_biometric_service
from ..services.biometric_service import BiometricService  # Direct import for fallback
from ..services.emotion_service import EmotionAnalyzer
from ..utils.common import generate_auth_log_id
from ..utils.rate_limiter import RateLimiter
from ..utils.security import (authenticate_user, create_access_token,
                              get_current_user, get_password_hash)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Rate limiter for authentication attempts
auth_rate_limiter = RateLimiter(window_size=60)  # 60 seconds window

# Initialize emotion analyzer
emotion_analyzer = EmotionAnalyzer()

# Local fallback service for authentication
_local_biometric_service = None

def get_biometric_service():
    """Get or create a biometric service instance"""
    global _local_biometric_service, biometric_service
    
    # First try the global service
    if biometric_service is not None:
        logger.info("Using global biometric service")
        return biometric_service
    
    # Try to initialize the global service
    logger.warning("Global biometric service not available, attempting to initialize")
    try:
        new_service = initialize_biometric_service()
        if new_service is not None:
            logger.info("Successfully initialized global biometric service")
            return new_service
    except Exception as e:
        logger.error(f"Failed to initialize global biometric service: {e}")
    
    # If still not available, use local service
    if _local_biometric_service is None:
        logger.warning("Creating local biometric service as fallback")
        try:
            _local_biometric_service = BiometricService()
            logger.info("Local biometric service created")
        except Exception as e:
            logger.error(f"Failed to create local biometric service: {e}", exc_info=True)
    
    return _local_biometric_service

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
            raise last_error
        return wrapper
    return decorator

async def process_authentication(db, image_data, email=None, password=None, device_info=""):
    """
    Process an authentication request using facial recognition and/or password
    """
    try:
        # Get biometric service
        local_biometric_svc = get_biometric_service()
        
        if local_biometric_svc is None:
            logger.error("Could not obtain a biometric service")
            return {
                "success": False,
                "error": "Biometric service unavailable",
                "authenticated_at": datetime.utcnow().isoformat()
            }
        
        # Get directory structure first
        try:
            partitioned_path = Path("/app/data/partitioned")
            client_dir = partitioned_path / "client1"
            if client_dir.exists():
                # Match exact logic from create_client_dbs.py
                user_dirs = sorted([d for d in client_dir.iterdir() 
                                 if d.is_dir() and d.name.isdigit()], key=lambda x: int(x.name))
                dir_classes = [d.name for d in user_dirs]
                logger.info(f"DIRECTORY STRUCTURE (client1): {dir_classes}")
                
                # Initialize mapping service with directory structure
                local_biometric_svc.mapping_service.initialize_mapping()
                logger.info(f"Updated mapping from directory structure")
        except Exception as e:
            logger.error(f"Error updating mapping from directory: {e}")
        
        # Run debug mapping to check for inconsistencies
        local_biometric_svc.debug_mapping()
        
        # Print model info to verify which model is loaded
        model_info = local_biometric_svc.get_model_info()
        logger.info(f"LOADED MODEL INFO: {model_info}")
        
        # Predict identity using facial recognition
        prediction = local_biometric_svc.predict_identity(image_data, db=None, email=None)  # Don't pass db or email to avoid overrides
        
        # Print raw prediction data for debugging
        logger.info("=============== PREDICTION DEBUG ===============")
        logger.info(f"Raw prediction: {prediction}")
        
        # Try to get directory listing for client1
        try:
            partitioned_path = Path("/app/data/partitioned")
            client_dir = partitioned_path / "client1"
            if client_dir.exists():
                # Match exact logic from create_client_dbs.py
                user_dirs = sorted([d for d in client_dir.iterdir() 
                                 if d.is_dir() and d.name.isdigit()])
                logger.info(f"DIRECTORY STRUCTURE (client1): {[d.name for d in user_dirs]}")
        except Exception as e:
            logger.error(f"Error listing directory: {e}")
            
        # Get top 5 class predictions with confidence scores
        if hasattr(local_biometric_svc, 'get_top_predictions'):
            top_preds = local_biometric_svc.get_top_predictions(image_tensor=None, image_data=image_data, top_k=5)
            logger.info(f"TOP 5 PREDICTIONS: {top_preds}")
        
        # Compare prediction with expected user
        if email:
            try:
                expected_user = db.query(User).filter(User.email == email).first()
                if expected_user:
                    logger.info(f"EXPECTED USER: {expected_user.id} based on email {email}")
                    logger.info(f"MATCH: {'✓' if prediction.get('user_id') == expected_user.id else '✗'}")
            except Exception as e:
                logger.error(f"Error finding expected user: {e}")
        logger.info("===============================================")
        
        # Get predicted user ID and confidence
        predicted_user_id = prediction.get('user_id')
        confidence = prediction.get('confidence', 0.0)
        threshold = 0.5  # Default threshold
        
        # IMPORTANT: Always use the model's prediction
        logger.info(f"Using model's predicted user ID: {predicted_user_id}")
        
        # Try to find user in database for additional info only
        user = None
        try:
            # Look up the predicted user in the database
            user = db.query(User).filter(User.id == predicted_user_id).first()
            
            # If we can't find the predicted user, create a temporary user object
            if not user:
                logger.warning(f"Predicted user {predicted_user_id} not found in database, creating temporary user")
                from ..db.database import User as UserModel
                user = UserModel(
                    id=predicted_user_id,
                    name=f"User {predicted_user_id}",
                    email=f"user{predicted_user_id}@example.com",
                    department="Unknown",
                    role="User",
                    password_hash="",  # Empty password for temporary user
                    created_at=datetime.utcnow()
                )
        except Exception as db_error:
            logger.error(f"Database error while finding user: {db_error}", exc_info=True)
            # Create a minimal user object if database lookup fails
            from ..db.database import User as UserModel
            user = UserModel(
                id=predicted_user_id,
                name=f"User {predicted_user_id}",
                email=f"user{predicted_user_id}@example.com",
                department="Unknown",
                role="User",
                password_hash="",  # Empty password for temporary user
                created_at=datetime.utcnow()
            )
        
        logger.info(f"Using user: {user.id} for authentication")
        
        # Create authentication log
        auth_time = datetime.utcnow()
        auth_log = AuthenticationLog(
            id=generate_auth_log_id(),
            user_id=user.id,
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
        try:
            db.add(auth_log)
            db.commit()
        except Exception as db_error:
            logger.error(f"Failed to save authentication log: {db_error}")
            # Continue even if log saving fails
        
        # Generate a token that includes user information
        token_data = {
            "sub": user.id,  # Subject (user id)
            "name": user.name,
            "email": user.email,
            "department": user.department,
            "role": user.role,
            "created_at": user.created_at.isoformat() if user.created_at else None,
            "last_authenticated": user.last_authenticated.isoformat() if user.last_authenticated else None
        }
        
        # Create access token valid for 24 hours
        access_token = create_access_token(
            data=token_data,
            expires_delta=timedelta(hours=24)
        )
        
        # Update user's last_authenticated timestamp if it's a real database user
        try:
            if hasattr(user, '_sa_instance_state'):  # Check if it's a real SQLAlchemy object
                user.last_authenticated = auth_time
                db.commit()
        except Exception as update_error:
            logger.error(f"Failed to update last_authenticated: {update_error}")
        
        # Return the authentication result
        return {
            "success": True,
            "message": "Authentication successful",
            "user": {
                "id": user.id,
                "name": user.name,
                "email": user.email,
                "department": user.department,
                "role": user.role,
                "created_at": user.created_at.isoformat() if user.created_at else None,
                "last_authenticated": user.last_authenticated.isoformat() if user.last_authenticated else None
            },
            "token": access_token,
            "token_type": "bearer",
            "confidence": confidence,
            "emotion_data": json.loads(auth_log.emotion_data) if auth_log.emotion_data else None
        }
    except Exception as e:
        logger.error(f"Authentication processing failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": f"Authentication processing failed: {str(e)}",
            "authenticated_at": datetime.utcnow().isoformat()
        }

@router.post("/authenticate")
@with_db_retry(max_retries=3, retry_delay=1)
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
            logger.warning(f"Rate limit exceeded for client {client_ip}")
            return JSONResponse(
                status_code=429,
                content={
                    "success": False, 
                    "error": "Too many authentication attempts. Please try again later."
                }
            )
        
        # Read image
        image_data = await image.read()
        
        # Check if biometric service is initialized
        if get_biometric_service() is None:
            logger.error("Biometric service not available")
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": "Biometric service not available"}
            )
        
        # Process the image for face recognition
        try:
            result = await process_authentication(db, image_data, email, password, device_info)
            # Return the authentication result
            return JSONResponse(content=result)
        except Exception as auth_error:
            logger.error(f"Authentication processing error: {str(auth_error)}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": f"Authentication processing error: {str(auth_error)}"}
            )
        
    except SQLAlchemyError as e:
        logger.error(f"Database error during authentication: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "Database connection error. Please try again."}
        )
    except HTTPException as http_exc:
        logger.error(f"HTTP exception during authentication: {http_exc.detail}")
        return JSONResponse(
            status_code=http_exc.status_code,
            content={"success": False, "error": http_exc.detail}
        )
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}", exc_info=True)
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