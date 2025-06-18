"""
Authentication routes with privacy-preserving features
"""

from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, status, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import Dict, Optional
import torch
import numpy as np
from datetime import datetime, timedelta
import jwt
from PIL import Image
import io
import logging
import requests
from urllib.parse import urljoin
import threading
from concurrent.futures import ThreadPoolExecutor
import os
import json

from ..db.database import get_db, User, AuthenticationLog, log_authentication, SessionLocal
from ..privacy.privacy_engine import PrivacyEngine
from ..privacy.federated_manager import FederatedManager
from ..utils.face_recognition import FaceRecognizer
from ..utils.security import verify_password, get_password_hash, get_current_user
from ..services.emotion_service import EmotionAnalyzer, analyze_emotion

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Initialize OAuth2
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# JWT settings
SECRET_KEY = "your-secret-key"  # Change in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Initialize components
face_recognizer = FaceRecognizer()
privacy_engine = PrivacyEngine()
federated_manager = FederatedManager()

# Add emotion API URL configuration
EMOTION_API_URL = os.getenv("EMOTION_API_URL", "http://emotion-api:8080")  # Use container name from docker-compose

logger.info(f"Using emotion API at: {EMOTION_API_URL}")

# Thread pool for async tasks
thread_pool = ThreadPoolExecutor(max_workers=4)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def analyze_emotion(image_data: bytes) -> Dict:
    """Make call to emotion API"""
    try:
        logger.info(f"Making request to emotion API at {EMOTION_API_URL}")
        
        # Create multipart form data
        files = {
            'file': ('face.jpg', image_data, 'image/jpeg')
        }
        
        # Make request to emotion API
        logger.info("Sending request to emotion API...")
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

def update_emotion_data(db: Session, log_id: str, emotion_data: Dict):
    """Update authentication log with emotion data in a separate thread"""
    # Create a new session for this update
    new_session = SessionLocal()
    try:
        logger.info(f"Updating emotion data for auth log {log_id}")
        log_entry = new_session.query(AuthenticationLog).filter(AuthenticationLog.id == log_id).first()
        if log_entry:
            log_entry.emotion_data = emotion_data
            new_session.commit()
            logger.info(f"Successfully updated emotion data: {emotion_data}")
        else:
            logger.warning(f"Auth log {log_id} not found")
    except Exception as e:
        logger.error(f"Failed to update emotion data: {e}")
        if new_session.is_active:
            new_session.rollback()
    finally:
        new_session.close()

@router.post("/face")
async def face_authentication(
    image: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Face recognition authentication with privacy preservation"""
    try:
        # Read image data once
        contents = await image.read()
        if not contents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty image file"
            )
        
        # Start emotion analysis in a thread
        emotion_future = thread_pool.submit(analyze_emotion, contents)
        
        try:
            pil_image = Image.open(io.BytesIO(contents))
            pil_image.verify()  # Verify it's a valid image
            pil_image = Image.open(io.BytesIO(contents))  # Reopen after verify
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid image file: {str(e)}"
            )
        
        # Convert to tensor
        try:
            image_tensor = face_recognizer.preprocess_image(pil_image)
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process image"
            )
        
        # Extract features
        try:
            with torch.no_grad():
                features = face_recognizer.extract_features(image_tensor)
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to extract face features"
            )
        
        # Apply privacy features only if available
        try:
            if privacy_engine.context is not None:
                # Apply differential privacy
                noised_features = privacy_engine.add_noise_to_gradients(features)
                # Encrypt features
                encrypted_features = privacy_engine.encrypt_biometric(noised_features)
            else:
                # Skip privacy features if not initialized
                logger.warning("Privacy features not initialized, proceeding without encryption")
                noised_features = features
                encrypted_features = None
        except Exception as e:
            logger.warning(f"Privacy features failed: {e}, proceeding without encryption")
            noised_features = features
            encrypted_features = None
        
        # Find matching user
        confidence = 0.0
        matched_user = None
        
        try:
            for user in db.query(User).all():
                if user.face_encoding:
                    try:
                        # Decrypt stored features if encrypted
                        if privacy_engine.context is not None and encrypted_features is not None:
                            stored_features = privacy_engine.decrypt_biometric(user.face_encoding)
                        else:
                            # Load features directly if not encrypted
                            stored_features = face_recognizer.load_features(user.face_encoding)
                        
                        # Compute similarity
                        similarity = face_recognizer.compute_similarity(
                            noised_features,
                            stored_features
                        )
                        
                        if similarity > confidence:
                            confidence = similarity
                            matched_user = user
                    except Exception as e:
                        logger.warning(f"Failed to process user {user.id}: {e}")
                        continue
        except Exception as e:
            logger.error(f"User matching failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to match user"
            )
        
        # Authentication result
        success = confidence >= 0.85
        
        # Get emotion analysis results
        try:
            emotion_result = await emotion_future
            if emotion_result:
                logger.info(f"Got emotion analysis result: {emotion_result}")
            else:
                logger.warning("No emotion analysis results")
                emotion_result = {
                    "happiness": 0.0,
                    "neutral": 1.0,
                    "surprise": 0.0,
                    "sadness": 0.0,
                    "anger": 0.0,
                    "disgust": 0.0,
                    "fear": 0.0
                }
        except Exception as e:
            logger.error(f"Failed to get emotion analysis: {e}")
            emotion_result = {
                "happiness": 0.0,
                "neutral": 1.0,
                "surprise": 0.0,
                "sadness": 0.0,
                "anger": 0.0,
                "disgust": 0.0,
                "fear": 0.0
            }
        
        # Create authentication log entry
        try:
            log_entry = AuthenticationLog(
                user_id=matched_user.id if matched_user else None,
                success=success,
                confidence=float(confidence),
                emotion_data=json.dumps(emotion_result),
                captured_image=encrypted_features.serialize() if encrypted_features else None,
                device_info=image.filename  # Store original filename as device info
            )
            db.add(log_entry)
            db.commit()
            db.refresh(log_entry)
            logger.info(f"Created authentication log entry with ID: {log_entry.id}")
            
        except Exception as e:
            logger.warning(f"Failed to log authentication: {e}")
            db.rollback()
        
        # Update user if successful
        if success and matched_user:
            try:
                matched_user.last_authenticated = datetime.utcnow()
                db.commit()
                
                # Create access token
                access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
                access_token = create_access_token(
                    data={"sub": str(matched_user.id)},
                    expires_delta=access_token_expires
                )
                
                response = {
                    "success": True,
                    "confidence": float(confidence),
                    "user": {
                        "id": str(matched_user.id),
                        "name": matched_user.name,
                        "email": matched_user.email,
                        "department": matched_user.department,
                        "role": matched_user.role
                    },
                    "access_token": access_token,
                    "token_type": "bearer",
                    "emotions": emotion_result
                }
                
                return response
            except Exception as e:
                logger.error(f"Failed to update user or create token: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to complete authentication"
                )
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Face authentication failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/verify")
async def verify_credentials(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """Verify username/password credentials"""
    try:
        user = db.query(User).filter(User.email == form_data.username).first()
        if not user or not verify_password(form_data.password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password"
            )
        
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": str(user.id)},
            expires_delta=access_token_expires
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "id": str(user.id),
                "name": user.name,
                "email": user.email,
                "department": user.department,
                "role": user.role
            }
        }
        
    except Exception as e:
        logger.error(f"Credential verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/register")
async def register_user(
    image: UploadFile = File(...),
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    department: str = Form(...),
    role: str = Form(...),
    db: Session = Depends(get_db)
):
    """Register new user with biometric data"""
    try:
        # Check if user exists
        if db.query(User).filter(User.email == email).first():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Process face image
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents))
        image_tensor = face_recognizer.preprocess_image(pil_image)
        
        # Extract and encrypt features
        with torch.no_grad():
            features = face_recognizer.extract_features(image_tensor)
            encrypted_features = privacy_engine.encrypt_biometric(features)
        
        # Create user
        user = User(
            name=name,
            email=email,
            password_hash=get_password_hash(password),
            department=department,
            role=role,
            face_encoding=encrypted_features.serialize()
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        
        return {
            "success": True,
            "message": "User registered successfully",
            "user_id": str(user.id)
        }
        
    except Exception as e:
        logger.error(f"User registration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        ) 