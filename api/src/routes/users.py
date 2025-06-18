"""
User management routes for the biometric authentication system
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import logging
from collections import defaultdict
import json

from ..db.database import get_db, User, AuthenticationLog
from ..utils.security import get_current_user, get_password_hash, verify_password

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

def parse_emotion_data(emotion_data: Optional[Dict[str, float]]) -> Dict[str, float]:
    """Parse and normalize emotion data"""
    default_emotions = {
        "happiness": 0.0,
        "neutral": 1.0,
        "surprise": 0.0,
        "sadness": 0.0,
        "anger": 0.0,
        "disgust": 0.0,
        "fear": 0.0
    }
    
    if not emotion_data:
        return default_emotions
        
    try:
        if isinstance(emotion_data, str):
            emotion_data = json.loads(emotion_data)
            
        # If the data has probabilities field, use that
        if isinstance(emotion_data, dict) and 'probabilities' in emotion_data:
            probabilities = emotion_data['probabilities']
            
            # Map the emotion names
            emotion_map = {
                'happy': 'happiness',
                'surprised': 'surprise',
                'sad': 'sadness',
                'angry': 'anger',
                'disgusted': 'disgust',
                'fearful': 'fear',
                'neutral': 'neutral'
            }
            
            normalized = {}
            for emotion, value in probabilities.items():
                emotion_key = emotion_map.get(emotion.lower(), emotion.lower())
                if emotion_key in default_emotions:
                    # Ensure the value is between 0 and 1
                    try:
                        value = float(value)
                        normalized[emotion_key] = max(0.0, min(1.0, value))
                    except (ValueError, TypeError):
                        normalized[emotion_key] = default_emotions[emotion_key]
            
            # Fill in missing emotions with defaults
            for emotion in default_emotions:
                if emotion not in normalized:
                    normalized[emotion] = default_emotions[emotion]
                    
            return normalized
            
        # If it's already in the right format, just normalize the values
        normalized = {}
        for emotion, value in emotion_data.items():
            if emotion in default_emotions:
                try:
                    value = float(value)
                    normalized[emotion] = max(0.0, min(1.0, value))
                except (ValueError, TypeError):
                    normalized[emotion] = default_emotions[emotion]
                    
        # Fill in missing emotions with defaults
        for emotion in default_emotions:
            if emotion not in normalized:
                normalized[emotion] = default_emotions[emotion]
                
        return normalized
    except (json.JSONDecodeError, AttributeError, ValueError):
        return default_emotions

@router.get("/{user_id}")
async def get_user_info(
    user_id: str,
    db: Session = Depends(get_db)
):
    """Get user information by ID"""
    try:
        logger.info(f"Looking up user with ID: {user_id}")
        
        # Get user with detailed logging
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            logger.warning(f"User not found: {user_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User not found: {user_id}"
            )
        
        logger.info(f"Found user: {user.name} ({user.id})")
        
        # Get authentication history with logging
        auth_history = db.query(AuthenticationLog).filter(
            AuthenticationLog.user_id == user_id
        ).order_by(AuthenticationLog.created_at.desc()).all()
        
        logger.info(f"Found {len(auth_history)} authentication records for user {user_id}")
        
        # Calculate authentication stats
        total_attempts = len(auth_history)
        successful_attempts = sum(1 for auth in auth_history if auth.success)
        success_rate = (successful_attempts / total_attempts * 100) if total_attempts > 0 else 0
        avg_confidence = sum(auth.confidence for auth in auth_history) / total_attempts if total_attempts > 0 else 0
        
        # Get latest authentication and emotion data
        latest_auth = auth_history[0] if auth_history else None
        latest_emotion_data = None
        if latest_auth and latest_auth.emotion_data:
            latest_emotion_data = parse_emotion_data(latest_auth.emotion_data)
        
        # Calculate emotion trends
        emotion_sums = defaultdict(float)
        emotion_counts = defaultdict(int)
        
        for auth in auth_history:
            if auth.emotion_data:
                emotion_data = parse_emotion_data(auth.emotion_data)
                for emotion, value in emotion_data.items():
                    emotion_sums[emotion] += float(value)
                    emotion_counts[emotion] += 1
        
        # Calculate averages
        avg_emotions = {}
        for emotion in emotion_sums:
            if emotion_counts[emotion] > 0:
                avg_emotions[emotion] = round(emotion_sums[emotion] / emotion_counts[emotion], 16)
            else:
                avg_emotions[emotion] = 0.0
        
        # Find dominant emotion
        dominant_emotion = "neutral"
        if avg_emotions:
            dominant_emotion = max(avg_emotions.items(), key=lambda x: x[1])[0]
        
        # Prepare response
        response = {
            "user_id": str(user.id),
            "name": user.name,
            "email": user.email,
            "department": user.department or "Unknown",
            "role": user.role or "User",
            "enrolled_at": user.created_at.isoformat(),
            "last_authenticated": latest_auth.created_at.isoformat() if latest_auth else None,
            "authentication_stats": {
                "total_attempts": total_attempts,
                "successful_attempts": successful_attempts,
                "success_rate": success_rate,
                "average_confidence": avg_confidence
            },
            "recent_attempts": [
                {
                    "timestamp": auth.created_at.isoformat(),
                    "success": bool(auth.success),
                    "confidence": float(auth.confidence),
                    "emotion_data": parse_emotion_data(auth.emotion_data)
                }
                for auth in auth_history[:5]  # Last 5 attempts
            ],
            "latest_auth": {
                "timestamp": latest_auth.created_at.isoformat() if latest_auth else None,
                "confidence": float(latest_auth.confidence) if latest_auth else None,
                "emotion_data": latest_emotion_data or parse_emotion_data(None)
            },
            "emotional_state": latest_emotion_data or parse_emotion_data(None),
            "emotion_trends": {
                "average": avg_emotions,
                "dominant": dominant_emotion
            }
        }
        
        logger.info(f"Successfully retrieved user info for {user_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user info for {user_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/me")
async def get_current_user_info(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get current user information"""
    try:
        user = db.query(User).filter(User.id == current_user.id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        return {
            "id": str(user.id),
            "name": user.name,
            "email": user.email,
            "department": user.department,
            "role": user.role,
            "created_at": user.created_at.isoformat(),
            "last_authenticated": user.last_authenticated.isoformat() if user.last_authenticated else None
        }
    except Exception as e:
        logger.error(f"Failed to get user info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/department/{department}")
async def get_department_users(
    department: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get users in a department"""
    try:
        users = db.query(User).filter(User.department == department).all()
        return [
            {
                "id": str(user.id),
                "username": user.username,
                "email": user.email,
                "full_name": user.full_name,
                "created_at": user.created_at.isoformat(),
                "last_login": user.last_login.isoformat() if user.last_login else None
            }
            for user in users
        ]
    except Exception as e:
        logger.error(f"Failed to get department users: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.put("/me/password")
async def update_password(
    current_password: str,
    new_password: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update user password"""
    try:
        user = db.query(User).filter(User.id == current_user.id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        if not verify_password(current_password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect password"
            )
        
        user.hashed_password = get_password_hash(new_password)
        user.updated_at = datetime.utcnow()
        db.commit()
        
        return {"message": "Password updated successfully"}
        
    except Exception as e:
        logger.error(f"Failed to update password: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        ) 