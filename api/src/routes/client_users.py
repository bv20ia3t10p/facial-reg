"""
Client-specific user routes for the biometric authentication system
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
import logging
import json

from api.src.db.database import get_db, User, AuthenticationLog
from api.src.utils.security import get_current_user

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

def get_emotion_from_data(emotion_data):
    """Extract primary emotion from emotion_data JSON"""
    if not emotion_data:
        return None
    try:
        if isinstance(emotion_data, str):
            data = json.loads(emotion_data)
        else:
            data = emotion_data
        return data.get('emotion')
    except:
        return None

@router.get("/{user_id}")
async def get_user_info(
    user_id: str,
    db: Session = Depends(get_db)
):
    """Get user information by ID"""
    try:
        logger.info(f"Looking up user with ID: {user_id}")
        
        # Log database state
        total_users = db.query(User).count()
        logger.info(f"Total users in database: {total_users}")
        
        # Get user with detailed logging
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            logger.warning(f"User not found: {user_id}")
            # Log all user IDs for debugging
            all_users = db.query(User.id).all()
            user_ids = [u[0] for u in all_users]
            logger.info(f"Available user IDs: {user_ids}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User not found: {user_id}"
            )
        
        logger.info(f"Found user: {user.name} ({user.id})")
        
        # Prepare response
        response = {
            "user_id": str(user.id),
            "name": user.name,
            "email": user.email,
            "department": user.department,
            "role": user.role,
            "enrolled_at": user.created_at.isoformat(),
            "last_authenticated": user.last_authenticated.isoformat() if user.last_authenticated else None,
            "authentication_stats": {
                "total_attempts": 0,
                "successful_attempts": 0,
                "success_rate": 0,
                "average_confidence": 0
            },
            "recent_attempts": [],
            "latest_auth": {
                "timestamp": None,
                "confidence": None,
                "emotion": None
            },
            "emotional_state": {
                "happiness": 0.0,
                "neutral": 1.0,  # Default to neutral when no emotions are recorded
                "surprise": 0.0,
                "sadness": 0.0,
                "anger": 0.0,
                "disgust": 0.0,
                "fear": 0.0
            }
        }
        
        # Try to get authentication history
        try:
            auth_history = db.query(AuthenticationLog).filter(
                AuthenticationLog.user_id == user_id
            ).order_by(AuthenticationLog.created_at.desc()).all()
            
            if auth_history:
                logger.info(f"Found {len(auth_history)} authentication records for user {user_id}")
                
                # Calculate stats
                total_attempts = len(auth_history)
                successful_attempts = sum(1 for auth in auth_history if auth.success)
                success_rate = (successful_attempts / total_attempts * 100) if total_attempts > 0 else 0
                avg_confidence = sum(auth.confidence for auth in auth_history) / total_attempts if total_attempts > 0 else 0
                
                logger.info(f"User {user_id} stats - Total attempts: {total_attempts}, "
                        f"Success rate: {success_rate:.1f}%, Avg confidence: {avg_confidence:.3f}")
                
                # Get latest authentication
                latest_auth = auth_history[0] if auth_history else None
                if latest_auth:
                    logger.info(f"Latest auth for user {user_id}: {latest_auth.created_at}, "
                            f"Success: {latest_auth.success}, Confidence: {latest_auth.confidence}")
                
                # Update response with auth history
                response["authentication_stats"] = {
                    "total_attempts": total_attempts,
                    "successful_attempts": successful_attempts,
                    "success_rate": success_rate,
                    "average_confidence": avg_confidence
                }
                response["recent_attempts"] = [
                    {
                        "timestamp": auth.created_at.isoformat(),
                        "success": auth.success,
                        "confidence": auth.confidence,
                        "emotion": get_emotion_from_data(auth.emotion_data)
                    }
                    for auth in auth_history[:5]  # Last 5 attempts
                ]
                if latest_auth:
                    response["latest_auth"] = {
                        "timestamp": latest_auth.created_at.isoformat(),
                        "confidence": latest_auth.confidence,
                        "emotion": get_emotion_from_data(latest_auth.emotion_data)
                    }
                    
                    # Update emotional state if we have a latest emotion
                    latest_emotion = get_emotion_from_data(latest_auth.emotion_data)
                    if latest_emotion:
                        response["emotional_state"] = {
                            "happiness": 0.0,
                            "neutral": 0.0,
                            "surprise": 0.0,
                            "sadness": 0.0,
                            "anger": 0.0,
                            "disgust": 0.0,
                            "fear": 0.0
                        }
                        response["emotional_state"][latest_emotion.lower()] = 1.0
        except Exception as e:
            logger.warning(f"Failed to get authentication history for user {user_id}: {str(e)}")
            # Continue without auth history
        
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
    finally:
        db.close()  # Close the session after we're done with it 