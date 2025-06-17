"""
User management routes for the biometric authentication system
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import logging
from collections import defaultdict

from api.src.db.database import get_db, User, AuthenticationLog
from api.src.utils.security import get_current_user, get_password_hash, verify_password

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

def calculate_emotion_trends(auth_history: List[AuthenticationLog]) -> Dict:
    """Calculate emotion trends from authentication history"""
    if not auth_history:
        return {}
    
    # Initialize counters
    emotion_counts = defaultdict(int)
    emotion_confidences = defaultdict(list)
    
    # Process each authentication log
    for auth in auth_history:
        if auth.emotion_data and isinstance(auth.emotion_data, dict):
            # Get the primary emotion and its confidence
            emotion = auth.emotion_data.get('emotion')
            if emotion:
                emotion_counts[emotion] += 1
                if 'confidence' in auth.emotion_data:
                    emotion_confidences[emotion].append(auth.emotion_data['confidence'])
    
    # Calculate percentages and average confidences
    total_emotions = sum(emotion_counts.values())
    emotion_trends = {}
    
    if total_emotions > 0:
        for emotion, count in emotion_counts.items():
            emotion_trends[emotion] = {
                'percentage': (count / total_emotions) * 100,
                'average_confidence': sum(emotion_confidences[emotion]) / len(emotion_confidences[emotion])
                if emotion_confidences[emotion] else 0
            }
    
    return emotion_trends

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
        
        # Log the exact SQL query being executed
        query = db.query(User).filter(User.id == user_id)
        logger.info(f"Executing SQL query: {query.statement.compile(compile_kwargs={'literal_binds': True})}")
        
        # Get user with detailed logging
        user = query.first()
        if not user:
            logger.warning(f"User not found: {user_id}")
            # Log all user IDs for debugging
            all_users = db.query(User.id).all()
            user_ids = [u[0] for u in all_users]
            logger.info(f"Available user IDs: {user_ids}")
            # Log the exact format of the requested ID vs available IDs
            logger.info(f"Requested ID type: {type(user_id)}, value: '{user_id}'")
            logger.info(f"Available ID types: {[type(uid) for uid in user_ids]}")
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
        
        # Calculate stats
        total_attempts = len(auth_history)
        successful_attempts = sum(1 for auth in auth_history if auth.success)
        success_rate = (successful_attempts / total_attempts * 100) if total_attempts > 0 else 0
        avg_confidence = sum(auth.confidence for auth in auth_history) / total_attempts if total_attempts > 0 else 0
        
        logger.info(f"User {user_id} stats - Total attempts: {total_attempts}, "
                   f"Success rate: {success_rate:.1f}%, Avg confidence: {avg_confidence:.3f}")
        
        # Get latest authentication
        latest_auth = auth_history[0] if auth_history else None
        
        # Calculate emotion trends
        emotion_trends = calculate_emotion_trends(auth_history)
        
        # Get recent emotion data (last 24 hours)
        recent_cutoff = datetime.utcnow() - timedelta(hours=24)
        recent_emotions = [
            auth.emotion_data for auth in auth_history 
            if auth.created_at >= recent_cutoff and auth.emotion_data
        ]
        
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
                "total_attempts": total_attempts,
                "successful_attempts": successful_attempts,
                "success_rate": success_rate,
                "average_confidence": avg_confidence
            },
            "recent_attempts": [
                {
                    "timestamp": auth.created_at.isoformat(),
                    "success": auth.success,
                    "confidence": auth.confidence,
                    "emotion_data": auth.emotion_data
                }
                for auth in auth_history[:5]  # Last 5 attempts
            ],
            "latest_auth": {
                "timestamp": latest_auth.created_at.isoformat() if latest_auth else None,
                "confidence": latest_auth.confidence if latest_auth else None,
                "emotion_data": latest_auth.emotion_data if latest_auth else None
            },
            "emotion_analysis": {
                "trends": emotion_trends,
                "recent_emotions": recent_emotions[:10],  # Last 10 emotion readings
                "last_24h_summary": {
                    "total_readings": len(recent_emotions),
                    "dominant_emotion": max(emotion_trends.items(), key=lambda x: x[1]['percentage'])[0]
                    if emotion_trends else None
                }
            },
            "last_updated": datetime.utcnow().isoformat()
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
    finally:
        db.close()  # Close the session after we're done with it

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