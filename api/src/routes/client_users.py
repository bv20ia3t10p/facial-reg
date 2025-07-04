"""
Client-specific user routes for the biometric authentication system
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Dict, Union
from datetime import datetime
import logging
import json
from collections import defaultdict
from pathlib import Path
from sqlalchemy import text

from ..db.database import get_db

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

def parse_emotion_data(emotion_data: Union[str, Dict, None]) -> Dict:
    """Parse emotion data from either string or dict format"""
    default_emotions = {
        "happiness": 0.0,
        "neutral": 1.0,
        "surprise": 0.0,
        "sadness": 0.0,
        "anger": 0.0,
        "disgust": 0.0,
        "fear": 0.0
    }
    
    if emotion_data is None:
        return default_emotions
    
    # Handle string input (direct emotion label)
    if isinstance(emotion_data, str):
        try:
            # Try to parse as JSON first
            parsed = json.loads(emotion_data)
            return parse_emotion_data(parsed)  # Recursively handle the parsed data
        except json.JSONDecodeError:
            # If not JSON, treat as direct emotion label
            emotion_label = emotion_data.lower()
            # Map common variations
            emotion_map = {
                'surprised': 'surprise',
                'happy': 'happiness',
                'angry': 'anger',
                'sad': 'sadness',
                'disgusted': 'disgust',
                'fearful': 'fear'
            }
            emotion_label = emotion_map.get(emotion_label, emotion_label)
            
            if emotion_label in default_emotions:
                result = default_emotions.copy()
                result[emotion_label] = 1.0
                return result
            return default_emotions
    
    # Handle dictionary input
    if isinstance(emotion_data, dict):
        result = default_emotions.copy()
        
        # If the dict has an 'emotion' key, it's using the single emotion format
        if 'emotion' in emotion_data:
            emotion_label = emotion_data['emotion'].lower()
            # Map common variations
            emotion_map = {
                'surprised': 'surprise',
                'happy': 'happiness',
                'angry': 'anger',
                'sad': 'sadness',
                'disgusted': 'disgust',
                'fearful': 'fear'
            }
            emotion_label = emotion_map.get(emotion_label, emotion_label)
            
            if emotion_label in result:
                result[emotion_label] = 1.0
            return result
        
        # Otherwise, it should be a probability distribution
        for emotion, value in emotion_data.items():
            emotion_label = emotion.lower()
            # Map common variations
            emotion_map = {
                'surprised': 'surprise',
                'happy': 'happiness',
                'angry': 'anger',
                'sad': 'sadness',
                'disgusted': 'disgust',
                'fearful': 'fear'
            }
            emotion_label = emotion_map.get(emotion_label, emotion_label)
            
            if emotion_label in result and isinstance(value, (int, float)):
                result[emotion_label] = float(value)
        
        return result
    
    return default_emotions

@router.get("/{user_id}")
async def get_user_info(
    user_id: str,
    db: Session = Depends(get_db)
):
    """Get user information and authentication history"""
    try:
        logger.info(f"=== Starting get_user_info for user {user_id} ===")
        
        # Check if user folder exists
        user_folder = Path("/app/data") / user_id
        logger.debug(f"Checking user folder: {user_folder}")
        if not user_folder.exists() or not user_folder.is_dir():
            logger.warning(f"User folder not found: {user_folder}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User not found: {user_id}"
            )
        logger.debug(f"User folder found: {user_folder}")
        
        # Get user metadata from folder
        metadata_file = user_folder / "metadata.json"
        user_metadata = {}
        if metadata_file.exists():
            try:
                logger.debug(f"Reading metadata file: {metadata_file}")
                with open(metadata_file, 'r') as f:
                    user_metadata = json.load(f)
                logger.debug(f"Loaded metadata: {json.dumps(user_metadata, indent=2)}")
            except json.JSONDecodeError:
                logger.warning(f"Failed to read metadata file for user {user_id}")
        else:
            logger.debug("No metadata file found, using defaults")
        
        # Get folder creation time as enrollment date
        try:
            enrolled_at = datetime.fromtimestamp(user_folder.stat().st_ctime).isoformat()
            logger.debug(f"Got enrollment date: {enrolled_at}")
        except Exception as e:
            logger.warning(f"Could not get folder creation time: {e}")
            enrolled_at = None
        
        # Initialize response with default values
        response = {
            "user_id": str(user_id),
            "name": user_metadata.get("name", f"User {user_id}"),
            "email": user_metadata.get("email", f"user_{user_id}@company.com"),
            "department": user_metadata.get("department", "Unknown"),
            "role": user_metadata.get("role", "User"),
            "enrolled_at": enrolled_at,
            "last_authenticated": None,
            "authentication_stats": {
                "total_attempts": 0,
                "successful_attempts": 0,
                "success_rate": 0.0,
                "average_confidence": 0.0
            },
            "recent_attempts": [],
            "latest_auth": None,
            "emotional_state": parse_emotion_data(None),
            "emotion_trends": {
                "average": parse_emotion_data(None),
                "dominant": "neutral"
            }
        }
        logger.debug(f"Initialized response with defaults: {json.dumps(response, indent=2)}")
        
        # Get authentication history from database
        try:
            logger.info("=== Fetching authentication history ===")
            # Get authentication logs with emotion data
            stmt = text("""
                SELECT id, user_id, success, confidence, emotion_data, created_at, device_info, captured_image
                FROM authentication_logs
                WHERE user_id = :user_id
                ORDER BY created_at DESC
            """)
            logger.debug(f"Executing query with params: {{'user_id': {user_id}}}")
            
            # Execute query synchronously
            result = db.execute(stmt, {"user_id": user_id})
            auth_history = result.fetchall()
            logger.info(f"Retrieved {len(auth_history)} authentication records")
            
            if auth_history:
                logger.debug(f"Processing {len(auth_history)} authentication records")
                
                # Calculate stats with proper type handling
                total_attempts = len(auth_history)
                successful_attempts = sum(1 for auth in auth_history if auth.success)
                success_rate = float(successful_attempts) / float(total_attempts) * 100.0 if total_attempts > 0 else 0.0
                avg_confidence = float(sum(float(auth.confidence) for auth in auth_history)) / float(total_attempts) if total_attempts > 0 else 0.0
                
                logger.info(f"Authentication stats calculated - Total: {total_attempts}, "
                          f"Successful: {successful_attempts}, Rate: {success_rate:.1f}%, "
                          f"Avg Confidence: {avg_confidence:.3f}")
                
                # Get latest authentication
                latest_auth = auth_history[0] if auth_history else None
                if latest_auth:
                    # Parse datetime string if needed
                    if isinstance(latest_auth.created_at, str):
                        latest_auth_time = datetime.fromisoformat(latest_auth.created_at.replace('Z', '+00:00'))
                    else:
                        latest_auth_time = latest_auth.created_at
                        
                    logger.debug(f"Latest authentication found - Time: {latest_auth_time}, "
                               f"Success: {latest_auth.success}, Confidence: {latest_auth.confidence}")
                    response["last_authenticated"] = latest_auth_time.isoformat()
                
                # Update response with auth history
                response["authentication_stats"] = {
                    "total_attempts": total_attempts,
                    "successful_attempts": successful_attempts,
                    "success_rate": float(success_rate),
                    "average_confidence": float(avg_confidence)
                }
                logger.debug("Updated authentication stats in response")
                
                # Process recent attempts with emotion data
                logger.info("=== Processing recent attempts ===")
                response["recent_attempts"] = []
                for idx, auth in enumerate(auth_history[:5]):  # Last 5 attempts
                    try:
                        logger.debug(f"Processing attempt {idx + 1}")
                        emotion_data = json.loads(auth.emotion_data) if isinstance(auth.emotion_data, str) else auth.emotion_data
                        emotion_data = parse_emotion_data(emotion_data)
                        logger.debug(f"Parsed emotion data: {json.dumps(emotion_data, indent=2)}")
                        
                        # Parse datetime string if needed
                        if isinstance(auth.created_at, str):
                            auth_time = datetime.fromisoformat(auth.created_at.replace('Z', '+00:00'))
                        else:
                            auth_time = auth.created_at
                            
                        attempt = {
                            "timestamp": auth_time.isoformat(),
                            "success": bool(auth.success),
                            "confidence": float(auth.confidence),
                            "emotion_data": emotion_data
                        }
                        response["recent_attempts"].append(attempt)
                        logger.debug(f"Added attempt to response: {json.dumps(attempt, indent=2)}")
                    except (json.JSONDecodeError, TypeError, ValueError) as e:
                        logger.error(f"Error processing authentication record {idx + 1}: {e}")
                        continue
                
                # Update latest auth with emotion data
                if latest_auth:
                    try:
                        logger.info("=== Processing latest authentication ===")
                        latest_emotion_data = json.loads(latest_auth.emotion_data) if isinstance(latest_auth.emotion_data, str) else latest_auth.emotion_data
                        latest_emotion_data = parse_emotion_data(latest_emotion_data)
                        logger.debug(f"Latest emotion data: {json.dumps(latest_emotion_data, indent=2)}")
                        
                        # Parse datetime string if needed
                        if isinstance(latest_auth.created_at, str):
                            latest_auth_time = datetime.fromisoformat(latest_auth.created_at.replace('Z', '+00:00'))
                        else:
                            latest_auth_time = latest_auth.created_at
                            
                        response["latest_auth"] = {
                            "timestamp": latest_auth_time.isoformat(),
                            "confidence": float(latest_auth.confidence),
                            "emotion_data": latest_emotion_data
                        }
                        logger.debug("Updated latest_auth in response")
                        
                        # Update emotional state with latest emotion data
                        response["emotional_state"] = latest_emotion_data
                        logger.debug("Updated emotional_state in response")
                    except (json.JSONDecodeError, TypeError, ValueError) as e:
                        logger.error(f"Error processing latest authentication: {e}")
                
                # Calculate emotion trends
                logger.info("=== Calculating emotion trends ===")
                emotion_trends = defaultdict(list)
                for idx, auth in enumerate(auth_history):
                    try:
                        emotion_data = json.loads(auth.emotion_data) if isinstance(auth.emotion_data, str) else auth.emotion_data
                        emotion_data = parse_emotion_data(emotion_data)
                        for emotion, value in emotion_data.items():
                            emotion_trends[emotion].append(float(value))
                        logger.debug(f"Processed emotion data for record {idx + 1}")
                    except (json.JSONDecodeError, TypeError, ValueError, AttributeError) as e:
                        logger.error(f"Error processing emotion data for record {idx + 1}: {e}")
                        continue
                
                # Calculate average emotion values
                avg_emotions = {}
                for emotion, values in emotion_trends.items():
                    if values:
                        avg_emotions[emotion] = sum(values) / len(values)
                        logger.debug(f"Average {emotion}: {avg_emotions[emotion]:.3f}")
                
                if avg_emotions:
                    dominant_emotion = max(avg_emotions.items(), key=lambda x: x[1])[0]
                    response["emotion_trends"] = {
                        "average": avg_emotions,
                        "dominant": dominant_emotion
                    }
                    logger.info(f"Dominant emotion: {dominant_emotion}")
            else:
                logger.warning(f"No authentication records found for user {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to get authentication history for user {user_id}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get authentication history: {str(e)}"
            )
        
        logger.info(f"=== Successfully completed get_user_info for {user_id} ===")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        ) 