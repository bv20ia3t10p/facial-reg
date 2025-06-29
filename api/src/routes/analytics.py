"""
Analytics routes for the biometric authentication system
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import logging
import json
import uuid
import pytz
import ast

from ..db.database import get_db, User, AuthenticationLog
from ..utils.security import get_current_user

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(tags=["analytics"])

def safe_parse_emotion_data(emotion_data: Any) -> Dict:
    """
    Safely parse emotion data that might not be in valid JSON format.
    Tries multiple approaches to convert the data into a usable dictionary.
    """
    # If it's already a dict, return it
    if isinstance(emotion_data, dict):
        return emotion_data
        
    # If it's a string, try to parse it
    if isinstance(emotion_data, str) and emotion_data.strip():
        # Check if it's a simple emotion name
        if emotion_data.strip().lower() in ['happiness', 'sadness', 'anger', 'fear', 'disgust', 'surprise', 'neutral']:
            return {emotion_data.strip().lower(): 1.0}
            
        # Try standard JSON parsing first
        try:
            return json.loads(emotion_data)
        except json.JSONDecodeError as e:
            # Only log warnings for issues other than the common property name quoting issue
            if "Expecting property name enclosed in double quotes" not in str(e):
                logger.warning(f"Error parsing emotion data: {e}")
            
            # Try to fix common JSON format issues
            try:
                # Try to evaluate as Python dict literal (handles single quotes)
                # This is safer than using eval() directly
                return ast.literal_eval(emotion_data)
            except (ValueError, SyntaxError) as e:
                if "Expecting property name enclosed in double quotes" not in str(e):
                    logger.warning(f"Failed to parse emotion data as Python dict: {e}")
                
                # If it looks like a dict but with single quotes, try a simple replacement
                # Note: This is a simple approach and may not work for complex nested structures
                if emotion_data.startswith('{') and emotion_data.endswith('}'):
                    try:
                        # Replace single quotes with double quotes, but be careful with nested quotes
                        fixed_json = emotion_data.replace("'", '"')
                        return json.loads(fixed_json)
                    except json.JSONDecodeError:
                        if "Expecting property name enclosed in double quotes" not in str(e):
                            logger.warning(f"Failed to parse emotion data after quote replacement")

    # Return empty dict if all parsing attempts fail
    return {}

def normalize_emotion_data(emotion_data: Dict) -> Dict[str, float]:
    """Normalize emotion data to a consistent format with numeric values"""
    # Map of emotion name variations to standard names
    emotion_name_map = {
        'happy': 'happiness',
        'sad': 'sadness',
        'angry': 'anger',
        'surprised': 'surprise',
        'fearful': 'fear',
        'disgusted': 'disgust',
        'neutral': 'neutral'
    }
    
    # Set of all standard emotion names
    standard_emotions = set(emotion_name_map.values())
    
    normalized = {}
    
    # Skip processing if emotion_data is not a dictionary
    if not isinstance(emotion_data, dict):
        logger.warning(f"Invalid emotion data format: {type(emotion_data)}")
        return {emotion: 0.0 for emotion in standard_emotions}
    
    for key, value in emotion_data.items():
        # Skip timestamp-like values
        if isinstance(value, str) and ('T' in value and '+' in value and value.count(':') >= 2):
            logger.debug(f"Skipping timestamp-like value: {value}")
            continue
            
        # Normalize emotion name
        emotion_name = emotion_name_map.get(key.lower(), key.lower())
        
        # Handle different value formats
        if isinstance(value, (int, float)):
            normalized[emotion_name] = float(value)
        elif isinstance(value, str):
            # If the value is a string emotion name, treat it as 1.0
            if value.lower() in emotion_name_map:
                normalized[emotion_name_map[value.lower()]] = 1.0
            # If the value is the same as the key (e.g., "sadness": "sadness"), treat it as 1.0
            elif value.lower() == key.lower() or value.lower() == emotion_name:
                normalized[emotion_name] = 1.0
            # If the value is a standard emotion name, treat it as 1.0
            elif value.lower() in standard_emotions:
                normalized[value.lower()] = 1.0
            else:
                try:
                    normalized[emotion_name] = float(value)
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert emotion value to float: {value}")
                    # Instead of skipping, set a default value of 1.0 for string emotions
                    normalized[emotion_name] = 1.0
    
    # Ensure we have all emotion categories with at least 0
    for standard_emotion in standard_emotions:
        if standard_emotion not in normalized:
            normalized[standard_emotion] = 0.0
    
    return normalized

def calculate_wellbeing_metrics(emotion_data: Dict[str, float]) -> Dict[str, float]:
    """Calculate wellbeing metrics from emotion data"""
    # Convert emotion values to float and normalize
    total = sum(float(v) for v in emotion_data.values())
    if total == 0:
        return {
            "stressLevel": 0,
            "jobSatisfaction": 0,
            "emotionalBalance": 0,
            "wellbeingScore": 0
        }
    
    normalized = {k: float(v)/total for k, v in emotion_data.items()}
    
    # Calculate metrics
    stress_level = (
        normalized.get('anger', 0) * 1.0 +
        normalized.get('fear', 0) * 0.8 +
        normalized.get('disgust', 0) * 0.6 +
        normalized.get('sadness', 0) * 0.4
    ) * 100
    
    job_satisfaction = (
        normalized.get('happiness', 0) * 1.0 +
        normalized.get('neutral', 0) * 0.5
    ) * 100
    
    emotional_balance = (
        normalized.get('neutral', 0) * 1.0 +
        normalized.get('happiness', 0) * 0.8 -
        normalized.get('anger', 0) * 0.6 -
        normalized.get('fear', 0) * 0.4
    ) * 100
    
    wellbeing_score = (
        (100 - stress_level) * 0.3 +
        job_satisfaction * 0.4 +
        (emotional_balance + 100) * 0.3 / 2
    )
    
    return {
        "stressLevel": min(100, max(0, stress_level)),
        "jobSatisfaction": min(100, max(0, job_satisfaction)),
        "emotionalBalance": min(100, max(0, emotional_balance)),
        "wellbeingScore": min(100, max(0, wellbeing_score))
    }

def generate_alerts(dept_analytics: List[Dict]) -> List[Dict]:
    """Generate alerts based on department analytics"""
    alerts = []
    for dept in dept_analytics:
        metrics = dept['metrics']
        timestamp = datetime.now(pytz.UTC).isoformat()
        
        # Check stress levels
        if metrics['stressLevel'] > 70:
            alerts.append({
                "id": str(uuid.uuid4()),
                "type": "stress",
                "severity": "high",
                "department": dept['department'],
                "message": f"High stress levels detected in {dept['department']} department",
                "timestamp": timestamp
            })
        elif metrics['stressLevel'] > 50:
            alerts.append({
                "id": str(uuid.uuid4()),
                "type": "stress",
                "severity": "medium",
                "department": dept['department'],
                "message": f"Moderate stress levels detected in {dept['department']} department",
                "timestamp": timestamp
            })
        
        # Check job satisfaction
        if metrics['jobSatisfaction'] < 30:
            alerts.append({
                "id": str(uuid.uuid4()),
                "type": "satisfaction",
                "severity": "high",
                "department": dept['department'],
                "message": f"Low job satisfaction detected in {dept['department']} department",
                "timestamp": timestamp
            })
        elif metrics['jobSatisfaction'] < 50:
            alerts.append({
                "id": str(uuid.uuid4()),
                "type": "satisfaction",
                "severity": "medium",
                "department": dept['department'],
                "message": f"Declining job satisfaction in {dept['department']} department",
                "timestamp": timestamp
            })
        
        # Check overall wellbeing
        if metrics['wellbeingScore'] < 40:
            alerts.append({
                "id": str(uuid.uuid4()),
                "type": "wellbeing",
                "severity": "high",
                "department": dept['department'],
                "message": f"Critical wellbeing score in {dept['department']} department",
                "timestamp": timestamp
            })
        elif metrics['wellbeingScore'] < 60:
            alerts.append({
                "id": str(uuid.uuid4()),
                "type": "wellbeing",
                "severity": "medium",
                "department": dept['department'],
                "message": f"Below average wellbeing score in {dept['department']} department",
                "timestamp": timestamp
            })
    
    return alerts

@router.get("/hr")
async def get_hr_analytics(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get HR analytics data"""
    try:
        logger.info(f"Received HR analytics request from user: {current_user.id if current_user else 'Unknown'}")
        logger.info(f"User department: {current_user.department if current_user else 'Unknown'}")
        logger.info(f"User role: {current_user.role if current_user else 'Unknown'}")
        
        # Check if user is in HR department or has HR role
        is_hr_user = False
        if current_user:
            department = current_user.department.lower() if current_user.department else ""
            role = current_user.role.lower() if current_user.role else ""
            
            if ('hr' in department or 'human resources' in department or 
                'hr' in role or 'human resources' in role):
                is_hr_user = True
        
        if not current_user or not is_hr_user:
            logger.warning(f"Access denied: User {current_user.id if current_user else 'Unknown'} not in HR department/role")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only HR department members can access this data"
            )
        
        # Get all authentication logs from the past 7 days
        now = datetime.now(pytz.UTC)
        week_ago = now - timedelta(days=7)
        logs = db.query(AuthenticationLog).filter(
            AuthenticationLog.created_at >= week_ago
        ).all()
        logger.info(f"Found {len(logs)} authentication logs in the past 7 days")
        
        # Calculate overall emotion distribution
        overall_emotions = {}
        for log in logs:
            if log.emotion_data:
                try:
                    # Handle different emotion_data formats - SQLAlchemy already converts JSON column to dict
                    emotion_data = log.emotion_data
                    
                    # Skip if emotion_data is not in the expected format
                    if not isinstance(emotion_data, dict):
                        logger.warning(f"Unexpected emotion_data format: {type(emotion_data)}")
                        # Try to convert to dict if it's a simple string
                        if isinstance(emotion_data, str) and emotion_data.strip():
                            if emotion_data.strip().lower() in ['happiness', 'sadness', 'anger', 'fear', 'disgust', 'surprise', 'neutral']:
                                emotion_data = {emotion_data.strip().lower(): 1.0}
                            else:
                                try:
                                    emotion_data = safe_parse_emotion_data(emotion_data)
                                except Exception as e:
                                    logger.warning(f"Failed to parse emotion data as JSON: {emotion_data[:30]}...")
                                    continue
                        else:
                            continue
                    
                    normalized_emotions = normalize_emotion_data(emotion_data)
                    for emotion, value in normalized_emotions.items():
                        overall_emotions[emotion] = overall_emotions.get(emotion, 0) + value
                except Exception as e:
                    logger.warning(f"Error processing emotion data: {str(e)}")
                    continue
        
        # Calculate overall wellbeing metrics
        overall_wellbeing = calculate_wellbeing_metrics(overall_emotions)
        
        # Get department analytics
        departments = db.query(User.department).distinct().all()
        logger.info(f"Found {len(departments)} distinct departments")
        department_analytics = []
        
        for (dept,) in departments:
            if not dept:
                continue
                
            dept_users = db.query(User).filter(User.department == dept).all()
            dept_user_ids = [user.id for user in dept_users]
            logger.info(f"Department {dept}: Found {len(dept_users)} users")
            
            # Get authentication logs for department users
            dept_logs = [log for log in logs if log.user_id in dept_user_ids]
            total_dept_logs = len(dept_logs)
            logger.info(f"Department {dept}: Found {total_dept_logs} authentication logs")
            
            # Calculate department emotion distribution
            dept_emotions = {}
            for log in dept_logs:
                if log.emotion_data:
                    try:
                        # Handle different emotion_data formats - SQLAlchemy already converts JSON column to dict
                        emotion_data = log.emotion_data
                        
                        # Skip if emotion_data is not in the expected format
                        if not isinstance(emotion_data, dict):
                            logger.warning(f"Unexpected emotion_data format for department {dept}: {type(emotion_data)}")
                            # Try to convert to dict if it's a simple string
                            if isinstance(emotion_data, str) and emotion_data.strip():
                                if emotion_data.strip().lower() in ['happiness', 'sadness', 'anger', 'fear', 'disgust', 'surprise', 'neutral']:
                                    emotion_data = {emotion_data.strip().lower(): 1.0}
                                else:
                                    try:
                                        emotion_data = safe_parse_emotion_data(emotion_data)
                                    except Exception as e:
                                        logger.warning(f"Failed to parse emotion data as JSON for department {dept}: {emotion_data[:30]}...")
                                        continue
                            else:
                                continue
                        
                        normalized_emotions = normalize_emotion_data(emotion_data)
                        for emotion, value in normalized_emotions.items():
                            dept_emotions[emotion] = dept_emotions.get(emotion, 0) + value
                    except Exception as e:
                        logger.warning(f"Error processing emotion data for department {dept}: {str(e)}")
                        continue
            
            # Calculate department metrics
            dept_metrics = calculate_wellbeing_metrics(dept_emotions)
            
            # Create trend data (simplified - just current state for now)
            trend_data = [{
                "timestamp": now.isoformat(),
                "metrics": dept_metrics
            }]
            
            department_analytics.append({
                "department": dept,
                "metrics": dept_metrics,
                "trendData": trend_data
            })
        
        # Calculate recent emotional trends (last 7 days in daily intervals)
        recent_emotional_trends = []
        for i in range(7):
            day_start = now - timedelta(days=i)
            day_end = day_start + timedelta(days=1)
            day_logs = [log for log in logs if day_start <= log.created_at.replace(tzinfo=pytz.UTC) <= day_end]
            
            day_emotions = {}
            total_emotions = 0
            
            for log in day_logs:
                if log.emotion_data:
                    try:
                        # Handle different emotion_data formats
                        if isinstance(log.emotion_data, dict):
                            # Already a dictionary, no need to parse
                            emotion_data = log.emotion_data
                        elif isinstance(log.emotion_data, str):
                            try:
                                if not log.emotion_data.strip():
                                    # Skip empty strings
                                    continue
                                
                                emotion_data = safe_parse_emotion_data(log.emotion_data)
                            except Exception as e:
                                logger.warning(f"Error parsing emotion data: {str(e)}")
                                continue
                        else:
                            logger.warning(f"Unexpected emotion_data type in day logs: {type(log.emotion_data)}")
                            continue
                        
                        # Skip if emotion_data is not in the expected format
                        if not isinstance(emotion_data, dict):
                            # Try to convert to dict if it's a simple string
                            if isinstance(emotion_data, str) and emotion_data.strip():
                                emotion_data = {emotion_data.strip().lower(): 1.0}
                            else:
                                continue
                        
                        normalized_emotions = normalize_emotion_data(emotion_data)
                        for emotion, value in normalized_emotions.items():
                            day_emotions[emotion] = day_emotions.get(emotion, 0) + value
                            total_emotions += value
                    except Exception as e:
                        logger.warning(f"Error processing daily emotion data: {str(e)}")
                        continue
            
            if total_emotions > 0:
                emotion_distribution = [
                    {
                        "emotion": emotion,
                        "percentage": (value / total_emotions) * 100
                    }
                    for emotion, value in day_emotions.items()
                ]
                
                recent_emotional_trends.append({
                    "timestamp": day_start.isoformat(),
                    "emotionDistribution": emotion_distribution
                })
        
        # Generate alerts based on department analytics
        alerts = generate_alerts(department_analytics)
        
        logger.info("Successfully generated HR analytics")
        return {
            "overallWellbeing": overall_wellbeing,
            "departmentAnalytics": department_analytics,
            "recentEmotionalTrends": recent_emotional_trends,
            "alerts": alerts
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get HR analytics: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get HR analytics"
        )

@router.get("/auth")
async def get_auth_analytics(
    db: Session = Depends(get_db)
):
    """Get authentication analytics"""
    logger.info("Received request to /api/analytics/auth")
    try:
        # Get authentication logs from the past 24 hours
        day_ago = datetime.utcnow() - timedelta(days=1)
        
        logs = db.query(AuthenticationLog).filter(
            AuthenticationLog.created_at >= day_ago
        ).all()
        
        # Calculate metrics
        total_auths = len(logs)
        successful_auths = sum(1 for log in logs if log.success)
        
        # Calculate average confidence
        avg_confidence = 0
        if total_auths > 0:
            avg_confidence = sum(log.confidence for log in logs) / total_auths
        
        # Initialize emotion counts using standard emotion names
        standard_emotions = {
            "neutral": 0,
            "happiness": 0,
            "sadness": 0,
            "anger": 0,
            "surprise": 0,
            "fear": 0,
            "disgust": 0
        }
        
        total_emotions = 0
        for log in logs:
            if log.emotion_data:
                try:
                    # Parse emotion data
                    emotion_data = safe_parse_emotion_data(log.emotion_data)
                    
                    # Use the normalize_emotion_data function to handle all cases
                    normalized_emotions = normalize_emotion_data(emotion_data)
                    
                    # Find the dominant emotion
                    if normalized_emotions:
                        max_emotion = max(normalized_emotions.items(), key=lambda x: x[1])
                        emotion_name = max_emotion[0]
                        if emotion_name in standard_emotions:
                            standard_emotions[emotion_name] += 1
                            total_emotions += 1
                except Exception as e:
                    logger.error(f"Error parsing emotion data: {e}")
                    continue
        
        # Convert counts to probabilities
        emotion_distribution = {
            emotion: count / total_emotions if total_emotions > 0 else 0.0
            for emotion, count in standard_emotions.items()
        }
        
        logger.info(f"Analytics results - Total auths: {total_auths}, Emotions: {standard_emotions}")
        
        return {
            "dailyAuthentications": total_auths,
            "averageConfidence": avg_confidence,
            "emotionDistribution": emotion_distribution
        }
        
    except Exception as e:
        logger.error(f"Failed to get auth analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/stats")
async def get_analytics_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get general analytics stats"""
    try:
        # Get recent authentication logs
        recent_logs = db.query(AuthenticationLog).order_by(
            AuthenticationLog.created_at.desc()
        ).limit(10).all()
        
        return {
            "recentAuthentications": [
                {
                    "id": str(log.id),
                    "timestamp": log.created_at.isoformat(),
                    "success": log.success,
                    "confidence": log.confidence,
                    "emotion": log.emotion,
                    "user": {
                        "id": str(log.user.id),
                        "name": log.user.name,
                        "department": log.user.department
                    } if log.user else None
                }
                for log in recent_logs
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to get analytics stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/auth-logs")
async def get_auth_logs(
    log_id: Optional[str] = None,
    user_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Query authentication logs by ID or user ID"""
    try:
        query = db.query(AuthenticationLog)
        
        if log_id:
            # Query specific log by ID
            log = query.filter(AuthenticationLog.id == log_id).first()
            if not log:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Authentication log {log_id} not found"
                )
            return {
                "id": log.id,
                "user_id": log.user_id,
                "success": log.success,
                "confidence": log.confidence,
                "emotion_data": safe_parse_emotion_data(log.emotion_data) if log.emotion_data else None,
                "created_at": log.created_at.isoformat(),
                "device_info": log.device_info
            }
            
        elif user_id:
            # Query all logs for a specific user, ordered by most recent first
            logs = query.filter(AuthenticationLog.user_id == user_id)\
                       .order_by(desc(AuthenticationLog.created_at))\
                       .all()
            return [{
                "id": log.id,
                "user_id": log.user_id,
                "success": log.success,
                "confidence": log.confidence,
                "emotion_data": safe_parse_emotion_data(log.emotion_data) if log.emotion_data else None,
                "created_at": log.created_at.isoformat(),
                "device_info": log.device_info
            } for log in logs]
        
        else:
            # If no specific filters, return last 24 hours of logs
            yesterday = datetime.utcnow() - timedelta(days=1)
            logs = query.filter(AuthenticationLog.created_at >= yesterday)\
                       .order_by(desc(AuthenticationLog.created_at))\
                       .all()
            return [{
                "id": log.id,
                "user_id": log.user_id,
                "success": log.success,
                "confidence": log.confidence,
                "emotion_data": safe_parse_emotion_data(log.emotion_data) if log.emotion_data else None,
                "created_at": log.created_at.isoformat(),
                "device_info": log.device_info
            } for log in logs]
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        ) 