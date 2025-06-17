"""
Analytics routes for the biometric authentication system
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Dict, List
from datetime import datetime, timedelta
import logging

from api.src.db.database import get_db, User, AuthenticationLog
from api.src.utils.security import get_current_user

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

@router.get("/hr")
async def get_hr_analytics(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get HR analytics data"""
    try:
        # Get all authentication logs from the past 7 days
        week_ago = datetime.utcnow() - timedelta(days=7)
        logs = db.query(AuthenticationLog).filter(
            AuthenticationLog.created_at >= week_ago
        ).all()
        
        # Calculate emotion distribution
        emotion_counts = {}
        total_logs = len(logs)
        
        for log in logs:
            if log.emotion:
                emotion_counts[log.emotion] = emotion_counts.get(log.emotion, 0) + 1
        
        emotion_distribution = [
            {
                "emotion": emotion,
                "percentage": (count / total_logs) * 100 if total_logs > 0 else 0
            }
            for emotion, count in emotion_counts.items()
        ]
        
        # Get department analytics
        departments = db.query(User.department).distinct().all()
        department_analytics = []
        
        for (dept,) in departments:
            dept_logs = [log for log in logs if log.user and log.user.department == dept]
            dept_total = len(dept_logs)
            
            # Calculate department metrics
            stress_level = sum(1 for log in dept_logs if log.emotion == 'stressed') / dept_total * 100 if dept_total > 0 else 0
            satisfaction = sum(1 for log in dept_logs if log.emotion in ['happy', 'neutral']) / dept_total * 100 if dept_total > 0 else 0
            emotional_balance = sum(1 for log in dept_logs if log.emotion not in ['stressed', 'angry', 'sad']) / dept_total * 100 if dept_total > 0 else 0
            
            department_analytics.append({
                "department": dept,
                "metrics": {
                    "stressLevel": stress_level,
                    "jobSatisfaction": satisfaction,
                    "emotionalBalance": emotional_balance,
                    "wellbeingScore": (satisfaction + emotional_balance - stress_level) / 3
                }
            })
        
        # Generate alerts
        alerts = []
        for dept in department_analytics:
            if dept["metrics"]["stressLevel"] > 60:
                alerts.append({
                    "id": f"stress_{dept['department']}",
                    "type": "stress",
                    "severity": "high",
                    "department": dept["department"],
                    "message": f"High stress levels detected in {dept['department']} department",
                    "timestamp": datetime.utcnow().isoformat()
                })
            elif dept["metrics"]["jobSatisfaction"] < 50:
                alerts.append({
                    "id": f"satisfaction_{dept['department']}",
                    "type": "satisfaction",
                    "severity": "medium",
                    "department": dept["department"],
                    "message": f"Low job satisfaction in {dept['department']} department",
                    "timestamp": datetime.utcnow().isoformat()
                })
        
        # Calculate overall wellbeing
        overall_metrics = {
            "stressLevel": sum(d["metrics"]["stressLevel"] for d in department_analytics) / len(department_analytics) if department_analytics else 0,
            "jobSatisfaction": sum(d["metrics"]["jobSatisfaction"] for d in department_analytics) / len(department_analytics) if department_analytics else 0,
            "emotionalBalance": sum(d["metrics"]["emotionalBalance"] for d in department_analytics) / len(department_analytics) if department_analytics else 0,
            "wellbeingScore": sum(d["metrics"]["wellbeingScore"] for d in department_analytics) / len(department_analytics) if department_analytics else 0
        }
        
        return {
            "overallWellbeing": overall_metrics,
            "departmentAnalytics": department_analytics,
            "recentEmotionalTrends": [
                {
                    "timestamp": (datetime.utcnow() - timedelta(days=7)).isoformat(),
                    "emotionDistribution": emotion_distribution
                },
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "emotionDistribution": emotion_distribution
                }
            ],
            "alerts": alerts
        }
        
    except Exception as e:
        logger.error(f"Failed to get HR analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/auth")
async def get_auth_analytics(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get authentication analytics"""
    try:
        # Get authentication logs from the past 24 hours
        day_ago = datetime.utcnow() - timedelta(days=1)
        logs = db.query(AuthenticationLog).filter(
            AuthenticationLog.created_at >= day_ago
        ).all()
        
        # Calculate metrics
        total_auths = len(logs)
        successful_auths = sum(1 for log in logs if log.success)
        avg_confidence = sum(log.confidence for log in logs) / total_auths if total_auths > 0 else 0
        
        return {
            "dailyAuthentications": total_auths,
            "successRate": (successful_auths / total_auths * 100) if total_auths > 0 else 0,
            "averageConfidence": avg_confidence
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