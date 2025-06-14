"""
Federated Learning Integration for Biometric API
Connects the authentication API with the federated learning coordinator
"""

import os
import sys
import logging
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import hashlib

import requests
import torch
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import schedule
import time
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()

class FederatedModelVersion(Base):
    """Track federated model versions"""
    __tablename__ = "federated_model_versions"
    
    id = Column(Integer, primary_key=True)
    version_hash = Column(String(64), unique=True, nullable=False)
    round_id = Column(Integer, nullable=False)
    accuracy = Column(Float, nullable=True)
    participants = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=False)
    model_path = Column(String(255), nullable=True)
    metadata = Column(Text, nullable=True)

class FederatedIntegration:
    """Integration layer between API and federated learning"""
    
    def __init__(self, 
                 coordinator_url: str = "http://localhost:8001",
                 api_db_path: str = "D:/data/biometric.db",
                 federated_db_path: str = "D:/data/federated_integration.db"):
        
        self.coordinator_url = coordinator_url
        self.api_db_path = api_db_path
        self.federated_db_path = federated_db_path
        
        # Setup database
        self.engine = create_engine(f"sqlite:///{federated_db_path}")
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Model management
        self.current_model_hash = None
        self.model_update_interval = 300  # 5 minutes
        self.last_model_check = None
        
        # Start background tasks
        self.start_background_tasks()
    
    def get_db_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()
    
    def check_coordinator_health(self) -> bool:
        """Check if federated coordinator is healthy"""
        try:
            response = requests.get(f"{self.coordinator_url}/health", timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Coordinator health check failed: {str(e)}")
            return False
    
    def get_global_model_info(self) -> Optional[Dict]:
        """Get current global model information"""
        try:
            response = requests.get(f"{self.coordinator_url}/models/global", timeout=10)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                logger.info("No global model available yet")
                return None
            else:
                logger.error(f"Failed to get global model: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting global model info: {str(e)}")
            return None
    
    def download_global_model(self, model_hash: str) -> Optional[str]:
        """Download global model if available"""
        try:
            # In a real implementation, this would download the actual model
            # For now, we'll simulate by checking if the model exists
            model_path = f"D:/models/federated_global_{model_hash}.pth"
            
            # Check if we already have this model
            if os.path.exists(model_path):
                logger.info(f"Global model {model_hash} already exists locally")
                return model_path
            
            # In real implementation, download from coordinator
            # For now, copy from the federated model path
            federated_model_path = "D:/models/federated_model.pth"
            if os.path.exists(federated_model_path):
                import shutil
                shutil.copy2(federated_model_path, model_path)
                logger.info(f"Downloaded global model to {model_path}")
                return model_path
            else:
                logger.warning("Federated model not found for download")
                return None
                
        except Exception as e:
            logger.error(f"Failed to download global model: {str(e)}")
            return None
    
    def update_api_model(self, model_path: str, model_info: Dict) -> bool:
        """Update the API's biometric model with federated model"""
        try:
            # Copy model to API model location
            api_model_path = "D:/models/biometric_model.pth"
            
            import shutil
            shutil.copy2(model_path, api_model_path)
            
            # Update model version in database
            with self.get_db_session() as db:
                # Deactivate old versions
                db.query(FederatedModelVersion).update({"is_active": False})
                
                # Add new version
                new_version = FederatedModelVersion(
                    version_hash=model_info["model_hash"],
                    round_id=model_info.get("round", 0),
                    accuracy=model_info.get("global_accuracy"),
                    participants=model_info.get("participants"),
                    is_active=True,
                    model_path=api_model_path,
                    metadata=json.dumps(model_info)
                )
                
                db.add(new_version)
                db.commit()
            
            self.current_model_hash = model_info["model_hash"]
            logger.info(f"Updated API model with federated model (hash: {model_info['model_hash']})")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update API model: {str(e)}")
            return False
    
    def check_for_model_updates(self):
        """Check for new federated model updates"""
        try:
            if not self.check_coordinator_health():
                logger.warning("Coordinator not available, skipping model update check")
                return
            
            # Get current global model info
            model_info = self.get_global_model_info()
            
            if not model_info:
                logger.info("No global model available")
                return
            
            model_hash = model_info["model_hash"]
            
            # Check if this is a new model
            if model_hash != self.current_model_hash:
                logger.info(f"New federated model available: {model_hash}")
                
                # Download the model
                model_path = self.download_global_model(model_hash)
                
                if model_path:
                    # Update API model
                    success = self.update_api_model(model_path, model_info)
                    
                    if success:
                        logger.info("Successfully updated API with new federated model")
                        
                        # Trigger API model reload (if API supports it)
                        self.notify_api_model_update()
                    else:
                        logger.error("Failed to update API model")
                else:
                    logger.error("Failed to download new model")
            else:
                logger.debug("No new model updates available")
            
            self.last_model_check = datetime.now()
            
        except Exception as e:
            logger.error(f"Error checking for model updates: {str(e)}")
    
    def notify_api_model_update(self):
        """Notify the API that the model has been updated"""
        try:
            # Send signal to API to reload model
            api_url = "http://localhost:8000"  # Assuming API runs on port 8000
            
            response = requests.post(
                f"{api_url}/admin/reload-model",
                json={"reason": "federated_update", "model_hash": self.current_model_hash},
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("Successfully notified API of model update")
            else:
                logger.warning(f"Failed to notify API: {response.status_code}")
                
        except Exception as e:
            logger.warning(f"Could not notify API of model update: {str(e)}")
    
    def get_api_usage_stats(self) -> Dict:
        """Get usage statistics from API database"""
        try:
            import sqlite3
            
            with sqlite3.connect(self.api_db_path) as conn:
                cursor = conn.cursor()
                
                # Get authentication stats
                cursor.execute("""
                    SELECT COUNT(*) as total_auths,
                           SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_auths,
                           COUNT(DISTINCT user_id) as unique_users
                    FROM authentication_logs 
                    WHERE created_at > datetime('now', '-24 hours')
                """)
                
                auth_stats = cursor.fetchone()
                
                # Get enrollment stats
                cursor.execute("""
                    SELECT COUNT(*) as total_enrollments,
                           COUNT(DISTINCT user_id) as unique_enrollments
                    FROM users 
                    WHERE created_at > datetime('now', '-24 hours')
                """)
                
                enrollment_stats = cursor.fetchone()
                
                return {
                    "total_authentications": auth_stats[0] if auth_stats else 0,
                    "successful_authentications": auth_stats[1] if auth_stats else 0,
                    "unique_users": auth_stats[2] if auth_stats else 0,
                    "total_enrollments": enrollment_stats[0] if enrollment_stats else 0,
                    "unique_enrollments": enrollment_stats[1] if enrollment_stats else 0,
                    "success_rate": (auth_stats[1] / auth_stats[0]) if auth_stats and auth_stats[0] > 0 else 0.0
                }
                
        except Exception as e:
            logger.error(f"Failed to get API usage stats: {str(e)}")
            return {}
    
    def trigger_federated_round(self) -> bool:
        """Trigger a new federated learning round"""
        try:
            response = requests.post(f"{self.coordinator_url}/rounds/start", timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Successfully started federated round {result['round_id']}")
                return True
            else:
                logger.error(f"Failed to start federated round: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error triggering federated round: {str(e)}")
            return False
    
    def get_federated_status(self) -> Dict:
        """Get comprehensive federated learning status"""
        try:
            status = {
                "coordinator_healthy": self.check_coordinator_health(),
                "last_model_check": self.last_model_check.isoformat() if self.last_model_check else None,
                "current_model_hash": self.current_model_hash,
                "api_usage_stats": self.get_api_usage_stats()
            }
            
            if status["coordinator_healthy"]:
                # Get coordinator status
                response = requests.get(f"{self.coordinator_url}/health", timeout=10)
                if response.status_code == 200:
                    coordinator_status = response.json()
                    status.update({
                        "active_round": coordinator_status.get("active_round"),
                        "registered_clients": coordinator_status.get("registered_clients", 0),
                        "he_enabled": coordinator_status.get("he_enabled", False)
                    })
                
                # Get active clients
                response = requests.get(f"{self.coordinator_url}/clients/active", timeout=10)
                if response.status_code == 200:
                    clients_info = response.json()
                    status["active_clients"] = clients_info.get("count", 0)
                    status["client_details"] = clients_info.get("active_clients", [])
                
                # Get current round info
                response = requests.get(f"{self.coordinator_url}/rounds/current", timeout=10)
                if response.status_code == 200:
                    round_info = response.json()
                    status["current_round"] = round_info
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting federated status: {str(e)}")
            return {"error": str(e)}
    
    def schedule_periodic_tasks(self):
        """Schedule periodic federated learning tasks"""
        # Check for model updates every 5 minutes
        schedule.every(5).minutes.do(self.check_for_model_updates)
        
        # Trigger federated round every hour (if conditions are met)
        schedule.every().hour.do(self.conditional_federated_round)
        
        # Cleanup old model versions daily
        schedule.every().day.at("02:00").do(self.cleanup_old_models)
    
    def conditional_federated_round(self):
        """Trigger federated round if conditions are met"""
        try:
            # Get API usage stats
            stats = self.get_api_usage_stats()
            
            # Only trigger if there's been significant activity
            if stats.get("total_authentications", 0) > 10:
                logger.info("Triggering federated round due to API activity")
                self.trigger_federated_round()
            else:
                logger.info("Skipping federated round - insufficient activity")
                
        except Exception as e:
            logger.error(f"Error in conditional federated round: {str(e)}")
    
    def cleanup_old_models(self):
        """Clean up old model versions"""
        try:
            with self.get_db_session() as db:
                # Keep only the last 10 model versions
                old_versions = db.query(FederatedModelVersion)\
                    .filter(FederatedModelVersion.is_active == False)\
                    .order_by(FederatedModelVersion.created_at.desc())\
                    .offset(10)\
                    .all()
                
                for version in old_versions:
                    # Delete model file if it exists
                    if version.model_path and os.path.exists(version.model_path):
                        os.remove(version.model_path)
                        logger.info(f"Deleted old model file: {version.model_path}")
                    
                    # Delete database record
                    db.delete(version)
                
                db.commit()
                logger.info(f"Cleaned up {len(old_versions)} old model versions")
                
        except Exception as e:
            logger.error(f"Error cleaning up old models: {str(e)}")
    
    def start_background_tasks(self):
        """Start background task scheduler"""
        def run_scheduler():
            self.schedule_periodic_tasks()
            
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        
        # Start scheduler in background thread
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        
        logger.info("Background federated integration tasks started")
    
    def get_model_history(self) -> List[Dict]:
        """Get history of federated model updates"""
        try:
            with self.get_db_session() as db:
                versions = db.query(FederatedModelVersion)\
                    .order_by(FederatedModelVersion.created_at.desc())\
                    .limit(20)\
                    .all()
                
                return [
                    {
                        "version_hash": v.version_hash,
                        "round_id": v.round_id,
                        "accuracy": v.accuracy,
                        "participants": v.participants,
                        "created_at": v.created_at.isoformat(),
                        "is_active": v.is_active,
                        "metadata": json.loads(v.metadata) if v.metadata else {}
                    }
                    for v in versions
                ]
                
        except Exception as e:
            logger.error(f"Error getting model history: {str(e)}")
            return []

# Global integration instance
federated_integration = None

def get_federated_integration() -> FederatedIntegration:
    """Get or create federated integration instance"""
    global federated_integration
    
    if federated_integration is None:
        federated_integration = FederatedIntegration()
    
    return federated_integration

def initialize_federated_integration():
    """Initialize federated integration on startup"""
    try:
        integration = get_federated_integration()
        logger.info("Federated integration initialized successfully")
        return integration
    except Exception as e:
        logger.error(f"Failed to initialize federated integration: {str(e)}")
        return None
 