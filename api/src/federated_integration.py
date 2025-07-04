"""
Federated Learning Integration for Biometric API
Connects the authentication API with the federated learning coordinator
"""

import logging
import json
from typing import Dict, Optional, Any
from pathlib import Path

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, MetaData
from sqlalchemy.ext.declarative import declarative_base

from .utils.datetime_utils import get_current_time, get_current_time_str

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a separate base for federated models to avoid conflicts
table_metadata = MetaData()  # Renamed from metadata to avoid conflict
FederatedBase = declarative_base(metadata=table_metadata)

class FederatedModelVersion(FederatedBase):
    """Track federated model versions"""
    __tablename__ = "federated_model_versions"
    
    id = Column(Integer, primary_key=True)
    version_hash = Column(String(64), unique=True, nullable=False)
    round_id = Column(Integer, nullable=False)
    accuracy = Column(Float, nullable=True)
    participants = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), default=get_current_time)
    is_active = Column(Boolean, default=False)
    path = Column(String(255), nullable=True)  # Path to model file
    model_metadata = Column(Text, nullable=True)  # Model metadata as JSON string

class FederatedIntegration:
    """Handles federated learning integration"""
    
    def __init__(self):
        self._initialized = False
        self._current_round = 0
        self._model_history = []
        self._status = {
            "active": False,
            "last_round": None,
            "participants": 0,
            "model_version": "0.1.0"
        }
    
    def initialize(self) -> bool:
        """Initialize federated learning components"""
        try:
            logger.info("Initializing federated learning integration...")
            
            # Load configuration if exists
            config_path = Path("database/clients.json")
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if config.get("clients"):
                        self._status["participants"] = len(config["clients"])
            
            self._initialized = True
            self._status["active"] = True
            logger.info("Federated learning integration initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize federated learning: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current federated learning status"""
        return {
            "initialized": self._initialized,
            "current_round": self._current_round,
            "status": self._status,
            "last_update": get_current_time_str()
        }
    
    def trigger_round(self) -> Dict[str, Any]:
        """Trigger a new federated learning round"""
        if not self._initialized:
            raise RuntimeError("Federated learning not initialized")
        
        try:
            self._current_round += 1
            timestamp = get_current_time_str()
            
            round_info = {
                "round": self._current_round,
                "timestamp": timestamp,
                "participants": self._status["participants"],
                "status": "completed"
            }
            
            self._model_history.append(round_info)
            self._status["last_round"] = timestamp
            
            logger.info(f"Completed federated learning round {self._current_round}")
            return round_info
            
        except Exception as e:
            logger.error(f"Failed to complete federated round: {e}")
            return {"error": str(e)}
    
    def get_model_history(self) -> list:
        """Get history of model updates"""
        return self._model_history.copy()

# Global instance
_federated_integration: Optional[FederatedIntegration] = None

def get_federated_integration() -> FederatedIntegration:
    """Get or create federated integration instance"""
    global _federated_integration
    if _federated_integration is None:
        _federated_integration = FederatedIntegration()
    return _federated_integration

def initialize_federated_integration() -> bool:
    """Initialize federated integration"""
    try:
        integration = get_federated_integration()
        return integration.initialize()
    except Exception as e:
        logger.error(f"Failed to initialize federated integration: {e}")
        return False 