"""
Client service for managing client-specific operations
"""

import logging
from typing import Optional, Dict, Any
import json
from pathlib import Path
from datetime import datetime

from ..db.database import Client, get_db
from ..utils.security import generate_uuid

logger = logging.getLogger(__name__)

class ClientService:
    """Service for managing client operations"""
    
    def __init__(self):
        """Initialize client service"""
        self._initialized = False
        self._client_id: Optional[str] = None
        self._client_type: Optional[str] = None
        self._privacy_budget: float = 1.0
        self._metrics: Dict[str, Any] = {}
        
    @property
    def is_initialized(self) -> bool:
        """Check if service is initialized"""
        return self._initialized
        
    @property
    def client_id(self) -> Optional[str]:
        """Get client ID"""
        return self._client_id
        
    @property
    def client_type(self) -> Optional[str]:
        """Get client type"""
        return self._client_type
        
    @property
    def privacy_budget(self) -> float:
        """Get remaining privacy budget"""
        return self._privacy_budget
        
    async def initialize(self) -> bool:
        """Initialize the client service"""
        try:
            logger.info("Initializing client service...")
            
            # Load client configuration
            config_path = Path("database/clients.json")
            if not config_path.exists():
                logger.error("Client configuration file not found")
                return False
                
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # For now, just use the first client entry
            # TODO: Add proper client identification logic
            if not config.get("clients"):
                logger.error("No clients found in configuration")
                return False
                
            client = config["clients"][0]
            self._client_id = client["id"]
            self._client_type = client["client_type"]
            self._privacy_budget = float(client.get("privacy_budget", 1.0))
            self._metrics = client.get("metrics", {})
            
            logger.info(f"Client service initialized with ID: {self._client_id}, Type: {self._client_type}")
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize client service: {e}")
            return False
            
    def update_privacy_budget(self, used_budget: float) -> None:
        """Update privacy budget after usage"""
        if not self._initialized:
            raise RuntimeError("Client service not initialized")
            
        self._privacy_budget = max(0.0, self._privacy_budget - used_budget)
        logger.info(f"Updated privacy budget to {self._privacy_budget}")
        
    def log_metric(self, metric_name: str, value: Any) -> None:
        """Log a metric for the client"""
        if not self._initialized:
            raise RuntimeError("Client service not initialized")
            
        if metric_name not in self._metrics:
            self._metrics[metric_name] = []
            
        self._metrics[metric_name].append({
            "value": value,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get all client metrics"""
        if not self._initialized:
            raise RuntimeError("Client service not initialized")
        return self._metrics.copy()
        
    def save_state(self) -> bool:
        """Save current client state to configuration file"""
        try:
            if not self._initialized:
                raise RuntimeError("Client service not initialized")
                
            config_path = Path("database/clients.json")
            if not config_path.exists():
                logger.error("Client configuration file not found")
                return False
                
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Update client state
            for client in config["clients"]:
                if client["id"] == self._client_id:
                    client.update({
                        "privacy_budget": self._privacy_budget,
                        "metrics": self._metrics,
                        "last_update": datetime.utcnow().isoformat()
                    })
                    break
                    
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
                
            logger.info("Client state saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save client state: {e}")
            return False 