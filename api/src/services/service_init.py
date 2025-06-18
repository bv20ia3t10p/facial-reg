"""
Service initialization and management module
"""

import logging
from ..utils.face_recognition import FaceRecognizer
from ..privacy.privacy_engine import PrivacyEngine
from .client_service import ClientService

logger = logging.getLogger(__name__)

class ServiceManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ServiceManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.client_service = None
            self.face_recognizer = None
            self.privacy_engine = None
            self._initialized = True
    
    async def initialize_services(self):
        """Initialize all required services"""
        logger.info("=== Initializing services ===")
        try:
            self.client_service = ClientService()
            logger.info("Client service initialized")
            
            self.face_recognizer = FaceRecognizer()
            logger.info("Face recognition service initialized")
            
            self.privacy_engine = PrivacyEngine()
            logger.info("Privacy engine initialized")
            
            logger.info("=== All services initialized successfully ===")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            return False
    
    def get_services(self):
        """Get initialized services"""
        if not all([self.client_service, self.face_recognizer, self.privacy_engine]):
            raise RuntimeError("Services not properly initialized")
        return self.client_service, self.face_recognizer, self.privacy_engine

# Create singleton instance
service_manager = ServiceManager() 