"""
Emotion service for handling facial emotion analysis
"""

import logging
import aiohttp
import json
from typing import Dict, Any
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionService:
    def __init__(self):
        """Initialize the emotion service"""
        self.analyzer = EmotionAnalyzer()
        logger.info("Emotion service initialized")
    
    async def analyze_emotion(self, image_data: bytes) -> Dict[str, float]:
        """Analyze emotions in an image"""
        return await self.analyzer.analyze_emotion(image_data)

class EmotionAnalyzer:
    def __init__(self):
        """Initialize the emotion analyzer"""
        self.api_url = os.getenv('EMOTION_API_URL', 'http://emotion-api:8080')
        logger.info(f"Emotion analyzer initialized with API URL: {self.api_url}")
    
    async def analyze_emotion(self, image_data: bytes) -> Dict[str, float]:
        """Analyze emotions in an image using the emotion API"""
        try:
            endpoint = f"{self.api_url}/predict"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, data={'image': image_data}) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get('emotions', {})
                    else:
                        logger.error(f"Emotion API error: {response.status}")
                        return {}
        except Exception as e:
            logger.error(f"Failed to analyze emotion: {e}")
            return {} 