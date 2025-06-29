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
            logger.info(f"Calling emotion API at: {endpoint}")
            
            # Create form data with proper file upload format
            form_data = aiohttp.FormData()
            form_data.add_field('file', image_data, content_type='image/jpeg', filename='image.jpg')
            
            timeout = aiohttp.ClientTimeout(total=30)  # 30 second timeout
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(endpoint, data=form_data) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"Emotion API response: {result}")
                        
                        # The emotion API returns the emotions directly, not nested in 'emotions' key
                        # Filter out non-emotion fields like 'reliability'
                        emotion_fields = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
                        emotions = {key: value for key, value in result.items() if key in emotion_fields}
                        
                        if not emotions:
                            logger.warning("No valid emotion data in response, using full result")
                            emotions = result
                            
                        return emotions
                    else:
                        error_text = await response.text()
                        logger.error(f"Emotion API error: {response.status} - {error_text}")
                        return {}
        except aiohttp.ClientError as e:
            logger.error(f"HTTP client error calling emotion API: {e}")
            return {}
        except Exception as e:
            logger.error(f"Failed to analyze emotion: {e}")
            return {} 