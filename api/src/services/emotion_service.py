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

class EmotionAnalyzer:
    def __init__(self):
        """Initialize the emotion analyzer"""
        self.api_url = os.getenv('EMOTION_API_URL', 'http://emotion-api:8080')
        logger.info(f"Emotion analyzer initialized with API URL: {self.api_url}")
    
    async def analyze_emotion(self, image_data: bytes) -> Dict[str, float]:
        """Analyze emotions in an image using the emotion API"""
        try:
            endpoint = f"{self.api_url}/predict"
            logger.info(f"Sending request to emotion API: {endpoint}")
            
            async with aiohttp.ClientSession() as session:
                # Create form data with the image
                form = aiohttp.FormData()
                form.add_field('file', image_data, filename='image.jpg', content_type='image/jpeg')
                
                # Send request to emotion API
                async with session.post(endpoint, data=form) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Emotion API error: {error_text}")
                        raise Exception(f"Emotion API returned status {response.status}: {error_text}")
                    
                    result = await response.json()
                    logger.info(f"Emotion API response: {json.dumps(result, indent=2)}")
                    
                    # The result directly contains emotion probabilities
                    if not result or not any(k in result for k in ['neutral', 'happiness', 'sadness']):
                        raise Exception("Invalid emotion data in API response")
                    
                    # Remove reliability score if present (it's not an emotion)
                    if 'reliability' in result:
                        del result['reliability']
                    
                    return result
            
        except Exception as e:
            logger.error(f"Emotion analysis failed: {e}")
            # Return neutral emotion as fallback
            return {
                "neutral": 1.0,
                "happiness": 0.0,
                "surprise": 0.0,
                "sadness": 0.0,
                "anger": 0.0,
                "disgust": 0.0,
                "fear": 0.0
            } 