"""
Emotion service for handling facial emotion analysis
"""

import logging
import os
import httpx
from typing import Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionService:
    """Service for detecting emotions from facial images"""

    def __init__(self):
        """Initialize the emotion service"""
        # Use API URL from environment variable or default to local service
        self.api_url = os.getenv("EMOTION_API_URL", 'http://emotion-api:8080')
        
        self.client = httpx.AsyncClient(timeout=10.0, base_url=self.api_url, transport=httpx.AsyncHTTPTransport(http2=False))
        logger.info(f"Emotion service initialized to use API at: {self.api_url}")

    async def detect_emotion(self, image_data: bytes) -> Dict[str, float]:
        """Detect emotion from a given image"""
        try:
            files = {'file': ('image.jpg', image_data, 'image/jpeg')}
            
            logger.info(f"Calling emotion API at: {self.client.base_url}/predict")
            
            response = await self.client.post("/predict", files=files)
            response.raise_for_status() 
            
            result = response.json()
            logger.info(f"Emotion API response: {result}")

            emotion_fields = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
            emotions = {key: value for key, value in result.items() if key in emotion_fields}
            
            if not emotions:
                logger.warning("No valid emotion data in API response.")
                return {}
                
            return emotions

        except httpx.HTTPStatusError as e:
            logger.error(f"Emotion API returned an error: {e.response.status_code} - {e.response.text}")
            return {}
        except httpx.RequestError as e:
            logger.error(f"HTTP client error calling emotion API: {e}")
            return {}
        except Exception as e:
            logger.error(f"An unexpected error occurred in detect_emotion: {e}", exc_info=True)
            return {}

# You might not need the EmotionAnalyzer class if the service is just a client
# to an external API. I'm leaving it here for now but it seems unused.
class EmotionAnalyzer:
    def __init__(self):
        """Initialize a local emotion analyzer."""
        # This part would contain logic for a local model if you had one.
        # For now, it does nothing as we are using an external API.
        pass

    async def analyze_emotion(self, image_data: bytes) -> Dict[str, float]:
        """Placeholder for local emotion analysis."""
        logger.warning("Local EmotionAnalyzer.analyze_emotion is not implemented. Returning empty dict.")
        return {} 