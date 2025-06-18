"""
Emotion Analysis Service
"""

import logging
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

class EmotionAnalyzer:
    """Emotion analysis model"""
    
    def __init__(self):
        """Initialize emotion analyzer"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.emotions = ["neutral", "happy", "sad", "angry", "surprised", "fearful", "disgusted"]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # TODO: Load actual emotion model
        # For now, using a simple placeholder model
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 56 * 56, len(self.emotions))
        ).to(self.device)
        
        self.model.eval()
        logger.info(f"Initialized emotion analyzer on {self.device}")

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for model input"""
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply transformations
            tensor = self.transform(image)
            tensor = tensor.unsqueeze(0)
            tensor = tensor.to(self.device)
            
            return tensor
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise

    @torch.no_grad()
    def analyze(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze emotions in an image"""
        try:
            # Preprocess image
            tensor = self.preprocess_image(image)
            
            # Get model predictions
            outputs = self.model(tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
            # Get emotion predictions
            emotion_probs = {
                emotion: float(prob)
                for emotion, prob in zip(self.emotions, probabilities)
            }
            
            # Get primary emotion
            primary_emotion = max(emotion_probs.items(), key=lambda x: x[1])
            
            return {
                "primary_emotion": primary_emotion[0],
                "confidence": primary_emotion[1],
                "probabilities": emotion_probs,
                "timestamp": None  # Set by the caller
            }
            
        except Exception as e:
            logger.error(f"Emotion analysis failed: {e}")
            raise

# Global instance
_emotion_analyzer: Optional[EmotionAnalyzer] = None

def get_emotion_analyzer() -> EmotionAnalyzer:
    """Get or create emotion analyzer instance"""
    global _emotion_analyzer
    if _emotion_analyzer is None:
        _emotion_analyzer = EmotionAnalyzer()
    return _emotion_analyzer

async def analyze_emotion(image: Image.Image) -> Dict[str, Any]:
    """Analyze emotions in an image"""
    analyzer = get_emotion_analyzer()
    return analyzer.analyze(image) 