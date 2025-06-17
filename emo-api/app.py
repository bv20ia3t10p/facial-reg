import logging
import time
import argparse
import pickle
import torch.storage
import os

from typing import Dict, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info("Starting imports...")

logger.info("Importing FastAPI...")
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
logger.info("FastAPI imported successfully")

logger.info("Importing PyTorch...")
import torch
from torch import nn
import torch.nn.functional as F
logger.info("PyTorch imported successfully")
logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA version: {torch.version.cuda}")

logger.info("Importing timm...")
import timm
logger.info("timm imported successfully")
logger.info(f"timm version: {timm.__version__}")

logger.info("Importing torchvision...")
import torchvision
import torchvision.transforms as transforms
logger.info("torchvision imported successfully")
logger.info(f"torchvision version: {torchvision.__version__}")

logger.info("Importing PIL...")
from PIL import Image
logger.info("PIL imported successfully")

logger.info("Importing other utilities...")
import io
import numpy as np
from typing import List
logger.info("All imports completed successfully")

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "argparse" and name == "Namespace":
            return argparse.Namespace
        return super().find_class(module, name)
    
    def persistent_load(self, persistent_id):
        return None  # Ignore persistent storage

def custom_load(f):
    return CustomUnpickler(f).load()

# Check CUDA availability and compatibility at startup
def setup_device():
    if not torch.cuda.is_available():
        logger.info("CUDA is not available, using CPU")
        return torch.device('cpu')
    
    try:
        logger.info("CUDA is available, testing compatibility...")
        device = torch.device('cuda')
        test_tensor = torch.zeros(1, device=device)
        test_result = test_tensor + 1
        del test_tensor
        del test_result
        
        logger.info(f"CUDA is fully compatible")
        logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        return device
    except Exception as e:
        logger.warning(f"CUDA compatibility test failed: {str(e)}")
        logger.warning("Falling back to CPU for compatibility")
        return torch.device('cpu')

# Set up device at module level
DEVICE = setup_device()
logger.info(f"Using device: {DEVICE}")

app = FastAPI(title="Emotion Recognition API",
             description="API for emotion recognition using RAF-DB model")

logger.info("FastAPI application created")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("CORS middleware added")

# Global variables
model = None
# More careful CUDA device setup
if torch.cuda.is_available():
    try:
        # Test CUDA with a small tensor operation
        test_tensor = torch.zeros(1, device='cuda')
        del test_tensor
        device = torch.device("cuda")
        logger.info(f"CUDA is available and working. Using device: {device}")
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
    except Exception as e:
        logger.warning(f"CUDA is available but test failed: {str(e)}. Falling back to CPU.")
        device = torch.device("cpu")
else:
    device = torch.device("cpu")
    logger.info("CUDA is not available. Using CPU.")

# Define emotion labels consistently (only once)
EMOTIONS = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

@app.on_event("startup")
async def startup_event():
    """Initialize the model when the FastAPI app starts"""
    global model
    try:
        logger.info("Loading model on startup...")
        model = create_vit_model()
        
        # Try CUDA first, fall back to CPU if it fails
        try:
            if device.type == 'cuda':
                logger.info("Moving model to CUDA...")
                model = model.to(device)
                # Test CUDA with a small forward pass
                test_input = torch.zeros(1, 3, 224, 224, device=device)
                with torch.no_grad():
                    _ = model(test_input)
                logger.info("CUDA test successful")
            else:
                logger.info("Using CPU for model")
                model = model.to(device)
        except Exception as e:
            logger.warning(f"Failed to use CUDA: {str(e)}. Falling back to CPU.")
            device = torch.device("cpu")
            model = model.to(device)
            
        model.eval()
        
        # Load model weights
        model_path = os.getenv('MODEL_PATH', 'model.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        logger.info(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)  # Load to current device
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            logger.info("Loading from checkpoint dictionary format...")
            state_dict = checkpoint['model_state_dict']
            # Log checkpoint info
            logger.info(f"Checkpoint info: epoch={checkpoint.get('epoch', 'N/A')}, "
                       f"best_acc={checkpoint.get('best_acc', 'N/A')}")
        else:
            logger.info("Loading direct state dict...")
            state_dict = checkpoint
            
        # Load state dict with strict=False to handle missing keys
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Model loaded successfully on {next(model.parameters()).device}")
        
    except Exception as e:
        logger.error(f"Failed to load model during startup: {str(e)}")
        # Don't raise here, let the app start but handle missing model in endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint that also verifies model status"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device)
    }

# Load model (lazy loading on first request)
model = None
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

class GeometryModule(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.anchors = nn.Parameter(torch.randn(10, dim))
        nn.init.xavier_uniform_(self.anchors, gain=0.1)
        
        self.diversity_weight = nn.Parameter(torch.tensor(0.001))
        self.center_weight = nn.Parameter(torch.tensor(0.001))
        
        # Anchor refiner WITH bias in first layer
        self.anchor_refiner = nn.Sequential(
            nn.Linear(dim, dim // 2),  # Add bias back
            nn.ReLU(inplace=True),
            nn.Linear(dim // 2, dim),
            nn.Tanh()
        )
        
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # Feature enhancer WITH bias in first linear layer
        self.feature_enhancer = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),  # Add bias back
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim),
            nn.Dropout(0.1)
        )
        
        # Initialize projections
        for module in [self.query_proj, self.key_proj, self.value_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight, gain=0.1)
            nn.init.constant_(module.bias, 0)
            
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = features.size(0)
        
        # Normalize inputs for stability
        features_norm = F.normalize(features, p=2, dim=1, eps=1e-8)
        
        # Refine anchors adaptively
        anchor_refinements = self.anchor_refiner(self.anchors)
        refined_anchors = self.anchors + 0.1 * anchor_refinements  # Small refinement
        anchors_norm = F.normalize(refined_anchors, p=2, dim=1, eps=1e-8)
        
        # Attention mechanism
        q = self.query_proj(features_norm).unsqueeze(1)  # [B, 1, D]
        k = self.key_proj(anchors_norm).unsqueeze(0)     # [1, A, D]
        v = self.value_proj(anchors_norm).unsqueeze(0)   # [1, A, D]
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) / (features.size(-1) ** 0.5)  # [B, 1, A]
        attn_weights = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        attended = attn_weights @ v  # [B, 1, D]
        attended = self.out_proj(attended.squeeze(1))  # [B, D]
        
        # Enhance features
        enhanced_features = features_norm + 0.1 * attended
        enhanced_features = self.feature_enhancer(enhanced_features)
        
        # Add residual connection
        final_features = features_norm + 0.2 * enhanced_features
        
        # Compute geometry losses
        # 1. Diversity loss
        anchor_distances = torch.pdist(anchors_norm, p=2)
        if anchor_distances.numel() > 0:
            diversity_loss = torch.exp(-anchor_distances.mean().clamp(min=1e-6))
        else:
            diversity_loss = torch.tensor(0.0, device=features.device)
            
        # 2. Center loss
        feature_anchor_distances = torch.cdist(features_norm, anchors_norm)
        min_distances = feature_anchor_distances.min(dim=1)[0]
        center_loss = min_distances.mean()
        
        # Combined geometry loss
        geo_loss = (torch.abs(self.diversity_weight) * diversity_loss + 
                   torch.abs(self.center_weight) * center_loss)
        geo_loss = torch.clamp(geo_loss, min=0, max=0.01)
        
        return final_features, geo_loss

class ReliabilityModule(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        # Reliability net WITH bias in all layers
        self.reliability_net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),  # Add bias back
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(dim // 2, dim // 4),  # Add bias back
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(dim // 4, 64),  # Add bias back
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Uncertainty layers WITH bias
        self.uncertainty_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),  # Add bias back
                nn.GELU(),
                nn.Dropout(0.2)
            ) for _ in range(3)
        ])
        
        self.confidence_estimator = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Feature refiner WITH bias in first linear layer
        self.feature_refiner = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),  # Add bias back
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim),
            nn.Dropout(0.1)
        )
        
        # Initialize all networks
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
    def forward(self, features: torch.Tensor, predictions: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get reliability score
        reliability_score = self.reliability_net(features)  # [B, 1]
        
        # Estimate uncertainty through MC Dropout
        uncertainties = []
        for layer in self.uncertainty_layers:
            uncertain_features = layer(features)
            uncertainties.append(torch.var(uncertain_features, dim=1, keepdim=True))
        uncertainty = torch.stack(uncertainties).mean(dim=0)  # [B, 1]
        
        # Get confidence from predictions if available
        if predictions is not None:
            pred_probs = F.softmax(predictions, dim=1)
            confidence = self.confidence_estimator(pred_probs)  # [B, 1]
            
            # Combine reliability factors
            combined_reliability = (
                0.4 * reliability_score +
                0.3 * confidence +
                0.3 * (1.0 - uncertainty)  # Lower uncertainty = higher reliability
            )
        else:
            # Without predictions, use simpler reliability
            combined_reliability = (
                0.6 * reliability_score +
                0.4 * (1.0 - uncertainty)
            )
        
        # Clamp reliability to reasonable range
        final_reliability = torch.clamp(combined_reliability, min=0.1, max=0.95)
        
        # Refine features based on reliability
        reliability_weights = final_reliability.expand(-1, features.size(1))
        refined_features = self.feature_refiner(features)
        
        # Weighted combination
        enhanced_features = (
            reliability_weights * refined_features +
            (1.0 - reliability_weights) * features
        )
        
        # Add residual connection
        final_features = features + 0.1 * enhanced_features
        
        return final_features, final_reliability.squeeze(1)  # [B, D], [B]

def create_vit_model():
    """Create a Vision Transformer model matching the checkpoint architecture"""
    from torch import nn
    
    class EmotionViT(nn.Module):
        """Full GReFEL architecture with multi-scale features and geometry-aware learning"""
        def __init__(
            self,
            num_classes: int = 7,
            feature_dim: int = 768,
            num_anchors: int = 10,
            drop_rate: float = 0.15
        ):
            super().__init__()
            
            # Use standard ViT-Base without features_only to avoid spatial dimensions
            self.backbone = timm.create_model(
                'vit_base_patch16_224', 
                pretrained=True, 
                num_classes=0  # Remove classification head
            )
            
            self.backbone_dim = 768
            self.feature_dim = feature_dim
            
            # Since we can't easily get multi-scale from standard ViT, 
            # we'll create multiple projections from the final features
            self.multi_scale_projections = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(self.backbone_dim),
                    nn.Linear(self.backbone_dim, feature_dim),
                    nn.GELU(),
                    nn.Dropout(drop_rate * 0.5),
                    nn.Linear(feature_dim, feature_dim),
                    nn.GELU(),
                    nn.Dropout(drop_rate * 0.5)
                ) for _ in range(3)
            ])
            
            # Initialize projections
            for proj in self.multi_scale_projections:
                for module in proj.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight, gain=0.1)  # Slightly larger gain
                        if module.bias is not None:
                            nn.init.constant_(module.bias, 0)
            
            # Geometry-aware modules for each "scale"
            self.geometry_modules = nn.ModuleList([
                GeometryModule(feature_dim) for _ in range(3)
            ])
            
            # Feature fusion
            self.fusion = nn.Sequential(
                nn.LayerNorm(feature_dim * 3),
                nn.Linear(feature_dim * 3, feature_dim),
                nn.GELU(),
                nn.Dropout(drop_rate)
            )
            
            # Initialize fusion
            for module in self.fusion.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.1)  # Slightly larger gain
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            
            # Reliability balancing module
            self.reliability_module = ReliabilityModule(feature_dim)
            
            # Classification head
            self.classifier = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, feature_dim // 2),
                nn.GELU(),
                nn.Dropout(drop_rate),
                nn.Linear(feature_dim // 2, num_classes)
            )
            
            # Initialize classifier with proper scaling
            for i, module in enumerate(self.classifier.modules()):
                if isinstance(module, nn.Linear):
                    if i == len(list(self.classifier.modules())) - 1:  # Last layer
                        nn.init.xavier_uniform_(module.weight, gain=0.1)
                        if module.bias is not None:
                            nn.init.constant_(module.bias, 0)
                    else:
                        nn.init.xavier_uniform_(module.weight, gain=0.1)
                        if module.bias is not None:
                            nn.init.constant_(module.bias, 0)
        
        def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
            # Extract features from ViT backbone
            backbone_features = self.backbone(x)  # [B, 768]
            
            # Create "multi-scale" features by applying different projections
            processed_features = []
            total_geo_loss = 0
            
            for i, projection in enumerate(self.multi_scale_projections):
                # Apply different projections to simulate multi-scale
                projected = projection(backbone_features)
                
                # Apply geometry-aware module
                geo_features, geo_loss = self.geometry_modules[i](projected)
                
                processed_features.append(geo_features)
                total_geo_loss += geo_loss
            
            # Fuse multi-scale features
            fused_features = torch.cat(processed_features, dim=1)  # [B, feature_dim * 3]
            fused_features = self.fusion(fused_features)  # [B, feature_dim]
            
            # Reliability estimation
            reliability, reliability_score = self.reliability_module(fused_features, self.classifier(fused_features))
            
            # Classification
            logits = self.classifier(fused_features)
            
            return {
                'logits': logits,
                'reliability': reliability,
                'reliability_score': reliability_score,
                'geo_loss': total_geo_loss / 3,  # Average geometry loss
                'features': fused_features
            }
    
    return EmotionViT()

async def preprocess_image(file: UploadFile) -> torch.Tensor:
    """
    Preprocess image for model input.
    Args:
        file: UploadFile object containing the image
    Returns:
        torch.Tensor: Preprocessed image tensor on the same device as the model
    """
    try:
        logger.info("Opening image...")
        # Read file contents
        contents = await file.read()
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(contents))
        
        # Convert grayscale to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transformations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Transform and add batch dimension
        image_tensor = transform(image).unsqueeze(0)
        
        # Always start on CPU
        image_tensor = image_tensor.cpu()
        
        # Move to model's device if available
        if model is not None:
            try:
                model_device = next(model.parameters()).device
                logger.info(f"Moving input tensor to model device: {model_device}")
                image_tensor = image_tensor.to(model_device)
            except Exception as e:
                logger.warning(f"Failed to move tensor to model device: {str(e)}. Using CPU.")
                image_tensor = image_tensor.cpu()
        else:
            image_tensor = image_tensor.to(device)
            
        logger.info(f"Image preprocessed successfully, shape: {image_tensor.shape}, device: {image_tensor.device}")
        return image_tensor
        
    except Exception as e:
        logger.error(f"Failed to process image: {str(e)}")
        raise

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and preprocess the image
        image = await preprocess_image(file)
        
        # Ensure model is loaded
        if model is None:
            raise RuntimeError("Model not loaded")
            
        # Ensure model and input are on same device
        if image.device != next(model.parameters()).device:
            logger.info(f"Moving input tensor from {image.device} to {next(model.parameters()).device}")
            image = image.to(next(model.parameters()).device)
            
        logger.info("Running model forward pass...")
        # Get model predictions
        with torch.no_grad():
            output = model(image)
            logits = output['logits']  # Extract logits from output dict
            
            # Get probabilities for all emotions
            probs = F.softmax(logits, dim=1)
            probabilities = probs[0].cpu().numpy()  # Get first batch item
            
            # Get reliability score if available
            reliability = output.get('reliability_score', None)
            if reliability is not None:
                reliability = reliability.item()
        
        # Create response in the format expected by the frontend
        emotion_dict = {}
        for emotion, prob in zip(EMOTIONS, probabilities):
            emotion_dict[emotion.lower()] = float(prob)
        
        # Ensure all expected emotions are present with at least 0.0
        for emotion in EMOTIONS:
            if emotion.lower() not in emotion_dict:
                emotion_dict[emotion.lower()] = 0.0
        
        # Add reliability if available
        if reliability is not None:
            emotion_dict['reliability'] = float(reliability)
        
        logger.info(f"Predicted emotions: {emotion_dict}")
        return emotion_dict
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 