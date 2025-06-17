import torch
import logging
import pickle
import argparse
import io
import torch.storage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "argparse" and name == "Namespace":
            return argparse.Namespace
        return super().find_class(module, name)
    
    def persistent_load(self, persistent_id):
        return None  # Ignore persistent storage

def custom_load(f):
    return CustomUnpickler(f).load()

def inspect_checkpoint(path):
    logger.info(f"Attempting to load checkpoint from: {path}")
    try:
        # Try loading the checkpoint with torch's load
        logger.info("Loading checkpoint with torch.load...")
        checkpoint = torch.load(path, map_location='cpu', pickle_module=pickle)
        logger.info("Checkpoint loaded successfully")
        
        # Inspect the checkpoint type
        logger.info(f"Checkpoint type: {type(checkpoint)}")
        
        # If it's a dictionary, show all contents
        if isinstance(checkpoint, dict):
            logger.info("\nCheckpoint contents:")
            for key, value in checkpoint.items():
                logger.info(f"\nKey: {key}")
                logger.info(f"Value type: {type(value)}")
                if isinstance(value, torch.nn.Module):
                    logger.info(f"Model architecture: {value}")
                    total_params = sum(p.numel() for p in value.parameters())
                    logger.info(f"Total parameters: {total_params:,}")
                elif isinstance(value, dict):
                    logger.info(f"Dictionary keys: {list(value.keys())}")
                else:
                    try:
                        logger.info(f"Value: {value}")
                    except:
                        logger.info("Value not printable")
        else:
            logger.info("Checkpoint is not a dictionary")
            if isinstance(checkpoint, torch.nn.Module):
                logger.info(f"Direct model architecture: {checkpoint}")
                total_params = sum(p.numel() for p in checkpoint.parameters())
                logger.info(f"Total parameters: {total_params:,}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading checkpoint: {str(e)}")
        return False

if __name__ == "__main__":
    model_path = "model.pth"
    success = inspect_checkpoint(model_path)
    logger.info(f"Model loading {'successful' if success else 'failed'}") 