import os
import logging
from pathlib import Path
import json
import shutil
from collections import defaultdict
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_continuous_mapping(data_dir: Path):
    """Create a continuous mapping for class IDs"""
    # Collect all class IDs
    class_ids = set()
    for partition in ['server', 'client1', 'client2']:
        partition_dir = data_dir / partition
        if not partition_dir.exists():
            continue
        
        for class_dir in partition_dir.iterdir():
            if class_dir.is_dir():
                class_ids.add(int(class_dir.name))
    
    # Create continuous mapping
    sorted_ids = sorted(class_ids)
    id_mapping = {old_id: idx for idx, old_id in enumerate(sorted_ids)}
    
    # Save mapping
    mapping_file = data_dir / 'class_mapping.json'
    with open(mapping_file, 'w') as f:
        json.dump({
            'old_to_new': {str(k): str(v) for k, v in id_mapping.items()},
            'new_to_old': {str(v): str(k) for k, v in id_mapping.items()}
        }, f, indent=2)
    
    return id_mapping

def rename_class_directories(data_dir: Path, id_mapping: dict):
    """Rename class directories to use continuous IDs"""
    # Create backup
    backup_dir = data_dir.parent / (data_dir.name + '_backup')
    if not backup_dir.exists():
        shutil.copytree(data_dir, backup_dir)
        logger.info(f"Created backup at: {backup_dir}")
    
    # Track statistics
    stats = defaultdict(int)
    
    # Rename directories
    for partition in ['server', 'client1', 'client2']:
        partition_dir = data_dir / partition
        if not partition_dir.exists():
            continue
        
        logger.info(f"\nProcessing {partition}...")
        for class_dir in partition_dir.iterdir():
            if not class_dir.is_dir():
                continue
            
            old_id = int(class_dir.name)
            new_id = id_mapping[old_id]
            
            # Create new directory name with leading zeros
            new_name = f"{new_id:03d}"  # 3-digit format
            new_path = class_dir.parent / new_name
            
            # Rename directory
            class_dir.rename(new_path)
            stats['renamed'] += 1
            logger.info(f"Renamed {class_dir.name} -> {new_name}")
    
    logger.info(f"\nRenamed {stats['renamed']} directories")
    return stats

def update_model_mapping(model_path: Path, id_mapping: dict):
    """Update model's class mapping"""
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return
    
    try:
        # Load model state
        state_dict = torch.load(model_path)
        
        # Update final layer weights if needed
        if 'identity_head.4.weight' in state_dict:
            old_weights = state_dict['identity_head.4.weight']
            new_weights = torch.zeros_like(old_weights)
            
            for old_id, new_id in id_mapping.items():
                new_weights[new_id] = old_weights[old_id]
            
            state_dict['identity_head.4.weight'] = new_weights
            
            if 'identity_head.4.bias' in state_dict:
                old_bias = state_dict['identity_head.4.bias']
                new_bias = torch.zeros_like(old_bias)
                for old_id, new_id in id_mapping.items():
                    new_bias[new_id] = old_bias[old_id]
                state_dict['identity_head.4.bias'] = new_bias
        
        # Save updated model
        new_model_path = model_path.parent / f"remapped_{model_path.name}"
        torch.save(state_dict, new_model_path)
        logger.info(f"Saved remapped model to: {new_model_path}")
        
    except Exception as e:
        logger.error(f"Error updating model mapping: {e}")

def main():
    # Get project root directory
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data' / 'partitioned'
    
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return
    
    logger.info(f"Creating continuous class mapping for: {data_dir}")
    
    # Create new mapping
    id_mapping = create_continuous_mapping(data_dir)
    logger.info(f"Created mapping for {len(id_mapping)} classes")
    
    # Rename directories
    stats = rename_class_directories(data_dir, id_mapping)
    
    # Update model mappings
    model_files = [
        'best_pretrained_model.pth',
        'best_client1_pretrained_model.pth',
        'best_client2_pretrained_model.pth'
    ]
    
    for model_file in model_files:
        model_path = project_root / 'models' / model_file
        update_model_mapping(model_path, id_mapping)

if __name__ == "__main__":
    main() 