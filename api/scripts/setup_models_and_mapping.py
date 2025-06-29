#!/usr/bin/env python3
"""
Setup script for models and identity mapping
Ensures proper backup and availability of models and mapping files for app launch
"""

import os
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_directories():
    """Create necessary directories"""
    directories = [
        Path("/app/models"),
        Path("/app/models/backups"),
        Path("/app/data"),
        Path("/app/logs"),
        Path("/app/cache")
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")

def copy_identity_mapping():
    """Copy identity mapping to accessible locations"""
    # Source locations to check
    source_paths = [
        Path("./data/identity_mapping.json"),
        Path("/app/data/identity_mapping.json"),
        Path("./identity_mapping.json"),
        Path("/e:/Repos/facial-reg/data/identity_mapping.json")  # Local development
    ]
    
    # Destination locations
    dest_paths = [
        Path("/app/models/identity_mapping.json"),
        Path("/app/data/identity_mapping.json")
    ]
    
    mapping_copied = False
    
    # Find and copy the mapping file
    for source_path in source_paths:
        if source_path.exists():
            logger.info(f"Found identity mapping at: {source_path}")
            
            # Verify it's a valid mapping file
            try:
                with open(source_path, 'r') as f:
                    mapping_data = json.load(f)
                
                if "mapping" in mapping_data and "total_identities" in mapping_data:
                    logger.info(f"Valid identity mapping with {mapping_data['total_identities']} identities")
                    
                    # Copy to all destination locations
                    for dest_path in dest_paths:
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source_path, dest_path)
                        logger.info(f"Copied identity mapping to: {dest_path}")
                    
                    mapping_copied = True
                    break
                else:
                    logger.warning(f"Invalid mapping file format: {source_path}")
                    
            except Exception as e:
                logger.error(f"Error reading mapping file {source_path}: {e}")
                continue
    
    if not mapping_copied:
        logger.warning("No valid identity mapping found, creating default")
        create_default_identity_mapping()

def create_default_identity_mapping():
    """Create a default identity mapping if none exists"""
    default_mapping = {
        "version": "1.0.0",
        "mapping": {str(i): i for i in range(300)},  # 0-299
        "hash": "default",
        "total_identities": 300,
        "partition_stats": {
            "server": {"total_identities": 100},
            "client1": {"total_identities": 100}, 
            "client2": {"total_identities": 100}
        }
    }
    
    dest_paths = [
        Path("/app/models/identity_mapping.json"),
        Path("/app/data/identity_mapping.json")
    ]
    
    for dest_path in dest_paths:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, 'w') as f:
            json.dump(default_mapping, f, indent=2)
        logger.info(f"Created default identity mapping at: {dest_path}")

def backup_models():
    """Create backups of all model files"""
    models_dir = Path("/app/models")
    backups_dir = models_dir / "backups"
    backups_dir.mkdir(exist_ok=True)
    
    # Model files to backup
    model_files = [
        "server_model.pth",
        "client1_model.pth",
        "client2_model.pth", 
        "best_pretrained_model.pth"
    ]
    
    # Also check for models in the project directory (development)
    local_models_dir = Path("./models")
    if local_models_dir.exists():
        logger.info(f"Found local models directory: {local_models_dir}")
        
        # Copy models from local to container location
        for model_file in model_files:
            local_path = local_models_dir / model_file
            container_path = models_dir / model_file
            
            if local_path.exists() and not container_path.exists():
                shutil.copy2(local_path, container_path)
                logger.info(f"Copied model from local: {local_path} -> {container_path}")
    
    # Create backups
    backup_count = 0
    for model_file in model_files:
        source_path = models_dir / model_file
        backup_path = backups_dir / f"{model_file.replace('.pth', '_backup.pth')}"
        
        if source_path.exists():
            if not backup_path.exists():
                shutil.copy2(source_path, backup_path)
                logger.info(f"Created backup: {backup_path}")
                backup_count += 1
            else:
                logger.info(f"Backup already exists: {backup_path}")
            
            # Verify model file integrity
            try:
                import torch
                state_dict = torch.load(source_path, map_location='cpu')
                logger.info(f"Verified model integrity: {source_path}")
            except Exception as e:
                logger.error(f"Model file corrupted: {source_path} - {e}")
    
    logger.info(f"Created {backup_count} new model backups")

def setup_environment_variables():
    """Set up environment variables for the application"""
    env_vars = {
        "PYTHONPATH": "/app",
        "MODEL_PATH": "/app/models",
        "DATA_PATH": "/app/data",
        "MAPPING_FILE": "/app/models/identity_mapping.json"
    }
    
    for key, value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = value
            logger.info(f"Set environment variable: {key}={value}")

def verify_setup():
    """Verify that everything is set up correctly"""
    checks = []
    
    # Check identity mapping
    mapping_paths = [
        Path("/app/models/identity_mapping.json"),
        Path("/app/data/identity_mapping.json")
    ]
    
    mapping_found = False
    for path in mapping_paths:
        if path.exists():
            try:
                with open(path, 'r') as f:
                    mapping_data = json.load(f)
                logger.info(f"✓ Identity mapping found at {path} with {mapping_data.get('total_identities', 'unknown')} identities")
                mapping_found = True
                break
            except Exception as e:
                logger.error(f"✗ Invalid mapping file at {path}: {e}")
    
    checks.append(("Identity Mapping", mapping_found))
    
    # Check models
    models_dir = Path("/app/models")
    model_files = ["server_model.pth", "client1_model.pth", "client2_model.pth", "best_pretrained_model.pth"]
    models_found = 0
    
    for model_file in model_files:
        model_path = models_dir / model_file
        if model_path.exists():
            models_found += 1
            logger.info(f"✓ Model found: {model_path}")
    
    checks.append(("Models", models_found > 0))
    
    # Check backups
    backups_dir = models_dir / "backups"
    backup_files = list(backups_dir.glob("*_backup.pth")) if backups_dir.exists() else []
    checks.append(("Model Backups", len(backup_files) > 0))
    
    if len(backup_files) > 0:
        logger.info(f"✓ Found {len(backup_files)} backup files")
    
    # Summary
    passed_checks = sum(1 for name, passed in checks if passed)
    total_checks = len(checks)
    
    logger.info(f"\n=== Setup Verification: {passed_checks}/{total_checks} checks passed ===")
    for name, passed in checks:
        status = "✓" if passed else "✗"
        logger.info(f"{status} {name}")
    
    return passed_checks == total_checks

def main():
    """Main setup function"""
    logger.info("=== Starting Model and Mapping Setup ===")
    
    try:
        # Step 1: Ensure directories exist
        logger.info("Step 1: Creating directories...")
        ensure_directories()
        
        # Step 2: Copy identity mapping
        logger.info("Step 2: Setting up identity mapping...")
        copy_identity_mapping()
        
        # Step 3: Backup models
        logger.info("Step 3: Backing up models...")
        backup_models()
        
        # Step 4: Set environment variables
        logger.info("Step 4: Setting environment variables...")
        setup_environment_variables()
        
        # Step 5: Verify setup
        logger.info("Step 5: Verifying setup...")
        setup_success = verify_setup()
        
        if setup_success:
            logger.info("🎉 Setup completed successfully!")
            return True
        else:
            logger.error("❌ Setup completed with warnings")
            return False
            
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 