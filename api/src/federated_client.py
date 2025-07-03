"""
Federated Learning Client for Biometric API
Handles communication with coordinator and local model training
"""

import os
import logging
import json
import time
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import random
import hashlib
import io
import base64

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import requests
from fastapi import BackgroundTasks
import httpx
from PIL import Image
from torchvision import transforms

from .models.privacy_biometric_model import PrivacyBiometricModel
from .privacy.privacy_engine import PrivacyEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Suppress Opacus validator info messages
logging.getLogger('opacus.validators.module_validator').setLevel(logging.WARNING)
logging.getLogger('opacus.validators.batch_norm').setLevel(logging.WARNING)

class FederatedClient:
    """Client for federated learning integration"""

    def __init__(self):
        """Initialize federated learning client"""
        self.client_id = os.getenv("CLIENT_ID", "client1")
        self.coordinator_url = os.getenv("COORDINATOR_URL", "http://fl-coordinator:8000")
        self.device = torch.device('cpu')  # Force CPU usage
        self.registered = False
        self.active = False
        self.current_round = 0
        self.federated_config = None
        self.privacy_engine = None
        self.model = None
        self.dataset_size = 0
        self.mapping_service = None

    async def initialize(self):
        """Initialize federated client"""
        try:
            # Initialize mapping service first
            from .services.mapping_service import MappingService
            self.mapping_service = MappingService()
            
            # Validate mapping availability
            if not await self.validate_mapping():
                logger.error("Failed to validate mapping with coordinator")
                return False
            
            # Load configuration from coordinator
            config_response = requests.get(f"{self.coordinator_url}/config", timeout=10)
            if config_response.status_code == 200:
                self.federated_config = config_response.json()
            else:
                self.federated_config = {
                    'max_epsilon': 100.0,
                    'delta': 1e-5,
                    'noise_multiplier': 0.2,
                    'max_grad_norm': 5.0
                }
            
            # Initialize privacy engine
            self.privacy_engine = PrivacyEngine()
            await self.privacy_engine.initialize()
            
            # Initialize model
            self.model = await self.load_or_download_model()
            if not self.model:
                logger.error("Failed to initialize model")
                return False
            
            # Register with coordinator
            if await self.register_with_coordinator():
                self.registered = True
                self.active = True
            else:
                logger.warning(f"Failed to register client {self.client_id} with coordinator")
                return False
            
            # Get initial dataset size
            self.dataset_size = await self.get_dataset_size()
            
            # Start background federated loop
            asyncio.create_task(self.federated_learning_loop())
            
            logger.info(f"Federated client {self.client_id} initialized with {self.dataset_size} samples")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing federated client: {e}")
            return False

    async def validate_mapping(self) -> bool:
        """Validate mapping with coordinator"""
        try:
            # Initialize mapping from centralized source
            self.mapping_service.initialize_mapping()
            mapping = self.mapping_service.get_mapping()
            
            # Verify mapping format
            if not isinstance(mapping, dict):
                logger.error(f"Invalid mapping format: {type(mapping)}")
                return False
            
            # Verify mapping has entries
            if not mapping:
                logger.error("Empty mapping received from coordinator")
                return False
            
            logger.info(f"Successfully validated mapping with {len(mapping)} entries")
            return True
            
        except Exception as e:
            logger.error(f"Error validating mapping: {e}")
            return False

    async def load_or_download_model(self):
        """Load or download the model"""
        try:
            # Get current mapping
            self.mapping_service.initialize_mapping()
            mapping = self.mapping_service.get_mapping()
            if not mapping:
                logger.error("Cannot load model without valid mapping")
                return None
            
            num_identities = len(mapping)
            logger.info(f"Using {num_identities} identities from global mapping")
            
            model_path = Path(f"/app/models/best_{self.client_id}_pretrained_model.pth")
            
            if model_path.exists():
                # Load existing model
                state_dict = torch.load(model_path, map_location=self.device)
                
                # Handle metadata wrapper
                if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                    stored_num_identities = state_dict.get('num_identities', num_identities)
                    embedding_dim = state_dict.get('feature_dim', 512)
                    state_dict = state_dict['state_dict']
                    
                    # Verify mapping consistency
                    if stored_num_identities != num_identities:
                        logger.warning(f"Stored model has different number of identities ({stored_num_identities}) than current mapping ({num_identities})")
                        # We'll create a new model with correct size
                        raise ValueError("Model size mismatch with current mapping")
                else:
                    embedding_dim = 512
                
                # Create model with correct parameters
                model = PrivacyBiometricModel(
                    num_identities=num_identities,
                    embedding_dim=embedding_dim,
                    privacy_enabled=True
                ).to(self.device)
                
                # Load weights
                try:
                    model.load_state_dict(state_dict, strict=False)
                    logger.info(f"Successfully loaded model with {num_identities} identities")
                except Exception as e:
                    logger.error(f"Error loading state dict: {e}")
                    raise ValueError("Failed to load model weights")
            else:
                # Create new model with mapping-based parameters
                model = PrivacyBiometricModel(
                    num_identities=num_identities,
                    privacy_enabled=True
                ).to(self.device)
                logger.info(f"Created new model with {num_identities} identities")
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading/downloading model: {e}")
            return None

    async def register_with_coordinator(self):
        """Register client with coordinator"""
        try:
            response = requests.post(
                f"{self.coordinator_url}/clients/register",
                json={
                    "client_id": self.client_id,
                    "dataset_size": self.dataset_size
                }
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to register with coordinator: {e}")
            return False

    async def get_dataset_size(self):
        """Get size of local dataset"""
        try:
            # Allow override through env-var so users can mount data wherever they like
            data_root = os.getenv("CLIENT_DATA_DIR", f"/app/data/partitioned/{self.client_id}")
            data_dir = Path(data_root)
            if not data_dir.exists():
                logger.warning(f"[{self.client_id}] data directory {data_dir} not found")
                return 0
            
            total_samples = 0
            for class_dir in data_dir.iterdir():
                if class_dir.is_dir():
                    total_samples += len(list(class_dir.glob('*.jpg')))
            
            logger.info(f"[{self.client_id}] counted {total_samples} local samples in {data_dir}")
            return total_samples
        except Exception as e:
            logger.error(f"Error getting dataset size: {e}")
            return 0

    async def federated_learning_loop(self):
        """Main federated learning loop"""
        while self.active:
            try:
                # Get current round from coordinator
                response = requests.get(f"{self.coordinator_url}/rounds/current")
                if response.status_code != 200:
                    await asyncio.sleep(5)
                    continue
                
                round_info = response.json()
                round_number = round_info.get("round_id", 0)
                
                if round_number > self.current_round:
                    logger.info(f"Starting participation in round {round_number}")
                    
                    # Perform local training
                    start_time = time.time()
                    await self.train_local()
                    training_time = time.time() - start_time
                    
                    # Submit model update
                    if await self.submit_model_update(round_number):
                        self.current_round = round_number
                        logger.info(f"Completed federated round {round_number}")
                
                await asyncio.sleep(5)
            
            except Exception as e:
                logger.error(f"Error in federated learning loop: {e}")
                await asyncio.sleep(5)

    async def train_local(self):
        """Perform local training"""
        try:
            # Ensure mapping is up to date
            self.mapping_service.initialize_mapping()
            mapping = self.mapping_service.get_mapping()
            if not mapping:
                logger.error("Cannot train without valid mapping")
                return
            
            # Get mapping metadata
            mapping_size = len(mapping)
            mapping_hash = hashlib.sha256(json.dumps(sorted(mapping.items())).encode()).hexdigest()[:8]
            mapping_version = datetime.now().isoformat()
            
            logger.info(f"Using mapping version {mapping_version} (hash: {mapping_hash}) with {mapping_size} classes")
            
            # Verify model classifier size matches mapping
            if self.model.num_identities != mapping_size:
                logger.warning(f"Model classifier size ({self.model.num_identities}) doesn't match mapping size ({mapping_size})")
                # Reinitialize model with correct size
                self.model = await self.load_or_download_model()
                if not self.model:
                    logger.error("Failed to reinitialize model with correct mapping size")
                    return
            
            # Training logic would go here
            # For now, just simulate training
            await asyncio.sleep(0.1)
            
            # Add mapping metadata to model
            self.model.mapping_size = mapping_size
            self.model.mapping_hash = mapping_hash
            self.model.mapping_version = mapping_version
            
        except Exception as e:
            logger.error(f"Error during local training: {e}")

    def _serialize_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> str:
        """Serialize a state_dict to a base64-encoded string suitable for JSON payloads."""
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')

    async def submit_model_update(self, round_id: int) -> bool:
        """Submit model update to coordinator"""
        try:
            # Get current mapping metadata
            self.mapping_service.initialize_mapping()
            mapping = self.mapping_service.get_mapping()
            if not mapping:
                logger.error("Cannot submit update without valid mapping")
                return False
            
            mapping_size = len(mapping)
            mapping_hash = hashlib.sha256(json.dumps(sorted(mapping.items())).encode()).hexdigest()[:8]
            mapping_version = datetime.now().isoformat()
            
            # Prepare model update
            update = {
                # Encode weights so the payload is JSON-serialisable
                "weights": self._serialize_state_dict(self.model.state_dict()),
                "dataset_size": self.dataset_size,
                "mapping_size": mapping_size,
                "mapping_hash": mapping_hash,
                "mapping_version": mapping_version,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Submit update to coordinator
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.coordinator_url}/models/update/{self.client_id}/{round_id}/upload",
                    json=update,
                    headers={
                        "X-Client-ID": self.client_id,
                        "X-Mapping-Version": mapping_version,
                        "X-Mapping-Hash": mapping_hash
                    }
                )
                
                if response.status_code == 200:
                    logger.info(f"Successfully submitted model update for round {round_id}")
                    return True
                else:
                    logger.error(f"Failed to submit model update: {response.status_code}")
                    return False
                
        except Exception as e:
            logger.error(f"Error submitting model update: {e}")
            return False

    async def get_current_round(self) -> Optional[Dict]:
        """Get information about the current federated round"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.coordinator_url}/rounds/current")
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Failed to get current round: {response.status_code}")
                    return None
        
        except Exception as e:
            logger.error(f"Error getting current round: {e}")
            return None

    async def train_on_local_data(self, round_id: int) -> Tuple[Dict, Dict]:
        """Train model on local data for one federated round"""
        # In a real implementation, this would train on the actual dataset
        # Here, we simulate training by making slight modifications to the model
        
        training_start_time = time.time()
        
        logger.info(f"Starting local training for round {round_id}")
        
        # Simulate training metrics
        training_metrics = {
            "loss": random.uniform(0.1, 0.5),
            "accuracy": random.uniform(0.7, 0.95),
            "training_time": random.uniform(30, 120)
        }
        
        # Simulate privacy metrics
        privacy_metrics = {
            "epsilon_used": random.uniform(0.5, 2.0),
            "delta": self.federated_config.get('privacy', {}).get('delta', 1e-5),
            "noise_multiplier": self.federated_config.get('privacy', {}).get('noise_multiplier', 0.2)
        }
        
        # Simulate model updates (add small random changes)
        with torch.no_grad():
            for param in self.model.parameters():
                # Add small random changes to simulate training
                noise = torch.randn_like(param) * 0.01
                param.add_(noise)
        
        training_time = time.time() - training_start_time
        logger.info(f"Completed local training in {training_time:.2f} seconds")
        
        return training_metrics, privacy_metrics

    async def get_status(self) -> Dict:
        """Get status of federated client"""
        privacy_budget = await self.privacy_engine.get_remaining_budget()
        
        return {
            "client_id": self.client_id,
            "active": self.active,
            "registered": self.registered,
            "current_round": self.current_round,
            "privacy_budget_remaining": privacy_budget,
            "dataset_size": self.dataset_size,
            "coordinator_url": self.coordinator_url
        } 