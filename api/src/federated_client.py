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
import threading

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
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from torchvision.datasets import ImageFolder

# Disable NNPACK for better compatibility in environments without hardware support.
if hasattr(torch.backends, 'nnpack'):
    torch.backends.nnpack.enabled = False  # type: ignore

from .models.privacy_biometric_model import PrivacyBiometricModel
from .privacy.privacy_engine import PrivacyEngine
from .services.mapping_service import MappingService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Suppress Opacus validator info messages
logging.getLogger('opacus.validators.module_validator').setLevel(logging.WARNING)
logging.getLogger('opacus.validators.batch_norm').setLevel(logging.WARNING)

class ImageDataset(Dataset):
    """Custom dataset for loading client images for federated training."""

    def __init__(self, data_dir: str, mapping: Dict[str, int], transform=None):
        self.data_dir = Path(data_dir)
        self.mapping = mapping
        self.transform = transform
        self.image_paths = []
        self.labels = []

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        logger.info(f"Loading data from {self.data_dir}. Mapping has {len(mapping)} entries.")
        for user_dir in self.data_dir.iterdir():
            if user_dir.is_dir() and user_dir.name in self.mapping:
                class_idx = self.mapping[user_dir.name]
                for image_path in user_dir.glob('*.jpg'):
                    self.image_paths.append(image_path)
                    self.labels.append(class_idx)
        logger.info(f"Found {len(self.image_paths)} images for {len(set(self.labels))} users.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            label = self.labels[idx]

            if self.transform:
                image = self.transform(image)

            return image, label
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return a placeholder tensor if an image is corrupt
            return torch.zeros((3, 224, 224)), -1


class FederatedClient:
    """Client for federated learning integration"""

    def __init__(self):
        """Initialize federated learning client"""
        self.client_id = os.getenv("CLIENT_ID", "client1")
        self.server_url = os.getenv("SERVER_URL", "http://fl-coordinator:8080").rstrip('/')
        self.is_dp_enabled = os.getenv("ENABLE_DP", "true").lower() == "true"
        
        # Configure device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.registered = False
        self.active = False
        self.current_round = 0
        self.federated_config: Optional[Dict[str, Any]] = None
        self.privacy_engine: Optional[PrivacyEngine] = None
        self.model: Optional[PrivacyBiometricModel] = None
        self.dataset_size = 0
        self.mapping_service: Optional[MappingService] = None
        self.model_version = -1
        self.training_in_progress = False
        self.stop_event = threading.Event()
        self.client = httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(http2=False), timeout=60.0)
        self.data_loader: Optional[DataLoader] = None

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
            config_response = await self.client.get(f"{self.server_url}/config")
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
            
            # Setup DataLoader
            data_root = os.getenv("CLIENT_DATA_DIR", f"/app/data/partitioned/{self.client_id}")
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            try:
                # Use ImageFolder to automatically load data
                dataset = ImageFolder(root=data_root, transform=transform)
                if len(dataset) == 0:
                    logger.warning(f"No training data found for client {self.client_id} in {data_root}")
                
                # This is a temporary workaround. In a real scenario, you would
                # need to reconcile dataset.class_to_idx with the global mapping.
                logger.info(f"ImageFolder found {len(dataset.classes)} classes.")

                config = self.federated_config or {}
                batch_size = config.get('batch_size', 32)
                self.data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
                logger.info(f"DataLoader initialized with {len(dataset)} samples.")

            except FileNotFoundError as e:
                logger.error(f"Failed to create dataset: {e}")
                return False
            except Exception as e:
                logger.error(f"An unexpected error occurred during DataLoader setup: {e}", exc_info=True)
                return False

            # Register with coordinator
            if await self.register_with_coordinator():
                self.registered = True
                self.active = True
            else:
                logger.warning(f"Failed to register client {self.client_id} with coordinator")
                return False
            
            self.dataset_size = len(self.data_loader.dataset) if self.data_loader else 0  # type: ignore
            
            # Start background federated loop
            asyncio.create_task(self.federated_learning_loop())
            
            logger.info(f"Federated client {self.client_id} initialized with {self.dataset_size} samples")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing federated client: {e}")
            return False

    async def validate_mapping(self) -> bool:
        """Validate mapping with coordinator"""
        if not self.mapping_service:
            logger.error("Mapping service is not initialized.")
            return False
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
        if not self.mapping_service:
            logger.error("Mapping service is not initialized.")
            return None
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
            response = await self.client.post(
                f"{self.server_url}/clients/register",
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
                response = await self.client.get(f"{self.server_url}/rounds/current")
                if response.status_code != 200:
                    await asyncio.sleep(5)
                    continue
                
                round_info = response.json()
                round_number = round_info.get("round_id", 0)
                
                if round_number > self.current_round:
                    logger.info(f"Starting participation in round {round_number}")
                    
                    # Perform local training in a separate thread to avoid blocking
                    start_time = time.time()
                    await asyncio.to_thread(self.train_local)
                    training_time = time.time() - start_time
                    
                    # Submit model update
                    if await self.submit_model_update(round_number):
                        self.current_round = round_number
                        logger.info(f"Completed federated round {round_number}")
                
                await asyncio.sleep(5)
            
            except Exception as e:
                logger.error(f"Error in federated learning loop: {e}", exc_info=True)
                await asyncio.sleep(5)

    def train_local(self):
        """Perform local training for a specified number of epochs."""
        if not self.model:
            logger.error("Model not initialized, cannot train.")
            return
        if not self.data_loader:
            logger.warning("Data loader not available, skipping training round.")
            time.sleep(1) # Prevent busy-looping if no data
            return

        config = self.federated_config or {}
        epochs = config.get('epochs', 3)
        lr = config.get('learning_rate', 0.001)

        logger.info(f"[{self.client_id}] Starting local training for {epochs} epochs with LR={lr}.")
        
        self.model.to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        model_to_train = self.model
        data_loader_to_use = self.data_loader

        if self.is_dp_enabled and self.privacy_engine:
            try:
                model_to_train, optimizer, data_loader_to_use = self.privacy_engine.setup_differential_privacy(
                    model=self.model,
                    optimizer=optimizer,
                    data_loader=self.data_loader,
                    epochs=epochs
                )
                logger.info(f"[{self.client_id}] Attached DP privacy engine.")
            except Exception as e:
                logger.error(f"Failed to setup DP: {e}. Training proceeds without privacy guarantees.", exc_info=True)
        
        model_to_train.train()

        for epoch in range(epochs):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            num_batches = 0
            
            logger.info(f"[{self.client_id}] Starting Epoch {epoch+1}/{epochs}")
            
            for i, (images, labels) in enumerate(data_loader_to_use):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs, _ = model_to_train(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1

                if (i + 1) % 5 == 0:
                    logger.info(f"[{self.client_id}] Epoch {epoch+1}, Batch {i+1}/{len(data_loader_to_use)}, Loss: {loss.item():.4f}")

            epoch_time = time.time() - epoch_start_time
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            current_lr = optimizer.param_groups[0]['lr']
            
            log_msg = f"[{self.client_id}] Completed Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f} | LR: {current_lr} | Time: {epoch_time:.2f}s"
            
            if self.is_dp_enabled and self.privacy_engine and self.privacy_engine.dp_engine:
                privacy_spent = self.privacy_engine.compute_privacy_spent()
                epsilon = privacy_spent.get("epsilon", 0.0)
                if isinstance(epsilon, float):
                    log_msg += f" | Epsilon: {epsilon:.4f}"
            
            logger.info(log_msg)

        logger.info(f"[{self.client_id}] Finished local training.")

    def _serialize_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> str:
        """Serialize a state_dict to a base64-encoded string suitable for JSON payloads."""
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')

    async def submit_model_update(self, round_id: int) -> bool:
        """Submit model update to coordinator"""
        if not self.model:
            logger.error("Model is not initialized, cannot submit update.")
            return False
        if not self.mapping_service:
            logger.error("Mapping service is not initialized, cannot submit update.")
            return False
            
        try:
            # Get mapping info
            self.mapping_service.initialize_mapping()
            mapping = self.mapping_service.get_mapping()
            if not mapping:
                logger.error("Cannot submit update without valid mapping")
                return False
            
            mapping_size = len(mapping)
            mapping_hash = hashlib.sha256(json.dumps(sorted(mapping.items())).encode()).hexdigest()[:8]

            # Get latest state dict
            state_dict = self.model.state_dict()
            
            # Serialize state dict to base64
            serialized_state_dict = self._serialize_state_dict(state_dict)
            
            # Create payload
            payload = {
                "client_id": self.client_id,
                "round_id": round_id,
                "weights": serialized_state_dict,
                "dataset_size": self.dataset_size,
                "model_version": self.model_version + 1,
                "mapping_size": mapping_size,
                "mapping_hash": mapping_hash
            }
            
            # Submit to coordinator
            response = await self.client.post(
                f"{self.server_url}/models/update/{self.client_id}/{round_id}/upload",
                json=payload
            )
            
            if response.status_code == 200:
                self.model_version += 1
                logger.info(f"Successfully submitted model update for round {round_id}")
                return True
            else:
                logger.error(f"Error submitting model update: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error submitting model update: {e}")
            return False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(5),
        retry=retry_if_exception_type(httpx.RequestError)
    )
    async def get_current_round(self) -> Optional[Dict]:
        """Get current federated learning round from coordinator"""
        try:
            response = await self.client.get(f"{self.server_url}/rounds/current")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Error getting current round: {e.response.status_code} - {e.response.text}")
            return None
        except httpx.RequestError as e:
            logger.error(f"Request error getting current round: {e}")
            raise  # Re-raise to allow tenacity to retry
        except Exception as e:
            logger.error(f"An unexpected error occurred while getting current round: {e}")
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
        
        if not self.federated_config:
            logger.warning("Federated config not set, using default privacy metrics.")
            self.federated_config = {}

        # Simulate privacy metrics
        privacy_metrics = {
            "epsilon_used": random.uniform(0.5, 2.0),
            "delta": self.federated_config.get('privacy', {}).get('delta', 1e-5),
            "noise_multiplier": self.federated_config.get('privacy', {}).get('noise_multiplier', 0.2)
        }
        
        if not self.model:
            logger.error("Model not initialized, cannot simulate training.")
            return training_metrics, privacy_metrics

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
        if self.privacy_engine:
            privacy_budget = await self.privacy_engine.get_remaining_budget()
        else:
            privacy_budget = None
            logger.warning("Privacy engine not initialized, cannot report budget.")
        
        return {
            "client_id": self.client_id,
            "active": self.active,
            "registered": self.registered,
            "current_round": self.current_round,
            "privacy_budget_remaining": privacy_budget,
            "dataset_size": self.dataset_size,
            "server_url": self.server_url
        }

    def get_latest_model(self) -> Optional[Dict]:
        """
        Fetches the latest global model from the coordinator.
        """
        try:
            logger.info("Fetching latest model from coordinator...")
            response = requests.get(f"{self.server_url}/models/latest", timeout=10)
            
            if response.status_code == 200:
                model_info = response.json()
                # Implementation of get_latest_model method
                pass
            else:
                logger.error(f"Failed to fetch latest model: {response.status_code}")
                return None
        except requests.RequestException as e:
            logger.error(f"Error fetching latest model: {e}")
            return None

    # ... rest of the original methods ... 