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
import io

from .models.privacy_biometric_model import PrivacyBiometricModel
from .privacy.privacy_engine import PrivacyEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FederatedClient:
    """Client for federated learning integration"""

    def __init__(self):
        self.client_id = os.getenv("CLIENT_ID", "client1")
        self.coordinator_url = os.getenv("COORDINATOR_URL", "http://fl-coordinator:8000")
        self.model_path = os.getenv("MODEL_PATH", "/app/models/best_client1_pretrained_model.pth")
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.current_round = 0
        self.active = False
        self.last_update = None
        self.enable_dp = os.getenv("ENABLE_DP", "true").lower() == "true"
        self.registered = False
        self.model = None
        self.privacy_engine = None
        self.dataset_size = 0

    async def initialize(self):
        """Initialize federated client"""
        logger.info(f"Initializing federated client {self.client_id}")
        
        # Load configuration from coordinator
        try:
            config_response = requests.get(f"{self.coordinator_url}/config", timeout=10)
            if config_response.status_code == 200:
                self.federated_config = config_response.json()
                logger.info(f"Loaded coordinator config: {self.federated_config}")
            else:
                logger.warning(f"Failed to load coordinator config, using defaults")
                self.federated_config = {
                    'max_epsilon': 100.0,
                    'delta': 1e-5,
                    'noise_multiplier': 0.2,
                    'max_grad_norm': 5.0
                }
        except Exception as e:
            logger.error(f"Error loading coordinator config: {e}")
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
        
        # Register with coordinator
        if await self.register_with_coordinator():
            logger.info(f"Client {self.client_id} registered with coordinator")
            self.registered = True
            self.active = True
        else:
            logger.warning(f"Failed to register client {self.client_id} with coordinator")
        
        # Get initial dataset size
        self.dataset_size = await self.get_dataset_size()
        logger.info(f"Dataset size: {self.dataset_size} samples")
        
        # Start background federated loop
        asyncio.create_task(self.federated_learning_loop())
        
        logger.info(f"Federated client {self.client_id} initialized")
        return True

    async def load_or_download_model(self) -> nn.Module:
        """Load model from local storage or download from coordinator"""
        logger.info(f"Loading model from {self.model_path}")
        
        local_model_path = Path(self.model_path)
        try:
            if local_model_path.exists():
                # Load local model
                state_dict = torch.load(local_model_path, map_location=self.device)
                
                # Extract model parameters
                num_identities = 100  # Default
                if isinstance(state_dict, dict) and 'num_identities' in state_dict:
                    num_identities = state_dict['num_identities']
                elif isinstance(state_dict, dict) and 'state_dict' in state_dict:
                    if 'metadata' in state_dict and 'num_identities' in state_dict['metadata']:
                        num_identities = state_dict['metadata']['num_identities']
                
                # Create model
                model = PrivacyBiometricModel(
                    num_identities=num_identities, 
                    privacy_enabled=self.enable_dp
                ).to(self.device)
                
                # Load weights
                if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                    model.load_state_dict(state_dict['state_dict'])
                else:
                    model.load_state_dict(state_dict)
                
                logger.info(f"Successfully loaded local model with {num_identities} identities")
                return model
            else:
                # Try to download global model
                logger.info("Local model not found, downloading global model")
                return await self.download_global_model()
        
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Try to download global model as fallback
            logger.info("Attempting to download global model as fallback")
            return await self.download_global_model()

    async def download_global_model(self) -> nn.Module:
        """Download global model from coordinator"""
        try:
            # Get global model info
            async with httpx.AsyncClient() as client:
                info_response = await client.get(f"{self.coordinator_url}/models/global")
                
                if info_response.status_code != 200:
                    logger.error(f"Failed to get global model info: {info_response.status_code}")
                    raise Exception(f"Failed to get global model info: {info_response.status_code}")
                
                # Download model
                download_response = await client.get(f"{self.coordinator_url}/models/global/download")
                
                if download_response.status_code != 200:
                    logger.error(f"Failed to download model: {download_response.status_code}")
                    raise Exception(f"Failed to download model: {download_response.status_code}")
                
                # Save model locally
                model_data = download_response.content
                with open(self.model_path, 'wb') as f:
                    f.write(model_data)
                
                logger.info(f"Downloaded global model to {self.model_path}")
                
                # Load the downloaded model
                return await self.load_or_download_model()
        
        except Exception as e:
            logger.error(f"Error downloading global model: {e}")
            # Create a new model as last resort
            num_identities = 100  # Default
            model = PrivacyBiometricModel(
                num_identities=num_identities, 
                privacy_enabled=self.enable_dp
            ).to(self.device)
            logger.warning(f"Created new model with {num_identities} identities as fallback")
            return model

    async def register_with_coordinator(self) -> bool:
        """Register this client with the federated coordinator"""
        try:
            # Get privacy budget
            privacy_budget = await self.privacy_engine.get_remaining_budget()
            
            # Get dataset size
            dataset_size = await self.get_dataset_size()
            
            # Register with coordinator
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.coordinator_url}/clients/register",
                    json={
                        "client_id": self.client_id,
                        "dataset_size": dataset_size,
                        "privacy_budget_remaining": privacy_budget
                    }
                )
                
                if response.status_code == 200:
                    return True
                else:
                    logger.error(f"Failed to register: {response.status_code}, {response.text}")
                    return False
        
        except Exception as e:
            logger.error(f"Error registering client: {e}")
            return False

    async def get_dataset_size(self) -> int:
        """Get the size of the local dataset"""
        # In a real implementation, this would load the actual dataset
        # For simulation, return a random value between 100 and 1000
        return random.randint(100, 1000)

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

    async def submit_model_update(self, round_id: int, training_metrics: Dict, privacy_metrics: Dict) -> bool:
        """Submit model update to coordinator"""
        logger.info(f"Submitting model update for round {round_id}")
        
        try:
            # First notify coordinator of update
            async with httpx.AsyncClient() as client:
                update_response = await client.post(
                    f"{self.coordinator_url}/models/update",
                    json={
                        "client_id": self.client_id,
                        "round_id": round_id,
                        "dataset_size": await self.get_dataset_size(),
                        "training_metrics": training_metrics,
                        "privacy_metrics": privacy_metrics
                    }
                )
                
                if update_response.status_code != 200:
                    logger.error(f"Failed to submit update info: {update_response.status_code}")
                    return False
                
                # Get upload URL
                update_info = update_response.json()
                upload_url = update_info.get("upload_url")
                
                if not upload_url:
                    logger.error("No upload URL provided")
                    return False
                
                # Save model state dict to file
                temp_model_path = f"/app/cache/update_{self.client_id}_{round_id}.pth"
                torch.save(self.model.state_dict(), temp_model_path)
                
                # Upload model parameters
                with open(temp_model_path, "rb") as model_file:
                    files = {"model_file": model_file}
                    upload_response = await client.post(
                        f"{self.coordinator_url}{upload_url}",
                        files=files
                    )
                
                if upload_response.status_code == 200:
                    logger.info(f"Model update for round {round_id} submitted successfully")
                    self.last_update = datetime.utcnow().isoformat()
                    return True
                else:
                    logger.error(f"Failed to upload model: {upload_response.status_code}")
                    return False
        
        except Exception as e:
            logger.error(f"Error submitting model update: {e}")
            return False

    async def federated_learning_loop(self):
        """Background loop for participating in federated learning rounds"""
        while self.active:
            try:
                # Check for active round
                current_round_info = await self.get_current_round()
                
                if not current_round_info:
                    logger.info("No active round, waiting...")
                    await asyncio.sleep(30)  # Check again in 30 seconds
                    continue
                
                round_id = current_round_info.get("round_id", 0)
                status = current_round_info.get("status", "")
                
                # Skip if we already processed this round
                if round_id <= self.current_round:
                    await asyncio.sleep(30)  # Check again in 30 seconds
                    continue
                
                # Skip if round is not active
                if status != "active":
                    logger.info(f"Round {round_id} is not active (status: {status}), waiting...")
                    await asyncio.sleep(30)  # Check again in 30 seconds
                    continue
                
                logger.info(f"Starting participation in round {round_id}")
                
                # Train on local data
                training_metrics, privacy_metrics = await self.train_on_local_data(round_id)
                
                # Submit model update
                success = await self.submit_model_update(round_id, training_metrics, privacy_metrics)
                
                if success:
                    self.current_round = round_id
                    logger.info(f"Completed federated round {round_id}")
                else:
                    logger.warning(f"Failed to complete federated round {round_id}")
                
                # Wait before checking for new rounds
                await asyncio.sleep(60)  # Wait 1 minute
                
            except Exception as e:
                logger.error(f"Error in federated learning loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    async def get_status(self) -> Dict:
        """Get status of federated client"""
        privacy_budget = await self.privacy_engine.get_remaining_budget()
        
        return {
            "client_id": self.client_id,
            "active": self.active,
            "registered": self.registered,
            "current_round": self.current_round,
            "last_update": self.last_update,
            "privacy_budget_remaining": privacy_budget,
            "dataset_size": self.dataset_size,
            "coordinator_url": self.coordinator_url
        } 