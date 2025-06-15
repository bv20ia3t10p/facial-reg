"""
Federated Learning Client
Implements client-side FL with Homomorphic Encryption and Differential Privacy
"""

import os
import sys
import logging
import asyncio
import json
import hashlib
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import requests
from PIL import Image
import cv2

# Homomorphic Encryption
try:
    import tenseal as ts
except ImportError:
    print("Warning: tenseal not available. HE features disabled.")
    ts = None

# Differential Privacy
try:
    from opacus import PrivacyEngine
    from opacus.utils.batch_memory_manager import BatchMemoryManager
    from opacus.validators import ModuleValidator
except ImportError:
    print("Warning: opacus not available. DP features limited.")
    PrivacyEngine = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BiometricDataset(Dataset):
    """Dataset for biometric data"""
    
    def __init__(self, data_path: str, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.samples = []
        self.labels = []
        self.load_data()
    
    def load_data(self):
        """Load biometric data from directory"""
        if not os.path.exists(self.data_path):
            logger.warning(f"Data path {self.data_path} does not exist")
            return
        
        # Load identity folders
        identity_folders = [d for d in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, d))]
        
        for label, identity_folder in enumerate(identity_folders):
            identity_path = os.path.join(self.data_path, identity_folder)
            
            # Load images from identity folder
            for img_file in os.listdir(identity_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(identity_path, img_file)
                    self.samples.append(img_path)
                    self.labels.append(label)
        
        logger.info(f"Loaded {len(self.samples)} samples from {len(identity_folders)} identities")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        # Load and preprocess image
        try:
            image = cv2.imread(img_path)
            image = cv2.resize(image, (224, 224))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convert to tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            
            if self.transform:
                image = self.transform(image)
            
            return image, torch.tensor(label, dtype=torch.long)
            
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {str(e)}")
            # Return dummy data
            return torch.zeros(3, 224, 224), torch.tensor(0, dtype=torch.long)

class SimpleBiometricModel(nn.Module):
    """Simple biometric model for federated learning"""
    
    def __init__(self, num_classes: int = 100):
        super(SimpleBiometricModel, self).__init__()
        
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class FederatedClient:
    """Federated Learning Client with HE and DP"""
    
    def __init__(self, client_id: str, coordinator_url: str = "http://localhost:8001", 
                 data_path: str = None, client_type: str = "mobile"):
        self.client_id = client_id
        self.coordinator_url = coordinator_url
        self.data_path = data_path
        self.client_type = client_type
        
        # Model and training
        self.model = None
        self.optimizer = None
        self.privacy_engine = None
        self.data_loader = None
        
        # Privacy tracking
        self.privacy_spent = 0.0
        self.max_privacy_budget = 50.0
        
        # HE context
        self.he_context = None
        self.setup_he_context()
        
        # Initialize model
        self.setup_model()
        
        # Load data
        if data_path:
            self.setup_data()
    
    def setup_he_context(self):
        """Setup homomorphic encryption context"""
        if ts is None:
            logger.warning("TenSEAL not available. HE disabled.")
            return
        
        try:
            # Create CKKS context for client
            self.he_context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=8192,
                coeff_mod_bit_sizes=[60, 40, 40, 60]
            )
            self.he_context.global_scale = 2.0**40
            self.he_context.generate_galois_keys()
            
            logger.info("Client HE context initialized")
            
        except Exception as e:
            logger.error(f"Failed to setup HE context: {str(e)}")
            self.he_context = None
    
    def setup_model(self):
        """Setup model and optimizer"""
        try:
            # Determine number of classes from data
            num_classes = 100  # Default
            if self.data_path and os.path.exists(self.data_path):
                identity_folders = [d for d in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, d))]
                num_classes = len(identity_folders)
            
            # Initialize model
            self.model = SimpleBiometricModel(num_classes=num_classes)
            
            # Load pretrained model if available
            pretrained_path = os.getenv('PRETRAINED_MODEL_PATH')
            if pretrained_path and os.path.exists(pretrained_path):
                logger.info(f"Loading pretrained model from {pretrained_path}")
                try:
                    state_dict = torch.load(pretrained_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
                    self.model.load_state_dict(state_dict)
                    logger.info("Successfully loaded pretrained model")
                except Exception as e:
                    logger.error(f"Failed to load pretrained model: {str(e)}")
            
            # Move model to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                logger.info("Model moved to GPU")
            
            # Setup optimizer
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            
            # Validate model for DP if available
            if PrivacyEngine is not None:
                if not ModuleValidator.is_valid(self.model):
                    self.model = ModuleValidator.fix(self.model)
                    logger.info("Model fixed for differential privacy")
            
            logger.info(f"Model initialized with {num_classes} classes")
            
        except Exception as e:
            logger.error(f"Failed to setup model: {str(e)}")
            raise
    
    def setup_data(self):
        """Setup data loader"""
        try:
            dataset = BiometricDataset(self.data_path)
            
            if len(dataset) == 0:
                logger.warning("No data found in dataset")
                return
            
            # Create data loader with small batch size for memory efficiency
            self.data_loader = DataLoader(
                dataset, 
                batch_size=8,  # Small batch size for memory efficiency
                shuffle=True,
                num_workers=0,  # Avoid multiprocessing issues
                pin_memory=False
            )
            
            logger.info(f"Data loader created with {len(dataset)} samples")
            
        except Exception as e:
            logger.error(f"Failed to setup data: {str(e)}")
    
    def setup_differential_privacy(self):
        """Setup differential privacy"""
        if PrivacyEngine is None or self.data_loader is None:
            logger.warning("Cannot setup DP: missing dependencies or data")
            return False
        
        try:
            # Make model compatible with Opacus
            self.model = ModuleValidator.fix(self.model)
            
            # Create privacy engine
            self.privacy_engine = PrivacyEngine()
            
            # Make private
            self.model, self.optimizer, self.data_loader = self.privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.data_loader,
                epochs=1,
                target_epsilon=10.0,  # Per-round budget
                target_delta=1e-5,
                max_grad_norm=1.0,
            )
            
            logger.info("Differential privacy setup completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup DP: {str(e)}")
            return False
    
    def encrypt_model_weights(self) -> Optional[str]:
        """Encrypt model weights using homomorphic encryption"""
        if self.he_context is None:
            # Fallback: simple serialization (not actually encrypted)
            try:
                weights = []
                for param in self.model.parameters():
                    weights.extend(param.data.flatten().tolist())
                
                # Simulate encryption by encoding as hex
                weights_bytes = json.dumps(weights).encode()
                return weights_bytes.hex()
                
            except Exception as e:
                logger.error(f"Failed to serialize weights: {str(e)}")
                return None
        
        try:
            # Encrypt using CKKS
            all_weights = []
            for param in self.model.parameters():
                weights = param.data.flatten().tolist()
                all_weights.extend(weights)
            
            # Encrypt the flattened weights
            encrypted = ts.ckks_vector(self.he_context, all_weights)
            return encrypted.serialize().hex()
            
        except Exception as e:
            logger.error(f"Failed to encrypt weights: {str(e)}")
            return None
    
    def train_local_model(self, epochs: int = 1) -> Dict[str, Any]:
        """Train model locally with differential privacy"""
        if self.data_loader is None:
            logger.error("No data available for training")
            return {"success": False, "error": "No data"}
        
        try:
            self.model.train()
            total_loss = 0.0
            correct = 0
            total = 0
            
            criterion = nn.CrossEntropyLoss()
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                epoch_correct = 0
                epoch_total = 0
                
                for batch_idx, (data, target) in enumerate(self.data_loader):
                    self.optimizer.zero_grad()
                    
                    # Forward pass
                    output = self.model(data)
                    loss = criterion(output, target)
                    
                    # Backward pass
                    loss.backward()
                    self.optimizer.step()
                    
                    # Statistics
                    epoch_loss += loss.item()
                    _, predicted = output.max(1)
                    epoch_total += target.size(0)
                    epoch_correct += predicted.eq(target).sum().item()
                    
                    if batch_idx % 10 == 0:
                        logger.info(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                
                total_loss += epoch_loss
                correct += epoch_correct
                total += epoch_total
                
                logger.info(f"Epoch {epoch+1} completed - Loss: {epoch_loss/len(self.data_loader):.4f}, Acc: {100.*epoch_correct/epoch_total:.2f}%")
            
            # Calculate privacy spent
            if self.privacy_engine:
                privacy_spent = self.privacy_engine.get_epsilon(delta=1e-5)
                self.privacy_spent += privacy_spent
            else:
                privacy_spent = 1.0  # Simulate privacy cost
                self.privacy_spent += privacy_spent
            
            avg_loss = total_loss / (len(self.data_loader) * epochs)
            accuracy = correct / total if total > 0 else 0.0
            
            return {
                "success": True,
                "loss": avg_loss,
                "accuracy": accuracy,
                "privacy_spent": privacy_spent,
                "num_samples": total
            }
            
        except Exception as e:
            logger.error(f"Local training failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def register_with_coordinator(self) -> bool:
        """Register client with federated coordinator"""
        try:
            registration_data = {
                "client_id": self.client_id,
                "client_type": self.client_type,
                "capabilities": {
                    "has_data": self.data_loader is not None,
                    "data_samples": len(self.data_loader.dataset) if self.data_loader else 0,
                    "he_enabled": self.he_context is not None,
                    "dp_enabled": PrivacyEngine is not None
                },
                "public_key": None  # Could add public key for additional security
            }
            
            response = requests.post(
                f"{self.coordinator_url}/clients/register",
                json=registration_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Successfully registered with coordinator: {result}")
                
                # Get HE public context if available
                if result.get("he_public_context"):
                    try:
                        context_bytes = bytes.fromhex(result["he_public_context"])
                        self.he_context = ts.context_from(context_bytes)
                        logger.info("Received HE public context from coordinator")
                    except Exception as e:
                        logger.warning(f"Failed to load HE context: {str(e)}")
                
                return True
            else:
                logger.error(f"Registration failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Registration failed: {str(e)}")
            return False
    
    def check_for_rounds(self) -> Optional[Dict]:
        """Check for active federated rounds"""
        try:
            response = requests.get(f"{self.coordinator_url}/rounds/current", timeout=10)
            
            if response.status_code == 200:
                round_info = response.json()
                if round_info.get("round_id"):
                    return round_info
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to check for rounds: {str(e)}")
            return None
    
    def participate_in_round(self, round_info: Dict) -> bool:
        """Participate in a federated round"""
        round_id = round_info["round_id"]
        
        try:
            logger.info(f"Participating in round {round_id}")
            
            # Setup differential privacy for this round
            dp_setup = self.setup_differential_privacy()
            
            # Train local model
            training_result = self.train_local_model(epochs=1)
            
            if not training_result["success"]:
                logger.error(f"Local training failed: {training_result.get('error')}")
                return False
            
            # Encrypt model weights
            encrypted_weights = self.encrypt_model_weights()
            
            if encrypted_weights is None:
                logger.error("Failed to encrypt model weights")
                return False
            
            # Submit update to coordinator
            update_data = {
                "client_id": self.client_id,
                "round_id": round_id,
                "encrypted_weights": encrypted_weights,
                "num_samples": training_result["num_samples"],
                "privacy_spent": training_result["privacy_spent"],
                "metadata": {
                    "local_accuracy": training_result["accuracy"],
                    "local_loss": training_result["loss"],
                    "dp_enabled": dp_setup,
                    "he_enabled": self.he_context is not None
                }
            }
            
            response = requests.post(
                f"{self.coordinator_url}/rounds/{round_id}/submit",
                json=update_data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Successfully submitted update for round {round_id}")
                logger.info(f"Privacy remaining: {result.get('privacy_remaining', 'unknown')}")
                return True
            else:
                logger.error(f"Failed to submit update: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to participate in round {round_id}: {str(e)}")
            return False
    
    def run_client(self, check_interval: int = 30):
        """Run federated client main loop"""
        logger.info(f"Starting federated client {self.client_id}")
        
        # Register with coordinator
        if not self.register_with_coordinator():
            logger.error("Failed to register with coordinator")
            return
        
        logger.info(f"Client {self.client_id} running. Checking for rounds every {check_interval}s")
        
        try:
            while True:
                # Check privacy budget
                if self.privacy_spent >= self.max_privacy_budget:
                    logger.warning(f"Privacy budget exhausted ({self.privacy_spent}/{self.max_privacy_budget})")
                    break
                
                # Check for active rounds
                round_info = self.check_for_rounds()
                
                if round_info:
                    logger.info(f"Found active round: {round_info['round_id']}")
                    
                    # Check if we've already participated
                    already_participated = any(
                        update["client_id"] == self.client_id 
                        for update in round_info.get("updates", [])
                    )
                    
                    if not already_participated:
                        success = self.participate_in_round(round_info)
                        if success:
                            logger.info(f"Successfully participated in round {round_info['round_id']}")
                        else:
                            logger.error(f"Failed to participate in round {round_info['round_id']}")
                    else:
                        logger.info(f"Already participated in round {round_info['round_id']}")
                
                # Wait before checking again
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            logger.info("Client stopped by user")
        except Exception as e:
            logger.error(f"Client error: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument("--client-id", required=True, help="Unique client identifier")
    parser.add_argument("--coordinator-url", default="http://localhost:8001", help="Coordinator URL")
    parser.add_argument("--data-path", help="Path to client data")
    parser.add_argument("--client-type", default="mobile", help="Client type")
    parser.add_argument("--check-interval", type=int, default=30, help="Round check interval (seconds)")
    
    args = parser.parse_args()
    
    # Create and run client
    client = FederatedClient(
        client_id=args.client_id,
        coordinator_url=args.coordinator_url,
        data_path=args.data_path,
        client_type=args.client_type
    )
    
    client.run_client(check_interval=args.check_interval)

if __name__ == "__main__":
    main() 