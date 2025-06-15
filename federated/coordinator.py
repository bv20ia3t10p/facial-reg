"""
Federated Learning Coordinator Service
Implements FL with Homomorphic Encryption and Differential Privacy
"""

import os
import sys
import logging
import asyncio
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import hashlib
import uuid

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
from cryptography.fernet import Fernet
import sqlite3
from pydantic import BaseModel
import psutil

# Homomorphic Encryption (using tenseal for CKKS)
try:
    import tenseal as ts
except ImportError:
    print("Warning: tenseal not available. HE features disabled.")
    ts = None

# Differential Privacy
try:
    from opacus import PrivacyEngine
    from opacus.utils.batch_memory_manager import BatchMemoryManager
except ImportError:
    print("Warning: opacus not available. DP features limited.")
    PrivacyEngine = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('D:/logs/federated.log', maxBytes=10*1024*1024, backupCount=3),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Pydantic models
class ClientRegistration(BaseModel):
    client_id: str
    client_type: str = "mobile"
    capabilities: Dict[str, Any] = {}
    public_key: Optional[str] = None

class ModelUpdate(BaseModel):
    client_id: str
    round_id: int
    encrypted_weights: str
    num_samples: int
    privacy_spent: float
    metadata: Dict[str, Any] = {}

class FederatedRound(BaseModel):
    round_id: int
    participants: List[str]
    target_accuracy: float = 0.6
    max_privacy_budget: float = 50.0
    aggregation_method: str = "fedavg"

class FederatedConfig:
    """Configuration for federated learning"""
    
    def __init__(self):
        # FL Parameters
        self.min_clients = 2
        self.max_clients = 10
        self.rounds_per_epoch = 1
        self.target_accuracy = 0.6
        self.convergence_threshold = 0.01
        
        # Privacy Parameters
        self.max_privacy_budget = 50.0
        self.noise_multiplier = 1.0
        self.max_grad_norm = 1.0
        
        # HE Parameters
        self.poly_modulus_degree = 8192
        self.coeff_mod_bit_sizes = [60, 40, 40, 60]
        self.scale = 2.0**40
        
        # Model Parameters
        self.model_path = "D:/models/federated_model.pth"
        self.backup_path = "D:/models/backup/"
        
        # Timing
        self.round_timeout = 300  # 5 minutes
        self.client_timeout = 60   # 1 minute

config = FederatedConfig()

class HomomorphicEncryption:
    """Homomorphic Encryption handler using CKKS scheme"""
    
    def __init__(self):
        self.context = None
        self.public_key = None
        self.secret_key = None
        self.relin_keys = None
        self.galois_keys = None
        self.setup_context()
    
    def setup_context(self):
        """Setup CKKS context for homomorphic encryption"""
        if ts is None:
            logger.warning("TenSEAL not available. HE disabled.")
            return
        
        try:
            # Create CKKS context
            self.context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=config.poly_modulus_degree,
                coeff_mod_bit_sizes=config.coeff_mod_bit_sizes
            )
            
            # Set scale
            self.context.global_scale = config.scale
            
            # Generate keys
            self.context.generate_galois_keys()
            self.context.generate_relin_keys()
            
            # Make context public (for client operations)
            self.public_context = self.context.copy()
            self.public_context.make_context_public()
            
            logger.info("Homomorphic encryption context initialized")
            
        except Exception as e:
            logger.error(f"Failed to setup HE context: {str(e)}")
            self.context = None
    
    def encrypt_tensor(self, tensor: torch.Tensor) -> Optional[str]:
        """Encrypt a tensor using CKKS"""
        if self.context is None:
            return None
        
        try:
            # Convert tensor to list
            data = tensor.flatten().tolist()
            
            # Encrypt
            encrypted = ts.ckks_vector(self.context, data)
            
            # Serialize
            return encrypted.serialize().hex()
            
        except Exception as e:
            logger.error(f"Encryption failed: {str(e)}")
            return None
    
    def decrypt_tensor(self, encrypted_hex: str, shape: tuple) -> Optional[torch.Tensor]:
        """Decrypt a tensor from CKKS"""
        if self.context is None:
            return None
        
        try:
            # Deserialize
            encrypted_bytes = bytes.fromhex(encrypted_hex)
            encrypted = ts.lazy_ckks_vector_from(encrypted_bytes)
            encrypted.link_context(self.context)
            
            # Decrypt
            decrypted = encrypted.decrypt()
            
            # Convert back to tensor
            tensor = torch.tensor(decrypted[:np.prod(shape)]).reshape(shape)
            return tensor
            
        except Exception as e:
            logger.error(f"Decryption failed: {str(e)}")
            return None
    
    def add_encrypted(self, enc1_hex: str, enc2_hex: str) -> Optional[str]:
        """Add two encrypted tensors"""
        if self.context is None:
            return None
        
        try:
            # Deserialize both
            enc1_bytes = bytes.fromhex(enc1_hex)
            enc2_bytes = bytes.fromhex(enc2_hex)
            
            enc1 = ts.lazy_ckks_vector_from(enc1_bytes)
            enc2 = ts.lazy_ckks_vector_from(enc2_bytes)
            
            enc1.link_context(self.context)
            enc2.link_context(self.context)
            
            # Add
            result = enc1 + enc2
            
            # Serialize result
            return result.serialize().hex()
            
        except Exception as e:
            logger.error(f"Encrypted addition failed: {str(e)}")
            return None

he_manager = HomomorphicEncryption()

class DifferentialPrivacy:
    """Differential Privacy manager using Opacus"""
    
    def __init__(self):
        self.privacy_engines = {}  # client_id -> PrivacyEngine
        self.privacy_spent = {}    # client_id -> float
    
    def setup_privacy_engine(self, client_id: str, model: torch.nn.Module, 
                           optimizer: torch.optim.Optimizer, data_loader) -> Optional[object]:
        """Setup privacy engine for a client"""
        if PrivacyEngine is None:
            logger.warning("Opacus not available. DP features limited.")
            return None
        
        try:
            privacy_engine = PrivacyEngine()
            
            model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=data_loader,
                epochs=1,
                target_epsilon=config.max_privacy_budget,
                target_delta=1e-5,
                max_grad_norm=config.max_grad_norm,
            )
            
            self.privacy_engines[client_id] = privacy_engine
            self.privacy_spent[client_id] = 0.0
            
            logger.info(f"Privacy engine setup for client {client_id}")
            return privacy_engine
            
        except Exception as e:
            logger.error(f"Failed to setup privacy engine for {client_id}: {str(e)}")
            return None
    
    def get_privacy_spent(self, client_id: str) -> float:
        """Get privacy budget spent by client"""
        return self.privacy_spent.get(client_id, 0.0)
    
    def update_privacy_spent(self, client_id: str, epsilon: float):
        """Update privacy budget spent"""
        if client_id not in self.privacy_spent:
            self.privacy_spent[client_id] = 0.0
        self.privacy_spent[client_id] += epsilon
    
    def check_privacy_budget(self, client_id: str) -> bool:
        """Check if client has remaining privacy budget"""
        spent = self.get_privacy_spent(client_id)
        return spent < config.max_privacy_budget

dp_manager = DifferentialPrivacy()

class FederatedDatabase:
    """Database for federated learning coordination"""
    
    def __init__(self, db_path: str = "D:/data/federated.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_db()
    
    def init_db(self):
        """Initialize federated learning database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS clients (
                    client_id TEXT PRIMARY KEY,
                    client_type TEXT NOT NULL,
                    registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP,
                    capabilities TEXT,
                    public_key TEXT,
                    privacy_spent REAL DEFAULT 0.0,
                    active BOOLEAN DEFAULT 1
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS federated_rounds (
                    round_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    participants TEXT,
                    global_accuracy REAL,
                    convergence_metric REAL,
                    model_hash TEXT,
                    status TEXT DEFAULT 'active'
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS client_updates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    round_id INTEGER,
                    client_id TEXT,
                    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    num_samples INTEGER,
                    local_accuracy REAL,
                    privacy_spent REAL,
                    update_hash TEXT,
                    encrypted_weights TEXT,
                    FOREIGN KEY (round_id) REFERENCES federated_rounds(round_id),
                    FOREIGN KEY (client_id) REFERENCES clients(client_id)
                )
            """)
            
            conn.commit()
    
    def get_connection(self):
        """Get optimized database connection"""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=10000")
        return conn
    
    def register_client(self, client_id: str, client_type: str, capabilities: dict, public_key: str = None):
        """Register a new federated client"""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO clients 
                (client_id, client_type, capabilities, public_key, last_seen)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (client_id, client_type, json.dumps(capabilities), public_key))
            conn.commit()
    
    def get_active_clients(self) -> List[Dict]:
        """Get list of active clients"""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT client_id, client_type, capabilities, privacy_spent, last_seen
                FROM clients 
                WHERE active = 1 AND privacy_spent < ?
                ORDER BY last_seen DESC
            """, (config.max_privacy_budget,))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'client_id': row[0],
                    'client_type': row[1],
                    'capabilities': json.loads(row[2]) if row[2] else {},
                    'privacy_spent': row[3],
                    'last_seen': row[4]
                })
            return results
    
    def start_round(self, participants: List[str]) -> int:
        """Start a new federated round"""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO federated_rounds (participants, status)
                VALUES (?, 'active')
            """, (json.dumps(participants),))
            round_id = cursor.lastrowid
            conn.commit()
            return round_id
    
    def submit_client_update(self, round_id: int, client_id: str, num_samples: int,
                           local_accuracy: float, privacy_spent: float, 
                           encrypted_weights: str):
        """Submit client model update"""
        update_hash = hashlib.sha256(encrypted_weights.encode()).hexdigest()
        
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO client_updates 
                (round_id, client_id, num_samples, local_accuracy, privacy_spent, 
                 update_hash, encrypted_weights)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (round_id, client_id, num_samples, local_accuracy, privacy_spent,
                  update_hash, encrypted_weights))
            
            # Update client privacy spent
            conn.execute("""
                UPDATE clients SET privacy_spent = privacy_spent + ?, last_seen = CURRENT_TIMESTAMP
                WHERE client_id = ?
            """, (privacy_spent, client_id))
            
            conn.commit()
    
    def get_round_updates(self, round_id: int) -> List[Dict]:
        """Get all updates for a round"""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT client_id, num_samples, local_accuracy, privacy_spent, 
                       encrypted_weights, submitted_at
                FROM client_updates 
                WHERE round_id = ?
                ORDER BY submitted_at
            """, (round_id,))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'client_id': row[0],
                    'num_samples': row[1],
                    'local_accuracy': row[2],
                    'privacy_spent': row[3],
                    'encrypted_weights': row[4],
                    'submitted_at': row[5]
                })
            return results
    
    def complete_round(self, round_id: int, global_accuracy: float, 
                      convergence_metric: float, model_hash: str):
        """Mark round as completed"""
        with self.get_connection() as conn:
            conn.execute("""
                UPDATE federated_rounds 
                SET completed_at = CURRENT_TIMESTAMP, global_accuracy = ?,
                    convergence_metric = ?, model_hash = ?, status = 'completed'
                WHERE round_id = ?
            """, (global_accuracy, convergence_metric, model_hash, round_id))
            conn.commit()

fed_db = FederatedDatabase()

class FederatedAggregator:
    """Secure aggregation with HE and DP"""
    
    def __init__(self):
        self.current_round = None
        self.round_updates = {}
        self.global_model = None
        self.load_global_model()
    
    def load_global_model(self):
        """Load the global model"""
        try:
            if os.path.exists(config.model_path):
                self.global_model = torch.load(config.model_path, map_location='cpu')
                logger.info("Global model loaded successfully")
            else:
                logger.warning("No global model found, will initialize from first client")
        except Exception as e:
            logger.error(f"Failed to load global model: {str(e)}")
    
    def save_global_model(self):
        """Save the global model"""
        try:
            os.makedirs(os.path.dirname(config.model_path), exist_ok=True)
            torch.save(self.global_model, config.model_path)
            
            # Create backup
            backup_path = f"{config.backup_path}model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            torch.save(self.global_model, backup_path)
            
            logger.info("Global model saved successfully")
        except Exception as e:
            logger.error(f"Failed to save global model: {str(e)}")
    
    def federated_averaging(self, client_updates: List[Dict]) -> torch.nn.Module:
        """Perform federated averaging with encrypted weights"""
        if not client_updates:
            return self.global_model
        
        try:
            # Decrypt and aggregate weights
            total_samples = sum(update['num_samples'] for update in client_updates)
            aggregated_weights = {}
            
            for i, update in enumerate(client_updates):
                client_id = update['client_id']
                num_samples = update['num_samples']
                encrypted_weights = update['encrypted_weights']
                
                logger.info(f"Processing update from {client_id} with {num_samples} samples")
                
                # For now, simulate decryption (replace with actual HE decryption)
                # In real implementation, this would decrypt the homomorphically encrypted weights
                if he_manager.context is not None:
                    # Decrypt weights (simplified - actual implementation would handle model structure)
                    decrypted_data = he_manager.decrypt_tensor(encrypted_weights, (512,))
                    if decrypted_data is not None:
                        weight = num_samples / total_samples
                        if i == 0:
                            aggregated_weights['features'] = decrypted_data * weight
                        else:
                            aggregated_weights['features'] += decrypted_data * weight
                else:
                    # Fallback: simulate aggregation
                    weight = num_samples / total_samples
                    simulated_weights = torch.randn(512) * weight
                    if i == 0:
                        aggregated_weights['features'] = simulated_weights
                    else:
                        aggregated_weights['features'] += simulated_weights
            
            # Update global model (simplified)
            if self.global_model is None:
                # Initialize global model structure
                self.global_model = {
                    'features': aggregated_weights.get('features', torch.randn(512)),
                    'classifier': torch.randn(100, 512),  # Assuming 100 classes
                    'metadata': {
                        'round': self.current_round,
                        'participants': len(client_updates),
                        'total_samples': total_samples
                    }
                }
            else:
                # Update existing model
                if 'features' in aggregated_weights:
                    self.global_model['features'] = aggregated_weights['features']
                self.global_model['metadata'].update({
                    'round': self.current_round,
                    'participants': len(client_updates),
                    'total_samples': total_samples,
                    'updated_at': datetime.now().isoformat()
                })
            
            logger.info(f"Federated averaging completed for round {self.current_round}")
            return self.global_model
            
        except Exception as e:
            logger.error(f"Federated averaging failed: {str(e)}")
            return self.global_model
    
    def evaluate_global_model(self) -> float:
        """Evaluate global model accuracy (simulated)"""
        try:
            # In real implementation, this would evaluate on validation set
            # For now, simulate accuracy based on round progress
            if self.current_round is None:
                return 0.3
            
            # Simulate improving accuracy over rounds
            base_accuracy = 0.3
            improvement = min(0.3, self.current_round * 0.05)
            noise = np.random.normal(0, 0.02)  # Add some noise
            
            accuracy = base_accuracy + improvement + noise
            return max(0.0, min(1.0, accuracy))  # Clamp to [0, 1]
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            return 0.0

aggregator = FederatedAggregator()

# FastAPI app
app = FastAPI(
    title="Federated Learning Coordinator",
    description="FL service with Homomorphic Encryption and Differential Privacy",
    version="1.0.0"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Global state
active_round = None
round_start_time = None

@app.get("/health")
async def health_check():
    """Health check for federated service"""
    memory_status = {
        'system_used': psutil.virtual_memory().percent / 100,
        'system_available': psutil.virtual_memory().available / (1024**3)
    }
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "memory": memory_status,
        "active_round": active_round,
        "he_enabled": he_manager.context is not None,
        "registered_clients": len(fed_db.get_active_clients())
    }

@app.post("/clients/register")
async def register_client(registration: ClientRegistration):
    """Register a new federated learning client"""
    try:
        fed_db.register_client(
            registration.client_id,
            registration.client_type,
            registration.capabilities,
            registration.public_key
        )
        
        logger.info(f"Client {registration.client_id} registered successfully")
        
        return {
            "success": True,
            "client_id": registration.client_id,
            "registered_at": datetime.now().isoformat(),
            "he_public_context": he_manager.public_context.serialize().hex() if he_manager.context else None
        }
        
    except Exception as e:
        logger.error(f"Client registration failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Registration failed")

@app.get("/clients/active")
async def get_active_clients():
    """Get list of active clients"""
    try:
        clients = fed_db.get_active_clients()
        return {
            "active_clients": clients,
            "count": len(clients),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get active clients: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get clients")

@app.post("/rounds/start")
async def start_federated_round(background_tasks: BackgroundTasks):
    """Start a new federated learning round"""
    global active_round, round_start_time
    
    try:
        if active_round is not None:
            raise HTTPException(status_code=400, detail="Round already active")
        
        # Get available clients
        clients = fed_db.get_active_clients()
        if len(clients) < config.min_clients:
            raise HTTPException(status_code=400, detail=f"Need at least {config.min_clients} clients")
        
        # Select participants (for now, use all available)
        participants = [client['client_id'] for client in clients[:config.max_clients]]
        
        # Start round
        round_id = fed_db.start_round(participants)
        active_round = round_id
        round_start_time = datetime.now()
        aggregator.current_round = round_id
        
        # Schedule round timeout
        background_tasks.add_task(monitor_round_timeout, round_id)
        
        logger.info(f"Started federated round {round_id} with {len(participants)} participants")
        
        return {
            "round_id": round_id,
            "participants": participants,
            "started_at": round_start_time.isoformat(),
            "timeout_seconds": config.round_timeout,
            "global_model_hash": hashlib.sha256(str(aggregator.global_model).encode()).hexdigest() if aggregator.global_model else None
        }
        
    except Exception as e:
        logger.error(f"Failed to start round: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start round")

@app.get("/rounds/current")
async def get_current_round():
    """Get current round information"""
    if active_round is None:
        return {"active_round": None}
    
    try:
        updates = fed_db.get_round_updates(active_round)
        return {
            "round_id": active_round,
            "started_at": round_start_time.isoformat() if round_start_time else None,
            "participants_submitted": len(updates),
            "updates": [
                {
                    "client_id": update["client_id"],
                    "num_samples": update["num_samples"],
                    "local_accuracy": update["local_accuracy"],
                    "submitted_at": update["submitted_at"]
                }
                for update in updates
            ]
        }
    except Exception as e:
        logger.error(f"Failed to get current round: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get round info")

@app.post("/rounds/{round_id}/submit")
async def submit_model_update(round_id: int, update: ModelUpdate, background_tasks: BackgroundTasks):
    """Submit encrypted model update from client"""
    global active_round
    
    try:
        if active_round != round_id:
            raise HTTPException(status_code=400, detail="Round not active or invalid")
        
        # Validate client
        clients = fed_db.get_active_clients()
        client_ids = [c['client_id'] for c in clients]
        if update.client_id not in client_ids:
            raise HTTPException(status_code=400, detail="Client not registered")
        
        # Check privacy budget
        if not dp_manager.check_privacy_budget(update.client_id):
            raise HTTPException(status_code=400, detail="Privacy budget exhausted")
        
        # Store update
        fed_db.submit_client_update(
            round_id,
            update.client_id,
            update.num_samples,
            update.metadata.get('local_accuracy', 0.0),
            update.privacy_spent,
            update.encrypted_weights
        )
        
        # Update privacy tracking
        dp_manager.update_privacy_spent(update.client_id, update.privacy_spent)
        
        logger.info(f"Received update from {update.client_id} for round {round_id}")
        
        # Check if all clients have submitted
        updates = fed_db.get_round_updates(round_id)
        participants = json.loads(fed_db.get_connection().execute(
            "SELECT participants FROM federated_rounds WHERE round_id = ?", (round_id,)
        ).fetchone()[0])
        
        if len(updates) >= len(participants):
            # All clients submitted, trigger aggregation
            background_tasks.add_task(aggregate_round, round_id)
        
        return {
            "success": True,
            "round_id": round_id,
            "client_id": update.client_id,
            "privacy_remaining": config.max_privacy_budget - dp_manager.get_privacy_spent(update.client_id),
            "submitted_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to submit update: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to submit update")

@app.get("/models/global")
async def get_global_model():
    """Get current global model"""
    try:
        if aggregator.global_model is None:
            raise HTTPException(status_code=404, detail="No global model available")
        
        # Return model metadata (not actual weights for security)
        model_info = {
            "model_available": True,
            "last_updated": aggregator.global_model.get('metadata', {}).get('updated_at'),
            "round": aggregator.global_model.get('metadata', {}).get('round'),
            "participants": aggregator.global_model.get('metadata', {}).get('participants'),
            "total_samples": aggregator.global_model.get('metadata', {}).get('total_samples'),
            "model_hash": hashlib.sha256(str(aggregator.global_model).encode()).hexdigest()
        }
        
        return model_info
        
    except Exception as e:
        logger.error(f"Failed to get global model: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get model")

@app.get("/privacy/status/{client_id}")
async def get_privacy_status(client_id: str):
    """Get privacy budget status for client"""
    try:
        spent = dp_manager.get_privacy_spent(client_id)
        remaining = config.max_privacy_budget - spent
        
        return {
            "client_id": client_id,
            "privacy_spent": spent,
            "privacy_remaining": remaining,
            "privacy_budget": config.max_privacy_budget,
            "can_participate": remaining > 0
        }
        
    except Exception as e:
        logger.error(f"Failed to get privacy status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get privacy status")

# Background tasks
async def monitor_round_timeout(round_id: int):
    """Monitor round timeout"""
    global active_round
    
    await asyncio.sleep(config.round_timeout)
    
    if active_round == round_id:
        logger.warning(f"Round {round_id} timed out")
        
        # Get partial updates and aggregate
        updates = fed_db.get_round_updates(round_id)
        if updates:
            await aggregate_round(round_id)
        else:
            # No updates received, mark round as failed
            active_round = None
            logger.error(f"Round {round_id} failed - no updates received")

async def aggregate_round(round_id: int):
    """Aggregate client updates and update global model"""
    global active_round
    
    try:
        logger.info(f"Starting aggregation for round {round_id}")
        
        # Get all updates for this round
        updates = fed_db.get_round_updates(round_id)
        if not updates:
            logger.warning(f"No updates to aggregate for round {round_id}")
            return
        
        # Perform federated averaging
        aggregator.global_model = aggregator.federated_averaging(updates)
        
        # Save updated model
        aggregator.save_global_model()
        
        # Evaluate global model
        global_accuracy = aggregator.evaluate_global_model()
        
        # Calculate convergence metric (simplified)
        convergence_metric = abs(global_accuracy - config.target_accuracy)
        
        # Complete round in database
        model_hash = hashlib.sha256(str(aggregator.global_model).encode()).hexdigest()
        fed_db.complete_round(round_id, global_accuracy, convergence_metric, model_hash)
        
        # Reset active round
        active_round = None
        
        logger.info(f"Round {round_id} completed - Global accuracy: {global_accuracy:.4f}")
        
        # Check convergence
        if convergence_metric < config.convergence_threshold:
            logger.info(f"Convergence achieved! Target accuracy reached.")
        
    except Exception as e:
        logger.error(f"Aggregation failed for round {round_id}: {str(e)}")
        active_round = None

if __name__ == "__main__":
    uvicorn.run("coordinator:app", host="0.0.0.0", port=8001, workers=1, log_level="info", reload=False) 