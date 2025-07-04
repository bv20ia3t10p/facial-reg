"""
Federated Learning Manager for Privacy-Preserving Model Training
"""

import torch
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

class FederatedManager:
    """Manages federated learning across multiple clients"""
    
    def __init__(self):
        self.global_model = None
        self.client_models = {}
        self.aggregation_weights = {}
        self.round_metrics = defaultdict(list)
        self.current_round = 0
        self.total_rounds = 50
        self.min_clients = 2
        self.client_updates = {}
        self.last_aggregation = None
        
    async def initialize(self):
        """Initialize federated learning components"""
        logger.info("Initializing Federated Learning Manager...")
        self.last_aggregation = datetime.utcnow()
        logger.info("✓ Federated Learning Manager initialized")
    
    def register_client(self, client_id: str, num_samples: int):
        """Register a new client for federated learning"""
        try:
            if client_id not in self.client_models:
                self.aggregation_weights[client_id] = num_samples
                total_samples = sum(self.aggregation_weights.values())
                
                # Normalize weights
                for cid in self.aggregation_weights:
                    self.aggregation_weights[cid] /= total_samples
                
                logger.info(f"✓ Registered client {client_id} with {num_samples} samples")
                return True
        except Exception as e:
            logger.error(f"Client registration failed: {e}")
            return False
    
    def update_client_model(self, client_id: str, model_state: Dict[str, torch.Tensor]):
        """Update client model state"""
        try:
            self.client_models[client_id] = model_state
            self.client_updates[client_id] = datetime.utcnow()
            
            # Check if we have enough updates for aggregation
            if len(self.client_updates) >= self.min_clients:
                self._try_aggregate_models()
            
            return True
        except Exception as e:
            logger.error(f"Client model update failed: {e}")
            return False
    
    def _try_aggregate_models(self):
        """Attempt to aggregate models if conditions are met"""
        try:
            current_time = datetime.utcnow()
            
            # Check if we have enough recent updates
            recent_updates = [
                client_id for client_id, update_time in self.client_updates.items()
                if (current_time - update_time).total_seconds() < 3600  # 1 hour timeout
            ]
            
            if len(recent_updates) >= self.min_clients:
                self._aggregate_models(recent_updates)
                self.current_round += 1
                self.last_aggregation = current_time
                
                # Clear old updates
                self.client_updates = {}
                
                logger.info(f"✓ Completed federated round {self.current_round}/{self.total_rounds}")
                
        except Exception as e:
            logger.error(f"Model aggregation failed: {e}")
    
    def _aggregate_models(self, client_ids: List[str]):
        """Aggregate models using weighted averaging"""
        try:
            if not client_ids:
                return
            
            # Initialize aggregated model state
            aggregated_state = {}
            total_weight = sum(self.aggregation_weights[cid] for cid in client_ids)
            
            # Normalize weights for available clients
            weights = {
                cid: self.aggregation_weights[cid] / total_weight 
                for cid in client_ids
            }
            
            # Weighted average of model parameters
            for cid in client_ids:
                client_state = self.client_models[cid]
                for key in client_state:
                    if key not in aggregated_state:
                        aggregated_state[key] = torch.zeros_like(client_state[key])
                    aggregated_state[key] += weights[cid] * client_state[key]
            
            self.global_model = aggregated_state
            logger.info(f"✓ Aggregated models from {len(client_ids)} clients")
            
        except Exception as e:
            logger.error(f"Model aggregation failed: {e}")
            raise
    
    def get_global_model(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get current global model state"""
        return self.global_model
    
    def compute_round_metrics(self) -> Dict[str, float]:
        """Compute metrics for current round"""
        try:
            metrics = {
                "round": self.current_round,
                "num_clients": len(self.client_models),
                "avg_loss": np.mean(self.round_metrics["loss"]),
                "avg_accuracy": np.mean(self.round_metrics["accuracy"]),
                "participation_rate": len(self.client_models) / len(self.aggregation_weights)
            }
            return metrics
        except Exception as e:
            logger.error(f"Metrics computation failed: {e}")
            return {}
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current status of federated learning"""
        return {
            "current_round": self.current_round,
            "total_rounds": self.total_rounds,
            "active_clients": len(self.client_models),
            "total_clients": len(self.aggregation_weights),
            "last_aggregation": self.last_aggregation.isoformat() if self.last_aggregation else None,
            "metrics": self.compute_round_metrics()
        }
    
    def add_round_metrics(self, client_id: str, metrics: Dict[str, float]):
        """Add client metrics for current round"""
        try:
            for key, value in metrics.items():
                self.round_metrics[key].append(value)
        except Exception as e:
            logger.error(f"Failed to add metrics for client {client_id}: {e}")
    
    def check_convergence(self) -> bool:
        """Check if federated learning has converged"""
        try:
            if len(self.round_metrics["loss"]) < 3:
                return False
            
            # Check if loss has stabilized
            recent_losses = self.round_metrics["loss"][-3:]
            loss_diff = max(recent_losses) - min(recent_losses)
            
            # Check if accuracy is high enough
            recent_accuracy = np.mean(self.round_metrics["accuracy"][-3:])
            
            return loss_diff < 0.01 and recent_accuracy > 0.95
            
        except Exception as e:
            logger.error(f"Convergence check failed: {e}")
            return False 