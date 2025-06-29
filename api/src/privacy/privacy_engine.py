"""
Privacy Engine for Differential Privacy and Homomorphic Encryption
"""

import torch
from opacus import PrivacyEngine as OpacusPrivacyEngine
from opacus.validators import ModuleValidator
import tenseal as ts
import logging
from typing import Dict, Any, Optional, Union
import numpy as np
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Suppress specific Opacus validator info messages
logging.getLogger('opacus.validators.batch_norm').setLevel(logging.WARNING)
logging.getLogger('opacus.validators.module_validator').setLevel(logging.WARNING)

class PrivacyEngine:
    """Privacy Engine implementing DP and HE for biometric data"""
    
    def __init__(self):
        """Initialize privacy engine"""
        self.dp_engine = None
        self.he_context = None
        self.initialized = False
        
        # Get privacy parameters from environment
        self.epsilon = float(os.getenv('MAX_DP_EPSILON', '100.0'))
        self.delta = float(os.getenv('DP_DELTA', '1e-5'))
        self.max_grad_norm = float(os.getenv('DP_MAX_GRAD_NORM', '5.0'))
        self.noise_multiplier = float(os.getenv('DP_NOISE_MULTIPLIER', '0.2'))
        self.enable_dp = os.getenv('ENABLE_DP', 'true').lower() == 'true'
        self.max_epsilon = self.epsilon
        self.current_epsilon = 0.0
        
        logger.info(f"Privacy parameters: epsilon={self.epsilon}, delta={self.delta}, "
                   f"max_grad_norm={self.max_grad_norm}, noise_multiplier={self.noise_multiplier}")
    
    def _create_new_context(self) -> ts.Context:
        """Create a new HE context"""
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        context.global_scale = 2**40
        return context
    
    async def initialize(self):
        """Initialize privacy components"""
        try:
            # Create HE context if not exists
            context_path = Path("/app/api/data/he_context.pkl")
            context_loaded = False
            
            if context_path.exists():
                try:
                    with open(context_path, 'rb') as f:
                        # Use TenSEAL's load_context instead of pickle
                        serialized_context = f.read()
                        self.he_context = ts.Context.load(serialized_context)
                    logger.info("✓ Loaded existing HE context")
                    context_loaded = True
                except (EOFError, Exception) as e:
                    logger.warning(f"Failed to load existing HE context: {e}. Will create new one.")
            
            if not context_loaded:
                # Create new context
                self.he_context = self._create_new_context()
                
                # Save context using TenSEAL's serialization
                context_path.parent.mkdir(parents=True, exist_ok=True)
                with open(context_path, 'wb') as f:
                    # Use TenSEAL's serialize instead of pickle
                    serialized_context = self.he_context.serialize()
                    f.write(serialized_context)
                logger.info("✓ Created and saved new HE context")
            
            self.initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize privacy engine: {e}")
            raise
    
    def _ensure_context(self) -> bool:
        """Ensure HE context is initialized"""
        if not self.initialized or self.he_context is None:
            logger.warning("HE context not initialized")
            return False
        return True
    
    def encrypt_biometric(self, features: torch.Tensor) -> Union[ts.CKKSTensor, bytes]:
        """Encrypt biometric features using HE"""
        if not self._ensure_context():
            # Return raw features if encryption is not available
            return features.detach().cpu().numpy().tobytes()
        
        try:
            # Convert to numpy and flatten
            features_np = features.detach().cpu().numpy().flatten()
            
            # Encrypt using CKKS
            encrypted = ts.ckks_tensor(self.he_context, features_np)
            return encrypted
        except Exception as e:
            logger.warning(f"Encryption failed: {e}, using raw features")
            # Return raw features if encryption fails
            return features.detach().cpu().numpy().tobytes()
    
    def decrypt_biometric(self, encrypted_features: Union[ts.CKKSTensor, bytes]) -> torch.Tensor:
        """Decrypt biometric features"""
        if not self._ensure_context() or not isinstance(encrypted_features, ts.CKKSTensor):
            # If not encrypted or context not available, treat as raw bytes
            try:
                features_np = np.frombuffer(encrypted_features, dtype=np.float32)
                return torch.from_numpy(features_np)
            except Exception as e:
                logger.error(f"Failed to decode raw features: {e}")
                raise
        
        try:
            # Decrypt CKKS tensor
            decrypted_np = encrypted_features.decrypt()
            
            # Convert back to torch tensor
            return torch.from_numpy(decrypted_np)
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
    
    def setup_differential_privacy(
        self, 
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        data_loader: torch.utils.data.DataLoader,
        **kwargs
    ):
        """Setup differential privacy for training"""
        try:
            # Validate and fix model for DP
            if not ModuleValidator.is_valid(model):
                model = ModuleValidator.fix(model)
            
            # Create DP engine
            self.dp_engine = OpacusPrivacyEngine()
            
            # Make model private with demonstration parameters
            model, optimizer, data_loader = self.dp_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=data_loader,
                target_epsilon=self.epsilon,
                target_delta=self.delta,
                max_grad_norm=self.max_grad_norm,
                epochs=kwargs.get('epochs', 1),
                noise_multiplier=self.noise_multiplier
            )
            
            logger.info(f"✓ Differential privacy setup complete with: "
                       f"epsilon={self.epsilon}, delta={self.delta}, "
                       f"max_grad_norm={self.max_grad_norm}, "
                       f"noise_multiplier={self.noise_multiplier}")
            return model, optimizer, data_loader
            
        except Exception as e:
            logger.error(f"DP setup failed: {e}")
            raise
    
    def compute_privacy_spent(self) -> Dict[str, float]:
        """Compute current privacy budget spent"""
        try:
            if self.dp_engine is None:
                return {"epsilon": 0.0, "delta": self.delta}
            
            epsilon = self.dp_engine.get_epsilon(self.delta)
            return {
                "epsilon": epsilon,
                "delta": self.delta,
                "budget_remaining": max(0.0, self.epsilon - epsilon)
            }
        except Exception as e:
            logger.error(f"Privacy budget computation failed: {e}")
            return {"error": str(e)}
    
    def add_noise_to_gradients(self, gradients: torch.Tensor) -> torch.Tensor:
        """Add calibrated noise to gradients for DP-SGD"""
        try:
            if not self._ensure_context():
                # Return original gradients if privacy not available
                return gradients
            
            noise_scale = self.noise_multiplier * self.max_grad_norm
            noise = torch.normal(
                mean=0.0,
                std=noise_scale,
                size=gradients.shape,
                device=gradients.device
            )
            return gradients + noise
        except Exception as e:
            logger.warning(f"Gradient noise addition failed: {e}, using original gradients")
            # Return original gradients if noise addition fails
            return gradients
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current status of privacy components"""
        return {
            "homomorphic_encryption": {
                "active": self.he_context is not None,
                "scheme": "CKKS",
                "poly_modulus_degree": 8192 if self.he_context else None
            },
            "differential_privacy": {
                "active": self.dp_engine is not None,
                "privacy_budget": await self.get_privacy_budget()
            }
        }
    
    async def get_privacy_budget(self) -> Dict[str, float]:
        """Get current privacy budget status"""
        try:
            privacy_spent = self.compute_privacy_spent()
            return {
                "total_budget": self.epsilon,
                "spent": privacy_spent.get("epsilon", 0.0),
                "remaining": privacy_spent.get("budget_remaining", self.epsilon)
            }
        except Exception as e:
            logger.error(f"Privacy budget status check failed: {e}")
            return {"error": str(e)}
    
    def get_privacy_params(self) -> dict:
        """Get the current privacy parameters"""
        return {
            'epsilon': self.epsilon,
            'delta': self.delta,
            'max_grad_norm': self.max_grad_norm,
            'noise_multiplier': self.noise_multiplier
        }
        
    async def get_remaining_budget(self) -> float:
        """Get the remaining privacy budget"""
        # If privacy is not enabled, return a high value
        if not self.enable_dp:
            return float('inf')
            
        # If we're tracking epsilon, return the difference between max and current
        if hasattr(self, 'current_epsilon') and self.max_epsilon:
            return max(0.0, self.max_epsilon - self.current_epsilon)
            
        # Default to returning max epsilon if we're not tracking usage
        return self.max_epsilon or 100.0 