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

logger = logging.getLogger(__name__)

class PrivacyEngine:
    """Privacy Engine implementing DP and HE for biometric data"""
    
    def __init__(self):
        self.dp_engine = None
        self.context = None  # HE context
        self._context_initialized = False
        
        # Get privacy parameters from environment
        self.epsilon = float(os.getenv('MAX_DP_EPSILON', '100.0'))
        self.delta = float(os.getenv('DP_DELTA', '1e-5'))
        self.max_grad_norm = float(os.getenv('DP_MAX_GRAD_NORM', '5.0'))
        self.noise_multiplier = float(os.getenv('DP_NOISE_MULTIPLIER', '0.2'))
        
        logger.info(f"Privacy parameters: epsilon={self.epsilon}, delta={self.delta}, "
                   f"max_grad_norm={self.max_grad_norm}, noise_multiplier={self.noise_multiplier}")
    
    async def initialize(self):
        """Initialize privacy components"""
        logger.info("Initializing Privacy Engine...")
        
        # Initialize new HE context
        try:
            # Create TenSEAL context for homomorphic encryption
            self.context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=8192,
                coeff_mod_bit_sizes=[60, 40, 40, 60]
            )
            self.context.global_scale = 2**40
            self.context.generate_galois_keys()
            
            self._context_initialized = True
            logger.info("✓ Created new HE context")
        except Exception as e:
            logger.warning(f"Failed to initialize HE context: {e}, proceeding without encryption")
            self.context = None
            self._context_initialized = False
    
    def _ensure_context(self) -> bool:
        """Ensure HE context is initialized"""
        if not self._context_initialized or self.context is None:
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
            encrypted = ts.ckks_tensor(self.context, features_np)
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
                "active": self.context is not None,
                "scheme": "CKKS",
                "poly_modulus_degree": 8192 if self.context else None
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