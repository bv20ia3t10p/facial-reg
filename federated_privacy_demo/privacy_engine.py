"""
Privacy Engine for Biometric Data Processing
Implements HE and DP protections
"""

import os
import logging
from typing import Optional
import tenseal as ts
import torch
import numpy as np
from opacus import PrivacyEngine as OpacusPrivacyEngine

logger = logging.getLogger(__name__)

class PrivacyEngine:
    def __init__(self):
        self.context = None
        self.dp_engine = None
        self._setup_homomorphic_context()
    
    async def initialize(self):
        """Initialize privacy components"""
        try:
            await self._setup_homomorphic_context()
            logger.info("Privacy engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize privacy engine: {str(e)}")
            raise

    async def _setup_homomorphic_context(self):
        """Setup HE context with TenSEAL"""
        try:
            # Create TenSEAL context
            context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=8192,
                coeff_mod_bit_sizes=[60, 40, 40, 60]
            )
            context.global_scale = 2**40
            context.generate_galois_keys()
            self.context = context
            
            logger.info("HE context initialized successfully")
        except Exception as e:
            logger.error(f"Failed to setup HE context: {str(e)}")
            raise

    def encrypt_biometric(self, features: torch.Tensor) -> Optional[ts.CKKSTensor]:
        """Encrypt biometric features using HE"""
        try:
            if features is None:
                return None
                
            # Convert to numpy and flatten
            features_np = features.detach().cpu().numpy().flatten()
            
            # Encrypt using TenSEAL
            encrypted = ts.ckks_tensor(self.context, features_np)
            
            return encrypted
        except Exception as e:
            logger.error(f"Failed to encrypt biometric: {str(e)}")
            return None

    def decrypt_biometric(self, encrypted_features: ts.CKKSTensor) -> Optional[np.ndarray]:
        """Decrypt biometric features"""
        try:
            if encrypted_features is None:
                return None
                
            # Decrypt using TenSEAL
            decrypted = encrypted_features.decrypt()
            
            return np.array(decrypted)
        except Exception as e:
            logger.error(f"Failed to decrypt biometric: {str(e)}")
            return None

    def secure_compare(self, 
                      encrypted1: ts.CKKSTensor, 
                      encrypted2: ts.CKKSTensor,
                      threshold: float = 0.7) -> bool:
        """Compare two encrypted biometric features"""
        try:
            if encrypted1 is None or encrypted2 is None:
                return False
                
            # Compute encrypted distance
            diff = encrypted1 - encrypted2
            dist = diff.dot(diff)
            
            # Decrypt distance and compare
            distance = float(dist.decrypt()[0])
            similarity = 1.0 / (1.0 + distance)
            
            return similarity >= threshold
        except Exception as e:
            logger.error(f"Failed to compare biometrics: {str(e)}")
            return False

    def setup_dp_training(self, 
                         model: torch.nn.Module,
                         optimizer: torch.optim.Optimizer,
                         target_epsilon: float = 1.0,
                         target_delta: float = 1e-5) -> None:
        """Setup differential privacy for training"""
        try:
            privacy_engine = OpacusPrivacyEngine()
            
            model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=None,  # Will be set during training
                target_epsilon=target_epsilon,
                target_delta=target_delta,
                max_grad_norm=1.0,
                epochs=1
            )
            
            self.dp_engine = privacy_engine
            logger.info("DP training setup completed successfully")
        except Exception as e:
            logger.error(f"Failed to setup DP training: {str(e)}")
            raise 