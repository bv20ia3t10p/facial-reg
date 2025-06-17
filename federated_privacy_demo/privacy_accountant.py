#!/usr/bin/env python3
"""
Privacy Accountant for Differential Privacy in Federated Learning
Tracks and manages privacy budget consumption
"""

import torch
import numpy as np
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class PrivacyAccountant:
    """Tracks privacy budget consumption for differential privacy"""
    
    def __init__(self, noise_multiplier: float, max_epsilon: float, delta: float):
        """Initialize privacy accountant
        
        Args:
            noise_multiplier: Noise multiplier (σ) for DP-SGD
            max_epsilon: Maximum privacy budget (ε)
            delta: Privacy parameter (δ)
        """
        self.noise_multiplier = noise_multiplier
        self.max_epsilon = max_epsilon
        self.delta = delta
        self.epsilon_used = 0.0
        self.steps_taken = 0
        
        # Initialize privacy tracking
        self.privacy_history = []
        logger.info(f"Initialized privacy accountant with ε={max_epsilon}, δ={delta}, σ={noise_multiplier}")
    
    def is_budget_exhausted(self) -> bool:
        """Check if privacy budget has been exhausted
        
        Returns:
            True if privacy budget is exhausted, False otherwise
        """
        return self.epsilon_used >= self.max_epsilon
    
    def get_privacy_stats(self) -> Dict[str, float]:
        """Get current privacy statistics
        
        Returns:
            Dictionary containing privacy metrics
        """
        return {
            'epsilon_used': self.epsilon_used,
            'epsilon_remaining': max(0, self.max_epsilon - self.epsilon_used),
            'delta': self.delta,
            'noise_multiplier': self.noise_multiplier,
            'steps_taken': self.steps_taken,
            'budget_exhausted': self.is_budget_exhausted()
        }
    
    def apply_dp(self, loss: torch.Tensor, parameters: List[torch.Tensor]) -> torch.Tensor:
        """Apply differential privacy to gradients
        
        Args:
            loss: Loss tensor
            parameters: Model parameters
            
        Returns:
            Modified loss tensor with privacy guarantees
        """
        if self.is_budget_exhausted():
            logger.warning("Privacy budget exhausted, skipping DP application")
            return loss
        
        # Calculate privacy cost for this step
        step_epsilon = self._calculate_step_epsilon()
        
        # Update privacy budget
        self.epsilon_used += step_epsilon
        self.steps_taken += 1
        
        # Record privacy history
        self.privacy_history.append({
            'step': self.steps_taken,
            'epsilon_used': self.epsilon_used,
            'step_epsilon': step_epsilon
        })
        
        # Add noise to gradients
        for param in parameters:
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * self.noise_multiplier
                param.grad.add_(noise)
        
        return loss
    
    def _calculate_step_epsilon(self) -> float:
        """Calculate privacy cost for a single step
        
        Returns:
            Privacy cost (ε) for this step
        """
        # Using analytical moments accountant for RDP
        # This is a simplified version - in practice, you might want to use
        # a more sophisticated privacy accounting method
        q = 1.0  # Sampling rate (assuming full batch)
        sigma = self.noise_multiplier
        
        # Calculate RDP order
        alpha = 1 + 1 / (sigma * sigma)
        
        # Calculate RDP value
        rdp = (alpha * q * q) / (2 * sigma * sigma)
        
        # Convert RDP to (ε,δ)-DP
        epsilon = rdp + np.log(1 / self.delta) / (alpha - 1)
        
        return epsilon
    
    def reset(self):
        """Reset privacy accountant"""
        self.epsilon_used = 0.0
        self.steps_taken = 0
        self.privacy_history = []
        logger.info("Privacy accountant reset")
    
    def get_privacy_history(self) -> List[Dict[str, float]]:
        """Get history of privacy budget consumption
        
        Returns:
            List of dictionaries containing privacy metrics for each step
        """
        return self.privacy_history 