"""
Service for aggregating updates in federated learning.
"""
import numpy as np
from typing import List, Dict, Any, Optional

class FederatedAggregationService:
    """Service for aggregating model updates in federated learning."""
    
    def federated_averaging(self, weights_list: List[List[np.ndarray]], sample_counts: List[int]) -> Optional[List[np.ndarray]]:
        """
        Perform federated averaging on client model weights.
        
        Args:
            weights_list: List of client model weights
            sample_counts: List of number of samples used by each client
            
        Returns:
            Averaged weights, or None if aggregation failed
        """
        if not weights_list or not sample_counts:
            return None
        
        total_samples = sum(sample_counts)
        if total_samples == 0:
            return None
        
        # Initialize with zeros
        weighted_weights = [np.zeros_like(w) for w in weights_list[0]]
        
        # Perform weighted average
        for weights, sample_count in zip(weights_list, sample_counts):
            weight_factor = sample_count / total_samples
            for i, w in enumerate(weights):
                weighted_weights[i] += w * weight_factor
        
        return weighted_weights
    
    def median_aggregation(self, weights_list: List[List[np.ndarray]]) -> Optional[List[np.ndarray]]:
        """
        Perform element-wise median aggregation of model weights.
        This is more robust to Byzantine attacks than averaging.
        
        Args:
            weights_list: List of client model weights
            
        Returns:
            Median-aggregated weights, or None if aggregation failed
        """
        if not weights_list:
            return None
        
        # Initialize result array
        median_weights = []
        
        # For each layer's weights
        for layer_idx in range(len(weights_list[0])):
            # Stack weights from all clients for this layer
            stacked_weights = np.stack([client_weights[layer_idx] for client_weights in weights_list])
            
            # Compute median along the client dimension
            layer_median = np.median(stacked_weights, axis=0)
            median_weights.append(layer_median)
        
        return median_weights
    
    def trimmed_mean_aggregation(self, weights_list: List[List[np.ndarray]], trim_ratio: float = 0.1) -> Optional[List[np.ndarray]]:
        """
        Perform element-wise trimmed mean aggregation of model weights.
        This removes the highest and lowest values before averaging.
        
        Args:
            weights_list: List of client model weights
            trim_ratio: Ratio of values to trim from each end
            
        Returns:
            Trimmed-mean-aggregated weights, or None if aggregation failed
        """
        if not weights_list:
            return None
        
        # Must have at least 3 clients for trimming to make sense
        if len(weights_list) < 3:
            # Fall back to regular averaging with equal weights
            return self.federated_averaging(weights_list, [1] * len(weights_list))
        
        # Initialize result array
        trimmed_mean_weights = []
        
        # Number of clients to trim from each end
        n_trim = max(1, int(len(weights_list) * trim_ratio))
        
        # For each layer's weights
        for layer_idx in range(len(weights_list[0])):
            # Stack weights from all clients for this layer
            stacked_weights = np.stack([client_weights[layer_idx] for client_weights in weights_list])
            
            # Sort values along client dimension
            sorted_weights = np.sort(stacked_weights, axis=0)
            
            # Remove highest and lowest values
            trimmed_weights = sorted_weights[n_trim:-n_trim]
            
            # Compute mean of remaining values
            layer_mean = np.mean(trimmed_weights, axis=0)
            trimmed_mean_weights.append(layer_mean)
        
        return trimmed_mean_weights 