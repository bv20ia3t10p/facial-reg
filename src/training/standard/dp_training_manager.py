"""
Differential privacy enhanced training manager.
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
from typing import Dict, Tuple, Optional, Any

from src.training.standard.training_manager import StandardTrainingManager
from src.services.model_service import ModelService
from src.data.dataset_handler import FaceDatasetHandler
from src.config.model_config import LEARNING_RATE, MODEL_SAVE_PATH
from src.config.privacy_config import (
    DP_NOISE_MULTIPLIER, DP_L2_NORM_CLIP, DP_MICROBATCHES, DP_DELTA
)

class DPTrainingManager(StandardTrainingManager):
    """
    Manager for training face recognition models with differential privacy.
    """
    
    def __init__(self,
                model_service: ModelService,
                dataset_handler: FaceDatasetHandler,
                output_dir: str = MODEL_SAVE_PATH,
                noise_multiplier: float = DP_NOISE_MULTIPLIER,
                l2_norm_clip: float = DP_L2_NORM_CLIP,
                microbatches: int = DP_MICROBATCHES):
        """
        Initialize the DP training manager.
        
        Args:
            model_service: Service for managing the model
            dataset_handler: Handler for dataset operations
            output_dir: Directory to save training outputs
            noise_multiplier: Noise multiplier for differential privacy
            l2_norm_clip: L2 norm clipping threshold
            microbatches: Number of microbatches
        """
        super().__init__(model_service, dataset_handler, output_dir)
        self.noise_multiplier = noise_multiplier
        self.l2_norm_clip = l2_norm_clip
        self.microbatches = microbatches
        self.epsilon = None
    
    def _compute_epsilon(self, steps: int, num_examples: int, batch_size: int) -> float:
        """
        Compute the epsilon value for differential privacy.
        
        Args:
            steps: Number of steps taken
            num_examples: Number of examples in the dataset
            batch_size: Batch size used
            
        Returns:
            Epsilon value
        """
        if self.noise_multiplier == 0.0:
            return float('inf')
        
        # Calculate sampling rate
        sampling_probability = batch_size / num_examples
        
        # Calculate orders for RDP
        orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
        
        # Compute RDP
        rdp = compute_rdp(
            q=sampling_probability,
            noise_multiplier=self.noise_multiplier,
            steps=steps,
            orders=orders
        )
        
        # Convert to (epsilon, delta) pair
        eps, _, _ = get_privacy_spent(orders, rdp, target_delta=DP_DELTA)
        
        return eps
    
    def build_model(self, num_classes: int) -> tf.keras.Model:
        """
        Build the model for DP training.
        
        Args:
            num_classes: Number of identity classes
            
        Returns:
            Built model with DP optimization
        """
        # Create a DP model which is typically lighter than standard models
        model = self.model_service.build_model(num_classes, training=True)
        self.model = model
        return model
    
    def compile_model(self, 
                    num_examples: int, 
                    batch_size: int, 
                    epochs: int, 
                    learning_rate: float = LEARNING_RATE) -> None:
        """
        Compile the model with differential privacy optimizer.
        
        Args:
            num_examples: Number of examples in the training set
            batch_size: Batch size for training
            epochs: Number of epochs to train
            learning_rate: Learning rate for optimization
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Calculate steps per epoch
        steps_per_epoch = num_examples // batch_size
        total_steps = steps_per_epoch * epochs
        
        # Create DP optimizer
        optimizer = DPKerasSGDOptimizer(
            l2_norm_clip=self.l2_norm_clip,
            noise_multiplier=self.noise_multiplier,
            num_microbatches=self.microbatches,
            learning_rate=learning_rate
        )
        
        # Calculate privacy budget
        self.epsilon = self._compute_epsilon(
            steps=total_steps,
            num_examples=num_examples,
            batch_size=batch_size
        )
        print(f"Privacy budget epsilon: {self.epsilon:.2f} (lower is better)")
        
        # Compile the model
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, 
             train_ds: tf.data.Dataset, 
             val_ds: tf.data.Dataset, 
             epochs: int) -> Dict[str, list]:
        """
        Train the model with differential privacy.
        
        Args:
            train_ds: Training dataset
            val_ds: Validation dataset
            epochs: Number of epochs to train
            
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Apply DP optimizer if not already applied
        if not hasattr(self.model.optimizer, 'noise_multiplier'):
            raise ValueError("Model not compiled with DP optimizer. Call compile_model() first.")
        
        # Create callbacks
        callbacks = self.create_callbacks()
        
        # Train the model
        print("Starting training with differential privacy...")
        return super().train(train_ds, val_ds, epochs)
    
    def save_training_results(self, 
                             num_classes: int, 
                             batch_size: int, 
                             epochs: int, 
                             training_time: float,
                             val_metrics: Dict[str, float]) -> None:
        """
        Save training results with differential privacy information.
        
        Args:
            num_classes: Number of classes in the dataset
            batch_size: Batch size used for training
            epochs: Number of epochs trained
            training_time: Time taken for training in seconds
            val_metrics: Validation metrics (loss, accuracy)
        """
        super().save_training_results(num_classes, batch_size, epochs, training_time, val_metrics)
        
        # Append DP-specific information to the summary
        with open(os.path.join(self.output_dir, 'training_summary.txt'), 'a') as f:
            f.write("\nDifferential Privacy Parameters:\n")
            f.write(f"Noise multiplier: {self.noise_multiplier}\n")
            f.write(f"L2 norm clip: {self.l2_norm_clip}\n")
            f.write(f"Microbatches: {self.microbatches}\n")
            f.write(f"Privacy budget epsilon: {self.epsilon:.2f}\n")
            f.write(f"Privacy budget delta: {DP_DELTA}\n") 