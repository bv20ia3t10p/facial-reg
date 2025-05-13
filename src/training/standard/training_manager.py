"""
Standard supervised learning training manager.
"""
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from typing import Dict, Tuple, Optional, Any

from src.services.model_service import ModelService
from src.data.dataset_handler import FaceDatasetHandler
from src.config.model_config import LEARNING_RATE, MODEL_SAVE_PATH, EPOCHS

class StandardTrainingManager:
    """
    Manager for standard supervised training of face recognition models.
    """
    
    def __init__(self, 
                model_service: ModelService,
                dataset_handler: FaceDatasetHandler,
                output_dir: str = MODEL_SAVE_PATH):
        """
        Initialize the training manager.
        
        Args:
            model_service: Service for managing the model
            dataset_handler: Handler for dataset operations
            output_dir: Directory to save training outputs
        """
        self.model_service = model_service
        self.dataset_handler = dataset_handler
        self.output_dir = output_dir
        self.model = None
        self.history = None
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def prepare_data(self, 
                    data_dir: str, 
                    batch_size: int = 64,
                    validation_split: float = 0.2) -> Tuple[tf.data.Dataset, tf.data.Dataset, int]:
        """
        Prepare datasets for training.
        
        Args:
            data_dir: Directory containing the dataset
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            
        Returns:
            Tuple of (train_dataset, val_dataset, num_classes)
        """
        # Load dataset
        dataset_index = self.dataset_handler.load_dataset(data_dir)
        
        # Split into train/val
        train_df, val_df = self.dataset_handler.split_train_val(
            dataset_index, 
            val_split=validation_split
        )
        
        # Create TensorFlow datasets
        train_ds = self.dataset_handler.create_tf_dataset(
            train_df, 
            batch_size=batch_size, 
            training=True
        )
        
        val_ds = self.dataset_handler.create_tf_dataset(
            val_df, 
            batch_size=batch_size, 
            training=False
        )
        
        # Get number of classes
        num_classes = len(dataset_index['label_idx'].unique())
        
        return train_ds, val_ds, num_classes
    
    def build_model(self, num_classes: int) -> tf.keras.Model:
        """
        Build the model for training.
        
        Args:
            num_classes: Number of identity classes
            
        Returns:
            Built model
        """
        model = self.model_service.build_model(num_classes, training=True)
        self.model = model
        return model
    
    def create_callbacks(self) -> list:
        """
        Create callbacks for training.
        
        Returns:
            List of callbacks
        """
        # Learning rate scheduler
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=5, 
            min_lr=1e-5
        )
        
        # Model checkpoint
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(self.output_dir, 'face_recognition_model_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        
        return [lr_scheduler, model_checkpoint]
    
    def train(self, 
             train_ds: tf.data.Dataset, 
             val_ds: tf.data.Dataset, 
             epochs: int = EPOCHS) -> Dict[str, list]:
        """
        Train the model.
        
        Args:
            train_ds: Training dataset
            val_ds: Validation dataset
            epochs: Number of epochs to train
            
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Create callbacks
        callbacks = self.create_callbacks()
        
        # Train the model
        print("Starting training...")
        start_time = time.time()
        
        history = self.model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=callbacks
        )
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        self.history = history.history
        return history.history
    
    def save_model(self) -> str:
        """
        Save the trained model.
        
        Returns:
            Path where the model was saved
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Save the final model
        final_model_path = os.path.join(self.output_dir, 'face_recognition_model_final.h5')
        self.model.save(final_model_path)
        print(f"Model saved to {final_model_path}")
        
        return final_model_path
    
    def save_training_results(self, 
                             num_classes: int, 
                             batch_size: int, 
                             epochs: int, 
                             training_time: float,
                             val_metrics: Dict[str, float]) -> None:
        """
        Save training results, including history and summary.
        
        Args:
            num_classes: Number of classes in the dataset
            batch_size: Batch size used for training
            epochs: Number of epochs trained
            training_time: Time taken for training in seconds
            val_metrics: Validation metrics (loss, accuracy)
        """
        if self.history is None:
            raise ValueError("No training history. Call train() first.")
        
        # Save training history
        history_path = os.path.join(self.output_dir, 'training_history.npy')
        np.save(history_path, self.history)
        print(f"Training history saved to {history_path}")
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history['accuracy'])
        plt.plot(self.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history['loss'])
        plt.plot(self.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_history.png'))
        
        # Save a summary of training parameters and results
        with open(os.path.join(self.output_dir, 'training_summary.txt'), 'w') as f:
            f.write(f"Number of classes: {num_classes}\n")
            f.write(f"Batch size: {batch_size}\n")
            f.write(f"Epochs: {epochs}\n")
            f.write(f"Learning rate: {LEARNING_RATE}\n")
            f.write(f"Training time: {training_time:.2f} seconds\n")
            f.write(f"Final validation accuracy: {val_metrics['accuracy']:.4f}\n")
            f.write(f"Final validation loss: {val_metrics['loss']:.4f}\n") 