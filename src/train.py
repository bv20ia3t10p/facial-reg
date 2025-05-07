"""
Train the facial recognition model with differential privacy.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
import matplotlib.pyplot as plt
import argparse
import time

from config import (
    DATASET_PATH, BATCH_SIZE, EPOCHS, LEARNING_RATE, MODEL_SAVE_PATH,
    DP_NOISE_MULTIPLIER, DP_L2_NORM_CLIP, DP_MICROBATCHES, DP_DELTA
)
from data_utils import create_dataset_index, create_tf_dataset
from model import build_face_recognition_model, build_dp_model

def compute_epsilon(steps, noise_multiplier, batch_size, num_examples):
    """
    Compute the epsilon value for differential privacy.
    
    Args:
        steps: Number of steps taken
        noise_multiplier: Noise multiplier used
        batch_size: Batch size used
        num_examples: Number of examples in the dataset
        
    Returns:
        Epsilon value
    """
    if noise_multiplier == 0.0:
        return float('inf')
    
    # Calculate sampling rate
    sampling_probability = batch_size / num_examples
    
    # Calculate orders for RDP
    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
    
    # Compute RDP
    rdp = compute_rdp(
        q=sampling_probability,
        noise_multiplier=noise_multiplier,
        steps=steps,
        orders=orders
    )
    
    # Convert to (epsilon, delta) pair
    eps, _, _ = get_privacy_spent(orders, rdp, target_delta=DP_DELTA)
    
    return eps

def train_with_differential_privacy(args):
    """
    Train the facial recognition model with differential privacy.
    
    Args:
        args: Command line arguments
    """
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Create output directory
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
    # Create dataset
    print("Creating dataset index...")
    dataset_index = create_dataset_index(args.dataset_path)
    
    # Create TensorFlow datasets
    print("Creating TensorFlow datasets...")
    train_ds, val_ds, num_classes = create_tf_dataset(dataset_index)
    
    # Calculate steps per epoch
    steps_per_epoch = len(dataset_index) * (1 - args.validation_split) // args.batch_size
    
    # Build the model
    print("Building model...")
    if args.use_dp:
        model = build_dp_model(num_classes)
    else:
        model = build_face_recognition_model(num_classes, training=True)
    
    # Create optimizer
    if args.use_dp:
        print(f"Using differential privacy with noise multiplier: {args.noise_multiplier}")
        optimizer = DPKerasSGDOptimizer(
            l2_norm_clip=args.l2_norm_clip,
            noise_multiplier=args.noise_multiplier,
            num_microbatches=args.microbatches,
            learning_rate=args.learning_rate
        )
        
        # Calculate and print privacy budget
        total_steps = steps_per_epoch * args.epochs
        epsilon = compute_epsilon(
            steps=total_steps,
            noise_multiplier=args.noise_multiplier,
            batch_size=args.batch_size,
            num_examples=len(dataset_index) * (1 - args.validation_split)
        )
        print(f"Privacy budget epsilon: {epsilon:.2f} (lower is better)")
    else:
        print("Training without differential privacy")
        optimizer = tf.keras.optimizers.SGD(learning_rate=args.learning_rate, momentum=0.9)
    
    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Learning rate scheduler
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5
    )
    
    # Model checkpoint
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(MODEL_SAVE_PATH, 'face_recognition_model_best.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    
    # Train the model
    print("Starting training...")
    start_time = time.time()
    history = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        callbacks=[lr_scheduler, model_checkpoint]
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Save the final model
    final_model_path = os.path.join(MODEL_SAVE_PATH, 'face_recognition_model_final.h5')
    model.save(final_model_path)
    print(f"Model saved to {final_model_path}")
    
    # Save training history
    history_path = os.path.join(MODEL_SAVE_PATH, 'training_history.npy')
    np.save(history_path, history.history)
    print(f"Training history saved to {history_path}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_SAVE_PATH, 'training_history.png'))
    
    # Evaluate on validation set
    print("Evaluating model on validation set...")
    val_loss, val_accuracy = model.evaluate(val_ds)
    print(f"Validation accuracy: {val_accuracy:.4f}")
    print(f"Validation loss: {val_loss:.4f}")
    
    # Save a summary of training parameters and results
    with open(os.path.join(MODEL_SAVE_PATH, 'training_summary.txt'), 'w') as f:
        f.write(f"Number of classes: {num_classes}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Learning rate: {args.learning_rate}\n")
        f.write(f"Differential privacy: {args.use_dp}\n")
        if args.use_dp:
            f.write(f"DP noise multiplier: {args.noise_multiplier}\n")
            f.write(f"DP L2 norm clip: {args.l2_norm_clip}\n")
            f.write(f"DP microbatches: {args.microbatches}\n")
            f.write(f"DP epsilon: {epsilon:.2f}\n")
            f.write(f"DP delta: {DP_DELTA}\n")
        f.write(f"Training time: {training_time:.2f} seconds\n")
        f.write(f"Final validation accuracy: {val_accuracy:.4f}\n")
        f.write(f"Final validation loss: {val_loss:.4f}\n")
    
    print("Training complete!")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train facial recognition model with differential privacy')
    
    parser.add_argument('--dataset_path', type=str, default=DATASET_PATH,
                        help='Path to the CASIA-WebFace dataset')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help='Initial learning rate')
    parser.add_argument('--validation_split', type=float, default=0.2,
                        help='Fraction of data to use for validation')
    
    # Differential privacy parameters
    parser.add_argument('--use_dp', action='store_true',
                        help='Whether to use differential privacy')
    parser.add_argument('--noise_multiplier', type=float, default=DP_NOISE_MULTIPLIER,
                        help='Noise multiplier for DP-SGD')
    parser.add_argument('--l2_norm_clip', type=float, default=DP_L2_NORM_CLIP,
                        help='L2 norm clipping threshold for DP-SGD')
    parser.add_argument('--microbatches', type=int, default=DP_MICROBATCHES,
                        help='Number of microbatches for DP-SGD')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    train_with_differential_privacy(args) 