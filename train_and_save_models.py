#!/usr/bin/env python
"""
Script to train and save server and client models for the federated learning system.
This script should be run before deploying the models in a Docker environment.
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
import json

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

from src.model import build_face_recognition_model, build_dp_model
from src.data_partitioning import partition_dataset, create_client_unseen_classes
from src.data_utils import create_dataset_index
from config import MODEL_SAVE_PATH, BATCH_SIZE, LEARNING_RATE

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train and save models for federated learning')
    
    parser.add_argument('--dataset_dir', type=str, required=True,
                       help='Directory containing the extracted CASIA-WebFace dataset')
    parser.add_argument('--output_dir', type=str, default='data/partitioned',
                       help='Directory to store the partitioned dataset')
    parser.add_argument('--use_dp', action='store_true',
                       help='Use differential privacy during training')
    parser.add_argument('--initial_epochs', type=int, default=5,
                       help='Number of epochs for initial model training')
    parser.add_argument('--add_unseen_classes', action='store_true',
                       help='Add unseen classes to clients')
    parser.add_argument('--num_unseen_classes', type=int, default=10,
                       help='Number of unseen classes to add per client')

    return parser.parse_args()

def create_tf_dataset(data_dir, batch_size=BATCH_SIZE):
    """Create a TensorFlow dataset from a directory of images."""
    # Create dataset index
    dataset_index = create_dataset_index(data_dir)
    num_samples = len(dataset_index)
    
    # Get unique classes
    unique_classes = set(dataset_index['label_idx'].unique())
    num_classes = max(unique_classes) + 1 if unique_classes else 0
    
    print(f"Dataset has {num_samples} samples across {len(unique_classes)} classes")
    
    # Create TF dataset pipeline
    def _parse_function(img_path, label):
        img_str = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img_str, channels=3)
        img = tf.image.resize(img, (112, 112))
        img = tf.cast(img, tf.float32) / 255.0
        
        return img, label
    
    # Create train/val split
    train_indices = np.random.choice(dataset_index.index, size=int(0.8 * num_samples), replace=False)
    val_indices = np.array(list(set(dataset_index.index) - set(train_indices)))
    
    train_df = dataset_index.loc[train_indices]
    val_df = dataset_index.loc[val_indices]
    
    # Create TF datasets
    train_ds = tf.data.Dataset.from_tensor_slices((
        train_df['image_path'].values, 
        train_df['label_idx'].values
    ))
    train_ds = train_ds.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.shuffle(buffer_size=10000)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    val_ds = tf.data.Dataset.from_tensor_slices((
        val_df['image_path'].values, 
        val_df['label_idx'].values
    ))
    val_ds = val_ds.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.batch(batch_size)
    val_ds = val_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    return train_ds, val_ds, num_classes, num_samples, list(unique_classes)

def train_model(model, train_ds, val_ds, epochs=5):
    """Train a model with the given datasets."""
    # Create callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=[early_stopping]
    )
    
    # Evaluate the model
    val_loss, val_accuracy = model.evaluate(val_ds)
    
    print(f"Validation loss: {val_loss:.4f}, accuracy: {val_accuracy:.4f}")
    
    return history, val_loss, val_accuracy

def initialize_model(num_classes, use_dp=False):
    """Initialize and compile a facial recognition model."""
    if use_dp:
        model = build_dp_model(num_classes)
    else:
        model = build_face_recognition_model(num_classes, training=True)
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=0.9),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    
    return model

def save_class_info(classes, filepath):
    """Save class information to a JSON file."""
    class_info = {
        'num_classes': max(classes) + 1 if classes else 0,
        'class_list': sorted(classes)
    }
    
    with open(filepath, 'w') as f:
        json.dump(class_info, f)
    
    print(f"Class information saved to {filepath}")

def main():
    args = parse_args()
    
    # Create model save directory
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
    # Partition the dataset
    print("Partitioning dataset...")
    partitions = partition_dataset(
        args.dataset_dir, 
        args.output_dir,
        server_ratio=0.9,
        client_ratio=0.05
    )
    
    # Add unseen classes if requested
    if args.add_unseen_classes:
        print(f"Adding {args.num_unseen_classes} unseen classes to each client...")
        for client_name in ['client1', 'client2']:
            create_client_unseen_classes(
                args.dataset_dir, 
                partitions[client_name],
                args.num_unseen_classes
            )
    
    # Train server model
    print("\n=== Training Server Model ===")
    train_ds, val_ds, num_classes, num_samples, server_classes = create_tf_dataset(partitions['server'])
    server_model = initialize_model(num_classes, args.use_dp)
    server_history, server_val_loss, server_val_accuracy = train_model(
        server_model, train_ds, val_ds, args.initial_epochs
    )
    
    # Save server model
    server_model_path = os.path.join(MODEL_SAVE_PATH, 'server_initial_model.h5')
    server_model.save(server_model_path)
    print(f"Server model saved to {server_model_path}")
    
    # Save server class information
    save_class_info(server_classes, os.path.join(MODEL_SAVE_PATH, 'server_classes.json'))
    
    # Train client 1 model
    print("\n=== Training Client 1 Model ===")
    train_ds, val_ds, num_classes, num_samples, client1_classes = create_tf_dataset(partitions['client1'])
    client1_model = initialize_model(num_classes, args.use_dp)
    client1_history, client1_val_loss, client1_val_accuracy = train_model(
        client1_model, train_ds, val_ds, args.initial_epochs
    )
    
    # Save client 1 model
    client1_model_path = os.path.join(MODEL_SAVE_PATH, 'client1_initial_model.h5')
    client1_model.save(client1_model_path)
    print(f"Client 1 model saved to {client1_model_path}")
    
    # Save client 1 class information
    save_class_info(client1_classes, os.path.join(MODEL_SAVE_PATH, 'client1_classes.json'))
    
    # Train client 2 model
    print("\n=== Training Client 2 Model ===")
    train_ds, val_ds, num_classes, num_samples, client2_classes = create_tf_dataset(partitions['client2'])
    client2_model = initialize_model(num_classes, args.use_dp)
    client2_history, client2_val_loss, client2_val_accuracy = train_model(
        client2_model, train_ds, val_ds, args.initial_epochs
    )
    
    # Save client 2 model
    client2_model_path = os.path.join(MODEL_SAVE_PATH, 'client2_initial_model.h5')
    client2_model.save(client2_model_path)
    print(f"Client 2 model saved to {client2_model_path}")
    
    # Save client 2 class information
    save_class_info(client2_classes, os.path.join(MODEL_SAVE_PATH, 'client2_classes.json'))
    
    # Save a summary of all models and datasets
    summary = {
        'server': {
            'num_samples': num_samples,
            'num_classes': max(server_classes) + 1 if server_classes else 0,
            'val_accuracy': server_val_accuracy,
            'val_loss': server_val_loss,
            'model_path': server_model_path
        },
        'client1': {
            'num_samples': num_samples,
            'num_classes': max(client1_classes) + 1 if client1_classes else 0,
            'val_accuracy': client1_val_accuracy,
            'val_loss': client1_val_loss,
            'model_path': client1_model_path
        },
        'client2': {
            'num_samples': num_samples,
            'num_classes': max(client2_classes) + 1 if client2_classes else 0,
            'val_accuracy': client2_val_accuracy,
            'val_loss': client2_val_loss,
            'model_path': client2_model_path
        },
        'dataset': {
            'server_dir': partitions['server'],
            'client1_dir': partitions['client1'],
            'client2_dir': partitions['client2'],
            'use_dp': args.use_dp,
            'add_unseen_classes': args.add_unseen_classes,
            'num_unseen_classes': args.num_unseen_classes if args.add_unseen_classes else 0
        }
    }
    
    # Save summary
    summary_path = os.path.join(MODEL_SAVE_PATH, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nTraining summary saved to {summary_path}")
    print("\nTraining complete! Models are ready for federated learning in Docker.")

if __name__ == "__main__":
    main() 