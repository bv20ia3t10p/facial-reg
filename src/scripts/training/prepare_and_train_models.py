#!/usr/bin/env python
"""
Script to prepare data and train initial models for federated learning.

This script partitions a dataset for federated learning and trains initial
models for server and clients. It's designed to be run before deploying
the federated learning environment.
"""

import os
import sys
import argparse
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
import shutil

# Ensure we can import from the src folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.data.dataset_handler import FaceDatasetHandler
from src.services.model_service import ModelService
from src.models.standard_face_model import StandardFaceModel
from src.models.dp_face_model import DPFaceModel
from src.config.model_config import MODEL_SAVE_PATH, BATCH_SIZE, LEARNING_RATE

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Prepare data and train initial models for federated learning')
    
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Directory containing the extracted dataset')
    parser.add_argument('--output_dir', type=str, default='data/partitioned',
                        help='Directory to store the partitioned dataset')
    parser.add_argument('--model_dir', type=str, default=MODEL_SAVE_PATH,
                        help='Directory to save the trained models')
    parser.add_argument('--use_dp', action='store_true',
                        help='Use differential privacy during training')
    parser.add_argument('--initial_epochs', type=int, default=5,
                        help='Number of epochs for initial model training')
    parser.add_argument('--server_ratio', type=float, default=0.6,
                        help='Ratio of data for server (default: 0.6)')
    parser.add_argument('--client_ratio', type=float, default=0.2,
                        help='Ratio of data for each client (default: 0.2)')
    parser.add_argument('--add_unseen_classes', action='store_true',
                        help='Add unseen classes to clients')
    parser.add_argument('--num_unseen_classes', type=int, default=5,
                        help='Number of unseen classes to add per client')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help='Learning rate for training')
    parser.add_argument('--img_size', type=int, default=112,
                        help='Image size for training')
    
    return parser.parse_args()

def partition_dataset(dataset_dir, output_dir, server_ratio=0.6, client_ratio=0.2):
    """
    Partition a dataset for federated learning.
    
    Args:
        dataset_dir: Directory containing the dataset
        output_dir: Directory to save the partitioned data
        server_ratio: Fraction of data for the server
        client_ratio: Fraction of data for each client
        
    Returns:
        Dictionary with paths to the partitioned data
    """
    print(f"Partitioning dataset from {dataset_dir} to {output_dir}...")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    server_dir = os.path.join(output_dir, 'server')
    client1_dir = os.path.join(output_dir, 'client1')
    client2_dir = os.path.join(output_dir, 'client2')
    
    for directory in [server_dir, client1_dir, client2_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Get class directories
    class_dirs = [d for d in os.listdir(dataset_dir) 
                 if os.path.isdir(os.path.join(dataset_dir, d))]
    
    print(f"Found {len(class_dirs)} class directories in dataset")
    
    # Shuffle classes for random assignment
    np.random.shuffle(class_dirs)
    
    # Calculate number of classes for each partition
    num_server_classes = int(len(class_dirs) * server_ratio)
    num_client_classes = int(len(class_dirs) * client_ratio)
    
    # Assign classes to partitions
    server_classes = class_dirs[:num_server_classes]
    client1_classes = class_dirs[num_server_classes:num_server_classes + num_client_classes]
    client2_classes = class_dirs[num_server_classes + num_client_classes:num_server_classes + 2*num_client_classes]
    
    # Copy files to partitions
    partitions = [
        (server_classes, server_dir, 'server'),
        (client1_classes, client1_dir, 'client1'),
        (client2_classes, client2_dir, 'client2')
    ]
    
    for classes, target_dir, name in partitions:
        print(f"Copying {len(classes)} classes to {name} partition...")
        for class_dir in classes:
            src_dir = os.path.join(dataset_dir, class_dir)
            dst_dir = os.path.join(target_dir, class_dir)
            
            # Create class directory in destination
            os.makedirs(dst_dir, exist_ok=True)
            
            # Copy all images
            for img_file in os.listdir(src_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    src_path = os.path.join(src_dir, img_file)
                    dst_path = os.path.join(dst_dir, img_file)
                    shutil.copy2(src_path, dst_path)
    
    # Create and return partition info
    partitions = {
        'server': server_dir,
        'client1': client1_dir,
        'client2': client2_dir,
        'server_classes': server_classes,
        'client1_classes': client1_classes,
        'client2_classes': client2_classes
    }
    
    # Save partition summary
    summary = {
        'dataset': {
            'original_dir': os.path.abspath(dataset_dir),
            'partitioned_dir': os.path.abspath(output_dir),
            'server_dir': os.path.abspath(server_dir),
            'client1_dir': os.path.abspath(client1_dir),
            'client2_dir': os.path.abspath(client2_dir)
        },
        'partitions': {
            'server': {
                'num_classes': len(server_classes),
                'classes': server_classes
            },
            'client1': {
                'num_classes': len(client1_classes),
                'classes': client1_classes
            },
            'client2': {
                'num_classes': len(client2_classes),
                'classes': client2_classes
            }
        }
    }
    
    # Save summary to JSON
    summary_path = os.path.join(output_dir, 'partition_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Partition summary saved to {summary_path}")
    
    return partitions

def create_unseen_classes(partitions, dataset_dir, num_unseen_classes=5):
    """
    Add unseen classes to client partitions.
    
    Args:
        partitions: Dictionary with partition information
        dataset_dir: Directory containing the original dataset
        num_unseen_classes: Number of unseen classes to add per client
    """
    print(f"Adding {num_unseen_classes} unseen classes to each client...")
    
    # Get all classes
    all_classes = set([d for d in os.listdir(dataset_dir) 
                      if os.path.isdir(os.path.join(dataset_dir, d))])
    
    # Get classes already used in partitions
    used_classes = set(partitions['server_classes'] + 
                      partitions['client1_classes'] + 
                      partitions['client2_classes'])
    
    # Find unused classes
    unused_classes = list(all_classes - used_classes)
    
    if len(unused_classes) < 2 * num_unseen_classes:
        print(f"Warning: Not enough unused classes ({len(unused_classes)} available, " 
              f"needed {2*num_unseen_classes})")
        num_unseen_classes = len(unused_classes) // 2
    
    # Shuffle unused classes
    np.random.shuffle(unused_classes)
    
    # Assign unseen classes to clients
    client1_unseen = unused_classes[:num_unseen_classes]
    client2_unseen = unused_classes[num_unseen_classes:2*num_unseen_classes]
    
    # Copy unseen classes to client partitions
    for client_name, client_dir, unseen_classes in [
        ('client1', partitions['client1'], client1_unseen),
        ('client2', partitions['client2'], client2_unseen)
    ]:
        print(f"Adding {len(unseen_classes)} unseen classes to {client_name}")
        
        for class_dir in unseen_classes:
            src_dir = os.path.join(dataset_dir, class_dir)
            dst_dir = os.path.join(client_dir, class_dir)
            
            # Create class directory in destination
            os.makedirs(dst_dir, exist_ok=True)
            
            # Copy all images
            for img_file in os.listdir(src_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    src_path = os.path.join(src_dir, img_file)
                    dst_path = os.path.join(dst_dir, img_file)
                    shutil.copy2(src_path, dst_path)
    
    # Update partition information
    partitions['client1_classes'].extend(client1_unseen)
    partitions['client2_classes'].extend(client2_unseen)
    
    # Update partition summary
    summary_path = os.path.join(os.path.dirname(partitions['server']), 'partition_summary.json')
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    summary['partitions']['client1']['num_classes'] = len(partitions['client1_classes'])
    summary['partitions']['client1']['classes'] = partitions['client1_classes']
    summary['partitions']['client1']['unseen_classes'] = list(client1_unseen)
    
    summary['partitions']['client2']['num_classes'] = len(partitions['client2_classes'])
    summary['partitions']['client2']['classes'] = partitions['client2_classes']
    summary['partitions']['client2']['unseen_classes'] = list(client2_unseen)
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Updated partition summary saved to {summary_path}")

def train_initial_model(data_dir, model_dir, client_id, use_dp=False, epochs=5, 
                        batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, img_size=112):
    """
    Train an initial model for a client or server.
    
    Args:
        data_dir: Directory containing the training data
        model_dir: Directory to save the model
        client_id: ID of the client ('server', 'client1', or 'client2')
        use_dp: Whether to use differential privacy
        epochs: Number of epochs for training
        batch_size: Batch size for training
        learning_rate: Learning rate for training
        img_size: Image size for training
        
    Returns:
        Dictionary with training metrics and class information
    """
    print(f"\n=== Training {client_id.capitalize()} Model ===")
    
    # Initialize dataset handler and model service
    dataset_handler = FaceDatasetHandler()
    model_service = ModelService()
    
    # Load and prepare dataset
    dataset_index = dataset_handler.load_dataset(data_dir)
    train_df, val_df = dataset_handler.split_train_val(dataset_index)
    
    # Create TensorFlow datasets
    train_ds = dataset_handler.create_tf_dataset(train_df, batch_size=batch_size, training=True)
    val_ds = dataset_handler.create_tf_dataset(val_df, batch_size=batch_size, training=False)
    
    # Get class information
    class_info = dataset_handler.get_class_info(dataset_index)
    num_classes = class_info['num_classes']
    
    print(f"Dataset has {len(dataset_index)} samples across {num_classes} classes")
    print(f"Training set: {len(train_df)} samples, Validation set: {len(val_df)} samples")
    
    # Build model
    if use_dp:
        model = model_service.build_dp_model(num_classes, img_size=img_size)
    else:
        model = model_service.build_model(num_classes, img_size=img_size)
    
    # Train model
    print(f"Training for {epochs} epochs...")
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=3, restore_best_weights=True
        )
    ]
    
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks
    )
    
    # Evaluate model
    val_loss, val_accuracy = model.evaluate(val_ds)
    print(f"Validation loss: {val_loss:.4f}, accuracy: {val_accuracy:.4f}")
    
    # Save model
    model_path = os.path.join(model_dir, f'{client_id}_initial_model.h5')
    model.save(model_path)
    print(f"{client_id.capitalize()} model saved to {model_path}")
    
    # Save class information
    class_path = os.path.join(model_dir, f'{client_id}_classes.json')
    with open(class_path, 'w') as f:
        json.dump(class_info, f, indent=2)
    print(f"{client_id.capitalize()} class information saved to {class_path}")
    
    # Return training metrics and class information
    return {
        'model_path': model_path,
        'class_path': class_path,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
        'num_classes': num_classes,
        'num_samples': len(dataset_index)
    }

def main():
    """Run the main function to prepare data and train models."""
    args = parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Partition dataset
    partitions = partition_dataset(
        args.dataset_dir, 
        args.output_dir,
        server_ratio=args.server_ratio,
        client_ratio=args.client_ratio
    )
    
    # Add unseen classes if requested
    if args.add_unseen_classes:
        create_unseen_classes(
            partitions,
            args.dataset_dir,
            args.num_unseen_classes
        )
    
    # Train server model
    server_results = train_initial_model(
        partitions['server'],
        args.model_dir,
        'server',
        use_dp=args.use_dp,
        epochs=args.initial_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        img_size=args.img_size
    )
    
    # Train client 1 model
    client1_results = train_initial_model(
        partitions['client1'],
        args.model_dir,
        'client1',
        use_dp=args.use_dp,
        epochs=args.initial_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        img_size=args.img_size
    )
    
    # Train client 2 model
    client2_results = train_initial_model(
        partitions['client2'],
        args.model_dir,
        'client2',
        use_dp=args.use_dp,
        epochs=args.initial_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        img_size=args.img_size
    )
    
    # Create and save training summary
    training_summary = {
        'dataset': {
            'original_dir': os.path.abspath(args.dataset_dir),
            'partitioned_dir': os.path.abspath(args.output_dir),
            'server_dir': os.path.abspath(partitions['server']),
            'client1_dir': os.path.abspath(partitions['client1']),
            'client2_dir': os.path.abspath(partitions['client2'])
        },
        'models': {
            'server': {
                'model_path': server_results['model_path'],
                'class_path': server_results['class_path'],
                'val_loss': server_results['val_loss'],
                'val_accuracy': server_results['val_accuracy'],
                'num_classes': server_results['num_classes'],
                'num_samples': server_results['num_samples']
            },
            'client1': {
                'model_path': client1_results['model_path'],
                'class_path': client1_results['class_path'],
                'val_loss': client1_results['val_loss'],
                'val_accuracy': client1_results['val_accuracy'],
                'num_classes': client1_results['num_classes'],
                'num_samples': client1_results['num_samples']
            },
            'client2': {
                'model_path': client2_results['model_path'],
                'class_path': client2_results['class_path'],
                'val_loss': client2_results['val_loss'],
                'val_accuracy': client2_results['val_accuracy'],
                'num_classes': client2_results['num_classes'],
                'num_samples': client2_results['num_samples']
            }
        },
        'parameters': {
            'use_dp': args.use_dp,
            'initial_epochs': args.initial_epochs,
            'server_ratio': args.server_ratio,
            'client_ratio': args.client_ratio,
            'add_unseen_classes': args.add_unseen_classes,
            'num_unseen_classes': args.num_unseen_classes,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'img_size': args.img_size
        }
    }
    
    # Save training summary
    summary_path = os.path.join(args.model_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(training_summary, f, indent=2)
    
    print(f"\nTraining summary saved to {summary_path}")
    print("\nInitial model training complete!")

if __name__ == "__main__":
    main() 