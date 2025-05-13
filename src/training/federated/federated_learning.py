"""
Federated Learning implementation for the privacy-preserving facial recognition system.

This module provides utilities for training the face recognition model in a federated manner,
where multiple clients train locally and the server aggregates the model updates.
"""

import os
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
import collections
import time
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional
import json
import argparse

from config import (
    DATASET_PATH, BATCH_SIZE, EPOCHS, LEARNING_RATE, MODEL_SAVE_PATH,
    DP_NOISE_MULTIPLIER, DP_L2_NORM_CLIP, DP_MICROBATCHES, DP_DELTA,
    EMBEDDING_SIZE
)
from model import build_dp_model, build_face_recognition_model
from data_utils import preprocess_image, create_dataset_index

# Define client data specification
ClientData = collections.namedtuple('ClientData', ['client_id', 'dataset', 'num_samples', 'classes'])

def create_client_data_from_directory(base_dir, client_id, batch_size=BATCH_SIZE):
    """
    Create a client dataset from a directory of images.
    
    Args:
        base_dir: Base directory containing images
        client_id: Unique identifier for the client
        batch_size: Batch size for the dataset
        
    Returns:
        ClientData object containing the dataset and metadata
    """
    print(f"Creating client dataset for client {client_id} from {base_dir}")
    
    # Create dataset index
    dataset_index = create_dataset_index(base_dir)
    
    # Get unique classes
    classes = dataset_index['label_idx'].unique()
    num_classes = len(classes)
    
    # Create TF dataset
    def _parse_function(img_path, label):
        img_str = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img_str, channels=3)
        img = tf.image.resize(img, (112, 112))
        img = tf.cast(img, tf.float32) / 255.0
        return {'x': img, 'y': label}
    
    # Create dataset
    ds = tf.data.Dataset.from_tensor_slices((
        dataset_index['image_path'].values, 
        dataset_index['label_idx'].values
    ))
    ds = ds.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    return ClientData(
        client_id=client_id,
        dataset=ds,
        num_samples=len(dataset_index),
        classes=classes
    )

def create_federated_datasets(data_dir, num_clients=5, samples_per_client=None):
    """
    Create federated datasets by partitioning the data among clients.
    
    Args:
        data_dir: Directory containing the dataset
        num_clients: Number of clients to create
        samples_per_client: Number of samples per client (if None, use all available)
        
    Returns:
        List of ClientData objects
    """
    # Get all person directories
    person_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) 
                 if os.path.isdir(os.path.join(data_dir, d))]
    
    # Distribute person directories among clients (stratified by identity)
    client_person_dirs = [[] for _ in range(num_clients)]
    
    for i, person_dir in enumerate(person_dirs):
        client_idx = i % num_clients
        client_person_dirs[client_idx].append(person_dir)
    
    # Create temporary directories for each client
    client_data_dirs = []
    for i, dirs in enumerate(client_person_dirs):
        client_dir = os.path.join(data_dir, f"client_{i}")
        os.makedirs(client_dir, exist_ok=True)
        
        # Create symbolic links to person directories
        for person_dir in dirs:
            person_name = os.path.basename(person_dir)
            link_path = os.path.join(client_dir, person_name)
            
            # Create symbolic link if it doesn't exist
            if not os.path.exists(link_path):
                if os.name == 'nt':  # Windows
                    # On Windows, we need to copy the directory
                    import shutil
                    shutil.copytree(person_dir, link_path)
                else:  # Unix-like
                    os.symlink(person_dir, link_path)
        
        client_data_dirs.append(client_dir)
    
    # Create client datasets
    client_datasets = []
    for i, client_dir in enumerate(client_data_dirs):
        client_data = create_client_data_from_directory(client_dir, f"client_{i}")
        client_datasets.append(client_data)
        print(f"Client {i} has {client_data.num_samples} samples across {len(client_data.classes)} classes")
    
    return client_datasets

def create_keras_model(num_classes, use_dp=False):
    """
    Create a Keras model for federated learning.
    
    Args:
        num_classes: Number of classes in the dataset
        use_dp: Whether to use differential privacy
        
    Returns:
        Keras model
    """
    if use_dp:
        return build_dp_model(num_classes)
    else:
        return build_face_recognition_model(num_classes, training=True)

def model_fn(num_classes, use_dp=False):
    """
    Model function for TensorFlow Federated.
    
    Args:
        num_classes: Number of output classes
        use_dp: Whether to use differential privacy
        
    Returns:
        A compiled Keras model
    """
    keras_model = create_keras_model(num_classes, use_dp)
    
    # Use DP optimizer if requested
    if use_dp:
        optimizer = DPKerasSGDOptimizer(
            l2_norm_clip=DP_L2_NORM_CLIP,
            noise_multiplier=DP_NOISE_MULTIPLIER,
            num_microbatches=DP_MICROBATCHES,
            learning_rate=LEARNING_RATE
        )
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=0.9)
    
    # Compile the model
    keras_model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    
    return keras_model

def setup_federated_training(client_datasets, num_classes, use_dp=False):
    """
    Set up federated training using TensorFlow Federated.
    
    Args:
        client_datasets: List of ClientData objects
        num_classes: Number of output classes
        use_dp: Whether to use differential privacy
        
    Returns:
        Tuple of (fed_train, evaluate, initial_state)
    """
    # Convert client datasets to TFF datasets
    federated_data = [client.dataset for client in client_datasets]
    
    # Create model function
    def tff_model_fn():
        return tff.learning.from_keras_model(
            keras_model=model_fn(num_classes, use_dp),
            input_spec=federated_data[0].element_spec,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )
    
    # Build TFF iterative process
    iterative_process = tff.learning.build_federated_averaging_process(
        model_fn=tff_model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0)
    )
    
    # Create evaluation function
    evaluate = tff.learning.build_federated_evaluation(model_fn=tff_model_fn)
    
    # Initialize server state
    server_state = iterative_process.initialize()
    
    return iterative_process, evaluate, server_state

def train_federated_model(
    iterative_process, 
    server_state, 
    evaluate_fn, 
    client_datasets, 
    num_rounds=10, 
    fraction_fit=1.0
):
    """
    Train a federated model.
    
    Args:
        iterative_process: TFF iterative process
        server_state: Initial server state
        evaluate_fn: Evaluation function
        client_datasets: List of ClientData objects
        num_rounds: Number of federated rounds
        fraction_fit: Fraction of clients to use for training in each round
        
    Returns:
        Final server state and metrics
    """
    # Setup metrics tracking
    metrics = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    # Setup client selection
    federated_data = [client.dataset for client in client_datasets]
    num_clients = len(client_datasets)
    clients_per_round = max(1, int(num_clients * fraction_fit))
    
    # Train for multiple rounds
    for round_num in range(1, num_rounds + 1):
        print(f"\nRound {round_num}/{num_rounds}")
        
        # Select clients for this round
        client_indices = np.random.choice(
            np.arange(num_clients), 
            size=clients_per_round, 
            replace=False
        )
        selected_clients = [federated_data[i] for i in client_indices]
        
        # Train on selected clients
        start_time = time.time()
        result = iterative_process.next(server_state, selected_clients)
        server_state = result.state
        train_metrics = result.metrics
        
        # Extract and save training metrics
        train_loss = train_metrics['train']['loss']
        train_accuracy = train_metrics['train']['sparse_categorical_accuracy']
        metrics['train_loss'].append(train_loss)
        metrics['train_accuracy'].append(train_accuracy)
        
        # Evaluate on all clients
        evaluation = evaluate_fn(server_state.model, federated_data)
        val_loss = evaluation['loss']
        val_accuracy = evaluation['sparse_categorical_accuracy']
        metrics['val_loss'].append(val_loss)
        metrics['val_accuracy'].append(val_accuracy)
        
        # Print round results
        duration = time.time() - start_time
        print(f"Round {round_num} took {duration:.2f}s")
        print(f"Training loss: {train_loss:.4f}, accuracy: {train_accuracy:.4f}")
        print(f"Validation loss: {val_loss:.4f}, accuracy: {val_accuracy:.4f}")
    
    # Save metrics to file
    metrics_file = os.path.join(MODEL_SAVE_PATH, 'federated_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f)
    
    return server_state, metrics

def extract_keras_model(server_state, num_classes, use_dp=False):
    """
    Extract a Keras model from the federated server state.
    
    Args:
        server_state: TFF server state
        num_classes: Number of output classes
        use_dp: Whether to use differential privacy
        
    Returns:
        Keras model with weights from server state
    """
    # Create a new Keras model
    keras_model = create_keras_model(num_classes, use_dp)
    
    # Get model weights from server state
    model_weights = server_state.model.trainable
    
    # Set weights on the Keras model
    keras_model.set_weights([
        np.array(x) for x in tf.nest.flatten(model_weights)
    ])
    
    return keras_model

def plot_federated_metrics(metrics):
    """
    Plot metrics from federated training.
    
    Args:
        metrics: Dictionary of training metrics
    """
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(metrics['train_accuracy'], label='Train')
    plt.plot(metrics['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(metrics['train_loss'], label='Train')
    plt.plot(metrics['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_SAVE_PATH, 'federated_metrics.png'))
    plt.close()

def run_federated_learning(args):
    """
    Run federated learning on the facial recognition dataset.
    
    Args:
        args: Command line arguments
    """
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Create output directory
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
    # Create federated datasets
    print(f"Creating federated datasets from {args.data_dir} with {args.num_clients} clients")
    client_datasets = create_federated_datasets(
        args.data_dir, 
        num_clients=args.num_clients
    )
    
    # Determine number of classes (use all classes from all clients)
    all_classes = set()
    for client in client_datasets:
        all_classes.update(client.classes)
    num_classes = len(all_classes)
    print(f"Total number of classes across all clients: {num_classes}")
    
    # Setup federated training
    print("Setting up federated training...")
    iterative_process, evaluate_fn, server_state = setup_federated_training(
        client_datasets, 
        num_classes, 
        use_dp=args.use_dp
    )
    
    # Train federated model
    print(f"Training federated model for {args.num_rounds} rounds...")
    server_state, metrics = train_federated_model(
        iterative_process,
        server_state,
        evaluate_fn,
        client_datasets,
        num_rounds=args.num_rounds,
        fraction_fit=args.fraction_fit
    )
    
    # Plot training metrics
    plot_federated_metrics(metrics)
    
    # Extract Keras model
    print("Extracting final Keras model...")
    final_model = extract_keras_model(server_state, num_classes, use_dp=args.use_dp)
    
    # Save the final model
    final_model_path = os.path.join(MODEL_SAVE_PATH, 'federated_model.h5')
    final_model.save(final_model_path)
    print(f"Model saved to {final_model_path}")
    
    # Save training summary
    summary_path = os.path.join(MODEL_SAVE_PATH, 'federated_training_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Number of clients: {args.num_clients}\n")
        f.write(f"Number of rounds: {args.num_rounds}\n")
        f.write(f"Fraction of clients per round: {args.fraction_fit}\n")
        f.write(f"Number of classes: {num_classes}\n")
        f.write(f"Differential privacy: {args.use_dp}\n")
        if args.use_dp:
            f.write(f"DP noise multiplier: {DP_NOISE_MULTIPLIER}\n")
            f.write(f"DP L2 norm clip: {DP_L2_NORM_CLIP}\n")
            f.write(f"DP microbatches: {DP_MICROBATCHES}\n")
        f.write(f"Final validation accuracy: {metrics['val_accuracy'][-1]:.4f}\n")
    
    print("Federated training complete!")

def parse_args():
    """Parse command line arguments for federated learning."""
    parser = argparse.ArgumentParser(description='Federated learning for facial recognition system')
    
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing the extracted dataset')
    parser.add_argument('--num_clients', type=int, default=5,
                        help='Number of federated clients')
    parser.add_argument('--num_rounds', type=int, default=10,
                        help='Number of federated training rounds')
    parser.add_argument('--fraction_fit', type=float, default=1.0,
                        help='Fraction of clients to use in each round')
    parser.add_argument('--use_dp', action='store_true',
                        help='Whether to use differential privacy')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    run_federated_learning(args) 