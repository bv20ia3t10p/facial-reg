"""
Federated Learning Client for the privacy-preserving facial recognition system.

This module implements a client that participates in federated learning and can handle
non-existing classes without affecting the server.
"""

import os
import numpy as np
import tensorflow as tf
import json
import argparse
import time
import requests
import pickle
import threading
import sys
from pathlib import Path
import random

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    MODEL_SAVE_PATH, BATCH_SIZE, LEARNING_RATE, EMBEDDING_SIZE,
    DP_NOISE_MULTIPLIER, DP_L2_NORM_CLIP, DP_MICROBATCHES, DP_DELTA
)
from src.model import build_face_recognition_model, build_dp_model
from src.data_utils import create_dataset_index, preprocess_image

# Client configuration
DEFAULT_SERVER_HOST = 'localhost'
DEFAULT_SERVER_PORT = 8080
CLIENT_MODEL_PATH = os.path.join(MODEL_SAVE_PATH, 'client_model.h5')
MAX_RETRY_ATTEMPTS = 5
RETRY_DELAY_SECONDS = 5
EVALUATION_STEPS = 10

class ClientState:
    """Client state for federated learning."""
    
    def __init__(self, client_id, data_dir, allow_new_classes=False):
        self.client_id = client_id
        self.data_dir = data_dir
        self.allow_new_classes = allow_new_classes
        self.model = None
        self.local_classes = set()
        self.global_classes = set()
        self.num_samples = 0
        self.current_round = 0
        self.use_dp = False
        self.is_training = False
        self.train_dataset = None
        self.val_dataset = None
        self.class_mapping = {}  # Maps local class indices to global indices
        self.reverse_mapping = {}  # Maps global class indices to local indices
        
    def load_data(self):
        """Load and preprocess the client's local dataset."""
        print(f"Loading data from {self.data_dir}")
        
        # Create dataset index
        dataset_index = create_dataset_index(self.data_dir)
        self.num_samples = len(dataset_index)
        
        # Get unique classes
        self.local_classes = set(dataset_index['label_idx'].unique())
        print(f"Local dataset has {self.num_samples} samples across {len(self.local_classes)} classes")
        
        # Create TF dataset
        def _parse_function(img_path, label):
            img_str = tf.io.read_file(img_path)
            img = tf.image.decode_jpeg(img_str, channels=3)
            img = tf.image.resize(img, (112, 112))
            img = tf.cast(img, tf.float32) / 255.0
            
            # Apply class mapping if necessary
            if self.class_mapping and label.numpy() in self.class_mapping:
                mapped_label = self.class_mapping[label.numpy()]
                return {'x': img, 'y': tf.constant(mapped_label, dtype=tf.int64)}
            
            return {'x': img, 'y': label}
        
        # Create train/val split
        train_indices, val_indices = train_test_split(dataset_index.index)
        train_df = dataset_index.loc[train_indices]
        val_df = dataset_index.loc[val_indices]
        
        # Create TF datasets
        train_ds = tf.data.Dataset.from_tensor_slices((
            train_df['image_path'].values, 
            train_df['label_idx'].values
        ))
        train_ds = train_ds.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_ds = train_ds.shuffle(buffer_size=10000)
        train_ds = train_ds.batch(BATCH_SIZE)
        train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        
        val_ds = tf.data.Dataset.from_tensor_slices((
            val_df['image_path'].values, 
            val_df['label_idx'].values
        ))
        val_ds = val_ds.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        val_ds = val_ds.batch(BATCH_SIZE)
        val_ds = val_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        
        self.train_dataset = train_ds
        self.val_dataset = val_ds
        
        return train_ds, val_ds
    
    def update_class_mapping(self, max_global_classes):
        """
        Update the class mapping between local and global class indices.
        
        Args:
            max_global_classes: Maximum number of classes in the global model
        """
        if not self.allow_new_classes:
            # No remapping needed if we don't handle new classes
            return
        
        print(f"Updating class mapping (local classes: {self.local_classes}, max global: {max_global_classes})")
        
        # Initialize mappings
        self.class_mapping = {}
        self.reverse_mapping = {}
        
        # For existing classes, keep the same index
        for local_class in self.local_classes:
            if local_class < max_global_classes:
                # This class exists in the global model
                self.class_mapping[local_class] = local_class
                self.reverse_mapping[local_class] = local_class
        
        # For new classes, assign them indices beyond the global max
        new_global_idx = max_global_classes
        for local_class in sorted(self.local_classes):
            if local_class not in self.class_mapping:
                # This is a new class not in the global model
                self.class_mapping[local_class] = new_global_idx
                self.reverse_mapping[new_global_idx] = local_class
                new_global_idx += 1
        
        print(f"Class mapping updated: {self.class_mapping}")
    
    def train_local_model(self, epochs=1):
        """
        Train the local model for one federated round.
        
        Args:
            epochs: Number of local epochs to train
            
        Returns:
            Dictionary with training metrics
        """
        if self.model is None:
            print("Model not initialized. Cannot train.")
            return None
        
        self.is_training = True
        
        # Train for specified number of epochs
        history = self.model.fit(
            self.train_dataset,
            epochs=epochs,
            validation_data=self.val_dataset,
            verbose=1
        )
        
        # Extract metrics
        train_loss = history.history['loss'][-1]
        train_accuracy = history.history['sparse_categorical_accuracy'][-1]
        val_loss = history.history['val_loss'][-1]
        val_accuracy = history.history['val_sparse_categorical_accuracy'][-1]
        
        metrics = {
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        }
        
        self.is_training = False
        
        return metrics
    
    def evaluate_model(self):
        """
        Evaluate the local model on the validation dataset.
        
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None or self.val_dataset is None:
            print("Model or validation dataset not initialized. Cannot evaluate.")
            return None
        
        # Evaluate the model
        loss, accuracy = self.model.evaluate(self.val_dataset, verbose=0)
        
        metrics = {
            'val_loss': loss,
            'val_accuracy': accuracy
        }
        
        return metrics
    
    def apply_global_weights(self, global_weights, num_global_classes):
        """
        Apply global model weights to the local model, handling class dimension mismatches.
        
        Args:
            global_weights: List of weight arrays from the global model
            num_global_classes: Number of classes in the global model
            
        Returns:
            True if successful, False otherwise
        """
        if self.model is None:
            print("Local model not initialized. Cannot apply global weights.")
            return False
        
        try:
            # Get the current local weights
            local_weights = self.model.get_weights()
            
            # Check if the architecture is compatible (except for the output layer)
            if len(global_weights) != len(local_weights):
                print("Weight structure mismatch. Cannot apply global weights.")
                return False
            
            # Handle differently for the final classification layer if there's a class mismatch
            num_local_classes = max(self.local_classes) + 1 if self.local_classes else 0
            
            # Apply weights for all layers except the final classification layer
            new_weights = []
            for i, (global_w, local_w) in enumerate(zip(global_weights, local_weights)):
                # Check if this is the final classification layer weight or bias
                if i == len(global_weights) - 2:  # Final dense layer weights
                    # Keep embeddings layer weights the same
                    if global_w.shape[0] != local_w.shape[0]:
                        print(f"Embedding dimension mismatch: global {global_w.shape[0]} vs local {local_w.shape[0]}")
                        return False
                    
                    # If local model has more classes, initialize the extra weights
                    if self.allow_new_classes and num_local_classes > num_global_classes:
                        # For the weights connected to existing classes, use global weights
                        new_w = np.zeros(local_w.shape, dtype=local_w.dtype)
                        new_w[:, :num_global_classes] = global_w
                        
                        # Initialize weights for new classes with small random values
                        if num_local_classes > num_global_classes:
                            # Apply random initialization for new class weights
                            new_w[:, num_global_classes:] = np.random.normal(
                                0, 0.01, 
                                size=(global_w.shape[0], num_local_classes - num_global_classes)
                            ).astype(local_w.dtype)
                        
                        new_weights.append(new_w)
                    else:
                        # If no new classes, just use global weights
                        new_weights.append(global_w)
                        
                elif i == len(global_weights) - 1:  # Final dense layer bias
                    # If local model has more classes, initialize the extra biases
                    if self.allow_new_classes and num_local_classes > num_global_classes:
                        new_b = np.zeros(local_w.shape, dtype=local_w.dtype)
                        new_b[:num_global_classes] = global_w
                        
                        # Initialize biases for new classes with zeros
                        if num_local_classes > num_global_classes:
                            new_b[num_global_classes:] = np.zeros(num_local_classes - num_global_classes, dtype=local_w.dtype)
                        
                        new_weights.append(new_b)
                    else:
                        # If no new classes, just use global biases
                        new_weights.append(global_w)
                else:
                    # For all other layers, use global weights directly
                    if global_w.shape != local_w.shape:
                        print(f"Layer {i} shape mismatch: global {global_w.shape} vs local {local_w.shape}")
                        return False
                    new_weights.append(global_w)
            
            # Apply the new weights
            self.model.set_weights(new_weights)
            return True
            
        except Exception as e:
            print(f"Error applying global weights: {str(e)}")
            return False

def train_test_split(indices, test_size=0.2):
    """
    Split indices into training and testing sets.
    
    Args:
        indices: Array-like of indices to split
        test_size: Fraction of indices to use for testing
        
    Returns:
        Tuple of (train_indices, test_indices)
    """
    # Convert to numpy array if not already
    indices = np.array(indices)
    
    # Shuffle indices
    np.random.shuffle(indices)
    
    # Split indices
    test_count = int(len(indices) * test_size)
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    
    return train_indices, test_indices

def initialize_model(num_classes, use_dp=False):
    """
    Initialize a TensorFlow model for federated learning.
    
    Args:
        num_classes: Number of classes for the output layer
        use_dp: Whether to use differential privacy
        
    Returns:
        Compiled TensorFlow model
    """
    print(f"Initializing model with {num_classes} classes")
    
    if use_dp:
        model = build_dp_model(num_classes)
    else:
        model = build_face_recognition_model(num_classes, training=True)
    
    # Compile the model
    optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=0.9)
    
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    
    return model

def register_client(client_state, server_url):
    """
    Register the client with the federated learning server.
    
    Args:
        client_state: ClientState instance
        server_url: URL of the federated learning server
        
    Returns:
        Server response data or None if registration failed
    """
    register_url = f"{server_url}/register"
    
    # Prepare registration data
    data = {
        'client_id': client_state.client_id,
        'classes': sorted(list(client_state.local_classes)),
        'num_samples': client_state.num_samples
    }
    
    # Send registration request
    try:
        response = requests.post(register_url, json=data)
        response.raise_for_status()
        
        result = response.json()
        print(f"Client registered successfully. Server has {result.get('num_clients', 0)} clients.")
        return result
    
    except requests.exceptions.RequestException as e:
        print(f"Failed to register client: {str(e)}")
        return None

def fetch_global_model(client_state, server_url):
    """
    Fetch the global model from the federated learning server.
    
    Args:
        client_state: ClientState instance
        server_url: URL of the federated learning server
        
    Returns:
        True if model was fetched successfully, False otherwise
    """
    model_url = f"{server_url}/model"
    
    # Send request for global model
    try:
        response = requests.get(model_url)
        response.raise_for_status()
        
        # Parse binary response
        model_data = pickle.loads(response.content)
        global_weights = model_data.get('weights')
        round_number = model_data.get('round', 0)
        max_classes = model_data.get('max_classes', 0)
        
        print(f"Fetched global model for round {round_number} with {max_classes} classes")
        
        # Update client state
        client_state.current_round = round_number
        client_state.global_classes = set(range(max_classes))
        
        # Update class mapping
        client_state.update_class_mapping(max_classes)
        
        # Determine the number of classes needed for the local model
        if client_state.allow_new_classes:
            # If we allow new classes, use the maximum local class index
            num_local_classes = max(client_state.local_classes) + 1 if client_state.local_classes else max_classes
            # Ensure the model has at least as many classes as the global model
            num_local_classes = max(num_local_classes, max_classes)
        else:
            # Otherwise, use the global model's class count
            num_local_classes = max_classes
        
        # Initialize local model if needed
        if client_state.model is None:
            client_state.model = initialize_model(num_local_classes, client_state.use_dp)
        
        # Apply global weights to local model
        success = client_state.apply_global_weights(global_weights, max_classes)
        
        # Save local model
        if success:
            client_state.model.save(CLIENT_MODEL_PATH)
            print(f"Local model saved to {CLIENT_MODEL_PATH}")
        
        return success
    
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch global model: {str(e)}")
        return False

def send_model_update(client_state, server_url, metrics):
    """
    Send local model update to the federated learning server.
    
    Args:
        client_state: ClientState instance
        server_url: URL of the federated learning server
        metrics: Dictionary of training metrics
        
    Returns:
        True if update was sent successfully, False otherwise
    """
    update_url = f"{server_url}/update"
    
    # Get local model weights
    if client_state.model is None:
        print("Local model not initialized. Cannot send update.")
        return False
    
    local_weights = client_state.model.get_weights()
    
    # If handling new classes, we need to modify the weights to match the global model
    if client_state.allow_new_classes and client_state.global_classes:
        max_global_class = max(client_state.global_classes)
        
        # Only include weights for classes known to the server
        if len(local_weights) >= 2:  # Check if we have at least the final layer weights and biases
            # Extract the final layer weights (typically the last two elements are weights and biases)
            final_weights = local_weights[-2]
            final_biases = local_weights[-1]
            
            # If the final layer has more classes than the global model, truncate it
            if final_weights.shape[1] > max_global_class + 1:
                print(f"Truncating local model output from {final_weights.shape[1]} to {max_global_class + 1} classes")
                # Truncate the weights to only include global classes
                modified_weights = final_weights[:, :max_global_class + 1]
                modified_biases = final_biases[:max_global_class + 1]
                
                # Replace the weights and biases in the local_weights list
                local_weights[-2] = modified_weights
                local_weights[-1] = modified_biases
    
    # Prepare update data
    data = {
        'client_id': client_state.client_id,
        'weights': local_weights,
        'metrics': metrics,
        'num_samples': client_state.num_samples
    }
    
    # Serialize the data
    serialized_data = pickle.dumps(data)
    
    # Send update request
    try:
        response = requests.post(
            update_url, 
            data=serialized_data, 
            headers={'Content-Type': 'application/octet-stream'}
        )
        response.raise_for_status()
        
        result = response.json()
        print(f"Model update sent successfully for round {result.get('round', 0)}")
        return True
    
    except requests.exceptions.RequestException as e:
        print(f"Failed to send model update: {str(e)}")
        return False

def run_federated_client(args):
    """
    Run the federated learning client.
    
    Args:
        args: Command line arguments
    """
    # Set up client state
    client_state = ClientState(
        client_id=args.client_id,
        data_dir=args.data_dir,
        allow_new_classes=args.allow_new_classes
    )
    client_state.use_dp = args.use_dp
    
    # Set up server URL
    server_url = f"http://{args.server_host}:{args.server_port}"
    
    # Load local dataset
    client_state.load_data()
    
    # Register with server
    for _ in range(MAX_RETRY_ATTEMPTS):
        result = register_client(client_state, server_url)
        if result:
            break
        time.sleep(RETRY_DELAY_SECONDS)
    else:
        print(f"Failed to register after {MAX_RETRY_ATTEMPTS} attempts. Exiting.")
        return
    
    # Main federated learning loop
    while True:
        try:
            # Fetch global model
            success = fetch_global_model(client_state, server_url)
            if not success:
                print("Failed to fetch global model. Retrying...")
                time.sleep(RETRY_DELAY_SECONDS)
                continue
            
            # Check if we're in the first round and need to generate some additional classes
            if args.add_new_classes and client_state.current_round == 0:
                add_new_classes(client_state, args.num_new_classes)
            
            # Train local model
            print(f"Training local model for round {client_state.current_round}")
            metrics = client_state.train_local_model(epochs=args.local_epochs)
            
            # Evaluate local model
            eval_metrics = client_state.evaluate_model()
            
            # Combine metrics
            all_metrics = {**metrics, **eval_metrics} if metrics and eval_metrics else {}
            
            # Send model update
            success = send_model_update(client_state, server_url, all_metrics)
            if not success:
                print("Failed to send model update. Retrying...")
                time.sleep(RETRY_DELAY_SECONDS)
                continue
            
            # Wait for the next round
            time.sleep(args.wait_time)
        
        except KeyboardInterrupt:
            print("Client stopped by user")
            break
        
        except Exception as e:
            print(f"Error in federated learning client: {str(e)}")
            time.sleep(RETRY_DELAY_SECONDS)

def add_new_classes(client_state, num_new_classes=5):
    """
    Add new classes to the client's local dataset that don't exist in the global model.
    This is for demonstration purposes to show how clients can handle new classes.
    
    Args:
        client_state: ClientState instance
        num_new_classes: Number of new classes to add
    """
    if not client_state.allow_new_classes:
        print("Client is not configured to allow new classes.")
        return
    
    existing_max_class = max(client_state.local_classes) if client_state.local_classes else 0
    max_global_class = max(client_state.global_classes) if client_state.global_classes else 0
    
    # Start new class indices from max(local, global) + 1
    start_class = max(existing_max_class, max_global_class) + 1
    
    print(f"Adding {num_new_classes} new classes starting from index {start_class}")
    
    # Add new classes to local classes
    for i in range(num_new_classes):
        new_class = start_class + i
        client_state.local_classes.add(new_class)
    
    print(f"Local classes after adding new ones: {client_state.local_classes}")
    
    # Update the class mapping
    if client_state.global_classes:
        client_state.update_class_mapping(max(client_state.global_classes) + 1)
    
    # If model exists, add output layer for new classes
    if client_state.model is not None:
        # Get the original weights
        weights = client_state.model.get_weights()
        
        # Create new model with expanded output layer
        new_num_classes = max(client_state.local_classes) + 1
        
        old_model = client_state.model
        client_state.model = None  # Clear current model
        
        # Initialize new model with more classes
        new_model = initialize_model(new_num_classes, client_state.use_dp)
        
        # Transfer weights from old model to new model for shared layers
        new_weights = new_model.get_weights()
        
        # Copy weights for shared layers
        for i in range(len(weights) - 2):  # All layers except the final dense layer
            new_weights[i] = weights[i]
        
        # Copy weights for existing classes in the final dense layer
        final_layer_weights = weights[-2]  # Final dense layer weights
        final_layer_bias = weights[-1]     # Final dense layer bias
        
        # Copy existing weights
        embedding_dim = final_layer_weights.shape[0]
        old_classes = final_layer_weights.shape[1]
        
        new_weights[-2][:, :old_classes] = final_layer_weights
        new_weights[-1][:old_classes] = final_layer_bias
        
        # Initialize weights for new classes with small random values
        new_weights[-2][:, old_classes:] = np.random.normal(
            0, 0.01, 
            size=(embedding_dim, new_num_classes - old_classes)
        ).astype(weights[-2].dtype)
        
        # Initialize biases for new classes with zeros
        new_weights[-1][old_classes:] = np.zeros(new_num_classes - old_classes, dtype=weights[-1].dtype)
        
        # Set the new weights
        new_model.set_weights(new_weights)
        
        # Update client model
        client_state.model = new_model
        
        print(f"Model updated to handle {new_num_classes} classes")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Federated Learning Client')
    
    parser.add_argument('--client_id', type=str, required=True,
                        help='Unique identifier for this client')
    parser.add_argument('--server_host', type=str, default=DEFAULT_SERVER_HOST,
                        help='Hostname of the federated learning server')
    parser.add_argument('--server_port', type=int, default=DEFAULT_SERVER_PORT,
                        help='Port of the federated learning server')
    parser.add_argument('--data_dir', type=str, default='data/extracted',
                        help='Directory containing the client dataset')
    parser.add_argument('--local_epochs', type=int, default=5,
                        help='Number of local epochs to train per round')
    parser.add_argument('--wait_time', type=int, default=10,
                        help='Time to wait between rounds in seconds')
    parser.add_argument('--use_dp', action='store_true',
                        help='Whether to use differential privacy')
    parser.add_argument('--allow_new_classes', action='store_true',
                        help='Whether to allow new classes not present in the global model')
    parser.add_argument('--add_new_classes', action='store_true',
                        help='Whether to add synthetic new classes for testing')
    parser.add_argument('--num_new_classes', type=int, default=5,
                        help='Number of new classes to add if add_new_classes is True')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    run_federated_client(args) 