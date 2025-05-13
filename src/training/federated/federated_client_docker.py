#!/usr/bin/env python
"""
Federated Learning Client for Docker deployment, loading from pre-trained models.
This script extends federated_client.py with functionality to load pre-trained models.
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
from src.data_utils import create_dataset_index
from src.federated_client import (
    ClientState, train_test_split, initialize_model,
    register_client, fetch_global_model, send_model_update
)

# Docker specific paths
DOCKER_MODEL_PATH = "/app/shared/models"
CONFIG_PATH = "/app/shared/config.json"

# Client configuration
DEFAULT_SERVER_HOST = 'fl_server'
DEFAULT_SERVER_PORT = 8080
MAX_RETRY_ATTEMPTS = 5
RETRY_DELAY_SECONDS = 5
DEFAULT_CLIENT_ID = 1

def load_config():
    """Load configuration from the shared directory."""
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r') as f:
                config = json.load(f)
            print(f"Loaded configuration from {CONFIG_PATH}")
            return config
        else:
            print(f"Warning: Configuration file {CONFIG_PATH} not found.")
            return {
                'num_rounds': 10,
                'use_dp': False,
                'allow_new_classes': True
            }
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return {
            'num_rounds': 10,
            'use_dp': False,
            'allow_new_classes': True
        }

def load_client_model(client_id):
    """Load the client model from the shared directory."""
    model_path = os.path.join(DOCKER_MODEL_PATH, f"client{client_id}_model.h5")
    
    try:
        if os.path.exists(model_path):
            print(f"Loading client model from {model_path}")
            model = tf.keras.models.load_model(model_path)
            print("Client model loaded successfully")
            return model
        else:
            print(f"Warning: Client model {model_path} not found.")
            return None
    except Exception as e:
        print(f"Error loading client model: {e}")
        return None

def load_client_classes(client_id):
    """Load client class information from the shared directory."""
    classes_path = os.path.join(DOCKER_MODEL_PATH, f"client{client_id}_classes.json")
    
    try:
        if os.path.exists(classes_path):
            with open(classes_path, 'r') as f:
                class_info = json.load(f)
            
            num_classes = class_info.get('num_classes', 0)
            class_list = class_info.get('class_list', [])
            
            print(f"Loaded client classes from {classes_path}")
            print(f"Number of classes: {num_classes}")
            return set(class_list)
        else:
            print(f"Warning: Client classes file {classes_path} not found.")
            return set()
    except Exception as e:
        print(f"Error loading client classes: {e}")
        return set()

def get_data_dir_from_env(client_id, fallback_dir="data"):
    """Get the data directory from environment variables or use a fallback."""
    env_var = f"CLIENT{client_id}_DATA_DIR"
    data_dir = os.environ.get(env_var)
    
    if data_dir:
        return data_dir
    
    return fallback_dir

def run_federated_client_docker(args):
    """
    Run the federated learning client in Docker.
    
    Args:
        args: Command line arguments
    """
    # Load configuration
    config = load_config()
    
    # Set up client state
    client_id = args.client_id
    data_dir = args.data_dir or get_data_dir_from_env(client_id)
    allow_new_classes = args.allow_new_classes or config.get('allow_new_classes', False)
    
    print(f"Initializing client {client_id} with data from {data_dir}")
    print(f"Allow new classes: {allow_new_classes}")
    
    # Create client state
    client_state = ClientState(client_id, data_dir, allow_new_classes)
    client_state.use_dp = args.use_dp or config.get('use_dp', False)
    
    # Load data
    client_state.load_data()
    
    # Load pre-trained model if available
    pre_trained_model = load_client_model(client_id)
    if pre_trained_model is not None:
        client_state.model = pre_trained_model
        print("Using pre-trained client model")
    
    # Load pre-trained class information if available
    pre_trained_classes = load_client_classes(client_id)
    if pre_trained_classes:
        client_state.local_classes.update(pre_trained_classes)
        print(f"Updated local classes from pre-trained model, now has {len(client_state.local_classes)} classes")
    
    # Server URL
    server_url = f"http://{args.server_host}:{args.server_port}"
    
    # Wait for server to be available
    server_available = False
    retry_count = 0
    
    while not server_available and retry_count < MAX_RETRY_ATTEMPTS:
        try:
            response = requests.get(f"{server_url}/status", timeout=5)
            if response.status_code == 200:
                server_available = True
                print(f"Connected to federated server at {server_url}")
            else:
                retry_count += 1
                print(f"Server returned status {response.status_code}, retrying ({retry_count}/{MAX_RETRY_ATTEMPTS})...")
                time.sleep(RETRY_DELAY_SECONDS)
        except requests.exceptions.RequestException:
            retry_count += 1
            print(f"Server not available, retrying ({retry_count}/{MAX_RETRY_ATTEMPTS})...")
            time.sleep(RETRY_DELAY_SECONDS)
    
    if not server_available:
        print(f"Failed to connect to server after {MAX_RETRY_ATTEMPTS} attempts. Exiting.")
        return
    
    # Register client with server
    register_result = register_client(client_state, server_url)
    if register_result is None:
        print("Failed to register client with server. Exiting.")
        return
    
    # Main federated learning loop
    training_complete = False
    while not training_complete:
        # Fetch global model from server
        fetch_success = fetch_global_model(client_state, server_url)
        if not fetch_success:
            print("Failed to fetch global model. Retrying in 5 seconds...")
            time.sleep(5)
            continue
        
        # Train local model
        print(f"Training local model for client {client_id}, round {client_state.current_round}")
        metrics = client_state.train_local_model(epochs=1)
        
        if metrics is None:
            print("Failed to train local model. Retrying in 5 seconds...")
            time.sleep(5)
            continue
        
        # Send model update to server
        update_success = send_model_update(client_state, server_url, metrics)
        if not update_success:
            print("Failed to send model update. Retrying in 5 seconds...")
            time.sleep(5)
            continue
        
        # Check server status
        try:
            response = requests.get(f"{server_url}/status")
            status_data = response.json()
            
            if status_data.get('status') == 'training_complete':
                training_complete = True
                print("Federated learning complete!")
                
                # Save final model
                if client_state.model is not None:
                    save_path = os.path.join(DOCKER_MODEL_PATH, f"client{client_id}_final_model.h5")
                    client_state.model.save(save_path)
                    print(f"Final model saved to {save_path}")
            else:
                current_round = status_data.get('round', 0)
                max_rounds = status_data.get('max_rounds', 10)
                print(f"Federated learning progress: round {current_round}/{max_rounds}")
        
        except requests.exceptions.RequestException as e:
            print(f"Failed to check server status: {str(e)}")
        
        # Wait before next round
        time.sleep(2)
    
    print(f"Client {client_id} completed federated learning.")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Federated Learning Client for Docker')
    
    parser.add_argument('--client_id', type=int, default=DEFAULT_CLIENT_ID,
                        help='Client ID')
    parser.add_argument('--server_host', type=str, default=DEFAULT_SERVER_HOST,
                        help='Hostname or IP of the federated server')
    parser.add_argument('--server_port', type=int, default=DEFAULT_SERVER_PORT,
                        help='Port of the federated server')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory containing the client dataset')
    parser.add_argument('--use_dp', action='store_true',
                        help='Whether to use differential privacy')
    parser.add_argument('--allow_new_classes', action='store_true',
                        help='Whether to allow handling of non-existing classes')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_federated_client_docker(args) 