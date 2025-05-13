#!/usr/bin/env python
"""
Federated Learning Server for Docker deployment, loading from pre-trained models.
This script extends federated_server.py with functionality to load pre-trained models.
"""

import os
import numpy as np
import tensorflow as tf
import json
import argparse
import time
import socket
import pickle
import threading
import queue
import matplotlib.pyplot as plt
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import urllib.parse
import io
import sys
from pathlib import Path
import base64
import cv2

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    MODEL_SAVE_PATH, BATCH_SIZE, LEARNING_RATE, EMBEDDING_SIZE,
    DP_NOISE_MULTIPLIER, DP_L2_NORM_CLIP, DP_MICROBATCHES, DP_DELTA
)
from src.model import build_face_recognition_model, build_dp_model, get_embedding_model
from src.face_detection import process_image_bytes, visualize_face_detection, face_image_to_bytes
from src.federated_server import (
    ServerState, ThreadedHTTPServer, FederatedServerHandler,
    save_model, save_identity_mapping, load_identity_mapping,
    update_embeddings_db, extract_embedding, identify_face,
    federated_averaging, aggregate_client_models, plot_federated_metrics
)

# Docker specific paths
DOCKER_MODEL_PATH = "/app/shared/models"
SERVER_MODEL_PATH = os.path.join(DOCKER_MODEL_PATH, "server_model.h5")
SERVER_CLASSES_PATH = os.path.join(DOCKER_MODEL_PATH, "server_classes.json")
CONFIG_PATH = "/app/shared/config.json"

# Server configuration
DEFAULT_PORT = 8080
MAX_BUFFER_SIZE = 104857600  # 100MB
IDENTITY_MAPPING_FILE = os.path.join(DOCKER_MODEL_PATH, 'identity_mapping.json')
SIMILARITY_THRESHOLD = 0.7  # Threshold for identity matching

# Global server state (initialized from parent module)
server_state = ServerState()
client_queue = queue.Queue()
lock = threading.RLock()

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

def load_server_model():
    """Load the server model from the shared directory."""
    global server_state
    
    try:
        if os.path.exists(SERVER_MODEL_PATH):
            print(f"Loading server model from {SERVER_MODEL_PATH}")
            model = tf.keras.models.load_model(SERVER_MODEL_PATH)
            server_state.model = model
            
            # Create embedding model
            server_state.embedding_model = get_embedding_model(model)
            
            print("Server model loaded successfully")
            return True
        else:
            print(f"Warning: Server model {SERVER_MODEL_PATH} not found.")
            return False
    except Exception as e:
        print(f"Error loading server model: {e}")
        return False

def load_server_classes():
    """Load server class information from the shared directory."""
    global server_state
    
    try:
        if os.path.exists(SERVER_CLASSES_PATH):
            with open(SERVER_CLASSES_PATH, 'r') as f:
                class_info = json.load(f)
            
            num_classes = class_info.get('num_classes', 0)
            class_list = class_info.get('class_list', [])
            
            server_state.global_classes = set(class_list)
            
            print(f"Loaded server classes from {SERVER_CLASSES_PATH}")
            print(f"Number of classes: {num_classes}")
            return True
        else:
            print(f"Warning: Server classes file {SERVER_CLASSES_PATH} not found.")
            return False
    except Exception as e:
        print(f"Error loading server classes: {e}")
        return False

def initialize_server_state(config):
    """Initialize the server state using the config and pre-trained model."""
    global server_state
    
    # Set max rounds from config
    server_state.max_rounds = config.get('num_rounds', 10)
    
    # Load identity mapping if exists
    if not load_identity_mapping():
        print("No existing identity mapping found, using default.")
    
    # Load pre-trained model if exists
    if not load_server_model():
        # No pre-trained model, will initialize when first client registers
        print("No pre-trained model found, will initialize when clients register.")
    else:
        # Load server classes
        load_server_classes()
        
        # Update embeddings database
        update_embeddings_db()
    
    print(f"Server state initialized with {len(server_state.global_classes)} classes")
    print(f"Server will run for {server_state.max_rounds} rounds")

def start_server_docker(port=DEFAULT_PORT):
    """
    Start the federated learning server in Docker.
    
    Args:
        port: HTTP server port
    """
    global server_state
    
    # Load configuration
    config = load_config()
    
    # Create model save directories
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(DOCKER_MODEL_PATH, exist_ok=True)
    
    # Initialize server state
    initialize_server_state(config)
    
    # Start HTTP server
    server_address = ('', port)
    httpd = ThreadedHTTPServer(server_address, FederatedServerHandler)
    
    print(f"Starting federated learning server on port {port}")
    print(f"Server will run for {server_state.max_rounds} rounds")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("Server stopped by user")
    finally:
        httpd.server_close()
        
        # Save final model and metrics if training was in progress
        if server_state.model is not None and not server_state.training_complete:
            save_model(server_state.model, server_state.round_number)
            plot_federated_metrics()
            save_identity_mapping()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Federated Learning Server for Docker')
    
    parser.add_argument('--server_port', type=int, default=DEFAULT_PORT,
                        help='Port to run the server on')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    start_server_docker(port=args.server_port) 