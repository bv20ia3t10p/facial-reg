"""
Federated Learning Server for the privacy-preserving facial recognition system.

This module implements a server that coordinates federated learning among multiple clients,
provides identity recognition, and handles feedback for continuous learning.
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

# Server configuration
DEFAULT_PORT = 8080
MAX_BUFFER_SIZE = 104857600  # 100MB
IDENTITY_MAPPING_FILE = os.path.join(MODEL_SAVE_PATH, 'identity_mapping.json')
SIMILARITY_THRESHOLD = 0.7  # Threshold for identity matching

# Global server state
class ServerState:
    def __init__(self):
        self.model = None
        self.embedding_model = None
        self.num_clients = 0
        self.client_models = {}
        self.client_metadata = {}
        self.round_number = 0
        self.max_rounds = 10
        self.global_classes = set()
        self.client_ready = {}
        self.round_in_progress = False
        self.training_complete = False
        self.metrics = {
            'global_accuracy': [],
            'global_loss': [],
            'rounds': []
        }
        self.identity_mapping = {}  # Maps class indices to identity names
        self.embeddings_db = {}  # Stores face embeddings for known identities

server_state = ServerState()
client_queue = queue.Queue()
lock = threading.RLock()

def initialize_model(num_classes, use_dp=False):
    """Initialize the global model."""
    global server_state
    
    print(f"Initializing global model with {num_classes} classes")
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
    
    server_state.model = model
    
    # Create embedding model
    server_state.embedding_model = get_embedding_model(model)
    
    return model

def save_model(model, round_number):
    """Save the global model."""
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    model_path = os.path.join(MODEL_SAVE_PATH, f'federated_model_round_{round_number}.h5')
    model.save(model_path)
    print(f"Global model saved to {model_path}")
    
    # Save also as the latest model
    latest_model_path = os.path.join(MODEL_SAVE_PATH, 'federated_model_latest.h5')
    model.save(latest_model_path)
    
    return model_path

def save_identity_mapping():
    """Save the identity mapping to a file."""
    global server_state
    
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    with open(IDENTITY_MAPPING_FILE, 'w') as f:
        json.dump(server_state.identity_mapping, f)
    
    print(f"Identity mapping saved to {IDENTITY_MAPPING_FILE}")

def load_identity_mapping():
    """Load the identity mapping from a file."""
    global server_state
    
    if os.path.exists(IDENTITY_MAPPING_FILE):
        with open(IDENTITY_MAPPING_FILE, 'r') as f:
            server_state.identity_mapping = json.load(f)
        
        print(f"Identity mapping loaded from {IDENTITY_MAPPING_FILE}")
        return True
    
    return False

def update_embeddings_db():
    """Update the embeddings database based on the current model and identity mapping."""
    global server_state
    
    if server_state.embedding_model is None:
        print("Embedding model not initialized. Cannot update embeddings database.")
        return False
    
    # Clear existing embeddings
    server_state.embeddings_db = {}
    
    return True

def extract_embedding(face_img):
    """
    Extract embedding for a preprocessed face image.
    
    Args:
        face_img: Preprocessed face image
        
    Returns:
        Face embedding vector
    """
    global server_state
    
    if server_state.embedding_model is None:
        raise ValueError("Embedding model not initialized")
    
    # Ensure face_img has batch dimension
    if len(face_img.shape) == 3:
        face_img = np.expand_dims(face_img, axis=0)
    
    # Extract embedding
    embedding = server_state.embedding_model.predict(face_img)
    
    # Normalize embedding
    embedding = embedding / np.linalg.norm(embedding)
    
    return embedding

def identify_face(face_img):
    """
    Identify a face using the current model.
    
    Args:
        face_img: Preprocessed face image
        
    Returns:
        Tuple of (identity_id, identity_name, confidence_score)
    """
    global server_state
    
    if server_state.model is None:
        raise ValueError("Model not initialized")
    
    # Ensure face_img has batch dimension
    if len(face_img.shape) == 3:
        face_img = np.expand_dims(face_img, axis=0)
    
    # Get prediction
    prediction = server_state.model.predict(face_img)
    
    # Get highest confidence class
    class_id = np.argmax(prediction[0])
    confidence = prediction[0][class_id]
    
    # Get identity name from mapping
    identity_name = server_state.identity_mapping.get(str(class_id), f"Unknown_{class_id}")
    
    return class_id, identity_name, float(confidence)

def federated_averaging(weights_list, sample_counts):
    """
    Perform federated averaging on client model weights.
    
    Args:
        weights_list: List of client model weights
        sample_counts: List of number of samples used by each client
        
    Returns:
        Averaged weights
    """
    total_samples = sum(sample_counts)
    weighted_weights = []
    
    # Ensure all weight arrays have the same structure
    if not weights_list:
        return None
    
    # Initialize with zeros
    weighted_weights = [np.zeros_like(w) for w in weights_list[0]]
    
    # Perform weighted average
    for weights, sample_count in zip(weights_list, sample_counts):
        weight_factor = sample_count / total_samples
        for i, w in enumerate(weights):
            weighted_weights[i] += w * weight_factor
    
    return weighted_weights

def aggregate_client_models():
    """Aggregate client models using federated averaging."""
    global server_state
    
    with lock:
        if not server_state.client_models:
            print("No client models to aggregate")
            return False
        
        print(f"Aggregating {len(server_state.client_models)} client models")
        
        # Extract weights and sample counts
        weights_list = []
        sample_counts = []
        
        for client_id, client_update in server_state.client_models.items():
            weights = client_update['weights']
            num_samples = client_update['num_samples']
            weights_list.append(weights)
            sample_counts.append(num_samples)
        
        # Perform federated averaging
        if not weights_list:
            print("No weights to aggregate")
            return False
        
        aggregated_weights = federated_averaging(weights_list, sample_counts)
        if aggregated_weights is None:
            print("Failed to aggregate weights")
            return False
        
        # Update global model
        server_state.model.set_weights(aggregated_weights)
        
        # Update embedding model
        server_state.embedding_model = get_embedding_model(server_state.model)
        
        # Update global metrics
        global_accuracy = 0.0
        global_loss = 0.0
        for client_id, client_update in server_state.client_models.items():
            global_accuracy += client_update['metrics']['val_accuracy'] * (client_update['num_samples'] / sum(sample_counts))
            global_loss += client_update['metrics']['val_loss'] * (client_update['num_samples'] / sum(sample_counts))
        
        server_state.metrics['global_accuracy'].append(global_accuracy)
        server_state.metrics['global_loss'].append(global_loss)
        server_state.metrics['rounds'].append(server_state.round_number)
        
        print(f"Round {server_state.round_number} global metrics:")
        print(f"  - Accuracy: {global_accuracy:.4f}")
        print(f"  - Loss: {global_loss:.4f}")
        
        # Save the model
        save_model(server_state.model, server_state.round_number)
        
        # Reset for next round
        server_state.client_models = {}
        server_state.round_number += 1
        server_state.round_in_progress = False
        
        # Check if training is complete
        if server_state.round_number >= server_state.max_rounds:
            server_state.training_complete = True
            plot_federated_metrics()
            return True
        
        return True

def plot_federated_metrics():
    """Plot federated learning metrics."""
    global server_state
    
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(server_state.metrics['rounds'], server_state.metrics['global_accuracy'], marker='o')
    plt.title('Global Model Accuracy')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(server_state.metrics['rounds'], server_state.metrics['global_loss'], marker='o', color='r')
    plt.title('Global Model Loss')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save the plot
    metrics_path = os.path.join(MODEL_SAVE_PATH, 'federated_metrics.png')
    plt.savefig(metrics_path)
    plt.close()
    
    # Save metrics to JSON
    metrics_json_path = os.path.join(MODEL_SAVE_PATH, 'federated_metrics.json')
    with open(metrics_json_path, 'w') as f:
        json.dump(server_state.metrics, f)
    
    print(f"Metrics saved to {metrics_path} and {metrics_json_path}")

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
    pass

class FederatedServerHandler(BaseHTTPRequestHandler):
    """HTTP request handler for federated server."""
    
    def _send_json_response(self, data, status=200):
        """Send a JSON response."""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def _send_binary_response(self, data, status=200):
        """Send a binary response."""
        self.send_response(status)
        self.send_header('Content-Type', 'application/octet-stream')
        self.end_headers()
        self.wfile.write(data)
    
    def _send_image_response(self, image_bytes, status=200):
        """Send an image response."""
        self.send_response(status)
        self.send_header('Content-Type', 'image/jpeg')
        self.end_headers()
        self.wfile.write(image_bytes)
    
    def _parse_body(self):
        """Parse the request body based on content type."""
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length > MAX_BUFFER_SIZE:
            raise ValueError(f"Content length {content_length} exceeds maximum buffer size {MAX_BUFFER_SIZE}")
        
        body = self.rfile.read(content_length)
        content_type = self.headers.get('Content-Type', '')
        
        if 'application/json' in content_type:
            return json.loads(body.decode())
        elif 'application/octet-stream' in content_type:
            return pickle.loads(body)
        elif 'multipart/form-data' in content_type:
            # For multipart form data, we'll handle it case by case
            return body
        else:
            return body
    
    def do_POST(self):
        """Handle POST requests."""
        global server_state
        
        parsed_path = urllib.parse.urlparse(self.path)
        
        # Client registration
        if parsed_path.path == '/register':
            try:
                data = self._parse_body()
                client_id = data.get('client_id')
                client_classes = data.get('classes', [])
                num_samples = data.get('num_samples', 0)
                
                with lock:
                    server_state.client_metadata[client_id] = {
                        'classes': client_classes,
                        'num_samples': num_samples,
                        'last_seen': time.time()
                    }
                    server_state.global_classes.update(client_classes)
                    server_state.client_ready[client_id] = True
                    server_state.num_clients = len(server_state.client_metadata)
                
                # Initialize model if needed
                if server_state.model is None and server_state.num_clients > 0:
                    num_classes = max(server_state.global_classes) + 1 if server_state.global_classes else 10
                    initialize_model(num_classes)
                
                print(f"Client {client_id} registered with {num_samples} samples and {len(client_classes)} classes")
                self._send_json_response({
                    'status': 'success',
                    'message': 'Client registered successfully',
                    'num_clients': server_state.num_clients,
                    'max_classes': max(server_state.global_classes) + 1 if server_state.global_classes else 0
                })
            except Exception as e:
                print(f"Registration error: {str(e)}")
                self._send_json_response({
                    'status': 'error',
                    'message': str(e)
                }, status=400)
        
        # Client model update
        elif parsed_path.path == '/update':
            try:
                update_data = self._parse_body()
                client_id = update_data.get('client_id')
                client_weights = update_data.get('weights')
                client_metrics = update_data.get('metrics', {})
                num_samples = update_data.get('num_samples', 0)
                
                with lock:
                    server_state.client_models[client_id] = {
                        'weights': client_weights,
                        'metrics': client_metrics,
                        'num_samples': num_samples,
                        'timestamp': time.time()
                    }
                    server_state.client_ready[client_id] = True
                    
                    # Check if all clients have submitted updates
                    all_clients_ready = all(server_state.client_ready.values())
                    
                    if all_clients_ready and not server_state.round_in_progress:
                        server_state.round_in_progress = True
                        # Start aggregation in a new thread
                        threading.Thread(target=aggregate_client_models).start()
                
                print(f"Received update from client {client_id} with {num_samples} samples")
                self._send_json_response({
                    'status': 'success',
                    'message': 'Update received',
                    'round': server_state.round_number
                })
            except Exception as e:
                print(f"Update error: {str(e)}")
                self._send_json_response({
                    'status': 'error',
                    'message': str(e)
                }, status=400)
        
        # Identity recognition
        elif parsed_path.path == '/recognize':
            try:
                if server_state.model is None:
                    self._send_json_response({
                        'status': 'error',
                        'message': 'Model not initialized yet'
                    }, status=503)
                    return
                
                # Read multipart form data
                content_type = self.headers.get('Content-Type', '')
                
                # Parse content type to get boundary
                if 'multipart/form-data' in content_type:
                    image_data = self._parse_body()
                else:
                    # For JSON data with base64 encoded image
                    request_data = self._parse_body()
                    
                    if isinstance(request_data, dict) and 'image' in request_data:
                        # Decode base64 image
                        image_data = base64.b64decode(request_data['image'])
                    else:
                        # Treat as raw image data
                        image_data = request_data
                
                # Process image to extract face
                face_img, original_img, face_rect = process_image_bytes(image_data)
                
                # Identify the face
                class_id, identity_name, confidence = identify_face(face_img)
                
                # Create visualization
                viz_img = visualize_face_detection(original_img, face_rect, identity_name)
                viz_bytes = face_image_to_bytes(viz_img)
                
                # Return recognition results
                response_data = {
                    'status': 'success',
                    'identity_id': int(class_id),
                    'identity_name': identity_name,
                    'confidence': confidence,
                    'visualization': base64.b64encode(viz_bytes).decode('utf-8')
                }
                
                self._send_json_response(response_data)
            
            except Exception as e:
                print(f"Recognition error: {str(e)}")
                self._send_json_response({
                    'status': 'error',
                    'message': str(e)
                }, status=400)
        
        # Identity feedback
        elif parsed_path.path == '/feedback':
            try:
                if server_state.model is None:
                    self._send_json_response({
                        'status': 'error',
                        'message': 'Model not initialized yet'
                    }, status=503)
                    return
                
                # Parse the request data
                request_data = self._parse_body()
                
                # Extract identity info and image data
                identity_id = request_data.get('identity_id')
                identity_name = request_data.get('identity_name')
                
                if 'image' in request_data:
                    # Decode base64 image
                    image_data = base64.b64decode(request_data['image'])
                else:
                    self._send_json_response({
                        'status': 'error',
                        'message': 'No image provided'
                    }, status=400)
                    return
                
                # Process image to extract face
                face_img, original_img, face_rect = process_image_bytes(image_data)
                
                # Update identity mapping
                if identity_id is not None and identity_name:
                    server_state.identity_mapping[str(identity_id)] = identity_name
                    save_identity_mapping()
                
                # Return success response
                self._send_json_response({
                    'status': 'success',
                    'message': 'Feedback received and processed',
                    'identity_id': identity_id,
                    'identity_name': identity_name
                })
                
            except Exception as e:
                print(f"Feedback error: {str(e)}")
                self._send_json_response({
                    'status': 'error',
                    'message': str(e)
                }, status=400)
        
        # Unknown endpoint
        else:
            self._send_json_response({
                'status': 'error',
                'message': 'Unknown endpoint'
            }, status=404)
    
    def do_GET(self):
        """Handle GET requests."""
        global server_state
        
        parsed_path = urllib.parse.urlparse(self.path)
        
        # Get global model
        if parsed_path.path == '/model':
            try:
                if server_state.model is None:
                    self._send_json_response({
                        'status': 'error',
                        'message': 'Model not initialized yet'
                    }, status=404)
                    return
                
                # Serialize model weights
                weights = server_state.model.get_weights()
                model_data = {
                    'weights': weights,
                    'round': server_state.round_number,
                    'max_classes': max(server_state.global_classes) + 1 if server_state.global_classes else 0
                }
                serialized_model = pickle.dumps(model_data)
                
                self._send_binary_response(serialized_model)
            except Exception as e:
                print(f"Model serving error: {str(e)}")
                self._send_json_response({
                    'status': 'error',
                    'message': str(e)
                }, status=500)
        
        # Get training status
        elif parsed_path.path == '/status':
            try:
                status_data = {
                    'status': 'training_complete' if server_state.training_complete else 'in_progress',
                    'round': server_state.round_number,
                    'max_rounds': server_state.max_rounds,
                    'num_clients': server_state.num_clients,
                    'global_classes': list(server_state.global_classes),
                    'identity_mapping': server_state.identity_mapping,
                    'metrics': {
                        'accuracy': server_state.metrics['global_accuracy'][-1] if server_state.metrics['global_accuracy'] else 0,
                        'loss': server_state.metrics['global_loss'][-1] if server_state.metrics['global_loss'] else 0
                    } if server_state.metrics['rounds'] else {}
                }
                self._send_json_response(status_data)
            except Exception as e:
                print(f"Status error: {str(e)}")
                self._send_json_response({
                    'status': 'error',
                    'message': str(e)
                }, status=500)
        
        # Unknown endpoint
        else:
            self._send_json_response({
                'status': 'error',
                'message': 'Unknown endpoint'
            }, status=404)

def start_server(port=DEFAULT_PORT, num_rounds=10, data_dir=None):
    """
    Start the federated learning server.
    
    Args:
        port: HTTP server port
        num_rounds: Number of federated learning rounds
        data_dir: Directory containing the server dataset
    """
    global server_state
    
    # Initialize server state
    server_state.max_rounds = num_rounds
    
    # Create model save directory
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
    # Load identity mapping if exists
    load_identity_mapping()
    
    # Start HTTP server
    server_address = ('', port)
    httpd = ThreadedHTTPServer(server_address, FederatedServerHandler)
    
    print(f"Starting federated learning server on port {port}")
    print(f"Server will run for {num_rounds} rounds")
    
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
    parser = argparse.ArgumentParser(description='Federated Learning Server')
    
    parser.add_argument('--server_port', type=int, default=DEFAULT_PORT,
                        help='Port to run the server on')
    parser.add_argument('--num_rounds', type=int, default=10,
                        help='Number of federated learning rounds')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory containing the server dataset')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    start_server(port=args.server_port, num_rounds=args.num_rounds, data_dir=args.data_dir) 