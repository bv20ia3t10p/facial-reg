#!/usr/bin/env python
"""
Script to deploy the server and clients from saved models and run them concurrently in a Docker environment.
"""

import os
import sys
import argparse
import json
import subprocess
import time
import signal
import shutil
from pathlib import Path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Deploy federated learning system in Docker')
    
    parser.add_argument('--model_dir', type=str, default='models',
                       help='Directory containing the saved models')
    parser.add_argument('--data_dir', type=str, default='data/partitioned',
                       help='Directory containing the partitioned dataset')
    parser.add_argument('--num_rounds', type=int, default=10,
                       help='Number of federated learning rounds')
    parser.add_argument('--use_dp', action='store_true',
                       help='Use differential privacy during training')
    parser.add_argument('--allow_new_classes', action='store_true',
                       help='Allow clients to handle new classes')
    parser.add_argument('--setup_only', action='store_true',
                       help='Only set up environment without starting Docker')
    parser.add_argument('--clean', action='store_true',
                       help='Clean up Docker containers and shared volumes before starting')
    
    return parser.parse_args()

def setup_environment(args):
    """Set up the environment for federated learning."""
    # Create necessary directories
    os.makedirs('shared', exist_ok=True)
    os.makedirs('shared/models', exist_ok=True)
    
    # Check if training summary exists
    summary_path = os.path.join(args.model_dir, 'training_summary.json')
    if not os.path.exists(summary_path):
        print(f"Error: Training summary not found at {summary_path}")
        print("Please run train_and_save_models.py first.")
        sys.exit(1)
    
    # Load training summary
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    # Copy model files to shared directory
    model_files = [
        ('server_initial_model.h5', 'server_model.h5'),
        ('server_classes.json', 'server_classes.json'),
        ('client1_initial_model.h5', 'client1_model.h5'),
        ('client1_classes.json', 'client1_classes.json'),
        ('client2_initial_model.h5', 'client2_model.h5'),
        ('client2_classes.json', 'client2_classes.json')
    ]
    
    for src_name, dest_name in model_files:
        src_path = os.path.join(args.model_dir, src_name)
        dest_path = os.path.join('shared/models', dest_name)
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dest_path)
            print(f"Copied {src_path} to {dest_path}")
        else:
            print(f"Warning: {src_path} not found, skipping.")
    
    # Create configuration files for Docker
    config = {
        'num_rounds': args.num_rounds,
        'use_dp': args.use_dp,
        'allow_new_classes': args.allow_new_classes,
        'data_paths': {
            'server': summary['dataset']['server_dir'],
            'client1': summary['dataset']['client1_dir'],
            'client2': summary['dataset']['client2_dir']
        }
    }
    
    # Save configuration
    config_path = os.path.join('shared', 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved to {config_path}")
    
    # Create or update the Docker environment file
    env_vars = [
        "NUM_CLIENTS=2",
        f"NUM_ROUNDS={args.num_rounds}",
        f"USE_DP={'True' if args.use_dp else 'False'}",
        f"ALLOW_NEW_CLASSES={'True' if args.allow_new_classes else 'False'}"
    ]
    
    with open('.env', 'w') as f:
        f.write('\n'.join(env_vars))
    
    print("Docker environment file created: .env")
    return config

def update_docker_compose(config):
    """Update docker-compose.yml with the correct settings."""
    compose_file = 'docker-compose.yml'
    
    # Check if docker-compose.yml exists
    if not os.path.exists(compose_file):
        print(f"Error: {compose_file} not found.")
        print("Please ensure you have the correct docker-compose.yml in your workspace.")
        sys.exit(1)
    
    # Read the docker-compose.yml
    with open(compose_file, 'r') as f:
        compose_content = f.read()
    
    # We're not modifying the docker-compose file directly to avoid parsing YAML
    # Instead, we rely on the .env file and the config.json file
    
    print(f"Using existing {compose_file} with environment variables from .env")

def clean_docker_environment():
    """Clean up Docker containers and shared volumes."""
    try:
        # Stop and remove containers
        subprocess.run(['docker-compose', 'down', '--volumes', '--remove-orphans'], check=True)
        print("Stopped and removed existing Docker containers")
        
        # Clean shared directory
        if os.path.exists('shared'):
            for item in os.listdir('shared'):
                item_path = os.path.join('shared', item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            print("Cleaned shared directory")
    
    except subprocess.CalledProcessError as e:
        print(f"Error cleaning Docker environment: {e}")
        return False
    
    return True

def start_docker_environment():
    """Start the Docker environment."""
    try:
        # Start Docker containers
        subprocess.run(['docker-compose', 'up', '--build', '-d'], check=True)
        print("Started Docker containers in background")
        
        # Show logs
        print("\nContainer logs (press Ctrl+C to exit logs, containers will continue running):")
        # Using subprocess.Popen to allow catching KeyboardInterrupt
        process = subprocess.Popen(['docker-compose', 'logs', '-f'])
        
        try:
            process.wait()
        except KeyboardInterrupt:
            # User pressed Ctrl+C, terminate the logs process but keep containers running
            process.terminate()
            print("\nStopped showing logs. Containers are still running.")
            print("To stop containers, run: docker-compose down")
            print("To show logs again, run: docker-compose logs -f")
    
    except subprocess.CalledProcessError as e:
        print(f"Error starting Docker environment: {e}")
        return False
    
    return True

def main():
    args = parse_args()
    
    # Clean Docker environment if requested
    if args.clean:
        print("Cleaning Docker environment...")
        if not clean_docker_environment():
            print("Failed to clean Docker environment, exiting.")
            sys.exit(1)
    
    # Set up environment
    print("Setting up environment...")
    config = setup_environment(args)
    
    # Update docker-compose.yml
    update_docker_compose(config)
    
    # Start Docker environment if not setup_only
    if not args.setup_only:
        print("\nStarting Docker environment...")
        if not start_docker_environment():
            print("Failed to start Docker environment, exiting.")
            sys.exit(1)
    else:
        print("\nEnvironment setup complete. To start Docker containers, run:")
        print("docker-compose up -d")
        print("docker-compose logs -f")

if __name__ == "__main__":
    main() 