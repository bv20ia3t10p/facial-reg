#!/usr/bin/env python
"""
Runner script for the privacy-preserving facial recognition system.
This script serves as a convenient entry point to the various components.
"""

import os
import argparse
import sys
import subprocess

def setup_directories():
    """Create necessary directories for the project."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/extracted", exist_ok=True)
    os.makedirs("templates", exist_ok=True)

def check_requirements():
    """Check if requirements are installed."""
    try:
        import numpy
        import tensorflow
        import tensorflow_privacy
        import tensorflow_federated
        import cv2
        import matplotlib
        import pandas
        import tqdm
        import tenseal
        import pyfhel
        import mxnet
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install required packages using: pip install -r requirements.txt")
        return False
    return True

def run_data_extraction(args):
    """Run the data extraction process."""
    from src.data_extraction import create_dataset_structure
    
    # Create dataset structure
    print(f"Extracting CASIA-WebFace dataset from RecordIO files...")
    create_dataset_structure(
        args.lst_file,
        args.rec_file,
        args.idx_file,
        args.output_dir,
        args.labels_map
    )
    print(f"Dataset extraction complete. Files saved to {args.output_dir}")

def run_training(args):
    """Run the training process with differential privacy."""
    from src.train import train_with_differential_privacy, parse_args as train_parse_args
    
    # Set up training arguments
    train_args = train_parse_args()
    train_args.dataset_path = args.dataset_path
    train_args.use_dp = args.use_dp
    
    # Run training
    train_with_differential_privacy(train_args)

def run_federated_training(args):
    """Run federated training."""
    from src.federated_learning import run_federated_learning, parse_args as fed_parse_args
    
    # Set up federated training arguments
    fed_args = fed_parse_args()
    fed_args.data_dir = args.data_dir
    fed_args.num_clients = args.num_clients
    fed_args.num_rounds = args.num_rounds
    fed_args.fraction_fit = args.fraction_fit
    fed_args.use_dp = args.use_dp
    
    # Run federated training
    run_federated_learning(fed_args)

def run_benchmark(args):
    """Run benchmarks for homomorphic encryption."""
    from src.homomorphic_encryption import benchmark_encryption
    benchmark_encryption(trials=args.trials)

def run_secure_inference(args):
    """Run secure inference with homomorphic encryption."""
    from src.secure_inference import main as secure_inference_main, parse_args as inference_parse_args
    
    # Set up inference arguments
    inference_args = inference_parse_args()
    inference_args.model_path = args.model_path
    inference_args.dataset_path = args.dataset_path
    inference_args.database_path = args.database_path
    inference_args.query_image = args.query_image
    
    # Run inference
    secure_inference_main(inference_args)

def run_web_app(args):
    """Run the web application."""
    app_command = [
        sys.executable, 
        "src/app.py",
        "--model_path", args.model_path or "",
        "--database_path", args.database_path or "",
        "--host", args.host,
        "--port", str(args.port)
    ]
    
    if args.debug:
        app_command.append("--debug")
    
    try:
        subprocess.run(app_command)
    except KeyboardInterrupt:
        print("Web application stopped")

def main():
    """Main entry point for the runner script."""
    parser = argparse.ArgumentParser(description="Privacy-Preserving Facial Recognition System")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Extract dataset command
    extract_parser = subparsers.add_parser("extract", help="Extract CASIA-WebFace dataset from RecordIO format")
    extract_parser.add_argument("--rec_file", type=str, required=True,
                               help="Path to the .rec file")
    extract_parser.add_argument("--idx_file", type=str, required=True,
                               help="Path to the .idx file")
    extract_parser.add_argument("--lst_file", type=str, required=True,
                               help="Path to the .lst file")
    extract_parser.add_argument("--output_dir", type=str, default="data/extracted",
                               help="Directory to save extracted images")
    extract_parser.add_argument("--labels_map", type=str, default="data/labels_map.csv",
                               help="Path to save the labels mapping")
    extract_parser.add_argument("--max_images", type=int, default=None,
                               help="Maximum number of images to extract")
    
    # Training command
    train_parser = subparsers.add_parser("train", help="Train the facial recognition model")
    train_parser.add_argument("--dataset_path", type=str, required=True,
                             help="Path to the CASIA-WebFace dataset")
    train_parser.add_argument("--use_dp", action="store_true",
                             help="Whether to use differential privacy")
    
    # Federated training command
    fed_parser = subparsers.add_parser("federated", help="Train with federated learning")
    fed_parser.add_argument("--data_dir", type=str, required=True,
                           help="Directory containing the extracted dataset")
    fed_parser.add_argument("--num_clients", type=int, default=5,
                           help="Number of federated clients")
    fed_parser.add_argument("--num_rounds", type=int, default=10,
                           help="Number of federated training rounds")
    fed_parser.add_argument("--fraction_fit", type=float, default=1.0,
                           help="Fraction of clients to use in each round")
    fed_parser.add_argument("--use_dp", action="store_true",
                           help="Whether to use differential privacy")
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark homomorphic encryption")
    benchmark_parser.add_argument("--trials", type=int, default=10,
                                 help="Number of trials to run")
    
    # Inference command
    inference_parser = subparsers.add_parser("inference", help="Run secure inference")
    inference_parser.add_argument("--model_path", type=str, default=None,
                                 help="Path to the trained model")
    inference_parser.add_argument("--dataset_path", type=str, default=None,
                                 help="Path to the face dataset")
    inference_parser.add_argument("--database_path", type=str, default=None,
                                 help="Path to save/load the secure database")
    inference_parser.add_argument("--query_image", type=str, required=True,
                                 help="Path to the query image for identification")
    
    # Web app command
    webapp_parser = subparsers.add_parser("webapp", help="Run the web application")
    webapp_parser.add_argument("--model_path", type=str, default=None,
                              help="Path to the trained model")
    webapp_parser.add_argument("--database_path", type=str, default=None,
                              help="Path to the secure database")
    webapp_parser.add_argument("--host", type=str, default="0.0.0.0",
                              help="Host to run the server on")
    webapp_parser.add_argument("--port", type=int, default=5000,
                              help="Port to run the server on")
    webapp_parser.add_argument("--debug", action="store_true",
                              help="Run in debug mode")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Set up project directories
    setup_directories()
    
    # Check requirements
    if not check_requirements():
        return
    
    # Run the appropriate command
    if args.command == "extract":
        run_data_extraction(args)
    elif args.command == "train":
        run_training(args)
    elif args.command == "federated":
        run_federated_training(args)
    elif args.command == "benchmark":
        run_benchmark(args)
    elif args.command == "inference":
        run_secure_inference(args)
    elif args.command == "webapp":
        run_web_app(args)

if __name__ == "__main__":
    main() 