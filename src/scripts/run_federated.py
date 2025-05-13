#!/usr/bin/env python
"""
Script to run the entire workflow from data extraction to federated learning.

This script replaces the shell scripts (run_federated_training.sh and run_federated_training.ps1)
with a Python implementation that works consistently across platforms.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Ensure we can import from the src root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run federated learning workflow')
    parser.add_argument('--dataset_dir', type=str, default='data/extracted',
                        help='Path to extracted dataset (default: data/extracted)')
    parser.add_argument('--partitioned_dir', type=str, default='data/partitioned',
                        help='Path for partitioned data (default: data/partitioned)')
    parser.add_argument('--num_rounds', type=int, default=10,
                        help='Number of federated learning rounds (default: 10)')
    parser.add_argument('--use_dp', action='store_true',
                        help='Enable differential privacy')
    parser.add_argument('--no_allow_new_classes', action='store_true',
                        help='Disable handling of new classes by clients')
    parser.add_argument('--num_unseen_classes', type=int, default=5,
                        help='Number of unseen classes per client (default: 5)')
    parser.add_argument('--no_clean_docker', action='store_true',
                        help="Don't clean Docker environment before starting")
    
    return parser.parse_args()

def check_dataset_exists(dataset_dir):
    """Check if the dataset exists."""
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory {dataset_dir} does not exist.")
        print("Please extract the dataset first using:")
        print(f"python src/scripts/run_wrapper.py extract --rec_file path/to/train.rec --idx_file path/to/train.idx --output_dir {dataset_dir}")
        sys.exit(1)

def train_initial_models(args):
    """Train and save initial models."""
    print("\n==== Step 1: Training initial models ====")
    
    # Prepare command
    cmd = [
        'python', 'src/scripts/run_wrapper.py', 'train', 'federated-prep',
        '--dataset_dir', args.dataset_dir,
        '--output_dir', args.partitioned_dir
    ]
    
    if args.use_dp:
        cmd.append('--use_dp')
    
    if not args.no_allow_new_classes:
        cmd.append('--add_unseen_classes')
        cmd.append('--num_unseen_classes')
        cmd.append(str(args.num_unseen_classes))
    
    # Run command
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print("Error: Initial model training failed.")
        sys.exit(1)

def deploy_federated_learning(args):
    """Deploy federated learning in Docker."""
    print("\n==== Step 2: Deploying federated learning in Docker ====")
    
    # Prepare command
    cmd = [
        'python', 'src/scripts/run_wrapper.py', 'deploy',
        '--model_dir', 'models',
        '--data_dir', args.partitioned_dir,
        '--num_rounds', str(args.num_rounds)
    ]
    
    if args.use_dp:
        cmd.append('--use_dp')
    
    if not args.no_allow_new_classes:
        cmd.append('--allow_new_classes')
    
    if not args.no_clean_docker:
        cmd.append('--clean')
    
    # Run command
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print("Error: Deployment of federated learning failed.")
        sys.exit(1)

def main():
    """Run the main workflow."""
    args = parse_args()
    
    print("=========== Privacy-Preserving Facial Recognition Federated Learning ===========")
    print(f"Dataset directory: {args.dataset_dir}")
    print(f"Partitioned data directory: {args.partitioned_dir}")
    print(f"Number of rounds: {args.num_rounds}")
    print(f"Use differential privacy: {args.use_dp}")
    print(f"Allow new classes: {not args.no_allow_new_classes}")
    print(f"Number of unseen classes: {args.num_unseen_classes}")
    print(f"Clean Docker environment: {not args.no_clean_docker}")
    print("==============================================================================")
    
    # Check if dataset exists
    check_dataset_exists(args.dataset_dir)
    
    # Train and save initial models
    train_initial_models(args)
    
    # Deploy federated learning
    deploy_federated_learning(args)
    
    print("\nFederated learning deployment complete!")
    print("To view logs: docker-compose logs -f")
    print("To stop containers: docker-compose down")

if __name__ == "__main__":
    main() 