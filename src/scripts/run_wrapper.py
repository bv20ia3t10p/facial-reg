#!/usr/bin/env python
"""
Unified entry point script for facial recognition operations.

This script provides a command-line interface to run different components
of the facial recognition system, including:
- Data extraction and preparation
- Training models (standard, DP, federated)
- Running inference
- Deploying federated learning

Each function is implemented as a subcommand with its own arguments.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Ensure we can import from src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

def setup_extract_parser(subparsers):
    """Setup parser for data extraction."""
    parser = subparsers.add_parser('extract', help='Extract face data from dataset')
    parser.add_argument('--rec_file', type=str, required=True,
                      help='Path to record file (.rec)')
    parser.add_argument('--idx_file', type=str, required=True,
                      help='Path to index file (.idx)')
    parser.add_argument('--output_dir', type=str, default='data/extracted',
                      help='Output directory for extracted faces')
    parser.add_argument('--lst_file', type=str, default=None,
                      help='Path to list file (.lst) containing identities')
    parser.add_argument('--align', action='store_true',
                      help='Align faces using facial landmarks')
    parser.add_argument('--crop_size', type=int, default=112,
                      help='Size to crop faces to')
    parser.add_argument('--max_images', type=int, default=None,
                      help='Maximum number of images to extract (for testing)')
    return parser

def setup_prepare_parser(subparsers):
    """Setup parser for dataset preparation."""
    parser = subparsers.add_parser('prepare', help='Prepare dataset for training')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing the extracted dataset')
    parser.add_argument('--output_dir', type=str, default='data/partitioned',
                      help='Output directory for partitioned dataset')
    parser.add_argument('--validate', action='store_true',
                      help='Validate dataset integrity')
    parser.add_argument('--clean', action='store_true',
                      help='Remove corrupt or invalid images')
    parser.add_argument('--partition', action='store_true',
                      help='Partition dataset for federated learning')
    parser.add_argument('--visualize', action='store_true',
                      help='Visualize sample images from dataset')
    parser.add_argument('--server_ratio', type=float, default=0.6,
                      help='Ratio of data for the server (default: 0.6)')
    parser.add_argument('--min_images', type=int, default=5,
                      help='Minimum number of images per class (default: 5)')
    return parser

def setup_train_parser(subparsers):
    """Setup parser for model training."""
    parser = subparsers.add_parser('train', help='Train face recognition model')
    # Add subparsers for different training modes
    train_subparsers = parser.add_subparsers(dest='train_mode', required=True,
                                           help='Training mode')
    
    # Standard training
    standard_parser = train_subparsers.add_parser('standard', help='Standard supervised training')
    standard_parser.add_argument('--data_dir', type=str, required=True,
                               help='Directory containing training data')
    standard_parser.add_argument('--model_dir', type=str, default='models',
                               help='Directory to save model')
    standard_parser.add_argument('--epochs', type=int, default=50,
                               help='Number of epochs to train')
    standard_parser.add_argument('--batch_size', type=int, default=32,
                               help='Batch size for training')
    standard_parser.add_argument('--use_dp', action='store_true',
                               help='Use differential privacy')
    
    # Federated training preparation
    federated_prep_parser = train_subparsers.add_parser('federated-prep', 
                                                      help='Prepare data and models for federated learning')
    federated_prep_parser.add_argument('--dataset_dir', type=str, required=True,
                                     help='Directory containing the extracted dataset')
    federated_prep_parser.add_argument('--output_dir', type=str, default='data/partitioned',
                                     help='Directory to store the partitioned dataset')
    federated_prep_parser.add_argument('--model_dir', type=str, default='models',
                                     help='Directory to save the trained models')
    federated_prep_parser.add_argument('--use_dp', action='store_true',
                                     help='Use differential privacy during training')
    federated_prep_parser.add_argument('--initial_epochs', type=int, default=5,
                                     help='Number of epochs for initial model training')
    federated_prep_parser.add_argument('--add_unseen_classes', action='store_true',
                                     help='Add unseen classes to clients')
    federated_prep_parser.add_argument('--num_unseen_classes', type=int, default=5,
                                     help='Number of unseen classes to add per client')
    
    return parser

def setup_deploy_parser(subparsers):
    """Setup parser for deployment."""
    parser = subparsers.add_parser('deploy', help='Deploy federated learning system')
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
    return parser

def setup_parser():
    """Setup command-line argument parser."""
    parser = argparse.ArgumentParser(description='Facial Recognition System')
    subparsers = parser.add_subparsers(dest='command', required=True,
                                      help='Command to run')
    
    # Setup subparsers for each command
    setup_extract_parser(subparsers)
    setup_prepare_parser(subparsers)
    setup_train_parser(subparsers)
    setup_deploy_parser(subparsers)
    
    return parser

def run_extract(args):
    """Run data extraction command."""
    from src.scripts.data.data_extraction import extract_faces
    
    extract_faces(
        rec_file=args.rec_file,
        idx_file=args.idx_file,
        output_dir=args.output_dir,
        lst_file=args.lst_file,
        align=args.align,
        crop_size=args.crop_size,
        max_images=args.max_images
    )

def run_prepare(args):
    """Run dataset preparation command."""
    script_path = os.path.join('src', 'scripts', 'data', 'prepare_casia_dataset.py')
    
    cmd = [
        'python', script_path,
        '--data_dir', args.data_dir,
        '--output_dir', args.output_dir,
        '--server_ratio', str(args.server_ratio),
        '--min_images', str(args.min_images)
    ]
    
    if args.validate:
        cmd.append('--validate')
    if args.clean:
        cmd.append('--clean')
    if args.partition:
        cmd.append('--partition')
    if args.visualize:
        cmd.append('--visualize')
    
    subprocess.run(cmd, check=True)

def run_train_standard(args):
    """Run standard training command."""
    script_path = os.path.join('src', 'scripts', 'training', 'train_model.py')
    
    cmd = [
        'python', script_path,
        '--mode', 'server',
        '--data-dir', args.data_dir,
        '--model-dir', args.model_dir,
        '--epochs', str(args.epochs),
        '--batch-size', str(args.batch_size)
    ]
    
    if args.use_dp:
        # Need to use a different script for DP training
        dp_script_path = os.path.join('src', 'scripts', 'training', 'dp_train.py')
        cmd = [
            'python', dp_script_path,
            '--data-dir', args.data_dir,
            '--model-dir', args.model_dir,
            '--epochs', str(args.epochs),
            '--batch-size', str(args.batch_size)
        ]
    
    subprocess.run(cmd, check=True)

def run_train_federated_prep(args):
    """Run federated training preparation command."""
    script_path = os.path.join('src', 'scripts', 'training', 'prepare_and_train_models.py')
    
    cmd = [
        'python', script_path,
        '--dataset_dir', args.dataset_dir,
        '--output_dir', args.output_dir,
        '--model_dir', args.model_dir,
        '--initial_epochs', str(args.initial_epochs)
    ]
    
    if args.use_dp:
        cmd.append('--use_dp')
    if args.add_unseen_classes:
        cmd.append('--add_unseen_classes')
        cmd.extend(['--num_unseen_classes', str(args.num_unseen_classes)])
    
    subprocess.run(cmd, check=True)

def run_deploy(args):
    """Run deployment command."""
    script_path = os.path.join('src', 'scripts', 'deployment', 'deploy_federated_learning.py')
    
    cmd = [
        'python', script_path,
        '--model_dir', args.model_dir,
        '--data_dir', args.data_dir,
        '--num_rounds', str(args.num_rounds)
    ]
    
    if args.use_dp:
        cmd.append('--use_dp')
    if args.allow_new_classes:
        cmd.append('--allow_new_classes')
    if args.setup_only:
        cmd.append('--setup_only')
    if args.clean:
        cmd.append('--clean')
    
    subprocess.run(cmd, check=True)

def main():
    """Main entry point for the script."""
    parser = setup_parser()
    args = parser.parse_args()
    
    # Execute the appropriate command
    if args.command == 'extract':
        run_extract(args)
    elif args.command == 'prepare':
        run_prepare(args)
    elif args.command == 'train':
        if args.train_mode == 'standard':
            run_train_standard(args)
        elif args.train_mode == 'federated-prep':
            run_train_federated_prep(args)
    elif args.command == 'deploy':
        run_deploy(args)

if __name__ == '__main__':
    main() 