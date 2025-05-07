"""
Data partitioning utilities for federated learning with CASIA-WebFace dataset.

This module handles splitting the dataset between server (90%) and clients (5% each).
"""

import os
import numpy as np
import pandas as pd
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def partition_dataset(extracted_dir, output_base_dir, server_ratio=0.9, client_ratio=0.05):
    """
    Partition the CASIA-WebFace dataset into server and client subsets.
    
    Args:
        extracted_dir: Directory containing the extracted dataset
        output_base_dir: Base directory for the partitioned data
        server_ratio: Fraction of data to allocate to the server
        client_ratio: Fraction of data to allocate to each client (two clients)
        
    Returns:
        Dictionary with paths to the partitioned datasets
    """
    print(f"Partitioning dataset from {extracted_dir}")
    
    # Create output directories
    server_dir = os.path.join(output_base_dir, "server")
    client1_dir = os.path.join(output_base_dir, "client1")
    client2_dir = os.path.join(output_base_dir, "client2")
    
    os.makedirs(server_dir, exist_ok=True)
    os.makedirs(client1_dir, exist_ok=True)
    os.makedirs(client2_dir, exist_ok=True)
    
    # Get all person (class) directories
    person_dirs = [d for d in os.listdir(extracted_dir) 
                  if os.path.isdir(os.path.join(extracted_dir, d))]
    
    print(f"Found {len(person_dirs)} total class directories")
    
    # Shuffle the class directories for random assignment
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(person_dirs)
    
    # Calculate number of classes for each partition (90% server, 5% each client)
    total_classes = len(person_dirs)
    client1_size = int(total_classes * client_ratio)
    client2_size = int(total_classes * client_ratio)
    server_size = total_classes - client1_size - client2_size
    
    # Split class directories
    server_classes = person_dirs[:server_size]
    client1_classes = person_dirs[server_size:server_size+client1_size]
    client2_classes = person_dirs[server_size+client1_size:]
    
    print(f"Class distribution: Server: {len(server_classes)}, Client1: {len(client1_classes)}, Client2: {len(client2_classes)}")
    
    # Dictionary to track stats
    stats = {
        'server': {'classes': 0, 'images': 0},
        'client1': {'classes': 0, 'images': 0},
        'client2': {'classes': 0, 'images': 0}
    }
    
    # Process server classes
    for class_dir in tqdm(server_classes, desc="Processing server classes"):
        src_path = os.path.join(extracted_dir, class_dir)
        dst_path = os.path.join(server_dir, class_dir)
        
        # Skip if source doesn't exist
        if not os.path.exists(src_path):
            continue
            
        # Create class directory
        os.makedirs(dst_path, exist_ok=True)
        
        # Copy all images
        images = [img for img in os.listdir(src_path) if img.endswith(('.jpg', '.jpeg', '.png'))]
        for img in images:
            src_img = os.path.join(src_path, img)
            dst_img = os.path.join(dst_path, img)
            shutil.copy2(src_img, dst_img)
        
        stats['server']['classes'] += 1
        stats['server']['images'] += len(images)
    
    # Process client1 classes
    for class_dir in tqdm(client1_classes, desc="Processing client1 classes"):
        src_path = os.path.join(extracted_dir, class_dir)
        dst_path = os.path.join(client1_dir, class_dir)
        
        # Skip if source doesn't exist
        if not os.path.exists(src_path):
            continue
            
        # Create class directory
        os.makedirs(dst_path, exist_ok=True)
        
        # Copy all images
        images = [img for img in os.listdir(src_path) if img.endswith(('.jpg', '.jpeg', '.png'))]
        for img in images:
            src_img = os.path.join(src_path, img)
            dst_img = os.path.join(dst_path, img)
            shutil.copy2(src_img, dst_img)
        
        stats['client1']['classes'] += 1
        stats['client1']['images'] += len(images)
    
    # Process client2 classes
    for class_dir in tqdm(client2_classes, desc="Processing client2 classes"):
        src_path = os.path.join(extracted_dir, class_dir)
        dst_path = os.path.join(client2_dir, class_dir)
        
        # Skip if source doesn't exist
        if not os.path.exists(src_path):
            continue
            
        # Create class directory
        os.makedirs(dst_path, exist_ok=True)
        
        # Copy all images
        images = [img for img in os.listdir(src_path) if img.endswith(('.jpg', '.jpeg', '.png'))]
        for img in images:
            src_img = os.path.join(src_path, img)
            dst_img = os.path.join(dst_path, img)
            shutil.copy2(src_img, dst_img)
        
        stats['client2']['classes'] += 1
        stats['client2']['images'] += len(images)
    
    print(f"Dataset partitioning complete.")
    print(f"Server: {stats['server']['classes']} classes, {stats['server']['images']} total images")
    print(f"Client1: {stats['client1']['classes']} classes, {stats['client1']['images']} total images")
    print(f"Client2: {stats['client2']['classes']} classes, {stats['client2']['images']} total images")
    
    return {
        'server': server_dir,
        'client1': client1_dir,
        'client2': client2_dir
    }

def create_client_unseen_classes(extracted_dir, client_dir, num_unseen=10):
    """
    Add unseen classes to client dataset.
    
    Args:
        extracted_dir: Directory containing the full extracted dataset
        client_dir: Client directory to add unseen classes to
        num_unseen: Number of unseen classes to add
    """
    print(f"Adding {num_unseen} unseen classes to {client_dir}")
    
    # Get all person directories in the full dataset
    all_person_dirs = [d for d in os.listdir(extracted_dir) 
                     if os.path.isdir(os.path.join(extracted_dir, d))]
    
    # Get person directories already in client data
    client_person_dirs = [d for d in os.listdir(client_dir)
                        if os.path.isdir(os.path.join(client_dir, d))]
    
    # Find unseen persons (not in client data)
    unseen_persons = [p for p in all_person_dirs if p not in client_person_dirs]
    
    # If not enough unseen persons, just use what we have
    num_unseen = min(num_unseen, len(unseen_persons))
    
    if num_unseen == 0:
        print("No unseen classes available to add.")
        return
    
    # Randomly select unseen persons
    np.random.seed(42)  # For reproducibility
    selected_unseen = np.random.choice(unseen_persons, size=num_unseen, replace=False)
    
    # Add unseen classes to client
    unseen_added = 0
    total_images = 0
    
    for person_dir in tqdm(selected_unseen, desc="Adding unseen classes"):
        src_path = os.path.join(extracted_dir, person_dir)
        dst_path = os.path.join(client_dir, person_dir)
        
        # Skip if source doesn't exist
        if not os.path.exists(src_path):
            continue
            
        # Create class directory
        os.makedirs(dst_path, exist_ok=True)
        
        # Copy all images
        images = [img for img in os.listdir(src_path) if img.endswith(('.jpg', '.jpeg', '.png'))]
        for img in images:
            src_img = os.path.join(src_path, img)
            dst_img = os.path.join(dst_path, img)
            shutil.copy2(src_img, dst_img)
        
        unseen_added += 1
        total_images += len(images)
    
    print(f"Added {unseen_added} unseen classes with {total_images} images to {client_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Partition CASIA-WebFace dataset for federated learning")
    
    parser.add_argument('--extracted_dir', type=str, required=True,
                       help='Directory containing the extracted dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for partitioned data')
    parser.add_argument('--add_unseen', action='store_true',
                       help='Add unseen classes to clients')
    parser.add_argument('--num_unseen', type=int, default=10,
                       help='Number of unseen classes to add per client')
    
    args = parser.parse_args()
    
    # Partition the dataset
    partitions = partition_dataset(args.extracted_dir, args.output_dir)
    
    # Add unseen classes if requested
    if args.add_unseen:
        for client_name in ['client1', 'client2']:
            create_client_unseen_classes(
                args.extracted_dir, 
                partitions[client_name],
                args.num_unseen
            ) 