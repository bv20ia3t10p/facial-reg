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
    
    # For each person directory, split the images
    all_images = []
    for person_dir in tqdm(person_dirs, desc="Indexing classes"):
        person_path = os.path.join(extracted_dir, person_dir)
        images = [os.path.join(person_path, img) for img in os.listdir(person_path) 
                 if img.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Skip if no images
        if not images:
            continue
            
        # Add person_id and images to the list
        for img in images:
            all_images.append({
                'image_path': img,
                'person_id': person_dir
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(all_images)
    
    # Group by person_id to ensure stratified split
    partitioned_data = {'server': [], 'client1': [], 'client2': []}
    
    for person_id, group in tqdm(df.groupby('person_id'), desc="Partitioning data"):
        # Get all images for this person
        person_images = group['image_path'].tolist()
        
        # Split into server (90%) and remaining (10%)
        server_images, remaining_images = train_test_split(
            person_images, train_size=server_ratio, random_state=42)
        
        # Split remaining into client1 (5%) and client2 (5%)
        if len(remaining_images) > 1:
            client1_images, client2_images = train_test_split(
                remaining_images, train_size=0.5, random_state=42)
        else:
            # If only one image, assign to client1
            client1_images = remaining_images
            client2_images = []
        
        # Add to partitioned data
        partitioned_data['server'].extend([(img, person_id) for img in server_images])
        partitioned_data['client1'].extend([(img, person_id) for img in client1_images])
        partitioned_data['client2'].extend([(img, person_id) for img in client2_images])
    
    # Create symbolic links or copy files to destination directories
    partition_dirs = {
        'server': server_dir,
        'client1': client1_dir,
        'client2': client2_dir
    }
    
    # Process each partition
    for partition_name, image_list in partitioned_data.items():
        dest_dir = partition_dirs[partition_name]
        
        # Create class directories and copy/link images
        for img_path, person_id in tqdm(image_list, 
                                        desc=f"Creating {partition_name} partition"):
            # Create class directory if it doesn't exist
            class_dir = os.path.join(dest_dir, person_id)
            os.makedirs(class_dir, exist_ok=True)
            
            # Copy or link the image
            img_filename = os.path.basename(img_path)
            dest_path = os.path.join(class_dir, img_filename)
            
            # Use copy instead of link for better compatibility
            shutil.copy2(img_path, dest_path)
    
    print(f"Dataset partitioning complete:")
    print(f"  Server: {len(partitioned_data['server'])} images")
    print(f"  Client 1: {len(partitioned_data['client1'])} images")
    print(f"  Client 2: {len(partitioned_data['client2'])} images")
    
    return {
        'server': server_dir,
        'client1': client1_dir,
        'client2': client2_dir
    }

def create_client_unseen_classes(extracted_dir, client_dir, num_unseen_classes=10):
    """
    Create unseen classes for a client that don't exist in the server dataset.
    
    Args:
        extracted_dir: Directory containing the full extracted dataset
        client_dir: Directory for the client's data
        num_unseen_classes: Number of unseen classes to add
        
    Returns:
        List of unseen class IDs added
    """
    print(f"Adding {num_unseen_classes} unseen classes to {client_dir}")
    
    # Get existing class directories in client directory
    existing_classes = {d for d in os.listdir(client_dir) 
                      if os.path.isdir(os.path.join(client_dir, d))}
    
    # Get all available classes from the full dataset
    all_classes = {d for d in os.listdir(extracted_dir) 
                  if os.path.isdir(os.path.join(extracted_dir, d))}
    
    # Find classes not in the client directory
    candidate_classes = all_classes - existing_classes
    
    # Select random unseen classes
    if len(candidate_classes) < num_unseen_classes:
        unseen_classes = list(candidate_classes)
        print(f"Warning: Only {len(unseen_classes)} unseen classes available")
    else:
        unseen_classes = np.random.choice(
            list(candidate_classes), num_unseen_classes, replace=False)
    
    # Copy images from unseen classes
    for class_id in unseen_classes:
        src_class_dir = os.path.join(extracted_dir, class_id)
        dst_class_dir = os.path.join(client_dir, class_id)
        
        # Create destination directory
        os.makedirs(dst_class_dir, exist_ok=True)
        
        # Get all images for this class
        images = [img for img in os.listdir(src_class_dir) 
                if img.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Copy images
        for img in images:
            src_path = os.path.join(src_class_dir, img)
            dst_path = os.path.join(dst_class_dir, img)
            shutil.copy2(src_path, dst_path)
        
        print(f"Added unseen class {class_id} with {len(images)} images")
    
    return list(unseen_classes)

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