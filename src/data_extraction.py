"""
Utilities for extracting and processing the CASIA-WebFace dataset from MXNet RecordIO format.
"""

import os
import mxnet as mx
import numpy as np
from tqdm import tqdm
import cv2
import argparse
from pathlib import Path
import shutil
import pandas as pd

def parse_lst_file(lst_file_path):
    """
    Parse a .lst file which contains image metadata.
    
    Args:
        lst_file_path: Path to the .lst file
        
    Returns:
        List of dictionaries with image information
    """
    print(f"Parsing LST file: {lst_file_path}")
    images_info = []
    
    with open(lst_file_path, 'r') as f:
        for line in tqdm(f):
            line = line.strip()
            if not line:
                continue
                
            # Parse the line: format is usually "index\tlabel\tpath"
            parts = line.split('\t')
            if len(parts) >= 3:
                idx = int(parts[0])
                label = int(parts[1])
                path = parts[2]
                
                images_info.append({
                    'idx': idx,
                    'label': label,
                    'path': path
                })
    
    print(f"Found {len(images_info)} entries in the LST file")
    return images_info

def extract_record_file(rec_file_path, idx_file_path, output_dir, max_images=None):
    """
    Extract images from a RecordIO file.
    
    Args:
        rec_file_path: Path to the .rec file
        idx_file_path: Path to the .idx file
        output_dir: Directory to save extracted images
        max_images: Maximum number of images to extract (None for all)
    
    Returns:
        List of paths to extracted images
    """
    print(f"Extracting images from: {rec_file_path}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create an MXNet RecordIO object
    record = mx.recordio.MXIndexedRecordIO(idx_file_path, rec_file_path, 'r')
    
    # Get a list of all record keys
    record_keys = list(record.keys)
    if max_images:
        record_keys = record_keys[:min(max_images, len(record_keys))]
    
    extracted_paths = []
    
    for idx in tqdm(record_keys):
        # Read the record
        item = record.read_idx(idx)
        header, img = mx.recordio.unpack(item)
        
        # Convert to OpenCV format
        img = mx.image.imdecode(img).asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
        
        # Save image with label information in filename
        label = int(header.label)
        label_dir = os.path.join(output_dir, f"{label:05d}")
        os.makedirs(label_dir, exist_ok=True)
        
        img_path = os.path.join(label_dir, f"{idx:08d}.jpg")
        cv2.imwrite(img_path, img)
        extracted_paths.append(img_path)
    
    print(f"Extracted {len(extracted_paths)} images to {output_dir}")
    return extracted_paths

def create_dataset_structure(lst_file_path, record_file_path, idx_file_path, output_dir, labels_map_path=None):
    """
    Create a structured dataset from CASIA-WebFace RecordIO files.
    
    Args:
        lst_file_path: Path to the .lst file
        record_file_path: Path to the .rec file
        idx_file_path: Path to the .idx file
        output_dir: Directory to save extracted images
        labels_map_path: Optional path to save the labels mapping
        
    Returns:
        DataFrame with paths and labels
    """
    # Parse LST file to get labels
    images_info = parse_lst_file(lst_file_path)
    
    # Create a map of indices to labels
    idx_to_label = {info['idx']: info['label'] for info in images_info}
    
    # Extract images
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract images from recordio
    print(f"Extracting images from: {record_file_path}")
    extracted_paths = extract_record_file(record_file_path, idx_file_path, output_dir)
    
    # Create dataframe with extracted paths and labels
    data = []
    for img_path in extracted_paths:
        img_idx = int(os.path.basename(img_path).split('.')[0])
        if img_idx in idx_to_label:
            label = idx_to_label[img_idx]
            data.append({
                'image_path': img_path,
                'person_id': label,
                'label_idx': label
            })
    
    df = pd.DataFrame(data)
    
    # Save labels mapping if requested
    if labels_map_path:
        unique_labels = sorted(list(set(idx_to_label.values())))
        label_map = {i: label for i, label in enumerate(unique_labels)}
        pd.DataFrame([label_map]).to_csv(labels_map_path, index=False)
    
    return df

def main():
    """Main function for extracting CASIA-WebFace dataset."""
    parser = argparse.ArgumentParser(description="Extract CASIA-WebFace dataset from MXNet RecordIO format")
    
    parser.add_argument('--rec_file', type=str, required=True,
                        help='Path to the .rec file')
    parser.add_argument('--idx_file', type=str, required=True,
                        help='Path to the .idx file')
    parser.add_argument('--lst_file', type=str, required=True,
                        help='Path to the .lst file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save extracted images')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of images to extract')
    
    args = parser.parse_args()
    
    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create dataset structure
    create_dataset_structure(
        args.lst_file,
        args.rec_file,
        args.idx_file,
        args.output_dir
    )

if __name__ == "__main__":
    main() 