#!/usr/bin/env python3
# Simple JPEG extractor for CASIA-WebFace dataset
# This script scans for JPEG signatures in any binary file and extracts them

import os
import sys
import io
import argparse
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description='Extract JPEG images from any binary file by scanning')
    parser.add_argument('--input-file', type=str, required=True, help='Input binary file (e.g., .rec file)')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for extracted images')
    parser.add_argument('--lst-file', type=str, help='Path to .lst file (optional, for class information)')
    parser.add_argument('--min-size', type=int, default=1000, help='Minimum size in bytes for valid images')
    parser.add_argument('--max-size', type=int, default=200000, help='Maximum size in bytes for valid images')
    return parser.parse_args()

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def find_all_jpeg_signatures(data):
    """Find all JPEG start markers in the data"""
    # JPEG files start with bytes FF D8 FF
    signature = b'\xff\xd8\xff'
    
    positions = []
    pos = 0
    
    while True:
        pos = data.find(signature, pos)
        if pos == -1:
            break
        positions.append(pos)
        pos += 1
        
    return positions

def extract_images_by_scanning(input_file, output_dir, lst_file=None, min_size=1000, max_size=200000):
    """
    Extract JPEG images by scanning for signatures
    
    This function looks for JPEG signature markers in a binary file and
    extracts any valid images it finds.
    """
    print(f"Scanning {input_file} for JPEG images...")
    ensure_dir(output_dir)
    
    # Try to read class IDs from .lst file if provided
    class_ids = {}
    if lst_file and os.path.exists(lst_file):
        print(f"Reading class information from {lst_file}")
        try:
            with open(lst_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        img_id = int(parts[0])
                        class_id = int(parts[1])
                        class_ids[img_id] = class_id
            print(f"Found {len(class_ids)} image-to-class mappings")
        except Exception as e:
            print(f"Error reading .lst file: {str(e)}")
            class_ids = {}
    
    # Track unique classes found
    found_classes = set()
    
    # Statistics
    valid_images = 0
    bytes_scanned = 0
    
    # Read the file in chunks to handle large files
    chunk_size = 50 * 1024 * 1024  # 50MB chunks
    
    with open(input_file, 'rb') as f:
        # Get file size for progress reporting
        f.seek(0, 2)
        file_size = f.tell()
        f.seek(0)
        
        print(f"File size: {file_size / (1024*1024):.1f} MB")
        
        # Process file in chunks
        buffer = b''
        overlap = 1024  # Ensure we don't miss signatures at chunk boundaries
        
        while bytes_scanned < file_size:
            # Read a chunk
            f.seek(bytes_scanned)
            new_data = f.read(chunk_size)
            if not new_data:
                break
                
            # Create buffer with overlap from previous chunk
            data = buffer + new_data
            
            # Find all JPEG signatures in this chunk
            jpeg_start_positions = find_all_jpeg_signatures(data)
            
            for i, start_pos in enumerate(jpeg_start_positions):
                # Find end of JPEG (FF D9)
                end_marker = data.find(b'\xff\xd9', start_pos + 3)
                
                # If found end marker and image is valid size
                if end_marker > 0 and end_marker - start_pos + 2 >= min_size and end_marker - start_pos + 2 <= max_size:
                    # Extract JPEG data
                    jpeg_data = data[start_pos:end_marker + 2]
                    
                    # Verify it's a valid JPEG
                    try:
                        # Try to open with PIL
                        image = Image.open(io.BytesIO(jpeg_data))
                        
                        # Check if image is reasonable size for a face (at least 40x40)
                        width, height = image.size
                        if width < 40 or height < 40:
                            continue
                        
                        # Determine class ID - if we have class mapping, use it
                        # Otherwise use a heuristic based on image position
                        if valid_images in class_ids:
                            class_id = class_ids[valid_images]
                        else:
                            # CASIA-WebFace often has sequential images for same person
                            # Group every ~100 images as a class
                            class_id = valid_images // 100
                        
                        found_classes.add(class_id)
                        
                        # Create class directory
                        class_dir = os.path.join(output_dir, f"class_{class_id:05d}")
                        ensure_dir(class_dir)
                        
                        # Save to a class directory
                        output_path = os.path.join(class_dir, f"img_{valid_images:08d}.jpg")
                        
                        # Save the image
                        image.save(output_path)
                        valid_images += 1
                        
                        # Report progress
                        if valid_images % 100 == 0:
                            progress = bytes_scanned / file_size
                            print(f"Extracted {valid_images} images... ({progress:.1%} of file processed)")
                            
                    except Exception as e:
                        # Not a valid image
                        pass
            
            # Keep overlap from end of this chunk for next iteration
            buffer = data[-overlap:] if len(data) > overlap else data
            bytes_scanned += len(new_data)
            
            # Print scanning progress
            if bytes_scanned % (chunk_size * 10) == 0:
                print(f"Scanned {bytes_scanned / (1024*1024):.1f} MB ({bytes_scanned / file_size:.1%} of file)")
    
    print(f"\nExtraction complete. Saved {valid_images} valid JPEG images to {output_dir}")
    print(f"Found {len(found_classes)} unique classes.")
    
    return valid_images

if __name__ == "__main__":
    args = parse_args()
    extract_images_by_scanning(
        args.input_file, 
        args.output_dir, 
        args.lst_file,
        args.min_size,
        args.max_size
    ) 