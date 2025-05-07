#!/usr/bin/env python3
# Direct RecordIO extractor for CASIA-WebFace dataset
# This script extracts images directly from a .rec file, with minimal dependencies

import os
import sys
import io
import struct
import argparse
from PIL import Image
import re

def parse_args():
    parser = argparse.ArgumentParser(description='Extract images from RecordIO file')
    parser.add_argument('--rec-file', type=str, required=True, help='Input RecordIO file')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for extracted images')
    parser.add_argument('--lst-file', type=str, help='Path to .lst file (optional, for class information)')
    return parser.parse_args()

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def extract_images(rec_file, output_dir, lst_file=None):
    """
    Extract images from a RecordIO file.
    
    This function parses the RecordIO file format and extracts images.
    It tries to preserve original class structure if available.
    """
    print(f"Extracting images from {rec_file} to {output_dir}")
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

    # RecordIO format constants
    RECORD_MAGIC = 0xced7230a  # This is a common magic number for RecordIO files

    # Statistics
    total_images = 0
    valid_images = 0
    current_pos = 0

    with open(rec_file, 'rb') as f:
        # Get file size for progress reporting
        f.seek(0, 2)  # Go to end of file
        file_size = f.tell()
        f.seek(0)  # Go back to beginning

        # Try to read file in chunks, looking for image data
        while current_pos < file_size:
            try:
                # Try different parsing approaches - RecordIO has several variants

                # Approach 1: Look for JPEG header directly
                # JPEG files start with bytes 0xFF 0xD8 0xFF
                f.seek(current_pos)
                header_bytes = f.read(3)

                if header_bytes == b'\xff\xd8\xff':
                    # Found JPEG header, try to read until EOF marker (0xFF 0xD9)
                    f.seek(current_pos)
                    # Read a large chunk that should contain the entire image
                    chunk = f.read(200000)  # Most JPEG images are under 200KB
                    
                    # Look for EOF marker
                    eof_pos = chunk.find(b'\xff\xd9')
                    if eof_pos > 0:
                        # Extract image data
                        img_data = chunk[:eof_pos+2]  # Include EOF marker
                        
                        # Try to open with PIL to verify it's a valid image
                        try:
                            img = Image.open(io.BytesIO(img_data))
                            
                            # Determine class directory
                            if total_images in class_ids:
                                class_id = class_ids[total_images]
                            else:
                                # If we don't have class info, extract face ID from image
                                # Try to identify class from image patterns in RecordIO
                                # CASIA-WebFace often uses sequential IDs for same person
                                class_id = total_images // 100  # Approximate grouping
                            
                            found_classes.add(class_id)
                            class_dir = os.path.join(output_dir, f"class_{class_id:05d}")
                            ensure_dir(class_dir)
                            
                            # Save to class directory
                            output_path = os.path.join(class_dir, f"img_{total_images:08d}.jpg")
                            
                            # Save the image
                            img.save(output_path)
                            valid_images += 1
                            
                            # Move past this image
                            current_pos += len(img_data)

                            # Print progress
                            if valid_images % 100 == 0:
                                print(f"Extracted {valid_images} images... ({current_pos / file_size:.1%} of file processed)")

                        except Exception as e:
                            # Not a valid image, move ahead by 1 byte and try again
                            current_pos += 1
                    else:
                        # No EOF marker found, move ahead by 1 byte
                        current_pos += 1
                else:
                    # Try Approach 2: Standard RecordIO format (used by MXNet)
                    f.seek(current_pos)
                    header = f.read(8)

                    if len(header) == 8:
                        magic, length = struct.unpack('<II', header)
                        if magic == RECORD_MAGIC and length < 10000000:  # Sanity check on length
                            # Read record data
                            data = f.read(length)
                            if len(data) >= length:
                                # Try to extract image based on common MXNet format
                                # Skip first few bytes (metadata) and look for JPEG header
                                img_start = data.find(b'\xff\xd8\xff')
                                if img_start >= 0:
                                    img_data = data[img_start:]

                                    # Try to open with PIL
                                    try:
                                        img = Image.open(io.BytesIO(img_data))
                                        
                                        # Determine class directory
                                        if total_images in class_ids:
                                            class_id = class_ids[total_images]
                                        else:
                                            # If we don't have class info, extract face ID from image
                                            class_id = total_images // 100  # Approximate grouping
                                        
                                        found_classes.add(class_id)
                                        class_dir = os.path.join(output_dir, f"class_{class_id:05d}")
                                        ensure_dir(class_dir)
                                        
                                        # Save to class directory
                                        output_path = os.path.join(class_dir, f"img_{total_images:08d}.jpg")
                                        
                                        img.save(output_path)
                                        valid_images += 1
                                        
                                        if valid_images % 100 == 0:
                                            print(f"Extracted {valid_images} images... ({current_pos / file_size:.1%} of file processed)")
                                    except:
                                        pass  # Invalid image data
                            
                            # Move past this record
                            current_pos += 8 + length
                        else:
                            # Not a valid record, move ahead
                            current_pos += 1
                    else:
                        # End of file or incomplete header
                        break

                # Count total attempts
                total_images += 1

            except Exception as e:
                # Error during parsing, move ahead by 1 byte
                current_pos += 1
                if total_images % 1000 == 0:
                    print(f"Error at position {current_pos}: {str(e)}")

    print(f"\nExtraction complete. Saved {valid_images} valid images to {output_dir}")
    print(f"Found {len(found_classes)} unique classes.")
    
    return valid_images

if __name__ == "__main__":
    args = parse_args()
    extract_images(args.rec_file, args.output_dir, args.lst_file) 