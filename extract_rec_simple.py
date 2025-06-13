#!/usr/bin/env python
"""
Simple RecordIO extractor for CASIA WebFace dataset.
Extracts images from a RecordIO file using LST file for mapping.
"""

import os
import argparse
import struct
import io
import csv
from PIL import Image
import logging
import sys
import re
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_output_directory(output_dir):
    """Clean the output directory by removing all its contents."""
    if os.path.exists(output_dir):
        logger.info(f"Cleaning output directory: {output_dir}")
        try:
            # First try to remove the directory entirely
            shutil.rmtree(output_dir)
            logger.info(f"Successfully removed existing directory: {output_dir}")
        except Exception as e:
            logger.error(f"Error cleaning directory {output_dir}: {e}")
            
            # If rmtree fails, try to remove files individually
            try:
                for root, dirs, files in os.walk(output_dir, topdown=False):
                    for file in files:
                        try:
                            os.remove(os.path.join(root, file))
                        except Exception as file_e:
                            logger.warning(f"Could not remove file {file}: {file_e}")
                    
                    for dir in dirs:
                        try:
                            os.rmdir(os.path.join(root, dir))
                        except Exception as dir_e:
                            logger.warning(f"Could not remove directory {dir}: {dir_e}")
                
                # Try to remove the main directory again
                if os.path.exists(output_dir):
                    try:
                        os.rmdir(output_dir)
                    except:
                        pass
            except Exception as e2:
                logger.error(f"Failed to clean directory contents: {e2}")
    
    # Create a fresh directory
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created fresh output directory: {output_dir}")
    except Exception as e:
        logger.error(f"Could not create output directory {output_dir}: {e}")
        raise

def parse_lst_file(lst_file):
    """Parse MXNet LST file to get index to label/path mapping."""
    logger.info(f"Parsing LST file: {lst_file}")
    
    # Store index -> label mapping
    idx_to_label = {}
    idx_to_path = {}
    
    try:
        with open(lst_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Split by whitespace (not tab)
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        # Format appears to be:
                        # index path class_id [other fields...]
                        idx = int(parts[0])
                        path = parts[1]
                        
                        # Extract class ID from path
                        # Path format: /raid5data/dplearn/CASIA-WebFace/0000045/001.jpg
                        # Class ID is the directory name (0000045)
                        class_dir = os.path.basename(os.path.dirname(path))
                        try:
                            label = int(class_dir)
                            # Class directories seem to have leading zeros, preserve that format
                            class_name = class_dir
                        except ValueError:
                            # If directory name isn't a number, try to use the third column if available
                            if len(parts) >= 3:
                                try:
                                    label = int(parts[2])
                                    class_name = f"{label:05d}"
                                except ValueError:
                                    label = idx
                                    class_name = f"{label:05d}"
                            else:
                                label = idx
                                class_name = f"{label:05d}"
                        
                        idx_to_label[idx] = label
                        idx_to_path[idx] = {'path': path, 'class': class_name}
                    except ValueError as ve:
                        logger.warning(f"Error parsing line: {line} - {ve}")
                        continue
        
        logger.info(f"Parsed {len(idx_to_label)} entries from LST file")
        return idx_to_label, idx_to_path
    except Exception as e:
        logger.error(f"Error parsing LST file: {e}")
        return {}, {}

def read_idx_file(idx_file):
    """Read MXNet IDX file and return a list of (position, length) tuples."""
    logger.info(f"Reading IDX file: {idx_file}")
    idx_records = []
    
    try:
        with open(idx_file, 'rb') as f:
            idx_data = f.read()
            
        # IDX file format is a series of (position, length) pairs, each 4 bytes (uint32)
        for i in range(0, len(idx_data), 8):
            if i + 8 <= len(idx_data):
                pos, length = struct.unpack('<II', idx_data[i:i+8])
                idx_records.append((i // 8, pos, length))
        
        logger.info(f"Read {len(idx_records)} records from IDX file")
        return idx_records
    except Exception as e:
        logger.error(f"Error reading IDX file: {e}")
        return []

def extract_using_idx_and_lst(rec_file, idx_file, lst_file, output_dir, max_images=None):
    """
    Extract images using IDX and LST files for more reliable extraction.
    
    Args:
        rec_file: Path to the RecordIO file
        idx_file: Path to the IDX file
        lst_file: Path to the LST file
        output_dir: Directory to save extracted images
        max_images: Maximum number of images to extract
    """
    # Clean output directory before extraction
    clean_output_directory(output_dir)
    
    # Parse LST file to get index to label mapping
    idx_to_label, idx_to_path = parse_lst_file(lst_file)
    if not idx_to_label:
        logger.error("Failed to parse LST file, cannot extract images")
        return False
    
    # Read IDX file
    idx_records = read_idx_file(idx_file)
    if not idx_records:
        logger.error("Failed to read IDX file, cannot extract images")
        return False
    
    # Limit extraction if max_images is set
    if max_images is not None:
        idx_records = idx_records[:min(max_images, len(idx_records))]
    
    logger.info(f"Will extract up to {len(idx_records)} images")
    
    # Extract images using IDX and LST information
    image_count = 0
    classes = set()
    
    try:
        with open(rec_file, 'rb') as f:
            for record_idx, pos, length in idx_records:
                try:
                    # Seek to position
                    f.seek(pos)
                    
                    # Read content
                    content = f.read(length)
                    if not content or len(content) < length:
                        logger.warning(f"Incomplete record at index {record_idx}")
                        continue
                    
                    # Debug - print first few bytes to understand format
                    if record_idx < 3:
                        logger.info(f"Record {record_idx} - first 16 bytes: {content[:16].hex()}")
                    
                    # Get label from LST file if available
                    if record_idx in idx_to_label:
                        label = idx_to_label[record_idx]
                        path_info = idx_to_path.get(record_idx, {'class': f"{label:05d}"})
                        class_name = path_info['class']
                    else:
                        # If not in LST, use record index as label
                        label = record_idx
                        class_name = f"{label:05d}"
                    
                    # Create class directory
                    class_dir = os.path.join(output_dir, class_name)
                    os.makedirs(class_dir, exist_ok=True)
                    classes.add(class_name)
                    
                    # Get image filename from path or create one
                    if record_idx in idx_to_path:
                        orig_path = idx_to_path[record_idx]['path']
                        img_filename = os.path.basename(orig_path)
                    else:
                        img_filename = f"{record_idx:08d}.jpg"
                    
                    img_path = os.path.join(class_dir, img_filename)
                    
                    # There are many possible record formats in MXNet RecordIO
                    # Try multiple approaches to extract the image
                    
                    # Approach 1: Find JPEG header in the content
                    # JPEG starts with FF D8 FF
                    jpeg_start = content.find(b'\xff\xd8\xff')
                    if jpeg_start >= 0:
                        # Found JPEG header, extract from there
                        img_data = content[jpeg_start:]
                        try:
                            img = Image.open(io.BytesIO(img_data))
                            img.save(img_path)
                            image_count += 1
                            if image_count % 10 == 0:
                                logger.info(f"Extracted {image_count} images, {len(classes)} classes")
                            continue  # Success, go to next record
                        except Exception as e:
                            logger.warning(f"Found JPEG header but failed to process: {e}")
                    
                    # Approach 2: Find PNG header in the content  
                    # PNG starts with 89 50 4E 47 0D 0A 1A 0A
                    png_start = content.find(b'\x89\x50\x4e\x47\x0d\x0a\x1a\x0a')
                    if png_start >= 0:
                        # Found PNG header, extract from there
                        img_data = content[png_start:]
                        try:
                            img = Image.open(io.BytesIO(img_data))
                            img.save(img_path)
                            image_count += 1
                            if image_count % 10 == 0:
                                logger.info(f"Extracted {image_count} images, {len(classes)} classes")
                            continue  # Success, go to next record
                        except Exception as e:
                            logger.warning(f"Found PNG header but failed to process: {e}")
                    
                    # Approach 3: Try with different offsets
                    extracted = False
                    for offset in [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]:
                        if offset >= len(content):
                            break
                            
                        try:
                            # Try as JPEG
                            img = Image.open(io.BytesIO(content[offset:]))
                            img.save(img_path)
                            logger.info(f"Successfully extracted image at offset {offset}")
                            extracted = True
                            image_count += 1
                            if image_count % 10 == 0:
                                logger.info(f"Extracted {image_count} images, {len(classes)} classes")
                            break
                        except Exception:
                            # Just try the next offset
                            pass
                    
                    # Approach 4: Try to save raw data as bytes and let the OS figure it out
                    if not extracted:
                        try:
                            # Save the raw bytes to file
                            raw_path = img_path.replace('.jpg', '.bin')
                            with open(raw_path, 'wb') as raw_file:
                                raw_file.write(content)
                            logger.info(f"Saved raw data to {raw_path} for record {record_idx}")
                        except Exception as e:
                            logger.warning(f"Failed to save raw data: {e}")
                            
                except Exception as e:
                    logger.warning(f"Failed to process record {record_idx}: {e}")
        
        logger.info(f"Extraction complete. Extracted {image_count} images across {len(classes)} classes")
        return image_count > 0
    except Exception as e:
        logger.error(f"Error during extraction: {e}")
        return False

def analyze_recordio_file(rec_file, idx_file=None, num_records=5):
    """
    Analyze the structure of a RecordIO file to understand its format.
    
    Args:
        rec_file: Path to the RecordIO file
        idx_file: Path to the IDX file (optional)
        num_records: Number of records to analyze
    """
    logger.info(f"Analyzing RecordIO file: {rec_file}")
    
    # Scan for JPEG signatures to understand structure
    logger.info("Scanning for JPEG signatures in the file...")
    jpeg_signature = b'\xff\xd8\xff'
    signatures_found = []
    
    try:
        with open(rec_file, 'rb') as f:
            # Get file size
            f.seek(0, 2)  # Seek to end
            file_size = f.tell()
            f.seek(0)  # Back to beginning
            
            # Read first 10MB chunk to analyze signature positions
            chunk_size = 10 * 1024 * 1024  # 10 MB
            data = f.read(min(chunk_size, file_size))
            
            # Find all JPEG signatures in first chunk
            pos = 0
            while True:
                pos = data.find(jpeg_signature, pos)
                if pos == -1 or len(signatures_found) >= 20:
                    break
                signatures_found.append(pos)
                pos += 3  # Move past signature
            
            # If signatures found, analyze their patterns
            if signatures_found:
                logger.info(f"Found {len(signatures_found)} JPEG signatures in first {len(data) // (1024*1024)}MB")
                
                # Show first few positions
                logger.info(f"First signature positions: {signatures_found[:5]}")
                
                # Calculate distances between signatures
                if len(signatures_found) > 1:
                    distances = [signatures_found[i+1] - signatures_found[i] for i in range(len(signatures_found)-1)]
                    avg_distance = sum(distances) / len(distances)
                    logger.info(f"Average distance between signatures: {avg_distance:.1f} bytes")
                    logger.info(f"Distance samples: {distances[:5]}")
                
                # Show a bit of context around first signature
                if signatures_found:
                    first_sig_pos = signatures_found[0]
                    start = max(0, first_sig_pos - 8)
                    end = min(len(data), first_sig_pos + 24)
                    context = data[start:end]
                    logger.info(f"Context around first signature: {context.hex(' ')}")
            else:
                logger.warning("No JPEG signatures found in the first chunk")
    
    except Exception as e:
        logger.error(f"Error during signature analysis: {e}")
    
    # Now try using IDX file if available
    if idx_file and os.path.exists(idx_file):
        # If IDX file is available, use it to locate records
        idx_records = read_idx_file(idx_file)
        if idx_records:
            logger.info(f"Using IDX file to locate records. Found {len(idx_records)} records.")
            records_to_analyze = idx_records[:min(num_records, len(idx_records))]
            
            with open(rec_file, 'rb') as f:
                for i, (record_idx, pos, length) in enumerate(records_to_analyze):
                    try:
                        f.seek(pos)
                        content = f.read(length)
                        
                        logger.info(f"Record {i} (index {record_idx}):")
                        logger.info(f"  Position: {pos}, Length: {length}")
                        logger.info(f"  First 32 bytes: {content[:32].hex(' ')}")
                        
                        # Check for JPEG signature in record content
                        jpeg_pos = content.find(jpeg_signature)
                        if jpeg_pos >= 0:
                            logger.info(f"  JPEG signature found at offset {jpeg_pos}")
                            
                            # Show context around JPEG signature
                            start = max(0, jpeg_pos - 4)
                            end = min(len(content), jpeg_pos + 20)
                            logger.info(f"  Context around JPEG: {content[start:end].hex(' ')}")
                        else:
                            logger.info(f"  No JPEG signature found in this record")
                        
                        logger.info("")
                    except Exception as e:
                        logger.error(f"Error analyzing record {record_idx}: {e}")
        else:
            logger.warning("Failed to read IDX file")
    
    logger.info("Analysis complete")
    logger.info("Recommendation: Based on analysis, using the direct scanning method with '--scan' is likely to be most reliable")

def extract_images_using_scan(rec_file, output_dir, lst_file=None, max_images=None, max_classes=None):
    """
    Extract images by directly scanning through RecordIO files for JPEG signatures.
    
    Args:
        rec_file: Path to the RecordIO file
        output_dir: Directory to save extracted images
        lst_file: Path to LST file for class organization (optional)
        max_images: Maximum number of images to extract
        max_classes: Maximum number of classes to extract
    """
    # Clean output directory before extraction
    clean_output_directory(output_dir)
    
    # Read LST file to get class information if available
    class_info = {}
    if lst_file and os.path.exists(lst_file):
        logger.info(f"Reading class information from LST file: {lst_file}")
        try:
            with open(lst_file, 'r') as f:
                for line_idx, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        # Format: idx path [other fields...]
                        path = parts[1]
                        
                        # Extract class from path
                        # Path format: /raid5data/dplearn/CASIA-WebFace/0000045/001.jpg
                        class_name = os.path.basename(os.path.dirname(path))
                        
                        # Store class info for this index
                        class_info[line_idx] = class_name
            
            logger.info(f"Found {len(class_info)} class mappings in LST file")
        except Exception as e:
            logger.warning(f"Error parsing LST file for class information: {e}")
    
    # Get unique classes and limit them if max_classes is specified
    unique_classes = set()
    if class_info:
        unique_classes = set(class_info.values())
        logger.info(f"Found {len(unique_classes)} unique classes in LST file")
        
        if max_classes and len(unique_classes) > max_classes:
            # Take a subset of classes - sort for consistent ordering
            sorted_classes = sorted(list(unique_classes))
            unique_classes = set(sorted_classes[:max_classes])
            logger.info(f"Limited to first {max_classes} classes (sorted alphabetically)")
            
            # Filter class_info to only include these classes
            filtered_class_info = {}
            for idx, class_name in class_info.items():
                if class_name in unique_classes:
                    filtered_class_info[idx] = class_name
            class_info = filtered_class_info
            logger.info(f"Filtered class info to {len(class_info)} entries for {len(unique_classes)} classes")
    
    # Create directories for the selected classes
    if unique_classes:
        for class_name in unique_classes:
            class_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
        logger.info(f"Created directories for {len(unique_classes)} classes")
    else:
        # Create default classes if no LST file or if parsing failed
        logger.info("No valid class information found, using sequential classes")
        max_default_classes = max_classes if max_classes else 10
        for i in range(1, max_default_classes + 1):
            class_name = f"{i:05d}"
            class_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            unique_classes.add(class_name)
    
    # Recognized image signatures
    jpeg_signature = b'\xff\xd8\xff'
    
    logger.info(f"Scanning for JPEG images in {rec_file}")
    logger.info(f"Target: {len(unique_classes)} classes, stopping when target is reached")
    
    # Initialize counters and buffers
    image_count = 0
    buffer_size = 8 * 1024 * 1024  # 8 MB buffer
    classes_used = set()
    classes_count = {}  # Track count of images per class
    
    try:
        with open(rec_file, 'rb') as f:
            # Get file size
            f.seek(0, 2)  # Seek to end
            file_size = f.tell()
            f.seek(0)  # Back to beginning
            
            logger.info(f"File size: {file_size} bytes")
            
            # Read in chunks to handle large files
            buffer = b''
            position = 0
            
            while position < file_size:
                if max_images and image_count >= max_images:
                    logger.info(f"Reached maximum number of images: {max_images}")
                    break
                
                # IMPROVED: Stop extraction when we have the target number of classes with sufficient images
                if max_classes and len(classes_used) >= max_classes:
                    # Check if all classes have at least 5 images
                    classes_with_enough_images = sum(1 for c in classes_used if classes_count.get(c, 0) >= 5)
                    if classes_with_enough_images >= max_classes:
                        logger.info(f"Reached target of {max_classes} classes with at least 5 images each")
                        break
                    elif len(classes_used) >= max_classes * 1.5:  # Safety limit to prevent over-extraction
                        logger.info(f"Reached safety limit: {len(classes_used)} classes processed")
                        break
                
                # Read next chunk
                chunk = f.read(buffer_size)
                if not chunk:
                    break
                    
                # Combine with leftovers from previous chunk
                buffer += chunk
                
                # Process buffer for JPEG images
                start_pos = 0
                while True:
                    # Find JPEG signature
                    img_start = buffer.find(jpeg_signature, start_pos)
                    if img_start == -1:
                        break
                    
                    # Find next JPEG signature to estimate end of current image
                    next_img_start = buffer.find(jpeg_signature, img_start + 3)
                    
                    # Extract image data
                    if next_img_start == -1:
                        # If we're at the end of buffer, leave some data for next iteration
                        if len(buffer) - img_start > buffer_size - 1024:
                            # Keep this signature for next iteration if near the end
                            break
                        img_data = buffer[img_start:]
                    else:
                        img_data = buffer[img_start:next_img_start]
                    
                    # Assign to a class directory
                    if image_count in class_info:
                        class_name = class_info[image_count]
                        # Skip if this class is not in our limited set
                        if max_classes and class_name not in unique_classes:
                            start_pos = img_start + 3
                            continue
                    else:
                        # Round-robin assignment to available classes
                        available_classes = list(unique_classes)
                        if available_classes:
                            class_name = available_classes[image_count % len(available_classes)]
                        else:
                            class_name = f"{(image_count % 10) + 1:05d}"
                    
                    # Skip if we already have enough images for this class and we're trying to balance
                    if (max_classes and 
                        len(classes_used) >= max_classes and 
                        class_name not in classes_used and
                        len([c for c in classes_used if classes_count.get(c, 0) >= 10]) >= max_classes * 0.8):
                        start_pos = img_start + 3
                        continue
                    
                    class_dir = os.path.join(output_dir, class_name)
                    os.makedirs(class_dir, exist_ok=True)
                    classes_used.add(class_name)
                    
                    # Save the image
                    try:
                        img = Image.open(io.BytesIO(img_data))
                        img_path = os.path.join(class_dir, f"image_{image_count:08d}.jpg")
                        img.save(img_path)
                        image_count += 1
                        
                        # Update class count
                        classes_count[class_name] = classes_count.get(class_name, 0) + 1
                        
                        if image_count % 50 == 0:
                            logger.info(f"Extracted {image_count} images across {len(classes_used)} classes")
                            # Show progress toward target
                            if max_classes:
                                classes_with_min_images = sum(1 for c in classes_used if classes_count.get(c, 0) >= 5)
                                logger.info(f"Progress: {classes_with_min_images}/{max_classes} classes with ≥5 images")
                    except Exception as e:
                        # Skip invalid images silently - quite common when scanning binary files
                        pass
                    
                    # Move to position after current image start 
                    start_pos = img_start + 3
                
                # Keep the last 4KB in case a JPEG signature is split between chunks
                if buffer:
                    buffer = buffer[-4096:] if len(buffer) > 4096 else buffer
                    position = f.tell() - len(buffer)
                else:
                    position = f.tell()
    
        logger.info(f"Extraction complete. Extracted {image_count} images across {len(classes_used)} classes")
        
        # Print class distribution stats
        if classes_count:
            logger.info("Class distribution:")
            sorted_classes = sorted(classes_count.items(), key=lambda x: x[1], reverse=True)
            
            # Show top classes
            top_classes = sorted_classes[:20]
            for cls, count in top_classes:
                logger.info(f"  {cls}: {count} images")
            
            # Show classes with insufficient images
            insufficient_classes = [(cls, count) for cls, count in sorted_classes if count < 5]
            if insufficient_classes:
                logger.warning(f"Found {len(insufficient_classes)} classes with <5 images:")
                for cls, count in insufficient_classes[:10]:  # Show first 10
                    logger.warning(f"  {cls}: {count} images")
        
        # Verify we have extracted something
        if image_count == 0:
            logger.warning("No images were extracted. The RecordIO file may be in a different format.")
            return False
        
        # Final summary
        classes_with_min_images = sum(1 for c in classes_used if classes_count.get(c, 0) >= 5)
        logger.info(f"Final result: {classes_with_min_images} classes with at least 5 images")
        
        return True
    except Exception as e:
        logger.error(f"Error during scanning: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Extract images from MXNet RecordIO files')
    parser.add_argument('--rec_file', type=str, required=True, help='RecordIO file')
    parser.add_argument('--idx_file', type=str, help='Index file for RecordIO')
    parser.add_argument('--lst_file', type=str, help='LST file containing class information')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--max_images', type=int, help='Maximum number of images to extract (for testing)')
    parser.add_argument('--max_classes', type=int, help='Maximum number of classes to extract')
    parser.add_argument('--scan', action='store_true', help='Use direct scan method for extraction')
    parser.add_argument('--analyze', action='store_true', help='Analyze RecordIO file structure')
    parser.add_argument('--num_records', type=int, default=5, help='Number of records to analyze')
    
    args = parser.parse_args()
    
    # Analyze file if requested
    if args.analyze:
        analyze_recordio_file(args.rec_file, args.idx_file, args.num_records)
        return
    
    # Clean the output directory before extraction starts
    clean_output_directory(args.output_dir)
    
    # Always use scan method as it's the most reliable
    if args.scan or True:  # Always use scan method
        logger.info("Using direct scanning method to extract images")
        if extract_images_using_scan(args.rec_file, args.output_dir, args.lst_file, args.max_images, args.max_classes):
            logger.info(f"Successfully extracted images to {args.output_dir}")
        else:
            logger.error("Extraction failed")
            sys.exit(1)
        return
    
    # The code below is kept for reference but never used since we always use the scan method
    try:
        if args.idx_file and os.path.exists(args.idx_file):
            logger.info("Extracting using IDX and LST files")
            if extract_using_idx_and_lst(args.rec_file, args.idx_file, args.lst_file, args.output_dir, args.max_images):
                logger.info(f"Successfully extracted images to {args.output_dir}")
            else:
                logger.error("Extraction failed")
                sys.exit(1)
        else:
            logger.info("IDX file not available, using direct extraction")
            if extract_from_recordio(args.rec_file, args.output_dir, args.max_images):
                logger.info(f"Successfully extracted images to {args.output_dir}")
            else:
                logger.error("Extraction failed")
                sys.exit(1)
    except Exception as e:
        logger.error(f"Error during extraction: {e}")
        sys.exit(1)

if __name__ == "__main__":
    sys.exit(main()) 