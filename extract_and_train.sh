#!/bin/bash
# Comprehensive script to extract CASIA-WebFace from RecordIO format, partition it, and train models

echo "=== Facial Recognition Pipeline: Extract, Partition, and Train ==="

# Determine OS type and set platform-specific commands
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    echo "Detected Windows environment..."
    IS_WINDOWS=true
    # Use the Windows-compatible removal commands
    RM_CMD="rm -rf"
    FIND_CMD="find"
    # Check if we're in a Git Bash or similar environment
    if command -v powershell.exe &> /dev/null; then
        echo "PowerShell is available for Windows-specific operations if needed"
    fi
else
    echo "Detected Unix-like environment..."
    IS_WINDOWS=false
    RM_CMD="rm -rf"
    FIND_CMD="find"
fi

# Function to safely handle directory paths across platforms
function create_dir() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        echo "Creating directory: $dir"
        mkdir -p "$dir"
    fi
}

# Set up environment
EXTRACT_MODE="direct"  # Default to direct extraction method
SKIP_EXTRACTION=false  # Default to performing extraction
SKIP_PARTITIONING=false # Default to performing partitioning
SKIP_ENV_SETUP=false   # Default to performing environment setup
SKIP_SYMLINKS=false    # Default to creating symlinks
SKIP_TRAINING=false    # Default to performing model training

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --scan)
      EXTRACT_MODE="scan"
      shift
      ;;
    --direct)
      EXTRACT_MODE="direct"
      shift
      ;;
    --skip-extraction)
      SKIP_EXTRACTION=true
      echo "Skipping extraction step..."
      shift
      ;;
    --skip-partitioning)
      SKIP_PARTITIONING=true
      echo "Skipping partitioning step..."
      shift
      ;;
    --skip-env-setup)
      SKIP_ENV_SETUP=true
      echo "Skipping environment setup and package installation..."
      shift
      ;;
    --skip-symlinks)
      SKIP_SYMLINKS=true
      echo "Skipping symlink creation (for Windows compatibility)..."
      shift
      ;;
    --skip-training)
      SKIP_TRAINING=true
      echo "Skipping model training step..."
      shift
      ;;
    --help)
      echo "Usage: ./extract_and_train.sh [--scan|--direct] [--skip-extraction] [--skip-partitioning] [--skip-env-setup] [--skip-symlinks] [--skip-training]"
      echo "  --scan              Use JPEG scanning method (slower but more reliable)"
      echo "  --direct            Use direct extraction method (faster but less reliable)"
      echo "  --skip-extraction   Skip the extraction step (use if images are already extracted)"
      echo "  --skip-partitioning Skip the partitioning step (use if images are already partitioned)"
      echo "  --skip-env-setup    Skip virtual environment creation and package installation"
      echo "  --skip-symlinks     Skip symlink creation (use on Windows systems)"
      echo "  --skip-training     Skip model training step"
      exit 0
      ;;
    *)
      shift
      ;;
  esac
done

# Check if Python is installed
if ! command -v python3 &> /dev/null && ! command -v python3.11 &> /dev/null; then
    echo "Python 3 is not installed. Please install it first."
    exit 1
fi

# Determine Python command to use
PYTHON_CMD="python3"
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
    echo "Using Python 3.11..."
else
    echo "Using Python 3..."
fi

# Create a virtual environment and install packages (if not skipped)
if [ "$SKIP_ENV_SETUP" = false ]; then
    # Create a virtual environment (for Python 3.11)
    if [ "$PYTHON_CMD" = "python3.11" ]; then
        if [ -d "venv311" ]; then
            echo "Removing existing virtual environment..."
            rm -rf venv311
        fi
        
        echo "Creating new Python 3.11 environment..."
        python3.11 -m venv venv311
        
        if [ $? -ne 0 ]; then
            echo "Failed to create virtual environment. Using system Python..."
        else
            # Activate the environment
            echo "Activating Python 3.11 environment..."
            source ./venv311/bin/activate || { echo "Failed to activate environment, continuing with system Python"; }
            
            # Install required packages
            echo "Installing core packages..."
            $PYTHON_CMD -m pip install --upgrade pip
            $PYTHON_CMD -m pip install Pillow numpy==1.23.5 tensorflow==2.15.0 opencv-python==4.8.1.78 matplotlib==3.7.3 scikit-learn==1.3.2 pandas==2.0.3 tqdm==4.66.1 flask==2.3.3
        fi
    else
        # Install required packages system-wide
        echo "Installing required packages..."
        $PYTHON_CMD -m pip install Pillow numpy tensorflow opencv-python matplotlib scikit-learn pandas tqdm flask
    fi
else
    echo "Environment setup and package installation skipped."
    
    # Check if virtual environment exists and activate it if it does
    if [ -d "venv311" ] && [ -f "venv311/bin/activate" ]; then
        echo "Found existing virtual environment, activating..."
        source ./venv311/bin/activate || echo "Failed to activate existing environment, continuing with system Python"
    fi
fi

# Check Python version
echo "Using Python version:"
$PYTHON_CMD --version

# Step 1: Extract CASIA-WebFace dataset from RecordIO format (if not skipped)
if [ "$SKIP_EXTRACTION" = false ]; then
    echo "=== Step 1: Extracting CASIA-WebFace dataset ==="
    echo "Extraction mode: $EXTRACT_MODE"
    
    # Find RecordIO file
    echo "Looking for RecordIO file..."
    REC_FILE=""
    LST_FILE=""
    
    if [ -f "data/extracted/casia-webface/train.rec" ]; then
        REC_FILE="data/extracted/casia-webface/train.rec"
        # Check for corresponding lst file
        if [ -f "data/extracted/casia-webface/train.lst" ]; then
            LST_FILE="data/extracted/casia-webface/train.lst"
        fi
    elif [ -f "data/extracted/casia-webface-original/train.rec" ]; then
        REC_FILE="data/extracted/casia-webface-original/train.rec"
        # Check for corresponding lst file
        if [ -f "data/extracted/casia-webface-original/train.lst" ]; then
            LST_FILE="data/extracted/casia-webface-original/train.lst"
        fi
    else
        # Try to find it elsewhere
        REC_FILE=$(find . -name "train.rec" -type f | head -n 1)
        
        if [ -z "$REC_FILE" ]; then
            echo "No train.rec file found in this directory structure. Exiting."
            exit 1
        fi
        
        # Look for lst file in the same directory as the rec file
        REC_DIR=$(dirname "$REC_FILE")
        if [ -f "$REC_DIR/train.lst" ]; then
            LST_FILE="$REC_DIR/train.lst"
        fi
    fi
    
    echo "Found RecordIO file at: $REC_FILE"
    if [ -n "$LST_FILE" ]; then
        echo "Found LST file at: $LST_FILE"
    else
        echo "No LST file found. Will use heuristic class assignments."
    fi
    
    # Clean up previous extraction directory
    echo "Cleaning up previous extraction directories..."
    if [ -d "data/extracted/casia-images" ]; then
        echo "Removing existing extraction directory: data/extracted/casia-images"
        $RM_CMD "data/extracted/casia-images"
    fi
    
    # Create output directory
    echo "Creating fresh extraction directory..."
    create_dir "data/extracted/casia-images"
    
    # Run the appropriate extractor based on mode
    if [ "$EXTRACT_MODE" = "scan" ]; then
        echo "Scanning for JPEG images in RecordIO file..."
        if [ -n "$LST_FILE" ]; then
            $PYTHON_CMD scan_extract.py --input-file "$REC_FILE" --output-dir "data/extracted/casia-images" --lst-file "$LST_FILE"
        else
            $PYTHON_CMD scan_extract.py --input-file "$REC_FILE" --output-dir "data/extracted/casia-images"
        fi
    else
        echo "Extracting images from RecordIO file..."
        if [ -n "$LST_FILE" ]; then
            $PYTHON_CMD direct_extract.py --rec-file "$REC_FILE" --output-dir "data/extracted/casia-images" --lst-file "$LST_FILE"
        else
            $PYTHON_CMD direct_extract.py --rec-file "$REC_FILE" --output-dir "data/extracted/casia-images"
        fi
    fi
    
    # Verify extraction was successful
    if [ -d "data/extracted/casia-images" ]; then
        NUM_CLASSES=$(find data/extracted/casia-images -mindepth 1 -maxdepth 1 -type d | wc -l)
        TOTAL_IMAGES=$(find data/extracted/casia-images -type f -name "*.jpg" | wc -l)
        
        echo "Extraction complete."
        echo "Created $NUM_CLASSES class directories with a total of $TOTAL_IMAGES images."
        
        # Try the alternative method if no images were found
        if [ "$TOTAL_IMAGES" -eq 0 ] && [ "$EXTRACT_MODE" = "direct" ]; then
            echo "No images extracted with direct method. Trying JPEG scanning method..."
            if [ -n "$LST_FILE" ]; then
                $PYTHON_CMD scan_extract.py --input-file "$REC_FILE" --output-dir "data/extracted/casia-images" --lst-file "$LST_FILE"
            else
                $PYTHON_CMD scan_extract.py --input-file "$REC_FILE" --output-dir "data/extracted/casia-images"
            fi
            
            # Check again
            NUM_CLASSES=$(find data/extracted/casia-images -mindepth 1 -maxdepth 1 -type d | wc -l)
            TOTAL_IMAGES=$(find data/extracted/casia-images -type f -name "*.jpg" | wc -l)
            
            echo "JPEG scanning complete."
            echo "Created $NUM_CLASSES class directories with a total of $TOTAL_IMAGES images."
        fi
        
        # Only proceed if we found images
        if [ "$TOTAL_IMAGES" -gt 0 ]; then
            # Update symlink for training
            echo "Updating dataset directory..."
            if [ -d "data/extracted/casia-webface" ]; then
                # Only move it if it's not a symlink already
                if [ ! -L "data/extracted/casia-webface" ]; then
                    mv data/extracted/casia-webface data/extracted/casia-webface-original-$(date +%Y%m%d)
                else
                    rm data/extracted/casia-webface
                fi
            fi
            
            # Create symlink or copy directory based on flag
            if [ "$SKIP_SYMLINKS" = true ]; then
                echo "Copying extracted images to data/extracted/casia-webface (symlinks skipped)..."
                # For Windows compatibility, copy directory instead of creating symlink
                create_dir "data/extracted/casia-webface"
                if command -v cp &> /dev/null; then
                    cp -r data/extracted/casia-images/* data/extracted/casia-webface/
                else
                    # Fallback to manual copy if cp is not available
                    $PYTHON_CMD -c "
import os
import shutil

src_dir = 'data/extracted/casia-images'
dst_dir = 'data/extracted/casia-webface'

os.makedirs(dst_dir, exist_ok=True)

print('Copying directory contents...')
for item in os.listdir(src_dir):
    src_path = os.path.join(src_dir, item)
    dst_path = os.path.join(dst_dir, item)
    
    if os.path.isdir(src_path):
        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
    else:
        shutil.copy2(src_path, dst_path)

print('Directory contents copied successfully')
"
                fi
            else
                echo "Creating symlink to extracted images..."
                ln -sf $(pwd)/data/extracted/casia-images data/extracted/casia-webface
            fi
            
            echo "=== Extraction Complete ==="
            echo "The CASIA-WebFace dataset has been extracted with $TOTAL_IMAGES images."
        else
            echo "No images were extracted. The RecordIO file may be in a different format."
            echo "Please check the file format and try again."
            exit 1
        fi
    else
        echo "Extraction failed."
        exit 1
    fi
    
    # Verify the dataset structure
    echo "Verifying extracted dataset structure..."
    $PYTHON_CMD -c "
    import os
    import sys
    
    dataset_dir = 'data/extracted/casia-webface'
    print(f'Dataset directory exists: {os.path.exists(dataset_dir)}')
    
    if os.path.exists(dataset_dir):
        # List the contents of the directory
        contents = os.listdir(dataset_dir)
        print(f'Number of items in dataset directory: {len(contents)}')
        print(f'First 10 items: {contents[:10] if len(contents) >= 10 else contents}')
        
        # Check if it has the expected structure (directories for each identity)
        dirs = [d for d in contents if os.path.isdir(os.path.join(dataset_dir, d))]
        files = [f for f in contents if os.path.isfile(os.path.join(dataset_dir, f))]
        
        print(f'Contains {len(dirs)} directories and {len(files)} files')
        
        # Check a few directories to see if they contain images
        if dirs:
            sample_dir = os.path.join(dataset_dir, dirs[0])
            sample_contents = os.listdir(sample_dir)
            sample_images = [f for f in sample_contents if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            print(f'Sample directory {dirs[0]} contains {len(sample_contents)} items')
            print(f'Sample directory contains {len(sample_images)} image files')
            
            if sample_images:
                print(f'Sample image files: {sample_images[:5]}...')
        else:
            # If no directories, maybe the archive structure was different
            # Look for image files directly in the dataset directory
            image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            print(f'Found {len(image_files)} image files directly in the dataset directory')
            if image_files:
                print(f'Sample image files: {image_files[:5]}...')
    "
else
    echo "=== Extraction step skipped ==="
    echo "Using existing extracted dataset at data/extracted/casia-webface"
    
    # Verify the dataset exists
    if [ ! -d "data/extracted/casia-webface" ]; then
        echo "Error: Extracted dataset not found at data/extracted/casia-webface"
        echo "Please extract the dataset first or remove the --skip-extraction flag"
        exit 1
    fi
fi

# Step 2: Partition the dataset for federated learning with 600 classes and 1:1:1 split (if not skipped)
if [ "$SKIP_PARTITIONING" = false ]; then
    echo "=== Step 2: Partitioning dataset ==="
    echo "Creating balanced partition with 200 classes each for server, client1, and client2..."
    
    # Clean up previous partitioning directories to avoid mixing data
    echo "Cleaning up previous partitioned data..."
    
    # Create/clean partitioning directory
    create_dir "data/partitioned"
    
    # Remove existing partition folders
    $RM_CMD "data/partitioned/server"
    $RM_CMD "data/partitioned/client1"
    $RM_CMD "data/partitioned/client2"
    
    # Create empty partition directories to ensure clean start
    create_dir "data/partitioned/server"
    create_dir "data/partitioned/client1"
    create_dir "data/partitioned/client2"
    
    $PYTHON_CMD -c "
import os
import sys
import numpy as np
import random
import shutil
from pathlib import Path

# Define balanced partitioning function with 200 classes per partition
def partition_dataset_balanced(extracted_dir, output_base_dir, classes_per_partition=200):
    print(f'Using class-based partitioning with {classes_per_partition} classes per partition')
    print(f'Split ratio - Server: 1/3, Client1: 1/3, Client2: 1/3 (equal split)')
    
    # Create output directories
    os.makedirs(f'{output_base_dir}/server', exist_ok=True)
    os.makedirs(f'{output_base_dir}/client1', exist_ok=True)
    os.makedirs(f'{output_base_dir}/client2', exist_ok=True)
    
    # Get all class directories
    all_classes = [d for d in os.listdir(extracted_dir) 
                  if os.path.isdir(os.path.join(extracted_dir, d))]
    
    print(f'Found {len(all_classes)} total classes')
    
    # Sort classes to ensure deterministic behavior
    all_classes.sort()
    
    # Take first 600 classes (200 for each partition)
    total_classes_needed = classes_per_partition * 3
    if len(all_classes) >= total_classes_needed:
        selected_classes = all_classes[:total_classes_needed]
    else:
        print(f'Warning: Not enough classes ({len(all_classes)}) for the requested partitioning')
        print(f'Using all available classes, which will be split as evenly as possible')
        selected_classes = all_classes
    
    print(f'Selected {len(selected_classes)} classes for partitioning')
    
    # Calculate number of classes for each partition
    if len(selected_classes) >= 3:
        classes_per_partition = len(selected_classes) // 3
        remainder = len(selected_classes) % 3
    else:
        classes_per_partition = 1
        remainder = 0
    
    # Partition classes evenly (distribute remainder classes if needed)
    idx1 = classes_per_partition + (1 if remainder > 0 else 0)
    idx2 = idx1 + classes_per_partition + (1 if remainder > 1 else 0)
    
    server_classes = selected_classes[:idx1]
    client1_classes = selected_classes[idx1:idx2]
    client2_classes = selected_classes[idx2:]
    
    # Double-check there's no overlap between partitions
    assert len(set(server_classes).intersection(set(client1_classes))) == 0, 'Server and Client1 partitions should not overlap'
    assert len(set(server_classes).intersection(set(client2_classes))) == 0, 'Server and Client2 partitions should not overlap'
    assert len(set(client1_classes).intersection(set(client2_classes))) == 0, 'Client1 and Client2 partitions should not overlap'
    
    print(f'Server classes: {len(server_classes)}')
    print(f'Client1 classes: {len(client1_classes)}')
    print(f'Client2 classes: {len(client2_classes)}')
    
    # Print first few classes in each partition for verification
    print(f'Server classes (first 5): {server_classes[:5]}')
    print(f'Client1 classes (first 5): {client1_classes[:5]}')
    print(f'Client2 classes (first 5): {client2_classes[:5]}')
    
    # Save class mappings for reference
    with open(f'{output_base_dir}/class_partitions.txt', 'w') as f:
        f.write('Server Classes:\\n')
        f.write('\\n'.join(server_classes))
        f.write('\\n\\nClient1 Classes:\\n')
        f.write('\\n'.join(client1_classes))
        f.write('\\n\\nClient2 Classes:\\n')
        f.write('\\n'.join(client2_classes))
    
    # Copy class directories to respective partitions
    partitions = {
        'server': server_classes,
        'client1': client1_classes,
        'client2': client2_classes
    }
    
    for partition_name, classes in partitions.items():
        print(f'Copying {len(classes)} classes to {partition_name} partition...')
        for class_dir in classes:
            src_dir = os.path.join(extracted_dir, class_dir)
            dst_dir = os.path.join(output_base_dir, partition_name, class_dir)
            
            # Create class directory in partition folder
            os.makedirs(dst_dir, exist_ok=True)
            
            # Copy all images for this class
            images = [f for f in os.listdir(src_dir) 
                     if os.path.isfile(os.path.join(src_dir, f)) and 
                     f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            image_count = 0
            for img in images:
                shutil.copy(os.path.join(src_dir, img), os.path.join(dst_dir, img))
                image_count += 1
            
            print(f'  Copied {image_count} images for class {class_dir}', end='\\r')
        print(f'\\nCompleted copying {len(classes)} classes to {partition_name}')
    
    return partitions

# Apply the balanced partitioning
try:
    partitions = partition_dataset_balanced(
        extracted_dir='data/extracted/casia-webface',
        output_base_dir='data/partitioned',
        classes_per_partition=200
    )
    print('Dataset partitioning complete')
except Exception as e:
    print(f'Error partitioning dataset: {str(e)}')
    print('Dataset partitioning failed')
    sys.exit(1)
"

    # Final verification of partitioned dataset
    echo "Verifying partitioned dataset..."
    $PYTHON_CMD -c "
import os

partitioned_dir = 'data/partitioned'
for subset in ['server', 'client1', 'client2']:
    subset_dir = os.path.join(partitioned_dir, subset)
    if os.path.exists(subset_dir):
        classes = [d for d in os.listdir(subset_dir) if os.path.isdir(os.path.join(subset_dir, d))]
        total_images = sum(len([f for f in os.listdir(os.path.join(subset_dir, c)) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]) 
                        for c in classes)
        print(f'{subset}: {len(classes)} classes, {total_images} total images')
        
        # Print first 5 class directories for verification
        print(f'{subset} class examples: {classes[:5] if len(classes) >= 5 else classes}')
    else:
        print(f'{subset} directory does not exist')
"
else
    echo "=== Partitioning step skipped ==="
    echo "Using existing partitioned dataset at data/partitioned"
    
    # Verify the partitioned directories exist
    if [ ! -d "data/partitioned/server" ] || [ ! -d "data/partitioned/client1" ] || [ ! -d "data/partitioned/client2" ]; then
        echo "Error: Partitioned datasets not found at data/partitioned/{server,client1,client2}"
        echo "Please partition the dataset first or remove the --skip-partitioning flag"
        exit 1
    fi
fi

# Create the models directory
mkdir -p models

# Training steps (if not skipped)
if [ "$SKIP_TRAINING" = false ]; then
    # Step 3: Train the server model
    echo "=== Step 3: Training server model ==="
    echo "Using ResNet50 with Squeeze-and-Excitation and Efficient Channel Attention"
    # Silence TensorFlow warnings in terminal output
    export TF_CPP_MIN_LOG_LEVEL=3
    # Run the separate Python script for server model training with improved parameters
    $PYTHON_CMD train_server_model.py --data-dir data/partitioned/server --model-dir models --epochs 50 --img-size 224 --batch-size 32 --lr 0.001 --debug

    # Step 4: Train client1 model
    echo "=== Step 4: Training client1 model ==="
    echo "Using ResNet50 with Squeeze-and-Excitation and Efficient Channel Attention"
    # Silence TensorFlow warnings in terminal output
    export TF_CPP_MIN_LOG_LEVEL=3
    # Run the separate Python script for client1 model training with improved parameters
    $PYTHON_CMD train_client_model.py --client-id client1 --data-dir data/partitioned/client1 --model-dir models --epochs 50 --img-size 224 --batch-size 32 --lr 0.001 --debug

    # Step 5: Train client2 model
    echo "=== Step 5: Training client2 model ==="
    echo "Using ResNet50 with Squeeze-and-Excitation and Efficient Channel Attention"
    # Silence TensorFlow warnings in terminal output
    export TF_CPP_MIN_LOG_LEVEL=3
    # Run the separate Python script for client2 model training with improved parameters
    $PYTHON_CMD train_client_model.py --client-id client2 --data-dir data/partitioned/client2 --model-dir models --epochs 50 --img-size 224 --batch-size 32 --lr 0.001 --debug
else
    echo "=== Skipping model training steps ==="
    echo "To train the models separately, run:"
    echo "  $PYTHON_CMD train_server_model.py --img-size 224 --batch-size 32 --debug # Train server model"
    echo "  $PYTHON_CMD train_client_model.py --client-id client1 --img-size 224 --batch-size 32 --debug # Train client1 model" 
    echo "  $PYTHON_CMD train_client_model.py --client-id client2 --img-size 224 --batch-size 32 --debug # Train client2 model"
fi

echo "=== Complete Pipeline Finished ==="
if [ "$SKIP_EXTRACTION" = true ]; then
    echo "1. Image extraction: SKIPPED (used existing images)"
else
    echo "1. Images extracted from RecordIO format: data/extracted/casia-webface"
fi

if [ "$SKIP_PARTITIONING" = true ]; then
    echo "2. Data partitioning: SKIPPED (used existing partitioned data)"
else
    echo "2. Data partitioned with 600 classes using equal 1:1:1 split (200 classes each): data/partitioned/"
fi

echo "3. Models trained and saved to: models/"
echo "   - Server model: models/server_model.h5"
echo "   - Client1 model: models/client1_model.h5" 
echo "   - Client2 model: models/client2_model.h5" 