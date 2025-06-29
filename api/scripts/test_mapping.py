#!/usr/bin/env python3
"""
Test script to verify the mapping implementation in the API matches 
the approach in improved_privacy_training.py
"""

import os
import sys
import json
import logging
from pathlib import Path
import requests
import argparse

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_improved_privacy_mapping(api_url="http://localhost:8000"):
    """Test that the API mapping matches the improved_privacy_training.py approach"""
    logger.info("Testing improved privacy mapping consistency")
    
    # Step 1: Get the API mapping
    try:
        logger.info("Fetching API mapping...")
        api_response = requests.get(f"{api_url}/api/mapping/refresh")
        api_response.raise_for_status()
        
        api_mapping = api_response.json()
        if not api_mapping.get("success", False):
            logger.error(f"API mapping error: {api_mapping}")
            return False
        
        api_mapping_data = api_mapping.get("mapping", {})
        logger.info(f"API mapping fetched successfully with {len(api_mapping_data)} classes")
        
        # Print first few entries
        first_entries = dict(list(api_mapping_data.items())[:5])
        logger.info(f"API mapping sample: {first_entries}")
    except Exception as e:
        logger.error(f"Error fetching API mapping: {e}")
        return False
    
    # Step 2: Generate mapping using the improved_privacy_training.py approach
    try:
        logger.info("Generating mapping using improved_privacy_training.py approach...")
        
        # Access partitioned data if available
        data_path = Path("/app/data/partitioned")
        if not data_path.exists():
            data_path = Path("./data/partitioned")  # Try relative path
            
        if not data_path.exists():
            logger.error(f"Data directory not found at {data_path}")
            return False
            
        # Collect all unique classes across all partitions
        all_classes = set()
        
        for node_name in ['server', 'client1', 'client2']:
            node_path = data_path / node_name
            if node_path.exists():
                classes = [d.name for d in node_path.iterdir() if d.is_dir()]
                logger.info(f"Found {len(classes)} classes in {node_name}")
                all_classes.update(classes)
        
        # Sort classes for consistent ordering
        sorted_classes = sorted(list(all_classes))
        
        # Create class_to_idx mapping
        class_to_idx = {cls: idx for idx, cls in enumerate(sorted_classes)}
        
        # Create idx_to_class mapping
        idx_to_class = {str(idx): cls for cls, idx in class_to_idx.items()}
        
        logger.info(f"Generated mapping with {len(idx_to_class)} classes")
        
        # Print first few entries for comparison
        first_entries = dict(list(idx_to_class.items())[:5])
        logger.info(f"Generated mapping sample: {first_entries}")
        
    except Exception as e:
        logger.error(f"Error generating mapping: {e}")
        return False
    
    # Step 3: Compare the two mappings
    try:
        if len(api_mapping_data) != len(idx_to_class):
            logger.error(f"Mapping size mismatch: API has {len(api_mapping_data)} classes, generated has {len(idx_to_class)} classes")
            return False
        
        mismatches = 0
        for idx, cls in idx_to_class.items():
            if idx not in api_mapping_data:
                logger.error(f"Index {idx} missing in API mapping")
                mismatches += 1
                continue
                
            if api_mapping_data[idx] != cls:
                logger.error(f"Mapping mismatch at index {idx}: API has '{api_mapping_data[idx]}', generated has '{cls}'")
                mismatches += 1
        
        if mismatches > 0:
            logger.error(f"Found {mismatches} mismatches between API mapping and generated mapping")
            return False
        else:
            logger.info("✅ SUCCESS: API mapping matches the improved_privacy_training.py approach")
            return True
            
    except Exception as e:
        logger.error(f"Error comparing mappings: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test improved privacy mapping consistency")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API URL")
    args = parser.parse_args()
    
    if test_improved_privacy_mapping(args.api_url):
        logger.info("Mapping test passed!")
        sys.exit(0)
    else:
        logger.error("Mapping test failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 