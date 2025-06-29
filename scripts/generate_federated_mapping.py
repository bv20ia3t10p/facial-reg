#!/usr/bin/env python3
"""
Generate identity mappings for federated learning setup
Scans server and client data directories to create consistent mappings
"""

import json
from pathlib import Path
import logging
import hashlib
from collections import OrderedDict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def scan_identity_folders(data_dir: Path) -> set:
    """Scan directory for identity folders"""
    identities = set()
    if not data_dir.exists():
        logger.warning(f"Directory not found: {data_dir}")
        return identities
        
    for item in data_dir.iterdir():
        if item.is_dir():
            identities.add(item.name)
    return identities

def generate_mapping(base_dir: Path) -> dict:
    """Generate identity mapping from all partitions"""
    # Scan all partitions
    partitions = ['server', 'client1', 'client2']
    all_identities = set()
    
    for partition in partitions:
        partition_dir = base_dir / partition
        partition_identities = scan_identity_folders(partition_dir)
        logger.info(f"Found {len(partition_identities)} identities in {partition}")
        all_identities.update(partition_identities)
    
    # Sort identities for consistent indexing
    sorted_identities = sorted(all_identities)
    
    # Create mapping
    mapping = {identity: idx for idx, identity in enumerate(sorted_identities)}
    
    # Create partition-specific statistics
    partition_stats = {}
    for partition in partitions:
        partition_dir = base_dir / partition
        partition_identities = scan_identity_folders(partition_dir)
        
        # Count images per identity
        identity_counts = {}
        for identity in partition_identities:
            img_count = len(list((partition_dir / identity).glob("*.jpg")))
            img_count += len(list((partition_dir / identity).glob("*.png")))
            identity_counts[identity] = img_count
            
        partition_stats[partition] = {
            "total_identities": len(partition_identities),
            "total_images": sum(identity_counts.values()),
            "identity_counts": identity_counts
        }
    
    # Calculate mapping hash
    mapping_str = json.dumps(mapping, sort_keys=True)
    mapping_hash = hashlib.sha256(mapping_str.encode()).hexdigest()
    
    return {
        "version": "1.0.0",
        "mapping": mapping,
        "hash": mapping_hash,
        "total_identities": len(all_identities),
        "partition_stats": partition_stats
    }

def main():
    # Setup paths
    base_dir = Path("data/partitioned")
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    # Generate mapping
    logger.info("Generating federated identity mapping...")
    mapping_data = generate_mapping(base_dir)
    
    # Save mapping
    output_file = output_dir / "identity_mapping.json"
    with open(output_file, 'w') as f:
        json.dump(mapping_data, f, indent=2)
    
    # Log statistics
    logger.info(f"Generated mapping for {mapping_data['total_identities']} unique identities")
    for partition, stats in mapping_data['partition_stats'].items():
        logger.info(f"{partition}: {stats['total_images']} images across {stats['total_identities']} identities")
    
    logger.info(f"Mapping saved to {output_file}")

if __name__ == "__main__":
    main() 