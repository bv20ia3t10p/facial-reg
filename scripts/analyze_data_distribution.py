import os
import logging
from pathlib import Path
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_data_distribution(data_dir: Path):
    """Analyze the distribution of training data across classes"""
    stats = {
        'samples_per_class': defaultdict(int),
        'total_samples': 0,
        'class_counts': defaultdict(int),
        'class_ranges': defaultdict(list)
    }
    
    # Analyze each partition
    for partition in ['server', 'client1', 'client2']:
        partition_dir = data_dir / partition
        if not partition_dir.exists():
            logger.warning(f"Partition directory not found: {partition_dir}")
            continue
            
        # Count samples per class
        for class_dir in partition_dir.iterdir():
            if not class_dir.is_dir():
                continue
                
            class_id = class_dir.name
            num_samples = len(list(class_dir.glob('*.jpg')))
            stats['samples_per_class'][class_id] += num_samples
            stats['total_samples'] += num_samples
            stats['class_counts'][partition] += 1
            stats['class_ranges'][partition].append(int(class_id))
    
    # Create analysis plots directory
    plots_dir = Path(__file__).parent.parent / 'analysis_plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Plot sample distribution
    plt.figure(figsize=(15, 5))
    
    # Sort class IDs numerically
    sorted_classes = sorted(stats['samples_per_class'].keys(), key=lambda x: int(x))
    samples = [stats['samples_per_class'][c] for c in sorted_classes]
    
    plt.subplot(1, 2, 1)
    plt.bar(range(len(sorted_classes)), samples)
    plt.title('Samples per Class')
    plt.xlabel('Class Index (Sorted by ID)')
    plt.ylabel('Number of Samples')
    
    # Plot class ID ranges
    plt.subplot(1, 2, 2)
    for i, partition in enumerate(['server', 'client1', 'client2']):
        if partition in stats['class_ranges']:
            ranges = stats['class_ranges'][partition]
            plt.scatter([i]*len(ranges), ranges, alpha=0.5, label=partition)
    plt.title('Class ID Ranges by Partition')
    plt.xlabel('Partition')
    plt.ylabel('Class ID')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'data_distribution.png')
    
    # Save statistics
    stats_file = plots_dir / 'data_stats.json'
    with open(stats_file, 'w') as f:
        # Convert defaultdict to regular dict for JSON serialization
        json_stats = {
            'samples_per_class': dict(stats['samples_per_class']),
            'total_samples': stats['total_samples'],
            'class_counts': dict(stats['class_counts']),
            'class_ranges': {k: sorted(v) for k, v in stats['class_ranges'].items()}
        }
        json.dump(json_stats, f, indent=2)
    
    # Log summary statistics
    logger.info("\nData Distribution Summary:")
    logger.info(f"Total samples: {stats['total_samples']}")
    for partition in ['server', 'client1', 'client2']:
        if partition in stats['class_counts']:
            logger.info(f"{partition} classes: {stats['class_counts'][partition]}")
            if partition in stats['class_ranges']:
                ranges = stats['class_ranges'][partition]
                logger.info(f"{partition} ID range: {min(ranges)} - {max(ranges)}")
    
    # Analyze potential issues
    logger.info("\nPotential Issues:")
    
    # Check for class imbalance
    mean_samples = np.mean(list(stats['samples_per_class'].values()))
    std_samples = np.std(list(stats['samples_per_class'].values()))
    threshold = mean_samples - 2*std_samples
    
    imbalanced_classes = [
        (class_id, count) 
        for class_id, count in stats['samples_per_class'].items()
        if count < threshold
    ]
    
    if imbalanced_classes:
        logger.warning(f"Found {len(imbalanced_classes)} potentially underrepresented classes:")
        for class_id, count in sorted(imbalanced_classes, key=lambda x: x[1]):
            logger.warning(f"  Class {class_id}: {count} samples (mean: {mean_samples:.1f})")
    
    # Check for ID gaps
    all_ids = sorted([int(id_) for partition in stats['class_ranges'].values() for id_ in partition])
    gaps = []
    for i in range(len(all_ids)-1):
        if all_ids[i+1] - all_ids[i] > 1:
            gaps.append((all_ids[i], all_ids[i+1]))
    
    if gaps:
        logger.warning(f"\nFound {len(gaps)} gaps in class ID sequence:")
        for start, end in gaps:
            logger.warning(f"  Gap between {start} and {end}")

def main():
    # Get project root directory
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data' / 'partitioned'
    
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return
    
    logger.info(f"Analyzing data distribution in: {data_dir}")
    analyze_data_distribution(data_dir)

if __name__ == "__main__":
    main() 