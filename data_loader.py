"""
Federated Data Loader for Partitioned CASIA-WebFace Dataset
Supports privacy-preserving federated learning with differential privacy
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
from typing import Dict, List, Tuple, Optional
from opacus.utils.uniform_sampler import UniformWithReplacementSampler

class FederatedBiometricDataset(Dataset):
    """Dataset class for federated biometric learning with privacy support"""
    
    def __init__(self, 
                 data_root: str, 
                 node_name: str,
                 transform: Optional[transforms.Compose] = None,
                 emotion_labels: Optional[Dict] = None):
        """
        Initialize federated dataset for a specific node
        
        Args:
            data_root: Path to data/partitioned/
            node_name: 'server', 'client1', or 'client2'
            transform: Image transformations
            emotion_labels: Optional emotion labels for multi-task learning
        """
        self.data_root = data_root
        self.node_name = node_name
        self.node_path = os.path.join(data_root, node_name)
        self.transform = transform or self._get_default_transforms()
        self.emotion_labels = emotion_labels or {}
        
        # Load identity mapping and image paths
        self.identity_dirs = sorted([d for d in os.listdir(self.node_path) 
                                   if d.startswith('0') and os.path.isdir(os.path.join(self.node_path, d))])
        
        # Create identity to class mapping
        self.identity_to_class = {identity: idx for idx, identity in enumerate(self.identity_dirs)}
        self.class_to_identity = {idx: identity for identity, idx in self.identity_to_class.items()}
        
        # Build image paths and labels
        self.image_paths = []
        self.identity_labels = []
        self.emotion_labels_list = []
        
        self._build_dataset()
        
        print(f"Loaded {len(self.image_paths)} images from {len(self.identity_dirs)} identities for {node_name}")
        print(f"Identity range: {min(self.identity_dirs)} to {max(self.identity_dirs)}")
    
    def _get_default_transforms(self):
        """Default image transformations for face recognition"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _build_dataset(self):
        """Build the dataset by scanning all identity directories"""
        for identity_dir in self.identity_dirs:
            identity_path = os.path.join(self.node_path, identity_dir)
            identity_class = self.identity_to_class[identity_dir]
            
            # Get all images for this identity
            image_files = [f for f in os.listdir(identity_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for image_file in image_files:
                image_path = os.path.join(identity_path, image_file)
                self.image_paths.append(image_path)
                self.identity_labels.append(identity_class)
                
                # Generate synthetic emotion label (for demo purposes)
                # In real implementation, you'd have actual emotion annotations
                emotion_label = self._generate_synthetic_emotion(identity_dir, image_file)
                self.emotion_labels_list.append(emotion_label)
    
    def _generate_synthetic_emotion(self, identity_dir: str, image_file: str) -> int:
        """Generate synthetic emotion labels for demo purposes"""
        # Use hash of identity + image for consistent synthetic emotions
        hash_val = hash(identity_dir + image_file) % 7
        return hash_val  # 0-6 for 7 emotion classes
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        identity_label = self.identity_labels[idx]
        emotion_label = self.emotion_labels_list[idx]
        
        # Load and transform image
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            image = torch.zeros(3, 224, 224)
        
        return image, identity_label, emotion_label
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Get the distribution of samples per class"""
        distribution = {}
        for label in self.identity_labels:
            distribution[label] = distribution.get(label, 0) + 1
        return distribution
    
    def get_node_info(self) -> Dict:
        """Get information about this node's data"""
        return {
            "node_name": self.node_name,
            "num_identities": len(self.identity_dirs),
            "num_images": len(self.image_paths),
            "identity_range": (min(self.identity_dirs), max(self.identity_dirs)),
            "avg_images_per_identity": len(self.image_paths) / len(self.identity_dirs)
        }

class FederatedDataManager:
    """Manager for federated learning data across multiple nodes"""
    
    def __init__(self, data_root: str = "data/partitioned"):
        self.data_root = data_root
        self.nodes = ["server", "client1", "client2"]
        self.datasets = {}
        self.dataloaders = {}
        
        # Global identity mapping across all nodes
        self.global_identity_mapping = self._build_global_mapping()
        
    def _build_global_mapping(self) -> Dict[str, int]:
        """Build global identity mapping across all federated nodes"""
        all_identities = []
        
        for node in self.nodes:
            node_path = os.path.join(self.data_root, node)
            if os.path.exists(node_path):
                identities = sorted([d for d in os.listdir(node_path) 
                                   if d.startswith('0') and os.path.isdir(os.path.join(node_path, d))])
                all_identities.extend(identities)
        
        # Create global mapping
        unique_identities = sorted(list(set(all_identities)))
        global_mapping = {identity: idx for idx, identity in enumerate(unique_identities)}
        
        print(f"Global identity mapping: {len(unique_identities)} unique identities")
        return global_mapping
    
    def create_node_dataset(self, node_name: str, transform: Optional[transforms.Compose] = None) -> FederatedBiometricDataset:
        """Create dataset for a specific federated node"""
        dataset = FederatedBiometricDataset(
            data_root=self.data_root,
            node_name=node_name,
            transform=transform
        )
        self.datasets[node_name] = dataset
        return dataset
    
    def create_privacy_dataloader(self, 
                                node_name: str, 
                                batch_size: int = 32,
                                sample_rate: Optional[float] = None) -> DataLoader:
        """Create privacy-compatible DataLoader for differential privacy"""
        
        if node_name not in self.datasets:
            self.create_node_dataset(node_name)
        
        dataset = self.datasets[node_name]
        
        # Calculate sample rate for uniform sampling (required for DP)
        if sample_rate is None:
            sample_rate = batch_size / len(dataset)
        
        # Use uniform sampling for privacy guarantees
        sampler = UniformWithReplacementSampler(
            num_samples=len(dataset),
            sample_rate=sample_rate,
        )
        
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=0,  # Set to 0 for privacy compatibility
            pin_memory=True,
        )
        
        self.dataloaders[node_name] = dataloader
        return dataloader
    
    def get_federated_stats(self) -> Dict:
        """Get statistics about the federated data distribution"""
        stats = {
            "total_nodes": len(self.nodes),
            "global_identities": len(self.global_identity_mapping),
            "node_stats": {}
        }
        
        total_images = 0
        for node in self.nodes:
            if node in self.datasets:
                node_info = self.datasets[node].get_node_info()
                stats["node_stats"][node] = node_info
                total_images += node_info["num_images"]
        
        stats["total_images"] = total_images
        return stats
    
    def simulate_new_employee_enrollment(self, target_node: str = "client2") -> Dict:
        """Simulate adding a new employee to demonstrate dynamic expansion"""
        # This would be used for testing new employee enrollment
        # In real implementation, this would handle actual new data
        
        current_stats = self.get_federated_stats()
        new_employee_id = f"0000{len(self.global_identity_mapping):03d}"
        
        return {
            "new_employee_id": new_employee_id,
            "assigned_node": target_node,
            "current_global_identities": len(self.global_identity_mapping),
            "new_global_identities": len(self.global_identity_mapping) + 1,
            "expansion_required": True
        }

# Demo usage and testing
if __name__ == "__main__":
    # Initialize federated data manager
    data_manager = FederatedDataManager("data/partitioned")
    
    # Create datasets for all nodes
    for node in ["server", "client1", "client2"]:
        print(f"\n=== Creating dataset for {node} ===")
        dataset = data_manager.create_node_dataset(node)
        
        # Create privacy-compatible dataloader
        dataloader = data_manager.create_privacy_dataloader(node, batch_size=16)
        
        # Test loading a batch
        for batch_idx, (images, identity_labels, emotion_labels) in enumerate(dataloader):
            print(f"Batch {batch_idx}: Images {images.shape}, Identities {identity_labels.shape}, Emotions {emotion_labels.shape}")
            print(f"Identity range in batch: {identity_labels.min().item()} to {identity_labels.max().item()}")
            if batch_idx == 0:  # Only test first batch
                break
    
    # Print federated statistics
    print("\n=== Federated Learning Statistics ===")
    stats = data_manager.get_federated_stats()
    print(json.dumps(stats, indent=2))
    
    # Simulate new employee enrollment
    print("\n=== New Employee Enrollment Simulation ===")
    enrollment_info = data_manager.simulate_new_employee_enrollment()
    print(json.dumps(enrollment_info, indent=2)) 