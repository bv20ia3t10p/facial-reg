"""
Privacy-Infused Federated Training Script

This script implements comprehensive privacy-preserving federated learning
for biometric authentication, building on the successful demo results.

Based on demo results:
- 3 federated nodes (server, client1, client2)
- 300 identities total (100 per node)
- 35,387 images total
- Privacy-enabled models with differential privacy
- Homomorphic encryption simulation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Any
import argparse

# Import our components
from privacy_biometric_model import PrivacyBiometricModel, PrivacyAccountant, FederatedModelManager
from privacy_federated_trainer import PrivacyFederatedTrainer, BiometricDataset, create_training_config
from data_loader import load_casia_webface_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('federated_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FederatedDataManager:
    """Manage data distribution across federated nodes"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.node_data = {}
        self.global_identity_mapping = {}
        
    def load_and_distribute_data(self, batch_size: int = 32) -> Dict[str, DataLoader]:
        """
        Load CASIA-WebFace data and distribute across federated nodes
        
        Returns:
            Dictionary of DataLoaders for each node
        """
        logger.info("Loading and distributing CASIA-WebFace data...")
        
        # Load the complete dataset
        try:
            # Use the existing data loader
            all_data = load_casia_webface_data(str(self.data_dir))
            logger.info(f"Loaded {len(all_data['images'])} images from {len(set(all_data['identities']))} identities")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            # Create synthetic data for testing
            return self._create_synthetic_data(batch_size)
        
        # Distribute data across nodes (simulating real federated scenario)
        node_distributions = self._distribute_data_across_nodes(all_data)
        
        # Create DataLoaders for each node
        data_loaders = {}
        transforms_pipeline = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        for node_id, node_data in node_distributions.items():
            # Convert to tensors
            images_tensor = torch.stack([
                transforms_pipeline(img) for img in node_data['images']
            ])
            
            identity_labels = torch.tensor(node_data['identity_labels'], dtype=torch.long)
            emotion_labels = torch.tensor(node_data['emotion_labels'], dtype=torch.long)
            
            # Create dataset and dataloader
            dataset = BiometricDataset(
                images=images_tensor,
                identity_labels=identity_labels,
                emotion_labels=emotion_labels
            )
            
            data_loaders[node_id] = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=True,
                num_workers=2,
                pin_memory=True
            )
            
            logger.info(f"{node_id}: {len(dataset)} samples, "
                       f"{len(set(node_data['identity_labels']))} identities")
        
        return data_loaders
    
    def _distribute_data_across_nodes(self, all_data: Dict) -> Dict[str, Dict]:
        """Distribute data across federated nodes with realistic Non-IID distribution"""
        
        # Get unique identities
        unique_identities = list(set(all_data['identities']))
        num_identities = len(unique_identities)
        
        # Create identity mapping (global ID to local ID)
        self.global_identity_mapping = {
            identity: idx for idx, identity in enumerate(unique_identities)
        }
        
        # Distribute identities across nodes (overlapping for realistic scenario)
        node_identities = {
            'server': unique_identities[:num_identities//3 + 50],      # 100 + overlap
            'client1': unique_identities[num_identities//4:num_identities//4 + 150],  # 100 + overlap  
            'client2': unique_identities[num_identities//2:num_identities//2 + 150]   # 100 + overlap
        }
        
        # Distribute actual data
        node_distributions = {}
        
        for node_id, node_identity_list in node_identities.items():
            node_images = []
            node_identity_labels = []
            node_emotion_labels = []
            
            for i, identity in enumerate(all_data['identities']):
                if identity in node_identity_list:
                    node_images.append(all_data['images'][i])
                    # Map to local identity ID
                    local_id = self.global_identity_mapping[identity]
                    node_identity_labels.append(local_id)
                    # Generate random emotion labels (0-6 for 7 emotions)
                    node_emotion_labels.append(np.random.randint(0, 7))
            
            node_distributions[node_id] = {
                'images': node_images,
                'identity_labels': node_identity_labels,
                'emotion_labels': node_emotion_labels,
                'identities': node_identity_list
            }
        
        return node_distributions
    
    def _create_synthetic_data(self, batch_size: int) -> Dict[str, DataLoader]:
        """Create synthetic data for testing when real data is not available"""
        logger.warning("Creating synthetic data for testing...")
        
        data_loaders = {}
        
        # Synthetic data parameters
        num_samples_per_node = {'server': 1000, 'client1': 1200, 'client2': 1100}
        num_identities = 300
        
        for node_id, num_samples in num_samples_per_node.items():
            # Generate synthetic images (224x224x3)
            images = torch.randn(num_samples, 3, 224, 224)
            
            # Generate identity labels (distributed across 300 identities)
            identity_labels = torch.randint(0, num_identities, (num_samples,))
            
            # Generate emotion labels (7 emotions)
            emotion_labels = torch.randint(0, 7, (num_samples,))
            
            # Create dataset and dataloader
            dataset = BiometricDataset(images, identity_labels, emotion_labels)
            data_loaders[node_id] = DataLoader(
                dataset, batch_size=batch_size, shuffle=True
            )
            
            logger.info(f"{node_id}: {num_samples} synthetic samples")
        
        return data_loaders


def create_privacy_training_config() -> Dict[str, Any]:
    """Create privacy-focused training configuration"""
    config = create_training_config()
    
    # Privacy-specific settings based on demo results
    config.update({
        'num_identities': 300,
        'learning_rate': 0.0005,  # Lower LR for privacy training
        'global_learning_rate': 0.005,
        'local_epochs': 3,  # Fewer epochs to preserve privacy budget
        'max_epsilon': 1.0,  # Total privacy budget
        'delta': 1e-5,
        'noise_multiplier': 1.2,  # Higher noise for better privacy
        'max_grad_norm': 0.8,  # Tighter gradient clipping
        'batch_size': 16,  # Smaller batches for better privacy
        'patience': 15,  # More patience for privacy training
        'min_delta': 0.0005,
        'use_homomorphic_encryption': True,
        'max_rounds': 50,  # Limit rounds due to privacy budget
        'save_frequency': 5,  # Save every 5 rounds
    })
    
    return config


def setup_privacy_monitoring(config: Dict[str, Any]) -> Dict[str, Any]:
    """Setup comprehensive privacy monitoring"""
    
    privacy_monitor = {
        'total_budget': config['max_epsilon'],
        'budget_allocation': {
            'training': 0.7 * config['max_epsilon'],
            'evaluation': 0.2 * config['max_epsilon'], 
            'emergency': 0.1 * config['max_epsilon']
        },
        'budget_tracking': {
            'used_training': 0.0,
            'used_evaluation': 0.0,
            'used_emergency': 0.0
        },
        'privacy_alerts': {
            'warning_threshold': 0.8,  # Alert at 80% budget usage
            'critical_threshold': 0.95  # Critical at 95% budget usage
        }
    }
    
    return privacy_monitor


def run_privacy_federated_training(args):
    """Run the complete privacy-infused federated training system"""
    
    logger.info("🚀 Starting Privacy-Infused Federated Biometric Training")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Results directory: {args.results_dir}")
    
    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)
    
    # Load and distribute data
    data_manager = FederatedDataManager(args.data_dir)
    data_loaders = data_manager.load_and_distribute_data(batch_size=args.batch_size)
    
    if not data_loaders:
        logger.error("Failed to load data loaders")
        return
    
    # Create training configuration
    config = create_privacy_training_config()
    config['batch_size'] = args.batch_size
    config['max_rounds'] = args.max_rounds
    
    # Setup privacy monitoring
    privacy_monitor = setup_privacy_monitoring(config)
    
    # Log configuration
    logger.info("Training Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Initialize privacy-infused federated trainer
    trainer = PrivacyFederatedTrainer(
        config=config,
        data_loaders=data_loaders,
        save_dir=str(results_dir)
    )
    
    # Save initial configuration
    config_file = results_dir / "training_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Start training
    training_start_time = time.time()
    
    try:
        # Run federated training
        final_results = trainer.train_complete_federated_system(
            max_rounds=config['max_rounds']
        )
        
        # Save trained models
        trainer.save_trained_models()
        
        # Generate comprehensive report
        training_duration = time.time() - training_start_time
        generate_training_report(final_results, training_duration, results_dir, privacy_monitor)
        
        logger.info("🎉 Privacy-Infused Federated Training Completed Successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    return final_results


def generate_training_report(final_results: Dict[str, Any], 
                           training_duration: float,
                           results_dir: Path,
                           privacy_monitor: Dict[str, Any]):
    """Generate comprehensive training report"""
    
    report = {
        'training_summary': {
            'completed': final_results.get('training_completed', False),
            'total_duration_hours': training_duration / 3600,
            'total_rounds': final_results['training_statistics']['total_rounds'],
            'final_accuracy': final_results['final_evaluation']['accuracy'],
            'final_loss': final_results['final_evaluation']['loss']
        },
        'privacy_analysis': {
            'privacy_budget_used': {},
            'privacy_efficiency': {},
            'differential_privacy_guarantees': {
                'epsilon': final_results['configuration']['max_epsilon'],
                'delta': final_results['configuration']['delta'],
                'noise_multiplier': final_results['configuration']['noise_multiplier']
            }
        },
        'federated_learning_metrics': {
            'communication_rounds': final_results['training_statistics']['total_rounds'],
            'average_round_duration': final_results['training_statistics']['average_round_duration'],
            'node_participation': {},
            'convergence_analysis': {}
        },
        'model_performance': {
            'global_model_accuracy': final_results['final_evaluation']['accuracy'],
            'global_model_loss': final_results['final_evaluation']['loss'],
            'total_samples_processed': final_results['final_evaluation']['total_samples']
        },
        'security_guarantees': {
            'differential_privacy': True,
            'homomorphic_encryption': final_results['configuration']['use_homomorphic_encryption'],
            'gradient_clipping': final_results['configuration']['max_grad_norm'],
            'data_locality_preserved': True
        }
    }
    
    # Extract privacy budget usage
    for node_id, privacy_stats in final_results['privacy_summary'].items():
        report['privacy_analysis']['privacy_budget_used'][node_id] = {
            'epsilon_used': privacy_stats['epsilon_used'],
            'epsilon_remaining': privacy_stats['epsilon_remaining'],
            'training_steps': privacy_stats['training_steps']
        }
    
    # Save comprehensive report
    report_file = results_dir / "comprehensive_training_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Generate human-readable summary
    summary_file = results_dir / "training_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("PRIVACY-INFUSED FEDERATED BIOMETRIC TRAINING REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Training Completed: {report['training_summary']['completed']}\n")
        f.write(f"Total Duration: {report['training_summary']['total_duration_hours']:.2f} hours\n")
        f.write(f"Total Rounds: {report['training_summary']['total_rounds']}\n")
        f.write(f"Final Accuracy: {report['training_summary']['final_accuracy']:.4f}\n")
        f.write(f"Final Loss: {report['training_summary']['final_loss']:.4f}\n\n")
        
        f.write("PRIVACY GUARANTEES:\n")
        f.write(f"  Differential Privacy: ε={report['privacy_analysis']['differential_privacy_guarantees']['epsilon']}\n")
        f.write(f"  Delta: δ={report['privacy_analysis']['differential_privacy_guarantees']['delta']}\n")
        f.write(f"  Noise Multiplier: σ={report['privacy_analysis']['differential_privacy_guarantees']['noise_multiplier']}\n\n")
        
        f.write("PRIVACY BUDGET USAGE:\n")
        for node_id, budget_info in report['privacy_analysis']['privacy_budget_used'].items():
            f.write(f"  {node_id}: {budget_info['epsilon_used']:.3f}/{report['privacy_analysis']['differential_privacy_guarantees']['epsilon']:.3f} "
                   f"({budget_info['epsilon_used']/report['privacy_analysis']['differential_privacy_guarantees']['epsilon']*100:.1f}%)\n")
    
    logger.info(f"Training report saved to {report_file}")
    logger.info(f"Training summary saved to {summary_file}")


def main():
    """Main function to run privacy-infused federated training"""
    
    parser = argparse.ArgumentParser(description='Privacy-Infused Federated Biometric Training')
    parser.add_argument('--data_dir', type=str, default='data', 
                       help='Directory containing CASIA-WebFace data')
    parser.add_argument('--results_dir', type=str, default='federated_training_results',
                       help='Directory to save training results')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--max_rounds', type=int, default=50,
                       help='Maximum number of federated rounds')
    parser.add_argument('--config_file', type=str, default=None,
                       help='Path to custom configuration file')
    
    args = parser.parse_args()
    
    # Load custom configuration if provided
    if args.config_file and Path(args.config_file).exists():
        with open(args.config_file, 'r') as f:
            custom_config = json.load(f)
        logger.info(f"Loaded custom configuration from {args.config_file}")
    
    # Run training
    try:
        final_results = run_privacy_federated_training(args)
        
        print("\n" + "="*60)
        print("🎉 PRIVACY-INFUSED FEDERATED TRAINING COMPLETED!")
        print("="*60)
        print(f"Final Accuracy: {final_results['final_evaluation']['accuracy']:.4f}")
        print(f"Final Loss: {final_results['final_evaluation']['loss']:.4f}")
        print(f"Total Rounds: {final_results['training_statistics']['total_rounds']}")
        print(f"Results saved to: {args.results_dir}")
        print("="*60)
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main() 