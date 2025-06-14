"""
Simple Privacy-Infused Federated Training Runner

This script runs privacy-infused federated training using the existing
demo setup and CASIA-WebFace data distribution.

Usage:
    python run_privacy_training.py
    python run_privacy_training.py --rounds 20 --batch_size 8
"""

import torch
import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path
import logging
import argparse

# Import our components
from privacy_biometric_model import PrivacyBiometricModel, PrivacyAccountant, FederatedModelManager
from privacy_federated_trainer import PrivacyFederatedTrainer, BiometricDataset, create_training_config
from data_loader import FederatedDataManager, FederatedBiometricDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_federated_data_loaders(batch_size=16):
    """
    Create federated data loaders using the same distribution as the demo
    """
    logger.info("Creating federated data loaders...")
    
    try:
        # Try to load real CASIA-WebFace data using FederatedDataManager
        logger.info("Attempting to load CASIA-WebFace data...")
        data_manager = FederatedDataManager("data/partitioned")
        
        # Create data loaders for each node
        data_loaders = {}
        for node_id in ['server', 'client1', 'client2']:
            try:
                # Create privacy-compatible dataloader
                dataloader = data_manager.create_privacy_dataloader(
                    node_name=node_id,
                    batch_size=batch_size,
                    sample_rate=None  # Will be calculated automatically
                )
                data_loaders[node_id] = dataloader
                
                # Get node info
                dataset = data_manager.datasets[node_id]
                node_info = dataset.get_node_info()
                logger.info(f"{node_id}: {node_info['num_images']} images, "
                           f"{node_info['num_identities']} identities, "
                           f"range: {node_info['identity_range'][0]} to {node_info['identity_range'][1]}")
                
            except Exception as node_error:
                logger.warning(f"Could not load data for {node_id}: {node_error}")
        
        if data_loaders:
            logger.info(f"Successfully loaded real data for {len(data_loaders)} nodes")
            return data_loaders
        else:
            raise Exception("No nodes could load real data")
        
    except Exception as e:
        logger.warning(f"Could not load real data ({e}), creating synthetic data for training...")
        return create_synthetic_data_loaders(batch_size)



def create_synthetic_data_loaders(batch_size):
    """Create synthetic data loaders for testing"""
    from torch.utils.data import DataLoader
    
    data_loaders = {}
    
    # Synthetic data parameters (matching demo results)
    node_configs = {
        'server': {'samples': 1000, 'identities': 100},
        'client1': {'samples': 1200, 'identities': 100}, 
        'client2': {'samples': 1100, 'identities': 100}
    }
    
    for node_id, config in node_configs.items():
        # Generate synthetic images (224x224x3)
        images = torch.randn(config['samples'], 3, 224, 224)
        
        # Generate identity labels (0 to 299 for 300 total identities)
        identity_labels = torch.randint(0, 300, (config['samples'],))
        
        # Generate emotion labels (0 to 6 for 7 emotions)
        emotion_labels = torch.randint(0, 7, (config['samples'],))
        
        # Create dataset and dataloader
        dataset = BiometricDataset(images, identity_labels, emotion_labels)
        data_loaders[node_id] = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        logger.info(f"{node_id}: {config['samples']} synthetic samples, "
                   f"{config['identities']} identities")
    
    return data_loaders

def create_optimized_privacy_config():
    """Create optimized privacy configuration for training"""
    config = create_training_config()
    
    # Optimized settings for privacy training
    config.update({
        'num_identities': 300,
        'learning_rate': 0.001,
        'global_learning_rate': 0.01,
        'weight_decay': 1e-4,
        'local_epochs': 3,  # Fewer epochs to preserve privacy budget
        'max_epsilon': 1.0,  # Total privacy budget
        'delta': 1e-5,
        'noise_multiplier': 1.0,  # Balanced noise for privacy vs utility
        'max_grad_norm': 1.0,
        'patience': 10,
        'min_delta': 0.001,
        'use_homomorphic_encryption': True,
        'batch_size': 16
    })
    
    return config

def run_training(args):
    """Run the privacy-infused federated training"""
    
    logger.info("🚀 Starting Privacy-Infused Federated Training")
    logger.info(f"Configuration: {args.rounds} rounds, batch size {args.batch_size}")
    
    # Create results directory
    results_dir = Path("privacy_training_results")
    results_dir.mkdir(exist_ok=True)
    
    # Create federated data loaders
    data_loaders = create_federated_data_loaders(batch_size=args.batch_size)
    
    if not data_loaders:
        logger.error("Failed to create data loaders")
        return None
    
    # Create training configuration
    config = create_optimized_privacy_config()
    config['batch_size'] = args.batch_size
    
    # Log configuration
    logger.info("Training Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Save configuration
    config_file = results_dir / "training_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Initialize trainer
    trainer = PrivacyFederatedTrainer(
        config=config,
        data_loaders=data_loaders,
        save_dir=str(results_dir)
    )
    
    # Run training
    training_start_time = time.time()
    
    try:
        logger.info(f"Starting {args.rounds} rounds of federated training...")
        
        for round_num in range(1, args.rounds + 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"ROUND {round_num}/{args.rounds}")
            logger.info(f"{'='*50}")
            
            # Execute one federated round
            round_results = trainer.train_federated_round()
            
            # Check stopping conditions
            if round_results.get('stopped_early', False):
                logger.info("Training stopped early due to privacy budget exhaustion")
                break
            
            if round_results.get('converged', False):
                logger.info("Training stopped due to convergence")
                break
            
            # Check if any nodes can still train
            if round_results['privacy_status']['active_nodes'] == 0:
                logger.info("No active nodes remaining, stopping training")
                break
            
            # Log round summary
            if 'global_evaluation' in round_results and round_results['global_evaluation']:
                eval_results = round_results['global_evaluation']
                logger.info(f"Round {round_num} Results:")
                logger.info(f"  Global Accuracy: {eval_results['accuracy']:.4f}")
                logger.info(f"  Global Loss: {eval_results['loss']:.4f}")
                logger.info(f"  Active Nodes: {round_results['privacy_status']['active_nodes']}/3")
        
        # Training completed
        training_duration = time.time() - training_start_time
        
        # Final evaluation
        final_evaluation = trainer._evaluate_global_model()
        
        # Save trained models
        trainer.save_trained_models()
        
        # Generate final results
        final_results = {
            'training_completed': True,
            'timestamp': datetime.now().isoformat(),
            'total_rounds': trainer.current_round,
            'total_duration_seconds': training_duration,
            'final_evaluation': final_evaluation,
            'privacy_summary': {
                node_id: accountant.get_privacy_stats() 
                for node_id, accountant in trainer.privacy_accountants.items()
            },
            'configuration': config
        }
        
        # Save final results
        results_file = results_dir / "final_results.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("🎉 PRIVACY-INFUSED FEDERATED TRAINING COMPLETED!")
        print("="*60)
        print(f"Final Accuracy: {final_evaluation['accuracy']:.4f}")
        print(f"Final Loss: {final_evaluation['loss']:.4f}")
        print(f"Total Rounds: {trainer.current_round}")
        print(f"Training Duration: {training_duration/60:.1f} minutes")
        print(f"Results saved to: {results_dir}")
        
        # Privacy budget summary
        print("\nPrivacy Budget Usage:")
        for node_id, stats in final_results['privacy_summary'].items():
            usage_pct = (stats['epsilon_used'] / config['max_epsilon']) * 100
            print(f"  {node_id}: {stats['epsilon_used']:.3f}/{config['max_epsilon']:.1f} ({usage_pct:.1f}%)")
        
        print("="*60)
        
        return final_results
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return None
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Privacy-Infused Federated Training')
    parser.add_argument('--rounds', type=int, default=20, 
                       help='Number of federated rounds (default: 20)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training (default: 16)')
    parser.add_argument('--synthetic', action='store_true',
                       help='Force use of synthetic data')
    
    args = parser.parse_args()
    
    # Run training
    results = run_training(args)
    
    if results:
        logger.info("Training completed successfully!")
    else:
        logger.error("Training failed or was interrupted")

if __name__ == "__main__":
    main() 