"""
Quick Privacy Training Test

This script runs a quick test of the privacy-infused federated training
with a small number of rounds to verify everything works correctly.
"""

import torch
import numpy as np
import logging
from pathlib import Path

# Import our components
from privacy_biometric_model import PrivacyBiometricModel, PrivacyAccountant, FederatedModelManager
from privacy_federated_trainer import PrivacyFederatedTrainer, BiometricDataset, create_training_config
from torch.utils.data import DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_data_loaders(batch_size=8):
    """Create small synthetic data loaders for quick testing"""
    logger.info("Creating test data loaders...")
    
    data_loaders = {}
    
    # Small test data (much smaller than demo for quick testing)
    node_configs = {
        'server': {'samples': 100, 'identities': 50},
        'client1': {'samples': 120, 'identities': 50}, 
        'client2': {'samples': 110, 'identities': 50}
    }
    
    for node_id, config in node_configs.items():
        # Generate synthetic images (224x224x3)
        images = torch.randn(config['samples'], 3, 224, 224)
        
        # Generate identity labels (0 to 149 for 150 total identities)
        identity_labels = torch.randint(0, 150, (config['samples'],))
        
        # Generate emotion labels (0 to 6 for 7 emotions)
        emotion_labels = torch.randint(0, 7, (config['samples'],))
        
        # Create dataset and dataloader
        dataset = BiometricDataset(images, identity_labels, emotion_labels)
        data_loaders[node_id] = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        logger.info(f"{node_id}: {config['samples']} test samples")
    
    return data_loaders

def create_test_config():
    """Create test configuration for quick training"""
    config = create_training_config()
    
    # Test settings (smaller and faster)
    config.update({
        'num_identities': 150,  # Smaller for testing
        'learning_rate': 0.01,  # Higher LR for faster convergence
        'global_learning_rate': 0.05,
        'weight_decay': 1e-4,
        'local_epochs': 2,  # Fewer epochs for quick test
        'max_epsilon': 0.5,  # Smaller privacy budget for test
        'delta': 1e-5,
        'noise_multiplier': 0.8,  # Less noise for faster convergence
        'max_grad_norm': 1.0,
        'patience': 5,  # Less patience for quick test
        'min_delta': 0.01,  # Larger delta for quicker convergence detection
        'use_homomorphic_encryption': True,
        'batch_size': 8
    })
    
    return config

def run_quick_test():
    """Run a quick test of privacy-infused federated training"""
    
    logger.info("🧪 Starting Quick Privacy Training Test")
    logger.info("This is a small-scale test to verify the training pipeline works")
    
    # Create test results directory
    results_dir = Path("quick_test_results")
    results_dir.mkdir(exist_ok=True)
    
    # Create test data loaders
    data_loaders = create_test_data_loaders(batch_size=8)
    
    # Create test configuration
    config = create_test_config()
    
    logger.info("Test Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Initialize trainer
    trainer = PrivacyFederatedTrainer(
        config=config,
        data_loaders=data_loaders,
        save_dir=str(results_dir)
    )
    
    # Run 5 rounds of training for testing
    test_rounds = 5
    logger.info(f"Running {test_rounds} rounds of federated training...")
    
    try:
        for round_num in range(1, test_rounds + 1):
            logger.info(f"\n{'='*40}")
            logger.info(f"TEST ROUND {round_num}/{test_rounds}")
            logger.info(f"{'='*40}")
            
            # Execute one federated round
            round_results = trainer.train_federated_round()
            
            # Check results
            if round_results.get('stopped_early', False):
                logger.warning("Training stopped early due to privacy budget exhaustion")
                break
            
            if round_results.get('converged', False):
                logger.info("Training converged early!")
                break
            
            # Log round summary
            if 'global_evaluation' in round_results and round_results['global_evaluation']:
                eval_results = round_results['global_evaluation']
                logger.info(f"Round {round_num} Results:")
                logger.info(f"  Global Accuracy: {eval_results['accuracy']:.4f}")
                logger.info(f"  Global Loss: {eval_results['loss']:.4f}")
                logger.info(f"  Active Nodes: {round_results['privacy_status']['active_nodes']}/3")
                
                # Privacy budget status
                for node_id, budget_info in round_results['privacy_status']['node_budgets'].items():
                    usage_pct = (budget_info['epsilon_used'] / config['max_epsilon']) * 100
                    logger.info(f"  {node_id} Privacy: {budget_info['epsilon_used']:.3f}/{config['max_epsilon']:.1f} ({usage_pct:.1f}%)")
        
        # Final evaluation
        final_evaluation = trainer._evaluate_global_model()
        
        # Test summary
        print("\n" + "="*50)
        print("🎉 QUICK PRIVACY TRAINING TEST COMPLETED!")
        print("="*50)
        print(f"Final Accuracy: {final_evaluation['accuracy']:.4f}")
        print(f"Final Loss: {final_evaluation['loss']:.4f}")
        print(f"Rounds Completed: {trainer.current_round}")
        
        # Privacy budget summary
        print("\nPrivacy Budget Usage:")
        for node_id, accountant in trainer.privacy_accountants.items():
            stats = accountant.get_privacy_stats()
            usage_pct = (stats['epsilon_used'] / config['max_epsilon']) * 100
            print(f"  {node_id}: {stats['epsilon_used']:.3f}/{config['max_epsilon']:.1f} ({usage_pct:.1f}%)")
        
        print("="*50)
        
        # Verify key components worked
        success_checks = {
            'Models created': len(trainer.models) == 3,
            'Privacy accountants active': len(trainer.privacy_accountants) == 3,
            'Global model exists': trainer.global_model is not None,
            'Training completed': trainer.current_round > 0,
            'Final accuracy > 0': final_evaluation['accuracy'] > 0,
            'Privacy budget tracked': all(
                accountant.get_privacy_stats()['epsilon_used'] > 0 
                for accountant in trainer.privacy_accountants.values()
            )
        }
        
        print("\nComponent Verification:")
        all_passed = True
        for check, passed in success_checks.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {check}: {status}")
            if not passed:
                all_passed = False
        
        if all_passed:
            print("\n🎉 All components working correctly!")
            print("You can now run full training with: python run_privacy_training.py")
        else:
            print("\n⚠️  Some components need attention")
        
        return all_passed
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"\n❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = run_quick_test()
    if success:
        print("\n✅ Quick test passed! Ready for full training.")
    else:
        print("\n❌ Quick test failed. Please check the logs.") 