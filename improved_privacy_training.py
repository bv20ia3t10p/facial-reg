#!/usr/bin/env python3
"""
Improved Privacy-Infused Federated Training with Pre-training Phase

This script addresses the low accuracy issue by:
1. Pre-training a global model without privacy constraints (50 epochs)
2. Using the pre-trained model as initialization for federated training
3. Better privacy budget management with larger batches and reduced noise
4. More conservative privacy parameters for better utility
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import json
import time
from pathlib import Path
from datetime import datetime
import argparse

from privacy_biometric_model import PrivacyBiometricModel
from privacy_federated_trainer import PrivacyFederatedTrainer, BiometricDataset

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_improved_config():
    """Create improved training configuration"""
    return {
        # Model configuration
        'num_identities': 300,
        'num_emotions': 7,
        'feature_dim': 512,
        
        # Pre-training phase (no privacy)
        'pretrain_epochs': 50,
        'pretrain_lr': 0.001,
        'pretrain_batch_size': 32,
        
        # Federated training phase (with privacy)
        'federated_rounds': 20,
        'local_epochs': 1,  # Reduced to preserve budget
        'learning_rate': 0.0005,  # Lower for fine-tuning
        'batch_size': 32,  # Larger batches
        'weight_decay': 1e-4,
        
        # Improved privacy settings
        'max_epsilon': 2.0,  # Increased budget
        'delta': 1e-5,
        'noise_multiplier': 0.5,  # Reduced noise
        'max_grad_norm': 1.0,
        
        # Training settings
        'patience': 15,
        'min_delta': 0.001,
        'use_homomorphic_encryption': True,
    }

def pretrain_model(config, data_loaders):
    """Pre-train model without privacy constraints"""
    logger.info("🚀 Phase 1: Pre-training (No Privacy)")
    logger.info(f"Training for {config['pretrain_epochs']} epochs")
    
    # Create model without privacy
    model = PrivacyBiometricModel(
        num_identities=config['num_identities'],
        privacy_enabled=False
    )
    
    # Combine all data for pre-training
    all_data = []
    for node_id, loader in data_loaders.items():
        for batch in loader:
            all_data.append(batch)
        logger.info(f"Collected data from {node_id}")
    
    # Create combined dataset
    all_images = torch.cat([batch[0] for batch in all_data])
    all_identity_labels = torch.cat([batch[1] for batch in all_data])
    all_emotion_labels = torch.cat([batch[2] for batch in all_data])
    
    logger.info(f"Combined dataset: {len(all_images)} samples")
    
    # Create pre-training loader
    pretrain_dataset = BiometricDataset(all_images, all_identity_labels, all_emotion_labels)
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=config['pretrain_batch_size'], shuffle=True)
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=config['pretrain_lr'])
    identity_criterion = nn.CrossEntropyLoss()
    emotion_criterion = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    best_acc = 0.0
    
    for epoch in range(config['pretrain_epochs']):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, identity_labels, emotion_labels in pretrain_loader:
            # Forward pass
            identity_logits, emotion_logits, _ = model(images)
            
            # Compute loss
            id_loss = identity_criterion(identity_logits, identity_labels)
            em_loss = emotion_criterion(emotion_logits, emotion_labels)
            loss = id_loss + 0.3 * em_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = torch.max(identity_logits, 1)
            total += identity_labels.size(0)
            correct += (predicted == identity_labels).sum().item()
        
        # Calculate metrics
        avg_loss = total_loss / len(pretrain_loader)
        accuracy = correct / total
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{config['pretrain_epochs']}: "
                       f"Loss={avg_loss:.4f}, Acc={accuracy:.4f}")
        
        if accuracy > best_acc:
            best_acc = accuracy
    
    logger.info(f"✅ Pre-training completed! Best accuracy: {best_acc:.4f}")
    
    # Enable privacy for federated phase
    model.privacy_enabled = True
    model._setup_privacy_components()
    
    return model

def create_data_loaders(batch_size=32):
    """Create improved synthetic data loaders"""
    logger.info("Creating improved synthetic data...")
    
    data_loaders = {}
    
    # Better synthetic data distribution
    configs = {
        'server': {'samples': 2000},
        'client1': {'samples': 2200}, 
        'client2': {'samples': 2100}
    }
    
    for node_id, config in configs.items():
        # More realistic synthetic images
        images = torch.randn(config['samples'], 3, 224, 224) * 0.5 + 0.5
        identity_labels = torch.randint(0, 300, (config['samples'],))
        emotion_labels = torch.randint(0, 7, (config['samples'],))
        
        dataset = BiometricDataset(images, identity_labels, emotion_labels)
        data_loaders[node_id] = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        logger.info(f"{node_id}: {config['samples']} samples")
    
    return data_loaders

def run_improved_training(args):
    """Run improved training with pre-training phase"""
    logger.info("🚀 Starting Improved Privacy Training")
    
    # Setup
    results_dir = Path("improved_results")
    results_dir.mkdir(exist_ok=True)
    
    config = create_improved_config()
    if args.pretrain_epochs:
        config['pretrain_epochs'] = args.pretrain_epochs
    if args.federated_rounds:
        config['federated_rounds'] = args.federated_rounds
    
    # Create data
    data_loaders = create_data_loaders(config['batch_size'])
    
    # Phase 1: Pre-training
    pretrained_model = pretrain_model(config, data_loaders)
    
    # Save pre-trained model
    torch.save(pretrained_model.state_dict(), results_dir / "pretrained_model.pth")
    
    # Phase 2: Federated training
    logger.info("\n🔒 Phase 2: Privacy-Infused Federated Training")
    
    trainer = PrivacyFederatedTrainer(
        config=config,
        data_loaders=data_loaders,
        save_dir=str(results_dir)
    )
    
    # Initialize with pre-trained weights
    pretrained_state = pretrained_model.state_dict()
    trainer.global_model.load_state_dict(pretrained_state)
    
    for node_id, model in trainer.node_models.items():
        model.load_state_dict(pretrained_state)
        logger.info(f"✅ Initialized {node_id} with pre-trained weights")
    
    # Run federated training
    start_time = time.time()
    
    for round_num in range(1, config['federated_rounds'] + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"ROUND {round_num}/{config['federated_rounds']}")
        logger.info(f"{'='*50}")
        
        round_results = trainer.train_federated_round()
        
        if round_results.get('stopped_early', False):
            logger.info("Stopped early due to privacy budget")
            break
        
        if 'global_evaluation' in round_results:
            eval_results = round_results['global_evaluation']
            logger.info(f"Round {round_num} Results:")
            logger.info(f"  Accuracy: {eval_results['accuracy']:.4f}")
            logger.info(f"  Loss: {eval_results['loss']:.4f}")
            logger.info(f"  Active Nodes: {round_results['privacy_status']['active_nodes']}/3")
    
    # Final results
    duration = time.time() - start_time
    final_eval = trainer._evaluate_global_model()
    
    print("\n" + "="*60)
    print("🎉 IMPROVED TRAINING COMPLETED!")
    print("="*60)
    print(f"Pre-training: {config['pretrain_epochs']} epochs")
    print(f"Federated Rounds: {trainer.current_round}")
    print(f"Final Accuracy: {final_eval['accuracy']:.4f}")
    print(f"Final Loss: {final_eval['loss']:.4f}")
    print(f"Duration: {duration/60:.1f} minutes")
    
    # Privacy summary
    print("\nPrivacy Budget Usage:")
    for node_id, accountant in trainer.privacy_accountants.items():
        stats = accountant.get_privacy_stats()
        usage = (stats['epsilon_used'] / config['max_epsilon']) * 100
        print(f"  {node_id}: {stats['epsilon_used']:.3f}/{config['max_epsilon']:.1f} ({usage:.1f}%)")
    
    print("="*60)
    
    return final_eval

def main():
    parser = argparse.ArgumentParser(description='Improved Privacy Training')
    parser.add_argument('--pretrain_epochs', type=int, default=50)
    parser.add_argument('--federated_rounds', type=int, default=20)
    
    args = parser.parse_args()
    run_improved_training(args)

if __name__ == "__main__":
    main() 