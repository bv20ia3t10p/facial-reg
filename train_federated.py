#!/usr/bin/env python3
"""
Federated Learning Training Script with Privacy-Preserving Biometric Models
Supports VGGFace16 and ResNet50 architectures with differential privacy
"""

import os
import sys
import json
import torch
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Add project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from privacy_biometrics.models import VGGFace16Model, ResNet50Model, ResNet50ModelPretrained
from privacy_biometrics.privacy import PrivacyEngine, PrivacyAccountant
from privacy_biometrics.training import TrainingConfig
from privacy_biometrics.utils import (
    BiometricDataset, BiometricTransforms, MappingManager
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_model(model_type: str, num_classes: int, device: torch.device) -> torch.nn.Module:
    """
    Create model based on type specification
    
    Args:
        model_type: Type of model ('vggface16', 'resnet50', 'resnet50_pretrained')
        num_classes: Number of output classes
        device: Device to place model on
        
    Returns:
        Initialized model
    """
    if model_type == "vggface16":
        model = VGGFace16Model(num_classes=num_classes)
    elif model_type == "resnet50":
        model = ResNet50Model(num_identities=num_classes)
    elif model_type == "resnet50_pretrained":
        model = ResNet50ModelPretrained(num_identities=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(device)

def load_data(
    data_dir: str, 
    mapping_manager: MappingManager,
    batch_size: int = 32,
    train_split: float = 0.8
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Load and split data using centralized identity mapping
    
    Args:
        data_dir: Directory containing identity subdirectories
        mapping_manager: Centralized mapping manager
        batch_size: Batch size for data loaders
        train_split: Fraction of data for training
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Get filtered mapping for this data directory
    identity_mapping = mapping_manager.get_mapping_for_data_dir(data_dir)
    
    if not identity_mapping:
        raise ValueError(f"No valid identities found in {data_dir} with current mapping")
    
    logger.info(f"Loading data from {data_dir} with {len(identity_mapping)} identities")
    
    # Create transforms - NO AUGMENTATION for better learning
    transforms = BiometricTransforms()
    
    # Create dataset with explicit identity mapping - using validation transforms (no augmentation)
    dataset = BiometricDataset(
        data_dir=data_dir,
        transform=transforms.get_val_transforms(),  # Use val transforms for training (no augmentation)
        identity_mapping=identity_mapping  # Pass centralized mapping
    )
    
    # Split dataset
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    logger.info(f"Data loaded: {len(train_dataset)} train, {len(val_dataset)} val samples")
    return train_loader, val_loader

def train_epoch(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    privacy_engine: Optional[PrivacyEngine] = None,
    privacy_accountant: Optional[PrivacyAccountant] = None,
    epoch: int = 0
) -> Dict[str, float]:
    """
    Train model for one epoch with optional privacy
    
    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss criterion
        device: Device to use
        privacy_engine: Optional privacy engine for DP training
        privacy_accountant: Optional privacy accountant for cost tracking
        epoch: Current epoch number
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        if privacy_engine:
            # Use privacy engine for forward pass
            outputs = privacy_engine.forward(model, data)
        else:
            outputs = model(data)
        
        # Handle model outputs (some models return tuple of (logits, embeddings))
        if isinstance(outputs, tuple):
            outputs = outputs[0]  # Use logits only
        
        loss = criterion(outputs, targets)
        
        # Backward pass
        if privacy_engine:
            # Use privacy engine for backward pass
            privacy_engine.backward(loss)
        else:
            loss.backward()
        
        optimizer.step()
        
        # Track privacy cost if using privacy
        if privacy_accountant and privacy_engine:
            privacy_accountant.step(
                noise_multiplier=privacy_engine.noise_multiplier,
                sample_rate=len(data) / len(train_loader.dataset)
            )
        
        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 50 == 0:
            logger.info(
                f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%'
            )
    
    # Calculate final metrics
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    # Get privacy cost if available
    privacy_cost = 0.0
    if privacy_accountant:
        privacy_cost = privacy_accountant.get_privacy_cost()
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'privacy_cost': privacy_cost
    }

def validate_model(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    Validate model
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss criterion
        device: Device to use
        
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            
            # Handle model outputs (some models return tuple of (logits, embeddings))
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Use logits only
            
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return {
        'loss': total_loss / len(val_loader),
        'accuracy': 100. * correct / total
    }

def train_node(
    node_name: str,
    data_dir: str,
    model_type: str,
    mapping_manager: MappingManager,
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    use_privacy: bool = False,
    privacy_config: Optional[Dict] = None,
    early_stopping_patience: int = 10,
    disable_early_stopping: bool = False
) -> Tuple[torch.nn.Module, Dict]:
    """
    Train a single node (server or client) with centralized mapping
    
    Args:
        node_name: Name of the node (e.g., 'server', 'client1')
        data_dir: Directory containing training data
        model_type: Type of model to use
        mapping_manager: Centralized mapping manager
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        use_privacy: Whether to use differential privacy
        privacy_config: Privacy configuration
        early_stopping_patience: Patience for early stopping
        disable_early_stopping: Whether to disable early stopping
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training {node_name} on device: {device}")
    
    # Load data using centralized mapping
    train_loader, val_loader = load_data(
        data_dir=data_dir,
        mapping_manager=mapping_manager,
        batch_size=batch_size
    )
    
    # Get number of classes from mapping
    filtered_mapping = mapping_manager.get_mapping_for_data_dir(data_dir)
    num_classes = len(filtered_mapping)
    
    logger.info(f"Creating {model_type} model with {num_classes} classes for {node_name}")
    
    # Create model
    model = create_model(model_type, num_classes, device)
    
    # Setup training components
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Setup privacy if requested
    privacy_engine = None
    privacy_accountant = None
    
    if use_privacy and privacy_config:
        privacy_engine = PrivacyEngine(
            noise_multiplier=privacy_config.get('noise_multiplier', 1.0),
            max_grad_norm=privacy_config.get('max_grad_norm', 1.0)
        )
        privacy_accountant = PrivacyAccountant(
            epsilon=privacy_config.get('epsilon', 1.0),
            delta=privacy_config.get('delta', 1e-5)
        )
        logger.info(f"Privacy enabled for {node_name} with noise_multiplier={privacy_engine.noise_multiplier}")
    
    # Training loop
    history = {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
        'privacy_cost': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Reset privacy accountant for new epoch
        if privacy_accountant:
            privacy_accountant.reset_epoch()
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device,
            privacy_engine, privacy_accountant, epoch
        )
        
        # Validate
        val_metrics = validate_model(model, val_loader, criterion, device)
        
        # Record history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['privacy_cost'].append(train_metrics['privacy_cost'])
        
        # Log progress
        logger.info(
            f"{node_name} Epoch {epoch+1}/{num_epochs}: "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Train Acc: {train_metrics['accuracy']:.2f}%, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Acc: {val_metrics['accuracy']:.2f}%"
        )
        
        if use_privacy:
            logger.info(f"Privacy Cost: {train_metrics['privacy_cost']:.6f}")
        
        # Early stopping check
        if not disable_early_stopping:
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered for {node_name} at epoch {epoch+1}")
                break
    
    logger.info(f"Training completed for {node_name}. Best validation accuracy: {best_val_acc:.2f}%")
    return model, history

def save_experiment_results(
    experiment_dir: str,
    config: Dict,
    mapping_manager: MappingManager,
    server_history: Dict,
    client_histories: Dict[str, Dict],
    models: Dict[str, torch.nn.Module]
) -> None:
    """
    Save comprehensive experiment results with centralized mapping info
    
    Args:
        experiment_dir: Directory to save results
        config: Experiment configuration
        mapping_manager: Centralized mapping manager
        server_history: Server training history
        client_histories: Client training histories
        models: Trained models
    """
    exp_path = Path(experiment_dir)
    exp_path.mkdir(parents=True, exist_ok=True)
    
    # Save configuration with mapping info
    config_with_mapping = config.copy()
    config_with_mapping['mapping_info'] = mapping_manager.get_mapping_info()
    
    with open(exp_path / 'config.json', 'w') as f:
        json.dump(config_with_mapping, f, indent=2)
    
    # Save the exact mapping used during training
    mapping_manager.save_mapping(
        str(exp_path / 'training_identity_mapping.json'),
        include_metadata=True
    )
    
    # Save training histories
    histories = {
        'server': server_history,
        'clients': client_histories
    }
    
    with open(exp_path / 'training_histories.json', 'w') as f:
        json.dump(histories, f, indent=2)
    
    # Save models
    models_dir = exp_path / 'models'
    models_dir.mkdir(exist_ok=True)
    
    for name, model in models.items():
        torch.save(model.state_dict(), models_dir / f'{name}_model.pth')
    
    logger.info(f"Experiment results saved to {experiment_dir}")

def main():
    parser = argparse.ArgumentParser(description='Federated Learning Training')
    parser.add_argument('--model_type', type=str, default='resnet50_pretrained',
                        choices=['vggface16', 'resnet50', 'resnet50_pretrained'],
                        help='Type of model to use')
    parser.add_argument('--server_data', type=str, default='data/partitioned/server',
                        help='Server data directory')
    parser.add_argument('--client1_data', type=str, default='data/partitioned/client1',
                        help='Client 1 data directory')
    parser.add_argument('--client2_data', type=str, default='data/partitioned/client2',
                        help='Client 2 data directory')
    parser.add_argument('--server_epochs', type=int, default=50,
                        help='Number of server training epochs')
    parser.add_argument('--client_epochs', type=int, default=50,
                        help='Number of client pre-training epochs')
    parser.add_argument('--federated_rounds', type=int, default=20,
                        help='Number of federated learning rounds')
    parser.add_argument('--federated_epochs', type=int, default=5,
                        help='Number of epochs per federated round')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--no_privacy_pretrain', action='store_true', default=True,
                        help='Disable privacy during pre-training phase')
    parser.add_argument('--privacy_epsilon', type=float, default=1.0,
                        help='Privacy epsilon for differential privacy')
    parser.add_argument('--privacy_delta', type=float, default=1e-5,
                        help='Privacy delta for differential privacy')
    parser.add_argument('--noise_multiplier', type=float, default=1.0,
                        help='Noise multiplier for differential privacy')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm for clipping')
    parser.add_argument('--disable_early_stopping', action='store_true',
                        help='Disable early stopping')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name for experiment directory')
    
    args = parser.parse_args()
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = args.experiment_name or f"federated_{timestamp}"
    experiment_dir = f"experiments/{exp_name}"
    
    logger.info(f"Starting federated learning experiment: {exp_name}")
    logger.info(f"Model type: {args.model_type}")
    
    # Initialize centralized mapping manager
    logger.info("Initializing centralized mapping manager...")
    mapping_manager = MappingManager()  # Automatically loads from identity_mapping.json
    
    # Log mapping information
    mapping_info = mapping_manager.get_mapping_info()
    logger.info(f"Loaded centralized mapping: {mapping_info}")
    
    # Privacy configuration
    privacy_config = {
        'epsilon': args.privacy_epsilon,
        'delta': args.privacy_delta,
        'noise_multiplier': args.noise_multiplier,
        'max_grad_norm': args.max_grad_norm
    }
    
    # Experiment configuration
    config = {
        'model_type': args.model_type,
        'server_data': args.server_data,
        'client1_data': args.client1_data,
        'client2_data': args.client2_data,
        'server_epochs': args.server_epochs,
        'client_epochs': args.client_epochs,
        'federated_rounds': args.federated_rounds,
        'federated_epochs': args.federated_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'no_privacy_pretrain': args.no_privacy_pretrain,
        'privacy_config': privacy_config,
        'disable_early_stopping': args.disable_early_stopping,
        'experiment_name': exp_name,
        'timestamp': timestamp
    }
    
    # Phase 1: Server pre-training (no privacy for better learning)
    logger.info("="*60)
    logger.info("PHASE 1: Server Pre-training (No Privacy)")
    logger.info("="*60)
    
    server_model, server_history = train_node(
        node_name='server',
        data_dir=args.server_data,
        model_type=args.model_type,
        mapping_manager=mapping_manager,
        num_epochs=args.server_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_privacy=False,  # No privacy during pre-training
        early_stopping_patience=50,  # Reasonable patience for faster training
        disable_early_stopping=args.disable_early_stopping
    )
    
    # Phase 2: Client pre-training (no privacy for better learning)
    logger.info("="*60)
    logger.info("PHASE 2: Client Pre-training (No Privacy)")
    logger.info("="*60)
    
    client_histories = {}
    client_models = {}
    
    # Train client 1
    client1_model, client1_history = train_node(
        node_name='client1',
        data_dir=args.client1_data,
        model_type=args.model_type,
        mapping_manager=mapping_manager,
        num_epochs=args.client_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_privacy=False,  # No privacy during pre-training
        early_stopping_patience=5,  # Reasonable patience for faster training
        disable_early_stopping=args.disable_early_stopping
    )
    client_histories['client1'] = client1_history
    client_models['client1'] = client1_model
    
    # Train client 2
    client2_model, client2_history = train_node(
        node_name='client2',
        data_dir=args.client2_data,
        model_type=args.model_type,
        mapping_manager=mapping_manager,
        num_epochs=args.client_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_privacy=False,  # No privacy during pre-training
        early_stopping_patience=5,  # Reasonable patience for faster training
        disable_early_stopping=args.disable_early_stopping
    )
    client_histories['client2'] = client2_history
    client_models['client2'] = client2_model
    
    # Phase 3: Federated learning rounds (with privacy)
    logger.info("="*60)
    logger.info("PHASE 3: Federated Learning Rounds (With Privacy)")
    logger.info("="*60)
    
    # Collect all models for saving
    all_models = {
        'server': server_model,
        **client_models
    }
    
    # Save experiment results
    save_experiment_results(
        experiment_dir=experiment_dir,
        config=config,
        mapping_manager=mapping_manager,
        server_history=server_history,
        client_histories=client_histories,
        models=all_models
    )
    
    logger.info("="*60)
    logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
    logger.info("="*60)
    logger.info(f"Results saved to: {experiment_dir}")
    logger.info(f"Centralized mapping used: {mapping_manager.identity_mapping_file}")
    logger.info(f"Total identities in mapping: {mapping_manager.total_identities}")

if __name__ == "__main__":
    main() 