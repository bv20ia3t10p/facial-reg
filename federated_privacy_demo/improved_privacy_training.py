#!/usr/bin/env python3
"""
Enhanced Privacy-Infused Federated Training with Stratified Validation

This script provides comprehensive training logs including:
1. Batch-level progress tracking with accuracy and loss
2. Detailed epoch summaries with validation metrics
3. Real-time training progress visualization
4. Stratified train/validation split for balanced evaluation
5. Enhanced model tuning to reach 60%+ accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import logging
import json
import time
from pathlib import Path
from datetime import datetime
import argparse
import sys
from tqdm import tqdm
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from privacy_biometric_model import PrivacyBiometricModel
from privacy_federated_trainer import PrivacyFederatedTrainer, BiometricDataset
from training_statistics_tracker import TrainingStatisticsTracker

# GPU Optimization Settings
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
    torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
    # Set memory allocation strategy
    torch.cuda.empty_cache()

# Setup enhanced logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

def create_improved_config():
    """Create improved training configuration"""
    server_epochs = 2  # Base server epochs
    return {
        'num_identities': 100,  # Will be updated with actual count
        'batch_size': 32,
        'pretrain_batch_size': 32,
        'pretrain_epochs': 50,  # Server pretraining epochs
        'client_epochs': 50,  # Client epochs = half of server e
        'weight_decay': 1e-5,  # Reduced weight decay
        'pretrain_lr': 0.001,  # Increased learning rate
        'pretrain_scheduler_step': 10,  # Less frequent LR decay
        'pretrain_scheduler_gamma': 0.7,  # Less aggressive decay
        'val_split': 0.2,
        'target_accuracy': 0.95,
        'patience': 15,  # Increased patience
        'min_delta': 0.001,
        'warmup_epochs': 3,  # Reduced warmup
        'log_frequency': 5,
        'use_label_smoothing': True,
        'label_smoothing': 0.1,
        'k_folds': 5,
        'federated_rounds': 10,
        'local_epochs': 5,
        # Differential Privacy Parameters
        'enable_differential_privacy_pretrain': False,
        'enable_differential_privacy_federated': True,
        'max_epsilon': 500.0,  # Adjusted privacy budget
        'delta': 1e-5,
        # DP-SGD Parameters
        'noise_multiplier': 0.15,  # Lower noise for better utility
        'max_grad_norm': 1.5,  # Increased slightly to accommodate lower noise
        'accounting_mode': 'rdp',  # RDP accounting for tighter bounds
        'min_sampling_rate': 0.02,  # Increased sampling rate
        'secure_aggregation': True,  # Keep secure aggregation enabled
        'dp_batch_size': 64,  # Increased batch size for better convergence with low noise
    }

def log_batch_progress(epoch, batch_idx, total_batches, loss, acc, lr, elapsed):
    """Log batch-level training progress"""
    print(f"\rEpoch {epoch:3d} | Batch {batch_idx:5d}/{total_batches:5d} | "
          f"Loss: {loss:.4f} | Acc: {acc:6.2f}% | "
          f"LR: {lr:.6f} | Time: {elapsed:6.1f}s", end="")

def log_epoch_summary(epoch, total_epochs, train_loss, train_acc, val_loss, val_acc, 
                     best_acc, lr, epoch_time, samples_processed):
    """Comprehensive epoch summary logging with GPU memory info"""
    print(f"\n{'='*80}")
    print(f"📈 EPOCH {epoch:2d}/{total_epochs:2d} SUMMARY")
    print(f"{'='*80}")
    print(f"🏋️  Training   | Loss: {train_loss:7.4f} | Accuracy: {train_acc:6.2f}%")
    print(f"🎯 Validation | Loss: {val_loss:7.4f} | Accuracy: {val_acc:6.2f}%")
    print(f"🏆 Best Acc   | {best_acc:6.2f}% {'✅ NEW BEST!' if val_acc > best_acc else ''}")
    print(f"⚙️  Learning Rate: {lr:.6f}")
    print(f"⏱️  Time: {epoch_time:5.1f}s | Samples: {samples_processed:,}")
    print(f"🚀 Speed: {samples_processed/epoch_time:,.0f} samples/sec")
    
    # GPU memory info
    if torch.cuda.is_available():
        gpu_memory_used = torch.cuda.memory_allocated() / 1024**3
        gpu_memory_cached = torch.cuda.memory_reserved() / 1024**3
        print(f"🎮 GPU Memory | Used: {gpu_memory_used:.2f}GB | Cached: {gpu_memory_cached:.2f}GB")
    
    print(f"{'='*80}\n")

def create_stratified_split(dataset, val_split=0.2, random_state=42):
    """Create stratified train/validation split ensuring balanced class distribution"""
    # Get all targets
    targets = []
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
    else:
        # Extract targets from dataset
        for i in range(len(dataset)):
            _, target = dataset[i]
            targets.append(target)
    
    targets = np.array(targets)
    indices = np.arange(len(dataset))
    
    # Create stratified split
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_split,
        stratify=targets,
        random_state=random_state
    )
    
    return train_idx, val_idx

def pretrain_server_model(config, data_loaders, stats_tracker):
    """Phase 1a: Server-side pretraining with stratified validation split"""
    # GPU Detection and Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🔧 Device Detection: {device}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"🎮 GPU: {gpu_name}")
        logger.info(f"💾 GPU Memory: {gpu_memory:.1f} GB")
        torch.cuda.empty_cache()  # Clear GPU cache
    else:
        logger.warning("⚠️  GPU not available, falling back to CPU (training will be slower)")
    
    logger.info("🚀 Phase 1a: SERVER Pre-training with Stratified Validation")
    logger.info(f"🎯 Target Accuracy: {config['target_accuracy']*100:.1f}%")
    logger.info(f"📚 Training for {config['pretrain_epochs']} epochs")
    logger.info(f"📊 Validation split: {config['val_split']*100:.0f}% (stratified)")
    
    # Data preparation
    print("📦 Preparing SERVER training data with stratified split...")
    
    if 'server' not in data_loaders:
        raise ValueError("Server data loader not found! Cannot proceed with server pretraining.")
    
    server_dataset = data_loaders['server'].dataset
    dataset_size = len(server_dataset)
    
    logger.info(f"📊 Dataset size: {dataset_size:,} samples")
    
    # Create stratified train/validation split
    train_idx, val_idx = create_stratified_split(
        server_dataset, 
        val_split=config['val_split'],
        random_state=42
    )
    
    logger.info(f"📊 Train samples: {len(train_idx):,}")
    logger.info(f"📊 Validation samples: {len(val_idx):,}")
    
    # Check class distribution in splits
    train_targets = [server_dataset.targets[i] for i in train_idx]
    val_targets = [server_dataset.targets[i] for i in val_idx]
    
    unique_train_classes = len(set(train_targets))
    unique_val_classes = len(set(val_targets))
    
    logger.info(f"🎯 Classes in train set: {unique_train_classes}")
    logger.info(f"🎯 Classes in validation set: {unique_val_classes}")
    
    # Create model
    model = PrivacyBiometricModel(
        num_identities=config['num_identities'],
        privacy_enabled=config.get('enable_differential_privacy_pretrain', False)
    ).to(device)
    
    # Create data subsets
    train_subset = Subset(server_dataset, train_idx)
    val_subset = Subset(server_dataset, val_idx)
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=config['pretrain_batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True  # Drop last incomplete batch to avoid BatchNorm issues
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=config['pretrain_batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False  # Keep all validation samples
    )
    
    # Training setup
    if config.get('use_label_smoothing', False):
        class LabelSmoothingCrossEntropy(nn.Module):
            def __init__(self, smoothing=0.1):
                super().__init__()
                self.smoothing = smoothing
                
            def forward(self, pred, target):
                confidence = 1. - self.smoothing
                logprobs = F.log_softmax(pred, dim=-1)
                nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
                nll_loss = nll_loss.squeeze(1)
                smooth_loss = -logprobs.mean(dim=-1)
                loss = confidence * nll_loss + self.smoothing * smooth_loss
                return loss.mean()
        
        criterion = LabelSmoothingCrossEntropy(config['label_smoothing'])
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['pretrain_lr'], 
        weight_decay=config['weight_decay']
    )
    
    # Improved learning rate scheduling with warmup
    def lr_lambda(epoch):
        if epoch < config.get('warmup_epochs', 5):
            # Warmup phase: gradually increase LR
            return (epoch + 1) / config.get('warmup_epochs', 5)
        else:
            # Decay phase: step decay
            decay_epoch = epoch - config.get('warmup_epochs', 5)
            return config['pretrain_scheduler_gamma'] ** (decay_epoch // config['pretrain_scheduler_step'])
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    training_history = []
    
    training_start_time = time.time()
    
    for epoch in range(config['pretrain_epochs']):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        num_samples = 0
        
        # Add progress bar for training
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["pretrain_epochs"]} [Train]',
                         leave=False, ncols=100)
        
        for batch_idx, (data, targets) in enumerate(train_pbar):
            # Skip batch if size is 1 (BatchNorm requires batch_size > 1)
            if data.size(0) == 1:
                continue
                
            data, targets = data.to(device), targets.to(device)
            batch_size = data.size(0)
            
            optimizer.zero_grad()
            outputs = model(data)
            # Extract identity logits (first element of tuple)
            identity_logits = outputs[0] if isinstance(outputs, tuple) else outputs
            loss = criterion(identity_logits, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_size  # Multiply by batch size
            num_samples += batch_size
            _, predicted = identity_logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            current_acc = 100. * correct / total
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.2f}%',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        train_pbar.close()
        train_loss = total_loss / num_samples  # Average by total samples
        train_acc = 100. * correct / total
        
        # Validation phase with progress bar
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_num_samples = 0
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{config["pretrain_epochs"]} [Val]',
                       leave=False, ncols=100)
        
        with torch.no_grad():
            for data, targets in val_pbar:
                # Skip batch if size is 1 (BatchNorm requires batch_size > 1)
                if data.size(0) == 1:
                    continue
                    
                data, targets = data.to(device), targets.to(device)
                batch_size = data.size(0)
                outputs = model(data)
                # Extract identity logits (first element of tuple)
                identity_logits = outputs[0] if isinstance(outputs, tuple) else outputs
                loss = criterion(identity_logits, targets)
                
                val_loss += loss.item() * batch_size  # Multiply by batch size
                val_num_samples += batch_size
                _, predicted = identity_logits.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
                # Update validation progress bar
                current_val_acc = 100. * val_correct / val_total
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{current_val_acc:.2f}%'
                })
        
        val_pbar.close()
        val_loss = val_loss / val_num_samples  # Average by total samples
        val_acc = 100. * val_correct / val_total
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Epoch summary
        epoch_time = time.time() - epoch_start_time
        samples_processed = len(train_loader.dataset)
        log_epoch_summary(epoch+1, config['pretrain_epochs'], train_loss, train_acc, 
                         val_loss, val_acc, best_val_acc, current_lr, epoch_time, samples_processed)
        
        # Track best model (improved early stopping)
        if epoch == 0:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            # Use validation loss for early stopping (more stable)
            if val_loss < best_val_loss - config['min_delta']:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_epoch = epoch + 1
                patience_counter = 0
                # Save best model to root directory
                torch.save(model.state_dict(), "best_pretrained_model.pth")
                logger.info(f"💾 Saved new best model (val_acc: {best_val_acc:.2f}%)")
            else:
                patience_counter += 1
        
        # Early stopping based on validation loss
        if patience_counter >= config['patience']:
            logger.info(f"⏹️  Early stopping at epoch {epoch+1} (patience: {config['patience']})")
            logger.info(f"   └─ Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
            break
        
        # Record history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': current_lr
        })
    
    training_time = time.time() - training_start_time
    
    # Load best model
    if os.path.exists("best_pretrained_model.pth"):
        model.load_state_dict(torch.load("best_pretrained_model.pth"))
        logger.info("✅ Loaded best model weights")
    
    # Save training results
    results_dir = Path("improved_results")
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / "server_training_results.json", 'w') as f:
        json.dump({
            'best_accuracy': best_val_acc / 100.0,  # Convert to decimal
            'best_epoch': best_epoch,
            'training_time': training_time,
            'training_history': training_history,
            'config': config
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"🎉 SERVER PRETRAINING COMPLETED!")
    print(f"{'='*80}")
    print(f"🏆 Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"📊 Best Epoch: {best_epoch}")
    print(f"⏱️  Training Time: {training_time/60:.1f} minutes")
    print(f"📈 Final Training Accuracy: {train_acc:.2f}%")
    print(f"🎯 Stratified validation ensured balanced class distribution")
    print(f"{'='*80}\n")
    
    return model, best_val_acc / 100.0

def pretrain_client_models(config, data_loaders, server_model_state, stats_tracker):
    """Phase 1b: Client-side personalized pretraining with server initialization"""
    logger.info("🔄 Phase 1b: CLIENT Pre-training (Local Personalization)")
    logger.info("📡 Distributing server weights to clients...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    client_models = {}
    client_histories = {}
    
    # Clear GPU cache before starting client training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Reduce batch size for client training to prevent OOM
    client_batch_size = max(config['pretrain_batch_size'] // 2, 2)  # Half the batch size
    logger.info(f"📊 Client batch size reduced to {client_batch_size} to prevent OOM")
    
    # Reduce epochs for client pretraining (they start from server weights)
    client_epochs = config['client_epochs']  # Use configured client epochs
    logger.info(f"📚 Client pretraining: {client_epochs} epochs per client")
    
    for node_id, original_data_loader in data_loaders.items():
        if node_id == 'server':
            continue  # Skip server, already trained
            
        print(f"\n🔧 Starting {node_id.upper()} personalized pretraining...")
        print(f"{'='*60}")
        
        # Clear GPU cache before each client
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Create client model with server initialization
        client_model = PrivacyBiometricModel(
            num_identities=config['num_identities'],
            privacy_enabled=False
        ).to(device)
        
        # Load server pretrained weights
        client_model.load_state_dict(server_model_state)
        logger.info(f"✅ {node_id}: Initialized with server weights")
        
        # Create new data loader with smaller batch size and no pin_memory
        client_dataset = original_data_loader.dataset
        client_data_loader = DataLoader(
            client_dataset, 
            batch_size=client_batch_size, 
            shuffle=True,
            num_workers=1,  # Reduce workers to save memory
            pin_memory=False,  # Disable pin_memory to save GPU memory
            drop_last=True  # Drop last incomplete batch to avoid BatchNorm issues
        )
        
        # Get dataset size for split calculation
        dataset_size = len(client_dataset)
        if dataset_size == 0:
            logger.warning(f"⚠️  No data for {node_id}, skipping...")
            continue
            
        logger.info(f"📊 {node_id} Data | Total samples: {dataset_size:,}")
        
        # Client-specific training setup (lower learning rate for fine-tuning)
        optimizer = optim.Adam(client_model.parameters(), lr=config['pretrain_lr'] * 0.5, weight_decay=config['weight_decay'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(client_epochs//3, 2), gamma=0.5)
        
        identity_criterion = nn.CrossEntropyLoss()
        
        # Client training tracking
        best_client_acc = 0.0
        client_history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rates': []
        }
        
        # Client training loop
        for epoch in range(client_epochs):
            epoch_start_time = time.time()
            
            # Training phase
            client_model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            num_samples = 0
            
            train_start_time = time.time()
            
            for batch_idx, (images, labels) in enumerate(client_data_loader):
                # Skip batch if size is 1 (BatchNorm requires batch_size > 1)
                if images.size(0) == 1:
                    continue
                
                # Clear cache periodically
                if batch_idx % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Move data to GPU
                images = images.to(device, non_blocking=True)
                identity_labels = labels.to(device, non_blocking=True)
                batch_size = images.size(0)
                
                # Forward pass - identity only
                identity_logits, _ = client_model(images)
                
                # Compute loss - identity only
                loss = identity_criterion(identity_logits, identity_labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Metrics
                running_loss += loss.item() * batch_size  # Multiply by batch size
                num_samples += batch_size
                _, predicted = torch.max(identity_logits, 1)
                total += identity_labels.size(0)
                correct += (predicted == identity_labels).sum().item()
                
                # Batch logging (less frequent for clients)
                if batch_idx % (config['log_frequency'] * 2) == 0:
                    current_acc = (correct / total) * 100
                    current_lr = optimizer.param_groups[0]['lr']
                    elapsed = time.time() - train_start_time
                    print(f"\r🔧 {node_id} Epoch {epoch+1:2d} | Batch {batch_idx+1:3d}/{len(client_data_loader):3d} | "
                            f"Loss: {loss.item():6.4f} | Acc: {current_acc:6.2f}% | LR: {current_lr:.6f}", 
                            end='', flush=True)
            
            # Training metrics
            train_loss = running_loss / num_samples if num_samples > 0 else 0  # Average by total samples
            train_acc = (correct / total) * 100 if total > 0 else 0
            
            # Simple validation using a subset of training data (since we can't afford to split)
            client_model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            val_num_samples = 0
            
            with torch.no_grad():
                # Use first few batches for validation
                val_batches = min(3, len(client_data_loader))
                for batch_idx, (images, labels) in enumerate(client_data_loader):
                    if batch_idx >= val_batches:
                        break
                    
                    # Skip batch if size is 1 (BatchNorm requires batch_size > 1)
                    if images.size(0) == 1:
                        continue
                        
                    images = images.to(device, non_blocking=True)
                    identity_labels = labels.to(device, non_blocking=True)
                    batch_size = images.size(0)
                    
                    identity_logits, _ = client_model(images)
                    loss = identity_criterion(identity_logits, identity_labels)
                    
                    val_loss += loss.item() * batch_size  # Multiply by batch size
                    val_num_samples += batch_size
                    _, predicted = torch.max(identity_logits, 1)
                    val_total += identity_labels.size(0)
                    val_correct += (predicted == identity_labels).sum().item()
            
            val_loss = val_loss / val_num_samples if val_num_samples > 0 else train_loss  # Average by total samples
            val_acc = (val_correct / val_total) * 100 if val_total > 0 else train_acc
            
            # Update learning rate
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Track history
            client_history['train_loss'].append(train_loss)
            client_history['train_acc'].append(train_acc)
            client_history['val_loss'].append(val_loss)
            client_history['val_acc'].append(val_acc)
            client_history['learning_rates'].append(current_lr)
            
            # Client epoch summary (compact)
            epoch_time = time.time() - epoch_start_time
            print(f"\n📈 {node_id} Epoch {epoch+1:2d}/{client_epochs:2d} | "
                  f"Train: {train_acc:5.2f}% | Val: {val_acc:5.2f}% | "
                  f"Loss: {train_loss:6.4f} | Time: {epoch_time:4.1f}s")
        
            # Update best accuracy
            if val_acc > best_client_acc:
                best_client_acc = val_acc
                torch.save(client_model.state_dict(), f"best_{node_id}_pretrained_model.pth")
                logger.info(f"💾 {node_id}: New best accuracy {best_client_acc:.2f}%")
        
        # Store client results
        client_models[node_id] = client_model
        client_histories[node_id] = client_history
        
        print(f"✅ {node_id} personalization completed! Best accuracy: {best_client_acc:.2f}%")
    
        # Save client history
        with open(f'{node_id}_training_history.json', 'w') as f:
            json.dump(client_history, f, indent=2)
        
        # Clear GPU cache after each client
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"\n{'='*80}")
    print(f"🎉 CLIENT PERSONALIZATION COMPLETED!")
    print(f"{'='*80}")
    
    # Final GPU cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return client_models

def create_data_loaders(batch_size=16):
    """Load real partitioned facial recognition data using PyTorch ImageFolder"""
    logger.info("📂 Loading real partitioned data using ImageFolder...")
    
    # Enhanced data preprocessing with augmentation for better generalization
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.3),  # Reduced for faces
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Use training transform for now (stratified split will handle train/val split)
    transform = train_transform
    
    data_loaders = {}
    partitioned_path = "./data/partitioned"
    
    # Check if partitioned data exists
    if not os.path.exists(partitioned_path):
        raise FileNotFoundError(f"Partitioned data not found at {partitioned_path}")
    
    print("📊 Loading real biometric data with ImageFolder...")
    
    # Collect all unique classes across all partitions for consistent labeling
    all_classes = set()
    for node_name in ['server', 'client1', 'client2']:
        node_path = os.path.join(partitioned_path, node_name)
        if os.path.exists(node_path):
            classes = [d for d in os.listdir(node_path) 
                      if os.path.isdir(os.path.join(node_path, d))]
            all_classes.update(classes)
    
    # Sort classes for consistent ordering
    sorted_classes = sorted(list(all_classes))
    class_to_idx = {cls: idx for idx, cls in enumerate(sorted_classes)}
    total_identities = len(sorted_classes)
    
    logger.info(f"🎯 Found {total_identities} unique identities across all partitions")
    
    # Create datasets for each node
    for node_name in ['server', 'client1', 'client2']:
        node_path = os.path.join(partitioned_path, node_name)
        
        if not os.path.exists(node_path):
            logger.warning(f"⚠️  {node_name} partition not found, skipping...")
            continue
        
        print(f"🔧 Processing {node_name}...")
        
        # Create ImageFolder dataset
        dataset = ImageFolder(
            root=node_path,
            transform=transform
        )
        
        # Remap class indices to global consistent mapping
        original_class_to_idx = dataset.class_to_idx
        dataset.class_to_idx = class_to_idx
        
        # Update targets to use global class indices
        new_targets = []
        for target in dataset.targets:
            original_class = dataset.classes[target]
            new_target = class_to_idx[original_class]
            new_targets.append(new_target)
        dataset.targets = new_targets
        
        # Update classes list
        dataset.classes = sorted_classes
        
        # Create data loader with conservative memory settings
        data_loaders[node_name] = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=1,  # Reduced workers to save memory
            pin_memory=False,  # Disable pin_memory to prevent OOM
            drop_last=True  # Drop last incomplete batch to avoid BatchNorm issues
        )
        
        # Statistics
        unique_classes_in_node = len(set(dataset.targets))
        avg_samples_per_class = len(dataset) / unique_classes_in_node if unique_classes_in_node > 0 else 0
        
        logger.info(f"✅ {node_name}: {len(dataset):,} samples loaded")
        logger.info(f"   └─ Classes in node: {unique_classes_in_node} | Avg samples/class: {avg_samples_per_class:.1f}")
    
    logger.info("🎯 ImageFolder data loading completed!")
    logger.info(f"📊 Total identities: {total_identities}")
    logger.info(f"📁 Partitions loaded: {list(data_loaders.keys())}")
    
    return data_loaders, total_identities

def run_improved_training(args):
    """Run improved privacy-preserving federated training with partitioned data"""
    logger.info("🚀 Starting Privacy-Preserving Federated Training")
    
    # Setup
    results_dir = Path("improved_results")
    results_dir.mkdir(exist_ok=True)
    
    # Initialize statistics tracker
    stats_tracker = TrainingStatisticsTracker(base_dir="thesis_statistics")
    
    config = create_improved_config()
    if args.pretrain_epochs:
        config['pretrain_epochs'] = args.pretrain_epochs
    if args.federated_rounds:
        config['federated_rounds'] = args.federated_rounds
    
    # Save configuration
    stats_tracker.save_configuration(config)
    
    # Create data
    data_loaders, total_identities = create_data_loaders(config['batch_size'])
    
    # Update config with actual number of identities from real data
    config['num_identities'] = total_identities
    logger.info(f"🔄 Updated config: num_identities = {total_identities}")
    
    # Phase 1a: Server Pre-training
    server_model_path = results_dir / "server_pretrained_model.pth"
    best_model_path = Path("best_pretrained_model.pth")  # In root directory
    server_results_path = results_dir / "server_training_results.json"
    
    if best_model_path.exists():  # Only check for best model existence
        logger.info("🔄 Found existing best model, loading...")
        server_model = PrivacyBiometricModel(
            num_identities=config['num_identities'],
            privacy_enabled=False
        )
        # Load the best model from root directory
        server_model.load_state_dict(torch.load("best_pretrained_model.pth"))
        
        # Try to load the best accuracy from results file
        server_best_acc = 0.0
        if server_results_path.exists():
            try:
                with open(server_results_path, 'r') as f:
                    results = json.load(f)
                    server_best_acc = results.get('best_accuracy', 0.0)
                logger.info(f"✅ Loaded previous best accuracy: {server_best_acc:.4f}")
            except Exception as e:
                logger.warning(f"⚠️  Could not load previous best accuracy: {e}")
    else:
        logger.info("🔧 No existing best model found, starting server pre-training...")
        server_model, server_best_acc = pretrain_server_model(config, data_loaders, stats_tracker)
        torch.save(server_model.state_dict(), server_model_path)
    
    server_state = server_model.state_dict()
    
    # Phase 1b: Client Personalized Pre-training
    client_model_paths = {
        'client1': Path("best_client1_pretrained_model.pth"),
        'client2': Path("best_client2_pretrained_model.pth")
    }
    
    existing_clients = {node_id: path.exists() for node_id, path in client_model_paths.items()}
    logger.info(f"📋 Client model status: {existing_clients}")
    
    if all(existing_clients.values()):
        logger.info("🔄 Found existing client pre-trained models, loading...")
        client_models = {}
        for node_id, path in client_model_paths.items():
            try:
                client_model = PrivacyBiometricModel(
                    num_identities=config['num_identities'],
                    privacy_enabled=False
                )
                client_model.load_state_dict(torch.load(path))
                client_models[node_id] = client_model
                logger.info(f"✅ {node_id} pre-trained model loaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load {node_id} model: {e}")
                logger.info("🔧 Falling back to training client models...")
                client_models = pretrain_client_models(config, data_loaders, server_state, stats_tracker)
                break
    else:
        missing_models = [node_id for node_id, exists in existing_clients.items() if not exists]
        logger.info(f"🔧 Missing client models: {missing_models}. Training client models...")
        client_models = pretrain_client_models(config, data_loaders, server_state, stats_tracker)
    
    # Phase 2: Federated training
    logger.info("\n🔒 Phase 2: Privacy-Infused Federated Training")
    
    trainer = PrivacyFederatedTrainer(
        config=config,
        data_loaders=data_loaders,
        save_dir=str(results_dir),
        stats_tracker=stats_tracker
    )
    
    # Initialize with personalized pre-trained weights
    trainer.global_model.load_state_dict(server_state)
    logger.info("✅ Initialized global model with server pre-trained weights")
    
    for node_id, model in trainer.models.items():
        if node_id in client_models:
            client_state = client_models[node_id].state_dict()
            model.load_state_dict(client_state)
            logger.info(f"✅ Initialized {node_id} with personalized pre-trained weights")
        else:
            model.load_state_dict(server_state)
            logger.info(f"✅ Initialized {node_id} with server pre-trained weights (fallback)")
    
    # Run federated training
    start_time = time.time()
    
    for round_num in range(1, config['federated_rounds'] + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"ROUND {round_num}/{config['federated_rounds']}")
        logger.info(f"{'='*50}")
        
        round_results = trainer.train_federated_round()
        
        # Log federated round statistics
        stats_tracker.log_federated_stats(round_num, {
            'global_loss': round_results.get('global_evaluation', {}).get('loss', 0),
            'global_acc': round_results.get('global_evaluation', {}).get('accuracy', 0),
            'avg_client_loss': round_results.get('avg_client_loss', 0),
            'avg_client_acc': round_results.get('avg_client_accuracy', 0),
            'active_clients': len(round_results.get('privacy_status', {}).get('active_nodes', []))
        })
        
        # Log privacy metrics for each client
        for client_id, privacy_stats in round_results.get('privacy_status', {}).get('client_privacy', {}).items():
            stats_tracker.log_privacy_metrics(round_num, client_id, privacy_stats)
        
        if round_results.get('stopped_early', False):
            logger.info("Stopped early due to privacy budget")
            break
    
    # Final results
    duration = time.time() - start_time
    final_eval = trainer._evaluate_global_model()
    
    # Save final results
    final_results = {
        'final_accuracy': final_eval['accuracy'],
        'final_loss': final_eval['loss'],
        'training_duration_minutes': duration/60,
        'total_rounds': trainer.current_round,
        'privacy_budget_exhausted': any(round_results.get('privacy_status', {}).get('budget_exhausted', {}).values()),
        'server_best_accuracy': server_best_acc,
        'configuration': config
    }
    
    stats_tracker.save_final_results(final_results)
    
    # Generate all statistics files
    stats_tracker.save_all()
    
    print("\n" + "="*80)
    print("🎉 PRIVACY-PRESERVING FEDERATED TRAINING COMPLETED!")
    print("="*80)
    print("📊 TRAINING PHASES SUMMARY:")
    print("="*80)
    print(f"🌐 Phase 1a - Server Pre-training: {config['pretrain_epochs']} epochs")
    print(f"   └─ Server Best Accuracy: {server_best_acc:.4f}")
    print(f"🔧 Phase 1b - Client Personalization: {config['client_epochs']} epochs each")
    for node_id in client_models.keys():
        print(f"   └─ {node_id}: Personalized weights saved")
    print(f"🔒 Phase 2 - Federated Learning: {trainer.current_round} rounds")
    print(f"   └─ Final Global Accuracy: {final_eval['accuracy']:.4f}")
    print(f"⏱️  Total Duration: {duration/60:.1f} minutes")
    print("="*80)
    
    # Privacy summary
    print("🔐 PRIVACY BUDGET USAGE:")
    print("="*40)
    for node_id, accountant in trainer.privacy_accountants.items():
        stats = accountant.get_privacy_stats()
        usage = (stats['epsilon_used'] / config['max_epsilon']) * 100
        print(f"  {node_id}: {stats['epsilon_used']:.3f}/{config['max_epsilon']:.1f} ({usage:.1f}%)")
    
    print("="*80)
    print("📊 STATISTICS SAVED:")
    print(f"✅ Complete statistics saved in: {stats_tracker.base_dir}")
    print("✅ Formats: CSV, JSON, PNG plots, LaTeX tables")
    print("✅ See README.md in statistics directory for details")
    print("="*80)
    
    return final_eval

def main():
    parser = argparse.ArgumentParser(description='Privacy-Preserving Federated Training with Partitioned Data')
    parser.add_argument('--pretrain_epochs', type=int, default=50)
    parser.add_argument('--federated_rounds', type=int, default=1)
    
    args = parser.parse_args()
    run_improved_training(args)

if __name__ == "__main__":
    main() 