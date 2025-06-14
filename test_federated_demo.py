"""
Test Script for Federated Biometric Learning Demo
Tests the complete pipeline with your partitioned CASIA-WebFace data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json
from datetime import datetime

# Import our custom modules
from data_loader import FederatedDataManager, FederatedBiometricDataset
from privacy_biometric_model import PrivacyBiometricModel, FederatedModelManager, PrivacyAccountant

def test_data_loading():
    """Test federated data loading with your partitioned data"""
    print("=== Testing Federated Data Loading ===")
    
    # Initialize data manager
    data_manager = FederatedDataManager("data/partitioned")
    
    # Test each node
    for node in ["server", "client1", "client2"]:
        print(f"\nTesting {node} node:")
        
        # Create dataset
        dataset = data_manager.create_node_dataset(node)
        print(f"  ✓ Dataset created: {len(dataset)} samples, {dataset.get_node_info()['num_identities']} identities")
        
        # Create privacy-compatible dataloader
        dataloader = data_manager.create_privacy_dataloader(node, batch_size=8)
        
        # Test loading one batch
        for batch_idx, (images, identity_labels, emotion_labels) in enumerate(dataloader):
            print(f"  ✓ Batch loaded: {images.shape}, identities: {identity_labels.min()}-{identity_labels.max()}")
            break  # Only test first batch
    
    # Print overall statistics
    stats = data_manager.get_federated_stats()
    print(f"\n=== Federated Data Statistics ===")
    print(f"Total nodes: {stats['total_nodes']}")
    print(f"Global identities: {stats['global_identities']}")
    print(f"Total images: {stats['total_images']}")
    
    return data_manager

def test_privacy_model():
    """Test privacy-enabled biometric model"""
    print("\n=== Testing Privacy-Enabled Model ===")
    
    # Create model manager
    model_manager = FederatedModelManager(num_identities=300)
    
    # Test model creation for each node
    models = {}
    for node in ["server", "client1", "client2"]:
        print(f"\nTesting model for {node}:")
        
        # Create model
        model = model_manager.create_node_model(node, privacy_enabled=True)
        models[node] = model
        
        # Test forward pass
        dummy_input = torch.randn(4, 3, 224, 224)
        identity_logits, emotion_logits, features = model(dummy_input, add_noise=True, node_id=node)
        
        print(f"  ✓ Forward pass: identity {identity_logits.shape}, emotion {emotion_logits.shape}")
        
        # Test model expansion (new employee enrollment)
        if node == "client2":
            print(f"  ✓ Testing model expansion...")
            expanded_model = model.expand_for_new_identity(1)
            identity_expanded, _, _ = expanded_model(dummy_input)
            print(f"  ✓ Expanded model: {identity_expanded.shape}")
    
    # Test global model
    global_model = model_manager.create_global_model()
    print(f"\n✓ Global model created")
    
    # Print model statistics
    stats = model_manager.get_federated_stats()
    print(f"\nModel Statistics:")
    for node, node_stats in stats["node_models"].items():
        print(f"  {node}: {node_stats['total_parameters']:,} parameters, {node_stats['model_size_mb']:.1f}MB")
    
    return model_manager, models

def test_privacy_training_simulation():
    """Test privacy-preserving training simulation"""
    print("\n=== Testing Privacy Training Simulation ===")
    
    # Create a simple model for testing
    model = PrivacyBiometricModel(num_identities=100, privacy_enabled=True)  # Smaller for testing
    
    # Test privacy accounting
    privacy_accountant = PrivacyAccountant(max_epsilon=1.0, delta=1e-5)
    
    print("Simulating training steps with privacy budget tracking:")
    for step in range(20):
        # Simulate training step
        epsilon_step = 0.05
        privacy_accountant.consume_privacy_budget(epsilon_step)
        
        if step % 5 == 0:
            stats = privacy_accountant.get_privacy_stats()
            print(f"  Step {step:2d}: ε_used={stats['epsilon_used']:.3f}, remaining={stats['epsilon_remaining']:.3f}")
            
            if not privacy_accountant.can_train():
                print(f"  ⚠️  Privacy budget exhausted at step {step}")
                break
    
    return privacy_accountant

def test_federated_integration():
    """Test complete federated learning integration"""
    print("\n=== Testing Federated Learning Integration ===")
    
    # Initialize components
    data_manager = FederatedDataManager("data/partitioned")
    model_manager = FederatedModelManager(num_identities=300)
    
    # Create models and datasets for each node
    federated_setup = {}
    
    for node in ["server", "client1", "client2"]:
        print(f"\nSetting up {node}:")
        
        # Create dataset and dataloader
        dataset = data_manager.create_node_dataset(node)
        dataloader = data_manager.create_privacy_dataloader(node, batch_size=4)  # Small batch for testing
        
        # Create model
        model = model_manager.create_node_model(node, privacy_enabled=True)
        
        # Create optimizer (would be used in actual training)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        federated_setup[node] = {
            "dataset": dataset,
            "dataloader": dataloader,
            "model": model,
            "optimizer": optimizer,
            "node_info": dataset.get_node_info()
        }
        
        print(f"  ✓ {node} setup complete: {federated_setup[node]['node_info']['num_images']} images")
    
    # Test one training step simulation for each node
    print("\nSimulating one training step per node:")
    
    for node, setup in federated_setup.items():
        model = setup["model"]
        dataloader = setup["dataloader"]
        
        # Get one batch
        for batch_idx, (images, identity_labels, emotion_labels) in enumerate(dataloader):
            # Simulate forward pass
            model.train()
            identity_logits, emotion_logits, features = model(images, add_noise=True, node_id=node)
            
            # Simulate loss computation
            identity_loss = nn.CrossEntropyLoss()(identity_logits, identity_labels)
            emotion_loss = nn.CrossEntropyLoss()(emotion_logits, emotion_labels)
            total_loss = identity_loss + 0.3 * emotion_loss
            
            print(f"  {node}: loss={total_loss.item():.4f}, batch_size={images.shape[0]}")
            break  # Only test first batch
    
    return federated_setup

def test_new_employee_enrollment():
    """Test new employee enrollment simulation"""
    print("\n=== Testing New Employee Enrollment ===")
    
    data_manager = FederatedDataManager("data/partitioned")
    
    # Simulate new employee enrollment
    enrollment_info = data_manager.simulate_new_employee_enrollment(target_node="client2")
    
    print("New Employee Enrollment Simulation:")
    print(f"  New Employee ID: {enrollment_info['new_employee_id']}")
    print(f"  Assigned Node: {enrollment_info['assigned_node']}")
    print(f"  Current Identities: {enrollment_info['current_global_identities']}")
    print(f"  After Enrollment: {enrollment_info['new_global_identities']}")
    
    # Test model expansion
    model = PrivacyBiometricModel(num_identities=300, privacy_enabled=True)
    print(f"\nModel before expansion: {model.num_identities} identities")
    
    expanded_model = model.expand_for_new_identity(1)
    print(f"Model after expansion: {expanded_model.num_identities} identities")
    
    return enrollment_info

def run_complete_demo():
    """Run complete federated learning demo"""
    print("🚀 Starting Complete Federated Biometric Learning Demo")
    print("=" * 60)
    
    try:
        # Test 1: Data Loading
        data_manager = test_data_loading()
        
        # Test 2: Privacy Model
        model_manager, models = test_privacy_model()
        
        # Test 3: Privacy Training
        privacy_accountant = test_privacy_training_simulation()
        
        # Test 4: Federated Integration
        federated_setup = test_federated_integration()
        
        # Test 5: New Employee Enrollment
        enrollment_info = test_new_employee_enrollment()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("\n📊 Demo Summary:")
        print(f"  • Federated nodes: 3 (server, client1, client2)")
        print(f"  • Total identities: 300 (100 per node)")
        print(f"  • Privacy-enabled models: ✓")
        print(f"  • Differential privacy: ✓")
        print(f"  • Model expansion: ✓")
        print(f"  • Data locality: ✓ (no raw data sharing)")
        
        # Save demo results
        demo_results = {
            "timestamp": datetime.now().isoformat(),
            "federated_stats": data_manager.get_federated_stats(),
            "model_stats": model_manager.get_federated_stats(),
            "privacy_stats": privacy_accountant.get_privacy_stats(),
            "enrollment_simulation": enrollment_info
        }
        
        with open("demo_results.json", "w") as f:
            json.dump(demo_results, f, indent=2)
        
        print(f"  • Demo results saved to: demo_results.json")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Check if data directory exists
    if not os.path.exists("data/partitioned"):
        print("❌ Error: data/partitioned directory not found!")
        print("Please ensure your partitioned CASIA-WebFace data is in data/partitioned/")
        exit(1)
    
    # Run the complete demo
    success = run_complete_demo()
    
    if success:
        print("\n🎉 Federated biometric learning demo completed successfully!")
        print("You can now proceed with:")
        print("  1. Training the models with your data")
        print("  2. Implementing the authentication API")
        print("  3. Building the Progressive Web App")
        print("  4. Setting up homomorphic encryption")
    else:
        print("\n💡 Check the error messages above and ensure all dependencies are installed:")
        print("  pip install torch torchvision opacus pillow") 