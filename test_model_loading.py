#!/usr/bin/env python3
"""
Test script to verify model loading from /models directory
"""

import os
import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_model_loading():
    """Test loading models from the models directory"""
    print("=" * 60)
    print("TESTING MODEL LOADING FROM /models DIRECTORY")
    print("=" * 60)
    
    models_dir = Path("models")
    
    if not models_dir.exists():
        print(f"❌ Models directory not found: {models_dir}")
        return False
    
    print(f"✅ Models directory found: {models_dir}")
    
    # List all model files
    model_files = list(models_dir.glob("*.pth"))
    print(f"📁 Found {len(model_files)} model files:")
    
    for model_file in model_files:
        file_size = model_file.stat().st_size / (1024 * 1024)  # Size in MB
        print(f"  - {model_file.name} ({file_size:.1f} MB)")
    
    print("\n" + "=" * 60)
    print("TESTING MODEL FILE LOADING")
    print("=" * 60)
    
    success_count = 0
    
    for model_file in model_files:
        print(f"\n🔍 Testing: {model_file.name}")
        
        try:
            # Test loading the model file
            print(f"  Loading state dict...")
            state_dict = torch.load(model_file, map_location='cpu')
            
            # Check if it's a wrapped state dict or direct
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                print(f"  ✅ Found wrapped state dict with metadata")
                actual_state_dict = state_dict['state_dict']
                
                # Print metadata if available
                metadata_keys = [k for k in state_dict.keys() if k != 'state_dict']
                if metadata_keys:
                    print(f"  📋 Metadata keys: {metadata_keys}")
                    for key in metadata_keys:
                        print(f"    - {key}: {state_dict[key]}")
            else:
                print(f"  ✅ Found direct state dict")
                actual_state_dict = state_dict
            
            # Analyze the model structure
            print(f"  📊 Model analysis:")
            print(f"    - Total parameters: {len(actual_state_dict)}")
            
            # Look for identity/classification layers
            identity_layers = [k for k in actual_state_dict.keys() if 'identity' in k.lower() or 'classifier' in k.lower()]
            if identity_layers:
                print(f"    - Identity layers found: {len(identity_layers)}")
                for layer in identity_layers[:3]:  # Show first 3
                    if hasattr(actual_state_dict[layer], 'shape'):
                        print(f"      - {layer}: {actual_state_dict[layer].shape}")
            
            # Check for backbone layers
            backbone_layers = [k for k in actual_state_dict.keys() if 'backbone' in k.lower() or 'conv' in k.lower()]
            print(f"    - Backbone layers found: {len(backbone_layers)}")
            
            print(f"  ✅ Model loaded successfully!")
            success_count += 1
            
        except Exception as e:
            print(f"  ❌ Error loading model: {e}")
    
    print("\n" + "=" * 60)
    print("TESTING MODEL COMPATIBILITY")
    print("=" * 60)
    
    # Test loading with the actual model class
    try:
        from api.src.models.privacy_biometric_model import PrivacyBiometricModel
        print("✅ Successfully imported PrivacyBiometricModel")
        
        # Test creating a model instance
        print("🔧 Testing model instantiation...")
        model = PrivacyBiometricModel(num_identities=300, privacy_enabled=False)
        print(f"✅ Model created successfully")
        print(f"  - Device: {model.device}")
        print(f"  - Num identities: {model.num_identities}")
        print(f"  - Embedding dim: {model.embedding_dim}")
        print(f"  - Backbone output dim: {model.backbone_output_dim}")
        
        # Test with a sample model file
        if model_files:
            test_model = model_files[0]
            print(f"\n🔧 Testing loading state dict into model: {test_model.name}")
            
            state_dict = torch.load(test_model, map_location='cpu')
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            # Try to load the state dict
            try:
                model.load_state_dict(state_dict, strict=False)
                print(f"✅ State dict loaded successfully (strict=False)")
            except Exception as e:
                print(f"⚠️  State dict loading with warnings: {e}")
                
                # Try to identify mismatched keys
                model_keys = set(model.state_dict().keys())
                loaded_keys = set(state_dict.keys())
                
                missing_keys = model_keys - loaded_keys
                unexpected_keys = loaded_keys - model_keys
                
                if missing_keys:
                    print(f"  📝 Missing keys in loaded model: {len(missing_keys)}")
                    for key in list(missing_keys)[:5]:  # Show first 5
                        print(f"    - {key}")
                
                if unexpected_keys:
                    print(f"  📝 Unexpected keys in loaded model: {len(unexpected_keys)}")
                    for key in list(unexpected_keys)[:5]:  # Show first 5
                        print(f"    - {key}")
        
    except Exception as e:
        print(f"❌ Error testing model compatibility: {e}")
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {success_count}/{len(model_files)} models loaded successfully")
    print("=" * 60)
    
    if success_count == len(model_files):
        print("🎉 All models can be loaded successfully!")
        print("\n💡 Docker Compose Configuration:")
        print("   - Coordinator: MODEL_PATH=/app/models/server_model.pth")
        print("   - Client1: MODEL_PATH=/app/models/client1_model.pth") 
        print("   - Client2: MODEL_PATH=/app/models/client2_model.pth")
        return True
    else:
        print("⚠️  Some models had loading issues.")
        return False

if __name__ == "__main__":
    success = test_model_loading()
    exit(0 if success else 1) 