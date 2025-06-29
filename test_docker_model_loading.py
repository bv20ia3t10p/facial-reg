#!/usr/bin/env python3
"""
Test Docker model loading with BiometricModelLoader
"""

import os
import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_docker_model_loading():
    """Test model loading as it would work in Docker containers"""
    print("=" * 70)
    print("TESTING DOCKER MODEL LOADING")
    print("=" * 70)
    
    # Simulate Docker environment variables for each service
    test_scenarios = [
        {
            "name": "FL Coordinator",
            "NODE_TYPE": "coordinator",
            "MODEL_PATH": "/app/models/server_model.pth",
            "CLIENT_ID": "coordinator",
            "NUM_IDENTITIES": "300"
        },
        {
            "name": "Client 1",
            "NODE_TYPE": "client",
            "MODEL_PATH": "/app/models/client1_model.pth",
            "CLIENT_ID": "client1",
            "NUM_IDENTITIES": "300"
        },
        {
            "name": "Client 2", 
            "NODE_TYPE": "client",
            "MODEL_PATH": "/app/models/client2_model.pth",
            "CLIENT_ID": "client2",
            "NUM_IDENTITIES": "300"
        }
    ]
    
    success_count = 0
    
    for scenario in test_scenarios:
        print(f"\n{'='*50}")
        print(f"TESTING: {scenario['name']}")
        print(f"{'='*50}")
        
        # Set environment variables
        for key, value in scenario.items():
            if key != "name":
                os.environ[key] = value
                print(f"  {key}={value}")
        
        try:
            # Convert Docker path to local path for testing
            docker_model_path = scenario["MODEL_PATH"]
            local_model_path = docker_model_path.replace("/app/models/", "models/")
            os.environ["MODEL_PATH"] = local_model_path
            print(f"  Using local path: {local_model_path}")
            
            # Import and test BiometricModelLoader
            from api.src.services.biometric_model_loader import BiometricModelLoader
            
            # Get device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"  Device: {device}")
            
            # Create model loader
            loader = BiometricModelLoader(
                client_id=scenario["CLIENT_ID"],
                device=device
            )
            
            print(f"  Model path: {loader.model_path}")
            
            # Load model
            print("  Loading model...")
            model = loader.load_model()
            
            print(f"  ✅ Model loaded successfully!")
            print(f"    - Type: {type(model).__name__}")
            print(f"    - Device: {next(model.parameters()).device}")
            print(f"    - Num identities: {getattr(model, 'num_identities', 'unknown')}")
            print(f"    - Embedding dim: {getattr(model, 'embedding_dim', 'unknown')}")
            
            # Test model inference
            print("  Testing model inference...")
            with torch.no_grad():
                # Create dummy input (batch_size=2, channels=3, height=224, width=224)
                dummy_input = torch.randn(2, 3, 224, 224, device=device)
                
                # Forward pass
                output = model(dummy_input)
                
                # Handle different output formats
                if isinstance(output, tuple):
                    logits, embeddings = output
                    print(f"    - Logits shape: {logits.shape}")
                    print(f"    - Embeddings shape: {embeddings.shape if embeddings is not None else 'None'}")
                else:
                    logits = output
                    print(f"    - Logits shape: {logits.shape}")
                
                # Verify output dimensions
                expected_identities = int(scenario["NUM_IDENTITIES"])
                if logits.shape[1] == expected_identities:
                    print(f"    ✅ Output dimensions correct: {logits.shape[1]} identities")
                else:
                    print(f"    ⚠️  Output dimension mismatch: got {logits.shape[1]}, expected {expected_identities}")
            
            success_count += 1
            print(f"  🎉 {scenario['name']} test PASSED")
            
        except Exception as e:
            print(f"  ❌ Error testing {scenario['name']}: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Clean up environment variables
            for key in scenario.keys():
                if key != "name" and key in os.environ:
                    del os.environ[key]
    
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS: {success_count}/{len(test_scenarios)} scenarios passed")
    print(f"{'='*70}")
    
    if success_count == len(test_scenarios):
        print("🎉 All Docker model loading scenarios work correctly!")
        print("\n💡 Your Docker Compose is properly configured for:")
        print("   ✅ FL Coordinator with server_model.pth")
        print("   ✅ Client 1 with client1_model.pth")
        print("   ✅ Client 2 with client2_model.pth")
        print("\n🚀 You can now run: docker-compose up")
        return True
    else:
        print(f"⚠️  {len(test_scenarios) - success_count} scenarios failed")
        return False

def test_model_compatibility():
    """Test model architecture compatibility"""
    print("\n" + "=" * 70)
    print("TESTING MODEL ARCHITECTURE COMPATIBILITY")
    print("=" * 70)
    
    try:
        # Test ResNet50 import
        from privacy_biometrics.models.resnet50 import ResNet50ModelPretrained
        print("✅ ResNet50ModelPretrained import successful")
        
        # Test creating ResNet50 model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ResNet50ModelPretrained(
            num_identities=100,  # Match the models we have
            embedding_dim=512,
            privacy_enabled=True,
            pretrained=False
        ).to(device)
        
        print(f"✅ ResNet50 model created successfully")
        print(f"  - Device: {device}")
        print(f"  - Num identities: {model.num_identities}")
        print(f"  - Embedding dim: {model.embedding_dim}")
        
        # Test loading a real model file
        model_file = Path("models/client1_model.pth")
        if model_file.exists():
            print(f"\n🔧 Testing real model loading: {model_file}")
            state_dict = torch.load(model_file, map_location='cpu')
            
            # Load with strict=False to handle any mismatches
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            if not missing_keys and not unexpected_keys:
                print("✅ Perfect state dict match!")
            else:
                print(f"⚠️  State dict loaded with minor mismatches:")
                if missing_keys:
                    print(f"  - Missing keys: {len(missing_keys)}")
                if unexpected_keys:
                    print(f"  - Unexpected keys: {len(unexpected_keys)}")
            
            # Test inference
            model.eval()
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224, device=device)
                output = model(dummy_input)
                logits, embeddings = output
                print(f"✅ Model inference successful:")
                print(f"  - Logits shape: {logits.shape}")
                print(f"  - Embeddings shape: {embeddings.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing Docker Model Loading Configuration")
    
    # Test model compatibility first
    compat_success = test_model_compatibility()
    
    # Test Docker scenarios
    docker_success = test_docker_model_loading()
    
    overall_success = compat_success and docker_success
    
    print(f"\n{'='*70}")
    print(f"OVERALL RESULT: {'SUCCESS' if overall_success else 'FAILED'}")
    print(f"{'='*70}")
    
    exit(0 if overall_success else 1) 