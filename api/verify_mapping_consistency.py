#!/usr/bin/env python3
"""
Verification script to ensure training and prediction mapping methods produce identical results
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append('/app/api/src')

from pathlib import Path

def test_training_mapping_logic():
    """Test MappingManager.get_mapping_for_data_dir() logic (training)"""
    # Simulate the training logic
    print("=== TRAINING MAPPING LOGIC (MappingManager) ===")
    
    # Mock the global mapping (from identity_mapping.json)
    mock_global_mapping = {
        "101": 20, "1009": 80, "1010": 81, "1012": 83, "1013": 84, "1015": 86,
        "10101": 200, "10134": 230, "10148": 245  # etc
    }
    
    # Mock available identities in client1 data directory
    mock_available_identities = {"101", "1009", "1012", "1013", "1015", "10101", "10134"}
    
    # Training logic: start with global mapping order, then filter
    valid_identities = [
        identity for identity in mock_global_mapping.keys()  # Global mapping order
        if identity in mock_available_identities
    ]
    
    # Sort identities to ensure consistent ordering
    valid_identities.sort(key=int)
    
    # Create filtered mapping with continuous indices starting from 0
    training_mapping = {
        identity: idx 
        for idx, identity in enumerate(valid_identities)
    }
    
    print(f"Valid identities (sorted): {valid_identities}")
    print(f"Training mapping: {training_mapping}")
    return training_mapping

def test_prediction_mapping_logic():
    """Test MappingService.get_filtered_mapping_for_client() logic (prediction)"""
    print("\n=== PREDICTION MAPPING LOGIC (MappingService) ===")
    
    # Mock the global mapping (from identity_mapping.json)
    mock_global_mapping = {
        "101": 20, "1009": 80, "1010": 81, "1012": 83, "1013": 84, "1015": 86,
        "10101": 200, "10134": 230, "10148": 245  # etc
    }
    
    # Mock available identities in client1 data directory
    mock_available_identities = ["101", "1009", "1012", "1013", "1015", "10101", "10134"]
    
    # Prediction logic: sort data directory identities first
    mock_available_identities.sort(key=int)
    
    # Filter to identities that exist in both global mapping and data directory
    valid_identities = [
        identity for identity in mock_available_identities  # Data directory order
        if identity in mock_global_mapping
    ]
    
    # Create filtered mapping with continuous indices starting from 0
    prediction_mapping = {
        identity: idx 
        for idx, identity in enumerate(valid_identities)
    }
    
    print(f"Valid identities (sorted): {valid_identities}")
    print(f"Prediction mapping: {prediction_mapping}")
    return prediction_mapping

def main():
    print("Testing mapping consistency between training and prediction...")
    
    training_mapping = test_training_mapping_logic()
    prediction_mapping = test_prediction_mapping_logic()
    
    print("\n=== COMPARISON ===")
    if training_mapping == prediction_mapping:
        print("✅ SUCCESS: Training and prediction mappings are IDENTICAL!")
        print("   Both methods produce the same filtered mapping.")
    else:
        print("❌ ERROR: Training and prediction mappings are DIFFERENT!")
        print("   This would cause prediction errors.")
        
        print("\nDifferences:")
        all_keys = set(training_mapping.keys()) | set(prediction_mapping.keys())
        for key in sorted(all_keys, key=int):
            train_val = training_mapping.get(key, "MISSING")
            pred_val = prediction_mapping.get(key, "MISSING")
            if train_val != pred_val:
                print(f"  Identity {key}: Training={train_val}, Prediction={pred_val}")
    
    print(f"\nFinal mapping (both should be identical):")
    for identity, model_class in sorted(training_mapping.items(), key=lambda x: int(x[0])):
        print(f"  Identity {identity} → Model Class {model_class}")

if __name__ == "__main__":
    main() 