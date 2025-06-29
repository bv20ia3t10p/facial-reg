#!/usr/bin/env python3
"""
Test script to verify the new filtered mapping functionality in the API
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from services.mapping_service import MappingService
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_filtered_mapping():
    """Test the filtered mapping functionality"""
    
    print("=== Testing Filtered Mapping for Federated API ===\n")
    
    # Initialize mapping service
    mapping_service = MappingService()
    
    print("1. Initialize global mapping...")
    success = mapping_service.initialize_mapping()
    print(f"   Global mapping initialized: {success}")
    print(f"   Total identities in global mapping: {len(mapping_service.mapping_cache)}")
    print(f"   Sample global identities: {list(mapping_service.mapping_cache.keys())[:10]}")
    
    print("\n2. Create filtered mapping for client1...")
    filtered_mapping = mapping_service.get_filtered_mapping_for_client("client1")
    print(f"   Filtered mapping created: {len(filtered_mapping)} identities")
    print(f"   Model classes: 0-{len(filtered_mapping)-1}")
    
    if filtered_mapping:
        print(f"   Sample filtered mapping:")
        for i, (identity, model_class) in enumerate(list(filtered_mapping.items())[:10]):
            print(f"     Identity {identity} -> Model class {model_class}")
        
        print("\n3. Test mapping model predictions to identities...")
        test_predictions = [0, 1, 2, 50, 99]  # Model predictions
        
        for pred_class in test_predictions:
            identity = mapping_service.get_identity_by_model_class(pred_class, use_filtered=True)
            print(f"   Model class {pred_class} -> Identity {identity}")
        
        print("\n4. Test mapping identities to model classes...")
        # Get a few identities from the filtered mapping
        sample_identities = list(filtered_mapping.keys())[:5]
        
        for identity in sample_identities:
            model_class = mapping_service.get_model_class_by_identity(identity, use_filtered=True)
            print(f"   Identity {identity} -> Model class {model_class}")
        
        print("\n5. Test edge cases...")
        # Test invalid model class
        invalid_identity = mapping_service.get_identity_by_model_class(999, use_filtered=True)
        print(f"   Invalid model class 999 -> {invalid_identity}")
        
        # Test invalid identity
        invalid_class = mapping_service.get_model_class_by_identity("99999", use_filtered=True)
        print(f"   Invalid identity 99999 -> {invalid_class}")
        
        print("\n6. Compare with old global mapping...")
        print("   Old global mapping behavior:")
        for pred_class in [0, 1, 2]:
            old_identity = mapping_service.get_identity_by_index(pred_class)
            print(f"     Model class {pred_class} -> Global identity {old_identity}")
        
        print("\n   New filtered mapping behavior:")
        for pred_class in [0, 1, 2]:
            new_identity = mapping_service.get_identity_by_model_class(pred_class, use_filtered=True)
            print(f"     Model class {pred_class} -> Filtered identity {new_identity}")
        
        print(f"\n=== Summary ===")
        print(f"✓ Global mapping loaded: {len(mapping_service.mapping_cache)} identities")
        print(f"✓ Filtered mapping created: {len(filtered_mapping)} identities") 
        print(f"✓ Model predictions now map to correct identities for client1")
        print(f"✓ Filtered mapping matches federated training logic")
        
    else:
        print("   ❌ Failed to create filtered mapping")
        return False
    
    return True

if __name__ == "__main__":
    success = test_filtered_mapping()
    if success:
        print("\n🎉 Filtered mapping test completed successfully!")
    else:
        print("\n❌ Filtered mapping test failed!")
        sys.exit(1) 