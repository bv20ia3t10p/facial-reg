# Federated Learning Centralized Mapping Integration

## Summary

Successfully updated the API federated learning components to use the centralized `identity_mapping.json` as the single source of truth for identity mappings.

## Components Updated

### ✅ 1. MappingService (`api/src/services/mapping_service.py`)
- **Centralized Integration**: Uses `MappingManager` from `privacy_biometrics.utils`
- **Constructor**: Simplified to `MappingService()` without parameters
- **Methods**: All methods now use centralized mapping backend
- **Caching**: Maintains performance cache while using centralized source

### ✅ 2. Coordinator (`api/src/coordinator.py`)
- **Centralized Functions**: All mapping functions use `MappingManager`
- **API Endpoints**: `/api/mapping` endpoints serve centralized mapping
- **Model Creation**: Uses centralized mapping for model initialization
- **Client Validation**: Validates client mappings against centralized source

### ✅ 3. Federated Client (`api/src/federated_client.py`)
- **Mapping Service**: Uses centralized `MappingService()`
- **Model Loading**: Determines model size from centralized mapping
- **Training**: Validates mapping consistency during training
- **Updates**: Includes mapping metadata in model updates

### ✅ 4. Identity Prediction (`api/src/services/identity_prediction.py`)
- **Centralized Lookup**: Uses centralized mapping for identity resolution
- **Method Updates**: Fixed all method calls to use correct API
- **Syntax**: Recreated file with proper indentation and structure

### ✅ 5. Biometric Service (`api/src/services/biometric_service.py`)
- **Service Integration**: Uses centralized `MappingService()`
- **Method Calls**: Updated all mapping method calls
- **Cache References**: Fixed all `cached_mapping` references

### ✅ 6. Service Initialization (`api/src/services/service_init.py`)
- **Constructor**: Updated to use parameterless `MappingService()`
- **Initialization**: Uses centralized mapping initialization
- **Legacy Cleanup**: Removed old server-based mapping logic

## Key Changes Made

### Constructor Updates
```python
# Before
self.mapping_service = MappingService(server_url, client_id)

# After  
self.mapping_service = MappingService()
```

### Method Call Updates
```python
# Before
mapping = self.mapping_service.cached_mapping
user_id = self.mapping_service.get_user_id(class_idx)

# After
mapping = self.mapping_service.get_mapping()
user_id = self.mapping_service.get_identity_by_index(class_idx)
```

### Centralized Mapping Usage
```python
# All components now use
from privacy_biometrics.utils import MappingManager
mapping_manager = MappingManager()  # Loads from identity_mapping.json
```

## Verification

### ✅ Test Script Created
- `test_api_centralized_mapping.py`: Comprehensive test suite
- Tests all major components for centralized mapping usage
- Validates mapping consistency across components

### ✅ Components Status
- **MappingService**: ✅ Uses centralized mapping
- **Coordinator**: ✅ Uses centralized mapping  
- **FederatedClient**: ✅ Uses centralized mapping
- **IdentityPrediction**: ✅ Fixed and uses centralized mapping
- **BiometricService**: ✅ Uses centralized mapping

## Benefits Achieved

1. **Single Source of Truth**: All components use `data/identity_mapping.json`
2. **Consistency**: No more mapping conflicts between components
3. **Centralized Management**: Easy to update mappings system-wide
4. **Backward Compatibility**: Legacy API formats maintained
5. **Performance**: Caching maintained while using centralized backend

## Files Modified

- `api/src/services/mapping_service.py` - Centralized backend
- `api/src/coordinator.py` - Centralized mapping functions
- `api/src/federated_client.py` - Constructor and method fixes
- `api/src/services/identity_prediction.py` - Recreated with fixes
- `api/src/services/biometric_service.py` - Method call updates
- `api/src/services/service_init.py` - Constructor updates
- `api/src/services/biometric_utils.py` - Method call updates

## Next Steps

The federated learning system now properly uses the centralized `identity_mapping.json`. All API components will:

1. Load identities from the centralized mapping file
2. Maintain consistent class indices across all clients
3. Validate mappings against the centralized source
4. Use the same identity-to-index mapping for training and inference

The system is ready for federated learning with centralized mapping coordination. 