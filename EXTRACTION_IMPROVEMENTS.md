# CASIA Dataset Extraction Script Improvements

## Overview
The extraction script has been significantly improved to efficiently extract exactly 100 classes for each split (server, client1, client2) instead of over-extracting and then discarding classes.

## Key Changes Made

### 1. PowerShell Script (`Extract-CasiaDataset.ps1`)

**New Parameters:**
- `classes_per_split`: Number of classes per split (default: 100)
- `max_classes`: Total classes to extract (default: 300, automatically calculated as `classes_per_split * 3`)

**Improved Logic:**
- Validates that `max_classes` matches the expected total for splits
- Ensures exactly the specified number of classes per split
- Provides better error handling when insufficient classes are available
- Shows progress and unused class information

### 2. Python Extraction Script (`extract_rec_simple.py`)

**Efficiency Improvements:**
- **Early Termination**: Stops extraction once target number of classes with sufficient images (≥5 per class) is reached
- **Better Class Selection**: Uses consistent alphabetical ordering for reproducible results  
- **Progress Tracking**: Shows progress toward target with detailed logging
- **Smart Balancing**: Prevents over-extraction by monitoring class distribution

**Key Algorithm Changes:**
```python
# Old approach: Extract many classes, then filter later
# New approach: Stop when target is reached
if max_classes and len(classes_used) >= max_classes:
    classes_with_enough_images = sum(1 for c in classes_used if classes_count.get(c, 0) >= 5)
    if classes_with_enough_images >= max_classes:
        logger.info(f"Reached target of {max_classes} classes with at least 5 images each")
        break
```

## Usage Examples

### Basic Usage - 100 Classes Per Split
```powershell
.\Extract-CasiaDataset.ps1
```
This will extract exactly 300 classes total (100 per split).

### Custom Classes Per Split
```powershell
.\Extract-CasiaDataset.ps1 -classes_per_split 50
```
This will extract exactly 150 classes total (50 per split).

### Skip Stages
```powershell
# Skip extraction if already done
.\Extract-CasiaDataset.ps1 -skip_extraction

# Skip partitioning if only want extraction
.\Extract-CasiaDataset.ps1 -skip_partitioning
```

## Performance Benefits

### Before (Old Script):
1. Extract up to 750 classes (potentially thousands of images)
2. Filter to valid classes (≥5 images each) 
3. Randomly partition into 3 groups
4. Copy ~250 classes per split
5. **Waste**: Unused classes and excessive extraction time

### After (Improved Script):
1. Extract exactly 300 classes (targeted extraction)
2. Stop early when target reached
3. Partition exactly 100 classes per split
4. **Efficiency**: No wasted extraction, faster processing

## Benefits Summary

✅ **Faster Extraction**: Stops when target is reached instead of over-extracting  
✅ **Predictable Output**: Exactly 100 classes per split, every time  
✅ **Less Storage**: No unused extracted classes taking up disk space  
✅ **Better Resource Usage**: CPU and memory used more efficiently  
✅ **Reproducible**: Consistent alphabetical class selection  
✅ **Clear Progress**: Detailed logging shows progress toward target  

## File Structure After Processing

```
data/
├── casia-extracted/           # Raw extracted images (300 classes)
│   ├── 00001/                # Class directories with images
│   ├── 00002/
│   └── ...
└── partitioned/               # Split for federated learning
    ├── server/                # Exactly 100 classes
    │   ├── 00001/
    │   └── ...
    ├── client1/               # Exactly 100 classes  
    │   ├── 00101/
    │   └── ...
    └── client2/               # Exactly 100 classes
        ├── 00201/
        └── ...
```

## Technical Details

### Class Selection Strategy
- Classes are sorted alphabetically for consistent ordering
- First N classes are selected (where N = max_classes)
- This ensures reproducible results across runs

### Stopping Criteria
The extraction stops when either:
1. Target number of classes with ≥5 images each is reached, OR
2. Safety limit of 1.5x target classes is processed (prevents infinite loops)
3. Maximum images limit is reached (if specified)

### Error Handling
- Validates sufficient classes are available before partitioning
- Adjusts class count if insufficient classes found
- Warns about unused classes after partitioning
- Provides clear error messages for debugging 