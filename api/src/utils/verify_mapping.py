#!/usr/bin/env python
"""
Utility script to verify user ID mapping sorting
"""
import logging
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def verify_numerical_sorting():
    """Verify numerical vs lexicographical sorting behavior"""
    # Example folders from the client directory
    example_ids = ['1009', '101', '1010', '10035', '10087']
    
    print("\nSORT COMPARISON FOR USER FOLDERS:")
    print(f"Original list: {example_ids}")
    
    # Compare sorting methods
    print(f"\nLexicographical sorting (string): {sorted(example_ids)}")
    print(f"Numerical sorting (int): {sorted(example_ids, key=lambda x: int(x) if x.isdigit() else x)}")
    
    print("\nThis shows that numerical sorting ('1009' before '10035') is different from string sorting")
    print("and matches the directory structure as seen by Windows Explorer.")

def verify_directory_mapping():
    """Check the actual directory structure and appropriate mapping"""
    partitioned_path = Path("/app/data/partitioned")
    
    if not partitioned_path.exists():
        partitioned_path = Path("data/partitioned")  # Try relative path
        if not partitioned_path.exists():
            print("\nCould not find partitioned data directory")
            return False
    
    print("\nCHECKING PARTITIONED DATA DIRECTORIES:")
    
    # Get all nodes
    nodes = [d for d in partitioned_path.iterdir() if d.is_dir()]
    
    for node_path in nodes:
        node_name = node_path.name
        print(f"\n=> {node_name} directory:")
        
        if not node_path.exists():
            print(f"  {node_name} directory not found")
            continue
        
        # Get all user folders in this node
        user_folders = [d.name for d in node_path.iterdir() 
                     if d.is_dir() and d.name.isdigit()]
        
        if not user_folders:
            print(f"  No user folders found in {node_name}")
            continue
        
        # Compare sorting methods
        lex_sorted = sorted(user_folders)
        num_sorted = sorted(user_folders, key=lambda x: int(x))
        
        # Print sample of folders (first 10)
        print(f"  Found {len(user_folders)} user folders")
        print(f"  Sample folders: {user_folders[:10]}...")
        
        # Check if sorts are different
        if lex_sorted != num_sorted:
            print("\n  IMPORTANT: Lexicographical and numerical sorts are DIFFERENT!")
            print(f"  First mismatch: Index {next((i for i, (a, b) in enumerate(zip(lex_sorted, num_sorted)) if a != b), -1)}")
            
            # Show some examples of mismatches
            for i, (a, b) in enumerate(zip(lex_sorted, num_sorted)):
                if a != b:
                    print(f"  Mismatch: lex_sorted[{i}]={a}, num_sorted[{i}]={b}")
                    if i > 5:  # Show at most 5 mismatches
                        print("  ...")
                        break
        else:
            print("  Lexicographical and numerical sorts are identical for this dataset")

def verify_saved_mapping():
    """Check any saved mapping files"""
    mapping_paths = [
        Path("/app/models/mappings/global_mapping.json"),
        Path("/app/models/user_id_mapping.json"),
        Path("models/mappings/global_mapping.json")
    ]
    
    found_mapping = False
    
    for path in mapping_paths:
        if path.exists():
            found_mapping = True
            print(f"\nFound mapping file: {path}")
            
            try:
                with open(path, 'r') as f:
                    mapping = json.load(f)
                
                print(f"  Contains {len(mapping)} entries")
                
                # Convert string keys back to integers
                mapping = {int(k): v for k, v in mapping.items()}
                
                # Check order of values
                values = [v for _, v in sorted(mapping.items())]
                if values:
                    print("  First 10 mapped values:", values[:10])
                    
                    # Check if values that look like integers maintain numerical order
                    numeric_values = [v for v in values if isinstance(v, str) and v.isdigit()]
                    if numeric_values:
                        is_numerically_sorted = all(int(numeric_values[i]) <= int(numeric_values[i+1]) 
                                                 for i in range(len(numeric_values)-1))
                        print(f"  Values preserve numerical order: {is_numerically_sorted}")
                        
                        if not is_numerically_sorted:
                            # Find first violation
                            for i in range(len(numeric_values)-1):
                                if int(numeric_values[i]) > int(numeric_values[i+1]):
                                    print(f"  First order violation: {numeric_values[i]} > {numeric_values[i+1]}")
                                    break
            except Exception as e:
                print(f"  Error reading mapping: {e}")
    
    if not found_mapping:
        print("\nNo mapping files found")

if __name__ == "__main__":
    print("====== USER ID MAPPING VERIFICATION ======")
    verify_numerical_sorting()
    verify_directory_mapping()
    verify_saved_mapping()
    print("\n=========================================")
    print("Verification complete! Check the output above to ensure")
    print("that the mappings are using numerical sorting, not lexicographical.") 