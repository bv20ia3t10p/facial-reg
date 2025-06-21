"""
Script to fix imports in the federated_integration.py file
"""

import os
import sys
import traceback

# Add the current directory to the path
sys.path.append(os.path.abspath('.'))

print(f"Current directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

# Try to import the federated_integration module
try:
    print("Attempting to import federated_integration...")
    from src.federated_integration import get_federated_integration, initialize_federated_integration
    print("Successfully imported federated_integration module")
except Exception as e:
    print(f"Failed to import federated_integration: {e}")
    traceback.print_exc()
    
    # Try to fix the issue
    try:
        # Read the file
        file_path = 'src/federated_integration.py'
        print(f"Checking file: {file_path}")
        
        if os.path.exists(file_path):
            print(f"File exists: {file_path}")
            with open(file_path, 'r') as f:
                content = f.read()
            
            print(f"File content length: {len(content)}")
            print(f"First 100 characters: {content[:100]}")
            
            # Check if the file still contains 'Base'
            if 'class FederatedModelVersion(Base):' in content:
                print("Found reference to 'Base' in federated_integration.py")
                
                # Replace it with FederatedBase
                content = content.replace('class FederatedModelVersion(Base):', 'class FederatedModelVersion(FederatedBase):')
                
                # Write the file back
                with open(file_path, 'w') as f:
                    f.write(content)
                    
                print("Fixed reference to 'Base' in federated_integration.py")
            else:
                print("No reference to 'Base' found in federated_integration.py")
                print("Checking for other possible issues...")
                
                # Check for other issues
                if 'metadata = MetaData()' in content and 'FederatedBase = declarative_base(metadata=metadata)' in content:
                    print("Found correct MetaData and FederatedBase declarations")
                else:
                    print("MetaData or FederatedBase declarations might be incorrect")
        else:
            print(f"File does not exist: {file_path}")
    except Exception as e:
        print(f"Failed to fix federated_integration.py: {e}")
        traceback.print_exc() 