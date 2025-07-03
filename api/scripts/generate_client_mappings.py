import json
import os
import sys
from pathlib import Path

# Add the project root to the Python path to allow importing from 'src'
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.services.mapping_service import MappingService

def generate_client_mappings():
    """
    Generates filtered mapping files for each client based on partition_stats
    in the main identity_mapping.json file.
    """
    output_dir = project_root / "data"
    output_dir.mkdir(exist_ok=True)

    # Initialize the mapping service, which will load identity_mapping.json
    mapping_service = MappingService()
    
    if not mapping_service.mapping_metadata.get("partition_stats"):
        print("Error: partition_stats not found in identity_mapping.json. Cannot generate client mappings.")
        return

    partition_stats = mapping_service.mapping_metadata["partition_stats"]
    client_ids = [key for key in partition_stats.keys() if key != "server"]

    print(f"Found client IDs in partition_stats: {client_ids}")

    for client_id in client_ids:
        print(f"Generating mapping for {client_id}...")
        
        filtered_mapping = mapping_service.get_filtered_mapping_for_client(client_id)
        
        if not filtered_mapping:
            print(f"Warning: Could not generate filtered mapping for {client_id}. Skipping.")
            continue

        reverse_filtered_mapping = {v: k for k, v in filtered_mapping.items()}

        client_mapping_data = {
            "version": "1.0.0-filtered",
            "client_id": client_id,
            "source_hash": mapping_service.mapping_metadata.get("hash"),
            "total_identities": len(filtered_mapping),
            "identity_to_class": filtered_mapping,
            "class_to_identity": reverse_filtered_mapping,
        }

        output_file = output_dir / f"{client_id}_mapping.json"
        with open(output_file, 'w') as f:
            json.dump(client_mapping_data, f, indent=2)
        
        print(f"Successfully generated mapping for {client_id} at: {output_file}")

if __name__ == "__main__":
    generate_client_mappings() 