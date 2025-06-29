"""
Identity mapping service for API endpoints
Uses centralized identity_mapping.json as single source of truth
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

# Removed privacy_biometrics dependency to make API independent

logger = logging.getLogger(__name__)

class MappingService:
    """
    Service for managing identity mappings (API-independent version)
    """
    
    def __init__(self):
        """Initialize mapping service with identity_mapping.json"""
        self.mapping_cache = {}
        self.reverse_mapping_cache = {}
        self.mapping_metadata = {}
        # Add filtered mapping support for federated models
        self.filtered_mapping_cache = {}
        self.filtered_reverse_mapping_cache = {}
        self.client_data_dir = None
        self.initialize_mapping()
    
    def initialize_mapping(self) -> bool:
        """
        Initialize mapping using identity_mapping.json file
        
        Returns:
            True if successfully initialized, False otherwise
        """
        try:
            # Try to load from coordinator first
            try:
                response = self._fetch_mapping_from_server()
                if response:
                    logger.info("Successfully fetched mapping from coordinator")
                    return True
            except Exception as e:
                logger.warning(f"Could not fetch mapping from coordinator: {e}")
            
            # Try to load from identity_mapping.json
            mapping_loaded = self._load_from_identity_mapping_file()
            if mapping_loaded:
                logger.info("Successfully loaded mapping from identity_mapping.json")
                return True
            
            # Fallback to basic mapping generation from database
            self._generate_basic_mapping()
            
            logger.info(f"Initialized mapping service with {len(self.mapping_cache)} identities")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing mapping service: {e}")
            return False
    
    def _load_from_identity_mapping_file(self) -> bool:
        """Load mapping from identity_mapping.json file"""
        # Try multiple potential locations for the identity mapping file
        potential_paths = [
            Path("/app/models/identity_mapping.json"),  # Docker location
            Path("/app/data/identity_mapping.json"),    # Docker data location
            Path("./data/identity_mapping.json"),       # Local data
            Path("./identity_mapping.json"),            # Project root
            Path("../data/identity_mapping.json"),      # Parent data
        ]
        
        for mapping_path in potential_paths:
            try:
                if mapping_path.exists():
                    logger.info(f"Found identity mapping file at: {mapping_path}")
                    
                    with open(mapping_path, 'r') as f:
                        mapping_data = json.load(f)
                    
                    # Extract the mapping
                    if "mapping" in mapping_data:
                        raw_mapping = mapping_data["mapping"]
                        
                        # Convert to proper format (string keys to int values)
                        self.mapping_cache = {str(k): int(v) for k, v in raw_mapping.items()}
                        self.reverse_mapping_cache = {int(v): str(k) for k, v in raw_mapping.items()}
                        
                        # Store metadata
                        self.mapping_metadata = {
                            "version": mapping_data.get("version", "1.0.0"),
                            "total_identities": mapping_data.get("total_identities", len(self.mapping_cache)),
                            "hash": mapping_data.get("hash", ""),
                            "source_file": str(mapping_path),
                            "partition_stats": mapping_data.get("partition_stats", {})
                        }
                        
                        logger.info(f"Loaded {len(self.mapping_cache)} identities from {mapping_path}")
                        return True
                    else:
                        logger.warning(f"Invalid mapping file format in {mapping_path}")
                        
            except Exception as e:
                logger.debug(f"Could not load mapping from {mapping_path}: {e}")
                continue
        
        logger.warning("Could not find or load identity_mapping.json from any location")
        return False
    
    def _fetch_mapping_from_server(self) -> bool:
        """Fetch mapping from coordinator server"""
        import requests
        import os
        
        server_url = os.getenv("SERVER_URL", "http://fl-coordinator:8000")
        
        try:
            response = requests.get(f"{server_url}/api/mapping", timeout=5)
            if response.status_code == 200:
                mapping_data = response.json()
                
                if "identity_to_index" in mapping_data:
                    self.mapping_cache = mapping_data["identity_to_index"]
                    self.reverse_mapping_cache = {v: k for k, v in self.mapping_cache.items()}
                    
                    # Store metadata from server
                    self.mapping_metadata = {
                        "version": mapping_data.get("version", "1.0.0"),
                        "total_identities": mapping_data.get("total_identities", len(self.mapping_cache)),
                        "source": "coordinator_server",
                        "server_url": server_url
                    }
                    return True
        except Exception as e:
            logger.debug(f"Failed to fetch mapping from server: {e}")
        
        return False
    
    def _generate_basic_mapping(self):
        """Generate basic mapping from available data"""
        # Try to find data directories
        data_dirs = [
            Path("/app/data/partitioned/client1"),
            Path("/app/data/client1"),
            Path("./data/partitioned/client1"),
        ]
        
        identities = []
        for data_dir in data_dirs:
            if data_dir.exists():
                # Find identity directories (numeric folders)
                for item in data_dir.iterdir():
                    if item.is_dir() and item.name.isdigit():
                        identities.append(int(item.name))
                break
        
        if identities:
            identities.sort()
            self.mapping_cache = {str(i): i for i in identities}
            self.reverse_mapping_cache = {i: str(i) for i in identities}
            
            self.mapping_metadata = {
                "version": "1.0.0",
                "total_identities": len(identities),
                "source": "data_directory_scan",
                "data_directory": str(data_dir)
            }
            
            logger.info(f"Generated basic mapping from data directories: {len(identities)} identities")
        else:
            # Fallback to default mapping (300 identities to match models)
            self.mapping_cache = {str(i): i for i in range(300)}
            self.reverse_mapping_cache = {i: str(i) for i in range(300)}
            
            self.mapping_metadata = {
                "version": "1.0.0",
                "total_identities": 300,
                "source": "default_fallback"
            }
            
            logger.warning("Using default mapping (0-299)")
    
    def get_mapping(self) -> Dict[str, int]:
        """
        Get current identity mapping
        
        Returns:
            Dictionary mapping identity names to indices
        """
        if not self.mapping_cache:
            logger.warning("Mapping cache empty, reinitializing...")
            self.initialize_mapping()
        
        return self.mapping_cache.copy()
    
    def get_reverse_mapping(self) -> Dict[int, str]:
        """
        Get reverse mapping (index to identity)
        
        Returns:
            Dictionary mapping indices to identity names
        """
        if not self.reverse_mapping_cache:
            logger.warning("Reverse mapping cache empty, reinitializing...")
            self.initialize_mapping()
        
        return self.reverse_mapping_cache.copy()
    
    def get_identity_by_index(self, index: int) -> Optional[str]:
        """
        Get identity name by index
        
        Args:
            index: Identity index
            
        Returns:
            Identity name if found, None otherwise
        """
        return self.reverse_mapping_cache.get(index)
    
    def get_index_by_identity(self, identity: str) -> Optional[int]:
        """
        Get index by identity name
        
        Args:
            identity: Identity name
            
        Returns:
            Index if found, None otherwise
        """
        return self.mapping_cache.get(str(identity))
    
    def get_all_identities(self) -> List[str]:
        """
        Get list of all identity names
        
        Returns:
            List of identity names
        """
        return list(self.mapping_cache.keys())
    
    def get_total_identities(self) -> int:
        """
        Get total number of identities
        
        Returns:
            Total number of identities
        """
        return len(self.mapping_cache)
    
    def validate_identity(self, identity: str) -> bool:
        """
        Check if identity exists in mapping
        
        Args:
            identity: Identity to validate
            
        Returns:
            True if identity exists, False otherwise
        """
        return str(identity) in self.mapping_cache
    
    def validate_index(self, index: int) -> bool:
        """
        Check if index exists in mapping
        
        Args:
            index: Index to validate
            
        Returns:
            True if index exists, False otherwise
        """
        return index in self.reverse_mapping_cache
    
    def get_mapping_info(self) -> Dict:
        """
        Get comprehensive mapping information
        
        Returns:
            Dictionary with mapping metadata
        """
        base_info = {
            "total_identities": len(self.mapping_cache),
            "cached_identities": len(self.mapping_cache),
            "version": self.mapping_metadata.get("version", "1.0.0"),
            "source": self.mapping_metadata.get("source", "API-independent mapping service")
        }
        
        # Add additional metadata if available
        base_info.update(self.mapping_metadata)
        
        return base_info
    
    def refresh_mapping(self) -> bool:
        """
        Refresh mapping from centralized source
        
        Returns:
            True if successfully refreshed, False otherwise
        """
        logger.info("Refreshing mapping from centralized source...")
        return self.initialize_mapping()
    
    def search_identities(self, query: str, limit: int = 10) -> List[str]:
        """
        Search for identities matching query
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching identity names
        """
        query_lower = query.lower()
        matches = []
        
        for identity in self.mapping_cache.keys():
            if query_lower in identity.lower():
                matches.append(identity)
                if len(matches) >= limit:
                    break
        
        return matches
    
    def get_identity_stats(self) -> Dict:
        """
        Get statistics about identities
        
        Returns:
            Dictionary with identity statistics
        """
        if not self.mapping_cache:
            return {"error": "No mapping data available"}
        
        indices = list(self.reverse_mapping_cache.keys())
        identities = list(self.mapping_cache.keys())
        
        return {
            "total_identities": len(identities),
            "index_range": {
                "min": min(indices) if indices else None,
                "max": max(indices) if indices else None
            },
            "identity_examples": identities[:5],  # First 5 as examples
            "mapping_source": self.mapping_metadata.get("source_file", 
                                                      self.mapping_metadata.get("source", "API-independent mapping service")),
            "metadata": self.mapping_metadata
        }
    
    def get_filtered_mapping_for_client(self, client_id: str = None, data_dir: str = None) -> Dict[str, int]:
        """
        Get filtered mapping for a specific client that matches federated training logic.
        This recreates the exact MappingManager.get_mapping_for_data_dir() behavior.
        
        Args:
            client_id: Client identifier (e.g., 'client1', 'client2') 
            data_dir: Override data directory path
            
        Returns:
            Filtered mapping with continuous indices 0-N for available identities
        """
        if not self.mapping_cache:
            logger.warning("Global mapping not loaded, initializing...")
            self.initialize_mapping()
        
        # Determine data directory
        if data_dir:
            target_data_dir = Path(data_dir)
        elif client_id:
            # Try common client data directory patterns
            potential_dirs = [
                Path(f"/app/data/partitioned/{client_id}"),
                Path(f"/app/data/{client_id}"),
                Path(f"./data/partitioned/{client_id}"),
                Path(f"../data/partitioned/{client_id}"),
            ]
            target_data_dir = None
            for dir_path in potential_dirs:
                if dir_path.exists():
                    target_data_dir = dir_path
                    break
        else:
            # Try to detect client from available data
            potential_dirs = [
                Path("/app/data/partitioned/client1"),
                Path("/app/data/client1"),
                Path("./data/partitioned/client1"),
                Path("../data/partitioned/client1"),
            ]
            target_data_dir = None
            for dir_path in potential_dirs:
                if dir_path.exists():
                    target_data_dir = dir_path
                    client_id = "client1"
                    break
        
        if not target_data_dir or not target_data_dir.exists():
            logger.error(f"Could not find data directory for client {client_id}")
            return {}
        
        self.client_data_dir = str(target_data_dir)
        logger.info(f"Creating filtered mapping for client {client_id} using data from {target_data_dir}")
        
        # Get available identities in the data directory (same as training logic)
        try:
            available_identities = [d.name for d in target_data_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        except (OSError, PermissionError) as e:
            logger.error(f"Error accessing directory {target_data_dir}: {e}")
            return {}
        
        # Sort identities to ensure consistent ordering (CRITICAL for model compatibility)
        available_identities.sort(key=int)
        
        # Filter to identities that exist in both global mapping and data directory
        # Start with the sorted data directory identities, not the global mapping order
        valid_identities = [
            identity for identity in available_identities 
            if identity in self.mapping_cache
        ]
        
        # Create filtered mapping with continuous indices starting from 0 (matches training)
        filtered_mapping = {
            identity: idx 
            for idx, identity in enumerate(valid_identities)
        }
        
        # Create reverse mapping for predictions
        filtered_reverse_mapping = {
            idx: identity 
            for identity, idx in filtered_mapping.items()
        }
        
        # Cache the filtered mappings
        self.filtered_mapping_cache = filtered_mapping
        self.filtered_reverse_mapping_cache = filtered_reverse_mapping
        
        logger.info(f"Created filtered mapping for {len(filtered_mapping)} identities:")
        logger.info(f"  Model classes 0-{len(filtered_mapping)-1} map to identities: {valid_identities[:5]}...{valid_identities[-5:] if len(valid_identities) > 5 else valid_identities}")
        
        # Log missing identities for debugging
        missing_in_data = set(self.mapping_cache.keys()) - set(available_identities)
        missing_in_mapping = set(available_identities) - set(self.mapping_cache.keys())
        
        if missing_in_data:
            logger.debug(f"Identities in global mapping but not in {target_data_dir}: {sorted(list(missing_in_data))[:10]}...")
        if missing_in_mapping:
            logger.warning(f"Identities in {target_data_dir} but not in global mapping: {sorted(list(missing_in_mapping))[:10]}...")
        
        return filtered_mapping
    
    def get_identity_by_model_class(self, model_class: int, use_filtered: bool = True) -> Optional[str]:
        """
        Map model prediction class to identity name using filtered mapping.
        This is the correct method for federated model predictions.
        
        Args:
            model_class: Model prediction class (0-based index)
            use_filtered: Whether to use filtered mapping (should be True for federated models)
            
        Returns:
            Identity name if found, None otherwise
        """
        if use_filtered:
            if not self.filtered_reverse_mapping_cache:
                logger.warning("Filtered mapping not initialized, creating default...")
                self.get_filtered_mapping_for_client()
            
            identity = self.filtered_reverse_mapping_cache.get(model_class)
            if identity:
                logger.debug(f"Mapped model class {model_class} -> identity {identity}")
                return identity
            else:
                logger.warning(f"Model class {model_class} not found in filtered mapping (valid range: 0-{len(self.filtered_reverse_mapping_cache)-1})")
                return None
        else:
            # Legacy behavior - use global mapping
            return self.reverse_mapping_cache.get(model_class)
    
    def get_model_class_by_identity(self, identity: str, use_filtered: bool = True) -> Optional[int]:
        """
        Map identity name to model class using filtered mapping.
        
        Args:
            identity: Identity name
            use_filtered: Whether to use filtered mapping (should be True for federated models)
            
        Returns:
            Model class index if found, None otherwise
        """
        if use_filtered:
            if not self.filtered_mapping_cache:
                logger.warning("Filtered mapping not initialized, creating default...")
                self.get_filtered_mapping_for_client()
            
            model_class = self.filtered_mapping_cache.get(str(identity))
            if model_class is not None:
                logger.debug(f"Mapped identity {identity} -> model class {model_class}")
                return model_class
            else:
                logger.warning(f"Identity {identity} not found in filtered mapping")
                return None
        else:
            # Legacy behavior - use global mapping
            return self.mapping_cache.get(str(identity))

# Global instance for API use
mapping_service = MappingService()
    