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
import httpx

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
        # Initialization must now be called asynchronously after creation
    
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
        server_url = os.getenv("SERVER_URL", "http://fl-coordinator:8080")
        
        try:
            with httpx.Client(transport=httpx.HTTPTransport(http2=False)) as client:
                response = client.get(f"{server_url}/api/mapping", timeout=5)
            
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
            logger.warning("Mapping cache is empty. Please call `await initialize_mapping()` first.")
        
        return self.mapping_cache.copy()
    
    def get_reverse_mapping(self) -> Dict[int, str]:
        """
        Get reverse mapping (index to identity)
        
        Returns:
            Dictionary mapping indices to identity names
        """
        if not self.reverse_mapping_cache:
            logger.warning("Reverse mapping cache is empty. Please call `await initialize_mapping()` first.")
        
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
        Get metadata about the current mapping
        
        Returns:
            Dictionary with mapping metadata
        """
        return {
            "total_identities": self.get_total_identities(),
            "metadata": self.mapping_metadata
        }
    
    def refresh_mapping(self) -> bool:
        """
        Refreshes the mapping from the source (file or server).

        Returns:
            True if mapping was refreshed successfully, False otherwise.
        """
        logger.info("Attempting to refresh mapping...")
        # First, try fetching from server, as it's the most dynamic source
        if self._fetch_mapping_from_server():
            logger.info("Successfully refreshed mapping from server.")
            return True
        
        # If server fails, fallback to loading from the identity mapping file
        if self._load_from_identity_mapping_file():
            logger.info("Successfully refreshed mapping from file.")
            return True

        logger.warning("Failed to refresh mapping from all available sources.")
        return False

    def search_identities(self, query: str, limit: int = 10) -> List[str]:
        """
        Search for identities matching a query string
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching identity names
        """
        return [identity for identity in self.mapping_cache.keys() if query.lower() in identity.lower()][:limit]
    
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
    
    def get_filtered_mapping_for_client(self, client_id: Optional[str] = None, data_dir: Optional[str] = None) -> Dict[str, int]:
        """
        Get a filtered mapping for a specific client based on their local data.
        This is crucial for federated learning where clients only know about
        a subset of users.

        Args:
            client_id: The ID of the client (e.g., 'client1').
            data_dir: The root directory of the client's partitioned data.

        Returns:
            A dictionary mapping user IDs to a contiguous range of integers (0 to N-1).
        """
        # If filtered mapping for this client is already cached, return it
        if client_id and client_id in self.filtered_mapping_cache:
            return self.filtered_mapping_cache[client_id]

        if not self.mapping_cache:
            logger.info("Main mapping not initialized, initializing now...")
            self.initialize_mapping()

        if client_id and not data_dir:
            self.client_data_dir = self.client_data_dir or Path(f"/app/data/partitioned/{client_id}")
            data_dir_path = self.client_data_dir
        elif data_dir:
            data_dir_path = Path(data_dir)
        else:
            data_dir_path = None

        available_identities = []
        if data_dir_path and data_dir_path.exists():
            for item in data_dir_path.iterdir():
                if item.is_dir() and item.name in self.mapping_cache:
                    available_identities.append(item.name)
        else:
            logger.warning(f"Data directory not found for client {client_id}, cannot create filtered mapping.")
            return {} # Return empty dict if no data dir

        # Sort identities for consistency
        available_identities.sort()

        # Create a new mapping for these identities from 0 to N-1
        filtered_mapping = {identity: i for i, identity in enumerate(available_identities)}

        # Cache the filtered mapping
        if client_id:
            self.filtered_mapping_cache[client_id] = filtered_mapping
            self.filtered_reverse_mapping_cache[client_id] = {v: k for k, v in filtered_mapping.items()}

        return filtered_mapping

    def get_identity_by_model_class(self, model_class: int, use_filtered: bool = True) -> Optional[str]:
        """
        Get identity by the model's output class index.
        This is the reverse of `get_model_class_by_identity`.

        Args:
            model_class: The class index predicted by the model.
            use_filtered: Whether to use the filtered mapping cache.

        Returns:
            The identity string (e.g., "user_123") or None if not found.
        """
        if use_filtered and self.filtered_reverse_mapping_cache:
            return self.filtered_reverse_mapping_cache.get(model_class)
        return self.reverse_mapping_cache.get(model_class)

    def get_model_class_by_identity(self, identity: str, use_filtered: bool = True) -> Optional[int]:
        """
        Get the model's output class index for a given identity.
        This is the reverse of `get_identity_by_model_class`.

        Args:
            identity: The identity string (e.g., "user_123").
            use_filtered: Whether to use the filtered mapping cache.

        Returns:
            The model class index or None if not found.
        """
        if use_filtered and self.filtered_mapping_cache:
            return self.filtered_mapping_cache.get(identity)
        return self.mapping_cache.get(identity)

# Global instance for API use
mapping_service = MappingService()
    