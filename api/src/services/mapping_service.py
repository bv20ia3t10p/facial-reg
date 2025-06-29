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

# Add privacy_biometrics to path for centralized mapping
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from privacy_biometrics.utils import MappingManager

logger = logging.getLogger(__name__)

class MappingService:
    """
    Service for managing identity mappings using centralized identity_mapping.json
    """
    
    def __init__(self):
        """Initialize mapping service with centralized manager"""
        self.mapping_manager = None
        self.mapping_cache = {}
        self.reverse_mapping_cache = {}
        self.initialize_mapping()
    
    def initialize_mapping(self) -> bool:
        """
        Initialize mapping using centralized MappingManager
        
        Returns:
            True if successfully initialized, False otherwise
        """
        try:
            # Initialize centralized mapping manager
            self.mapping_manager = MappingManager()
            
            if not self.mapping_manager.mapping:
                logger.error("Failed to load centralized identity mapping")
                return False
            
            # Cache the mappings for performance
            self.mapping_cache = self.mapping_manager.mapping.copy()
            self.reverse_mapping_cache = self.mapping_manager.reverse_mapping.copy()
            
            logger.info(f"Initialized mapping service with {len(self.mapping_cache)} identities")
            logger.info(f"Using centralized mapping from: {self.mapping_manager.identity_mapping_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing centralized mapping service: {e}")
            return False
    
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
        if self.mapping_manager:
            return self.mapping_manager.get_mapping_info()
        else:
            return {
                "error": "Mapping manager not initialized",
                "total_identities": len(self.mapping_cache),
                "cached_identities": len(self.mapping_cache)
            }
    
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
            "mapping_source": getattr(self.mapping_manager, 'identity_mapping_file', 'unknown')
        }

# Global instance for API use
mapping_service = MappingService()
    