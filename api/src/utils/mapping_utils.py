"""
Mapping generation utility for creating client-specific mappings.
"""

import logging
from typing import Dict, Tuple, Any, Optional

logger = logging.getLogger(__name__)

def create_client_specific_mapping(
    full_mapping_data: Dict[str, Any], client_id: str
) -> Optional[Tuple[Dict[str, int], Dict[int, str]]]:
    """
    Creates a client-specific, re-indexed mapping from the global mapping data.

    Args:
        full_mapping_data: The complete data loaded from identity_mapping.json.
        client_id: The ID of the client (e.g., 'client1').

    Returns:
        A tuple containing two dictionaries:
        - The new identity-to-index mapping for the client.
        - The new index-to-identity reverse mapping.
        Returns None if the mapping cannot be created.
    """
    try:
        partition_stats = full_mapping_data.get("partition_stats", {})
        client_partition_info = partition_stats.get(client_id)

        if not client_partition_info or "identity_counts" not in client_partition_info:
            logger.error(f"No partition info for client '{client_id}' in mapping data.")
            return None

        client_identities_set = set(client_partition_info["identity_counts"].keys())

        # Sort the client's identities numerically to ensure a consistent order
        try:
            # Convert to int for sorting, then back to str
            sorted_identities = sorted(list(client_identities_set), key=int)
        except ValueError:
            # Fallback to lexical sort if UIDs are not all numeric
            logger.warning("Could not sort identities numerically, falling back to lexical sort.")
            sorted_identities = sorted(list(client_identities_set))

        if not sorted_identities:
            logger.error(f"No matching identities found for client '{client_id}'.")
            return None

        # Re-index the numerically sorted list from 0 to N-1
        new_mapping = {identity: i for i, identity in enumerate(sorted_identities)}
        new_reverse_mapping = {i: identity for identity, i in new_mapping.items()}

        logger.info(
            f"Created new specific mapping for client '{client_id}' with "
            f"{len(new_mapping)} identities, sorted numerically."
        )
        return new_mapping, new_reverse_mapping

    except Exception as e:
        logger.error(f"Failed to create client-specific mapping for '{client_id}': {e}")
        return None 