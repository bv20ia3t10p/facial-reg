"""
Common utilities used across the application
"""

import uuid
import logging
from pathlib import Path
import json
import sqlite3
import os
import re

logger = logging.getLogger(__name__)

def get_max_user_id() -> int:
    """
    Get the maximum user ID from the database
    Returns 0 if no users exist or if there's an error.
    """
    try:
        # Get database path from environment variable or use default
        db_path = os.getenv("DATABASE_URL", "sqlite:////app/database/client1.db")
        if db_path.startswith("sqlite:///"):
            db_path = db_path[10:]
        
        # Check if the database file exists
        if not os.path.exists(db_path):
            logger.warning(f"Database file not found: {db_path}")
            return 0
            
        # Connect to the database and query for all user IDs
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all user IDs from the database
        cursor.execute("SELECT id FROM users")
        user_ids = [row[0] for row in cursor.fetchall()]
        
        if not user_ids:
            return 0
        
        # Extract numeric IDs (if any)
        numeric_ids = []
        for id_str in user_ids:
            # If it's a pure numeric ID
            if re.match(r'^\d+$', id_str):
                numeric_ids.append(int(id_str))
            # If it contains numeric parts (e.g., "user-123")
            elif re.search(r'\d+', id_str):
                # Extract all numbers from the string
                numbers = re.findall(r'\d+', id_str)
                if numbers:
                    numeric_ids.extend([int(num) for num in numbers])
        
        # Return the maximum numeric ID found, or 0 if none
        max_id = max(numeric_ids) if numeric_ids else 0
        logger.info(f"Maximum user ID found: {max_id}")
        return max_id
    
    except Exception as e:
        logger.error(f"Error getting max user ID: {e}")
        return 0

def get_max_auth_log_id() -> int:
    """
    Get the maximum authentication log ID from the database
    Returns 0 if no logs exist or if there's an error.
    """
    try:
        # Get database path from environment variable or use default
        db_path = os.getenv("DATABASE_URL", "sqlite:////app/database/client1.db")
        if db_path.startswith("sqlite:///"):
            db_path = db_path[10:]
        
        # Check if the database file exists
        if not os.path.exists(db_path):
            logger.warning(f"Database file not found: {db_path}")
            return 0
            
        # Connect to the database and query for all auth log IDs
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all authentication log IDs from the database
        cursor.execute("SELECT id FROM authentication_logs")
        log_ids = [row[0] for row in cursor.fetchall()]
        
        if not log_ids:
            return 0
        
        # Extract numeric IDs (if any)
        numeric_ids = []
        for id_str in log_ids:
            # If it's a pure numeric ID
            if re.match(r'^\d+$', id_str):
                numeric_ids.append(int(id_str))
            # If it contains numeric parts (e.g., "log-123")
            elif re.search(r'\d+', id_str):
                # Extract all numbers from the string
                numbers = re.findall(r'\d+', id_str)
                if numbers:
                    numeric_ids.extend([int(num) for num in numbers])
        
        # Return the maximum numeric ID found, or 0 if none
        max_id = max(numeric_ids) if numeric_ids else 0
        logger.info(f"Maximum auth log ID found: {max_id}")
        return max_id
    
    except Exception as e:
        logger.error(f"Error getting max auth log ID: {e}")
        return 0

def generate_uuid() -> str:
    """
    Generate a unique identifier
    If using numeric IDs, returns the next available ID (max + 1)
    If using UUIDs, generates a new UUID
    """
    # Check if we should use numeric IDs (default) or UUIDs
    use_uuids = os.getenv("USE_UUID_IDS", "false").lower() == "true"
    
    if use_uuids:
        return str(uuid.uuid4())
    else:
        # Get the maximum existing user ID and add 1
        max_id = get_max_user_id()
        new_id = max_id + 1
        logger.info(f"Generated new sequential user ID: {new_id}")
        return str(new_id)

def generate_auth_log_id() -> str:
    """
    Generate a unique identifier for authentication logs
    Uses a different sequence than user IDs to avoid conflicts
    """
    # Check if we should use UUIDs
    use_uuids = os.getenv("USE_UUID_IDS", "false").lower() == "true"
    
    if use_uuids:
        return str(uuid.uuid4())
    else:
        # Get the maximum existing auth log ID and add 1
        max_id = get_max_auth_log_id()
        # Add a prefix to distinguish from user IDs
        new_id = max_id + 1
        logger.info(f"Generated new sequential auth log ID: {new_id}")
        # Add "A" prefix to distinguish from user IDs
        return f"A{new_id}" 