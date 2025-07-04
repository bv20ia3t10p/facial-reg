"""
Database utility functions.
"""

import os
import logging
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Get database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///database/facial_reg.db")

def get_client_databases():
    """Get all available client database files"""
    client_dbs = []
    
    # Check database directory
    db_path = Path("/app/database")
    if db_path.exists():
        # Look for client-specific database files
        client_db_files = list(db_path.glob("client*.db"))
        if client_db_files:
            logger.info(f"Found {len(client_db_files)} client database files")
            for db_file in client_db_files:
                client_id = db_file.stem  # Get filename without extension
                client_dbs.append({
                    "client_id": client_id,
                    "db_path": str(db_file),
                    "db_url": f"sqlite:///{db_file}"
                })
    
    # If no client databases, use the default
    if not client_dbs:
        client_dbs.append({
            "client_id": "default",
            "db_path": DATABASE_URL.replace("sqlite:///", ""),
            "db_url": DATABASE_URL
        })
        logger.info(f"Using default database: {DATABASE_URL}")
    
    return client_dbs 