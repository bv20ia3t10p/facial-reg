"""
Database migration script to add new columns to authentication_logs table
"""

import logging
import sqlite3
import os
from pathlib import Path
import sys
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def migrate_database():
    """Add threshold and authenticated_at columns to authentication_logs table"""
    try:
        # Get database path from environment or use default
        db_path = os.getenv("DATABASE_URL", "sqlite:////app/database/client1.db")
        
        # Convert SQLAlchemy URL to actual file path
        if db_path.startswith("sqlite:///"):
            db_path = db_path.replace("sqlite:///", "")
            
        logger.info(f"Migrating database at: {db_path}")
        
        # Check if database file exists
        if not Path(db_path).exists():
            logger.error(f"Database file not found: {db_path}")
            return False
        
        # Run the database operations in a thread to avoid blocking
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(None, _execute_migration, db_path)
        
        return success
    except Exception as e:
        logger.error(f"Database migration failed: {e}")
        return False

def _execute_migration(db_path):
    """Execute the actual database migration in a separate thread"""
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if the columns already exist
        cursor.execute("PRAGMA table_info(authentication_logs)")
        columns = [column[1] for column in cursor.fetchall()]
        
        # Add threshold column if it doesn't exist
        if "threshold" not in columns:
            logger.info("Adding threshold column to authentication_logs table")
            cursor.execute("ALTER TABLE authentication_logs ADD COLUMN threshold REAL")
        else:
            logger.info("threshold column already exists")
        
        # Add authenticated_at column if it doesn't exist
        if "authenticated_at" not in columns:
            logger.info("Adding authenticated_at column to authentication_logs table")
            cursor.execute("ALTER TABLE authentication_logs ADD COLUMN authenticated_at TIMESTAMP")
        else:
            logger.info("authenticated_at column already exists")
        
        # Commit changes
        conn.commit()
        conn.close()
        
        logger.info("Database migration completed successfully")
        return True
    except Exception as e:
        logger.error(f"Database migration execution failed: {e}")
        return False

if __name__ == "__main__":
    # Run the migration synchronously when executed directly
    loop = asyncio.get_event_loop()
    success = loop.run_until_complete(migrate_database())
    sys.exit(0 if success else 1) 