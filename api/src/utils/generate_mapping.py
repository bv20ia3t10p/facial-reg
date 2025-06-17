"""
Script to generate class mapping from database
"""

import os
import json
from pathlib import Path
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from api.src.db.database import User

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/biometric.db")
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def generate_class_mapping():
    """Generate class mapping from database"""
    try:
        # Get session
        db = SessionLocal()
        
        try:
            # Get all users ordered by ID
            users = db.query(User).order_by(User.id).all()
            logger.info(f"Found {len(users)} users")
            
            # Create mapping
            mapping = {}
            for idx, user in enumerate(users):
                mapping[str(idx)] = user.id
                logger.info(f"Mapped class {idx} to user {user.id}")
            
            # Save mapping
            mapping_path = Path("class_mapping.json")
            with open(mapping_path, 'w') as f:
                json.dump(mapping, f, indent=2)
            logger.info(f"Saved mapping to {mapping_path}")
            
            return mapping
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error generating class mapping: {e}")
        raise

if __name__ == "__main__":
    generate_class_mapping() 