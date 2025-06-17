"""
Script to create client-specific databases with fake data and correct schema
"""

import os
import random
from datetime import datetime, timedelta
import logging
import sqlite3
from pathlib import Path
import json
import uuid
import sys
from faker import Faker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Faker
fake = Faker()

def generate_uuid():
    """Generate UUID as string"""
    return str(uuid.uuid4())

def get_client_user_ids(client_id: str) -> list:
    """Get list of user IDs from client's partitioned folder structure"""
    data_path = Path("data/partitioned") / client_id
    if not data_path.exists():
        logger.warning(f"No data directory found for {client_id}")
        return []
    
    return sorted([d.name for d in data_path.iterdir() if d.is_dir() and d.name.isdigit()])

def get_available_clients() -> list:
    """Get list of available clients from partitioned folder structure"""
    data_path = Path("data/partitioned")
    if not data_path.exists():
        logger.error("Partitioned data directory not found")
        return []
    
    return sorted([d.name for d in data_path.iterdir() if d.is_dir() and d.name.startswith("client")])

def create_database_schema(cursor):
    """Create fresh database schema"""
    logger.info("Creating database schema...")
    
    # Create users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            department TEXT NOT NULL,
            role TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            face_encoding BLOB,
            created_at TIMESTAMP NOT NULL,
            last_authenticated TIMESTAMP
        )
    """)
    
    # Create authentication_logs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS authentication_logs (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            success BOOLEAN NOT NULL,
            confidence FLOAT NOT NULL,
            emotion_data JSON,
            created_at TIMESTAMP NOT NULL,
            device_info TEXT,
            captured_image BLOB,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)
    
    # Create indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_auth_logs_user_id ON authentication_logs(user_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")

def generate_fake_user(user_id: str):
    """Generate fake user data"""
    departments = ["Engineering", "HR", "Finance", "Marketing", "Sales", "IT", "Operations"]
    roles = ["Employee", "Manager", "Senior Manager", "Director", "VP", "Intern"]
    
    return {
        "id": user_id,
        "name": fake.name(),
        "email": f"user_{user_id}@company.com",
        "department": random.choice(departments),
        "role": random.choice(roles),
        "password_hash": fake.sha256(),  # Simulated password hash
        "created_at": fake.date_time_between(start_date="-1y", end_date="now").isoformat(),
        "last_authenticated": fake.date_time_between(start_date="-1m", end_date="now").isoformat()
    }

def generate_fake_emotion():
    """Generate fake emotion data"""
    emotions = ["neutral", "happy", "sad", "angry", "surprised", "fearful", "disgusted"]
    primary_emotion = random.choice(emotions)
    
    # Generate probabilities for all emotions
    probabilities = {}
    remaining_prob = 1.0
    for emotion in emotions:
        if emotion == primary_emotion:
            prob = random.uniform(0.6, 0.9)
            remaining_prob -= prob
            probabilities[emotion] = prob
        else:
            if emotion == emotions[-1]:  # Last emotion gets remaining probability
                probabilities[emotion] = remaining_prob
            else:
                prob = random.uniform(0, remaining_prob)
                remaining_prob -= prob
                probabilities[emotion] = prob
    
    return {
        "emotion": primary_emotion,
        "confidence": probabilities[primary_emotion],
        "probabilities": probabilities,
        "timestamp": datetime.utcnow().isoformat()
    }

def generate_auth_logs(user_id: str, count: int = 10):
    """Generate fake authentication logs for a user"""
    logs = []
    for _ in range(count):
        success = random.random() > 0.1  # 90% success rate
        confidence = random.uniform(0.6, 0.99) if success else random.uniform(0.3, 0.6)
        
        log = {
            "id": generate_uuid(),
            "user_id": user_id,
            "success": success,
            "confidence": confidence,
            "emotion_data": json.dumps(generate_fake_emotion()) if success else None,
            "created_at": fake.date_time_between(start_date="-6m", end_date="now").isoformat(),
            "device_info": f"Camera {random.randint(1, 5)}" if success else None
        }
        logs.append(log)
    return logs

def create_client_database(client_id: str):
    """Create a database for a specific client"""
    try:
        # Get user IDs from folder structure
        user_ids = get_client_user_ids(client_id)
        if not user_ids:
            logger.error(f"No users found for {client_id}")
            return False
            
        # Create database directory if it doesn't exist
        db_dir = Path("database")
        db_dir.mkdir(exist_ok=True)
        
        # Set database path
        db_path = db_dir / f"{client_id}.db"
        
        logger.info(f"Creating database for {client_id} with {len(user_ids)} users...")
        
        # Remove existing database if it exists
        if db_path.exists():
            db_path.unlink()
            logger.info(f"Removed existing database for {client_id}")
        
        # Create new database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        try:
            # Start transaction
            cursor.execute("BEGIN TRANSACTION")
            
            # Create schema
            create_database_schema(cursor)
            
            # Create users and their auth logs
            for user_id in user_ids:
                # Generate and insert user
                user_data = generate_fake_user(user_id)
                cursor.execute("""
                    INSERT INTO users (id, name, email, department, role, password_hash, created_at, last_authenticated)
                    VALUES (:id, :name, :email, :department, :role, :password_hash, :created_at, :last_authenticated)
                """, user_data)
                
                # Generate and insert auth logs
                auth_logs = generate_auth_logs(user_id)
                for log in auth_logs:
                    cursor.execute("""
                        INSERT INTO authentication_logs 
                        (id, user_id, success, confidence, emotion_data, created_at, device_info)
                        VALUES (:id, :user_id, :success, :confidence, :emotion_data, :created_at, :device_info)
                    """, log)
                
                logger.info(f"Created user {user_id} with {len(auth_logs)} auth logs")
            
            # Commit transaction
            cursor.execute("COMMIT")
            logger.info(f"Database creation completed successfully for {client_id}")
            return True
            
        except Exception as e:
            cursor.execute("ROLLBACK")
            logger.error(f"Database creation failed for {client_id}, rolling back: {e}")
            return False
            
        finally:
            cursor.close()
            conn.close()
            
    except Exception as e:
        logger.error(f"Database creation failed for {client_id}: {e}")
        return False

def create_all_databases():
    """Create databases for all clients"""
    try:
        # Get available clients from folder structure
        clients = get_available_clients()
        if not clients:
            logger.error("No client directories found")
            return False
        
        # Create database directory if it doesn't exist
        db_dir = Path("database")
        db_dir.mkdir(exist_ok=True)
        
        # Save client configuration with user counts
        client_config = {}
        for client_id in clients:
            user_count = len(get_client_user_ids(client_id))
            client_config[client_id] = user_count
        
        config_path = db_dir / "clients.json"
        with open(config_path, 'w') as f:
            json.dump(client_config, f, indent=2)
        logger.info(f"Saved client configuration to {config_path}")
        
        # Create databases for each client
        success = True
        for client_id in clients:
            if not create_client_database(client_id):
                success = False
                logger.error(f"Failed to create database for {client_id}")
        
        return success
        
    except Exception as e:
        logger.error(f"Failed to create client databases: {e}")
        return False

if __name__ == "__main__":
    if create_all_databases():
        logger.info("✓ All client databases created successfully")
        sys.exit(0)
    else:
        logger.error("✗ Failed to create some client databases")
        sys.exit(1) 