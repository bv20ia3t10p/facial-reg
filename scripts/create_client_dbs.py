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
import math
from faker import Faker
from zoneinfo import ZoneInfo
import argparse
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Faker
fake = Faker()

# Get system timezone
system_timezone = ZoneInfo(os.environ.get('TZ', 'UTC'))
logger.info(f"Using system timezone: {system_timezone}")

# Default values
DEFAULT_DAYS = 30  # One month instead of one year
DEFAULT_AUTHS_PER_DAY = 500  # 500 auth logs per day instead of 10000

def generate_uuid():
    """Generate UUID as string for users"""
    # For this script, use sequential IDs for users
    return str(uuid.uuid4())
    
def generate_auth_log_id():
    """Generate UUID as string for authentication logs"""
    # For authentication logs, use a different prefix
    return f"A{str(uuid.uuid4())}"

def get_timestamp(offset_days=0, offset_minutes=0, offset_hours=0, offset_seconds=0):
    """Generate timestamp with system timezone"""
    current_time = datetime.now(system_timezone)
    if any([offset_days, offset_minutes, offset_hours, offset_seconds]):
        current_time += timedelta(days=offset_days, minutes=offset_minutes, 
                                 hours=offset_hours, seconds=offset_seconds)
    return current_time

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
    
    # Create authentication_logs table with OTP field
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
            otp TEXT,
            client_id TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)
    
    # Create indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_auth_logs_user_id ON authentication_logs(user_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_auth_logs_created_at ON authentication_logs(created_at)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")

def generate_fake_user(user_id: str):
    """Generate fake user data"""
    departments = ["Engineering", "HR", "Finance", "Marketing", "Sales", "IT", "Operations"]
    roles = ["Employee", "Manager", "Senior Manager", "Director", "VP", "Intern"]
    
    # Generate timestamps relative to current time
    created_at = get_timestamp(offset_days=-random.randint(1, 365))  # Random day in the last year
    last_authenticated = get_timestamp(offset_days=-random.randint(0, 30))  # Random time in the last month
    
    return {
        "id": user_id,
        "name": fake.name(),
        "email": f"user_{user_id}@company.com",
        "department": random.choice(departments),
        "role": random.choice(roles),
        "password_hash": fake.sha256(),  # Simulated password hash
        "created_at": created_at.isoformat(),
        "last_authenticated": last_authenticated.isoformat()
    }

def generate_fake_emotion():
    """Generate fake emotion data with RAF-DB categories and randomized values"""
    # RAF-DB emotion categories (excluding contempt as requested)
    emotion_keys = ["neutral", "happiness", "sadness", "surprise", "fear", "disgust", "anger"]
    
    # Create emotion data with random float values
    emotion_data = {}
    
    # Decide whether to have one dominant emotion or more balanced distribution
    if random.random() < 0.7:  # 70% chance of dominant emotion
        # Select a dominant emotion
        dominant_emotion = random.choice(emotion_keys)
        # Assign higher value to dominant emotion
        dominant_value = random.uniform(0.65, 0.95)
        
        # Distribute remaining probability among other emotions
        remaining = 1.0 - dominant_value
        other_emotions = [e for e in emotion_keys if e != dominant_emotion]
        random.shuffle(other_emotions)
        
        # Assign the dominant emotion
        emotion_data[dominant_emotion] = dominant_value
        
        # Distribute remaining probability
        for i, emotion in enumerate(other_emotions):
            if i == len(other_emotions) - 1:  # Last emotion gets remaining probability
                emotion_data[emotion] = remaining
            else:
                value = random.uniform(0, remaining * 0.5)
                emotion_data[emotion] = value
                remaining -= value
    else:
        # Create a more balanced distribution
        total = 0.0
        for emotion in emotion_keys[:-1]:  # All but the last emotion
            value = random.random()
            emotion_data[emotion] = value
            total += value
            
        # Last emotion gets a value that makes sum close to 1
        emotion_data[emotion_keys[-1]] = random.uniform(0.1, 0.3)
        
        # Normalize values to sum to 1
        sum_values = sum(emotion_data.values())
        for emotion in emotion_data:
            emotion_data[emotion] = emotion_data[emotion] / sum_values
    
    # 10% chance to add metadata
    if random.random() < 0.1:
        emotion_data["timestamp"] = get_timestamp().isoformat()
        
    # 5% chance to add a dominant_emotion field
    if random.random() < 0.05:
        max_emotion = max(emotion_data.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0)
        emotion_data["dominant_emotion"] = max_emotion[0]
    
    return emotion_data

def generate_otp():
    """Generate a 6-digit OTP"""
    return f"{random.randint(100000, 999999)}"

def generate_auth_logs_for_year(user_ids, client_id, days=DEFAULT_DAYS, auths_per_day=DEFAULT_AUTHS_PER_DAY):
    """Generate authentication logs for a year with specified number of authentications per day"""
    logs = []
    
    # Calculate total number of logs
    total_logs = days * auths_per_day
    
    # Calculate logs per user (distribute evenly)
    logs_per_user = math.ceil(total_logs / len(user_ids))
    
    logger.info(f"Generating {total_logs} logs across {len(user_ids)} users ({logs_per_user} per user)")
    
    # Generate logs for each day
    for day in range(days):
        # Distribute authentications throughout the day (24 hours)
        for _ in range(auths_per_day):
            # Pick a random user
            user_id = random.choice(user_ids)
            
            # Random time within the day
            hour = random.randint(0, 23)
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            
            # Success rate (90% success)
            success = random.random() > 0.1
            confidence = random.uniform(0.6, 0.99) if success else random.uniform(0.3, 0.6)
            
            # Generate timestamp for this specific day and time
            created_at = get_timestamp(
                offset_days=-(days - day),  # Count backwards from today
                offset_hours=hour,
                offset_minutes=minute,
                offset_seconds=second
            )
            
            # Generate emotion data with random storage format
            emotion_data = None
            if success:
                emotion_obj = generate_fake_emotion()
                # Randomly choose between proper JSON and corrupted formats
                storage_type = random.randint(1, 10)
                if storage_type <= 8:  # 80% chance of proper JSON string
                    emotion_data = json.dumps(emotion_obj)
                else:  # 20% chance of corrupted but still valid JSON
                    # Occasionally store timestamp as emotion value to simulate data issues
                    if random.random() < 0.5:
                        emotion_data = json.dumps({"happiness": created_at.isoformat()})
                    else:
                        # Other potential format issues that are still valid JSON
                        emotion_data = json.dumps(str(emotion_obj))
            
            # Generate log with a different ID format than users
            log = {
                "id": generate_auth_log_id(),
                "user_id": user_id,
                "success": success,
                "confidence": confidence,
                "emotion_data": emotion_data,  # This is now always None or a JSON string
                "created_at": created_at.isoformat(),
                "device_info": f"Camera {random.randint(1, 5)}" if success else None,
                "otp": generate_otp() if success else None,
                "client_id": client_id
            }
            logs.append(log)
    
    return logs

def create_client_database(client_id: str, days=DEFAULT_DAYS, auths_per_day=DEFAULT_AUTHS_PER_DAY):
    """Create a database for a specific client with a year of authentication logs"""
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
        logger.info(f"Generating {days} days of data with ~{auths_per_day} authentications per day")
        
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
            
            # Create users
            for user_id in tqdm(user_ids, desc="Creating users"):
                # Generate and insert user
                user_data = generate_fake_user(user_id)
                cursor.execute("""
                    INSERT INTO users (id, name, email, department, role, password_hash, created_at, last_authenticated)
                    VALUES (:id, :name, :email, :department, :role, :password_hash, :created_at, :last_authenticated)
                """, user_data)
            
            # Generate all auth logs for the year
            logger.info("Generating authentication logs for the year...")
            auth_logs = generate_auth_logs_for_year(user_ids, client_id, days, auths_per_day)
            
            # Insert auth logs in batches to improve performance
            batch_size = 10000
            for i in tqdm(range(0, len(auth_logs), batch_size), desc="Inserting auth logs"):
                batch = auth_logs[i:i+batch_size]
                cursor.executemany("""
                    INSERT INTO authentication_logs 
                    (id, user_id, success, confidence, emotion_data, created_at, device_info, otp, client_id)
                    VALUES (:id, :user_id, :success, :confidence, :emotion_data, :created_at, :device_info, :otp, :client_id)
                """, batch)
                
                # Commit every batch to avoid transaction timeout
                conn.commit()
                cursor.execute("BEGIN TRANSACTION")
            
            # Commit final transaction
            cursor.execute("COMMIT")
            logger.info(f"Database creation completed successfully for {client_id}")
            logger.info(f"Created {len(auth_logs)} authentication logs")
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

def create_all_databases(days=DEFAULT_DAYS, auths_per_day=DEFAULT_AUTHS_PER_DAY):
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
            if not create_client_database(client_id, days, auths_per_day):
                success = False
                logger.error(f"Failed to create database for {client_id}")
        
        return success
        
    except Exception as e:
        logger.error(f"Failed to create client databases: {e}")
        return False

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Create client databases with authentication logs")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS,
                        help=f"Number of days to generate data for (default: {DEFAULT_DAYS} days)")
    parser.add_argument("--auths-per-day", type=int, default=DEFAULT_AUTHS_PER_DAY,
                        help=f"Number of authentications per day (default: {DEFAULT_AUTHS_PER_DAY})")
    args = parser.parse_args()
    
    print(f"\n📊 Generating {args.days} days of authentication data")
    print(f"📈 {args.auths_per_day} authentications per day")
    print(f"🎭 Using randomized emotion data formats\n")
    
    # Create databases with specified parameters
    if create_all_databases(args.days, args.auths_per_day):
        logger.info("✓ All client databases created successfully")
        sys.exit(0)
    else:
        logger.error("✗ Failed to create some client databases")
        sys.exit(1) 