import os
import sys
import uuid
import random
from pathlib import Path
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime, timedelta

# Database connection parameters
DB_PARAMS = {
    'dbname': 'facial_recognition',
    'user': 'postgres',  # Change this to your PostgreSQL username
    'password': 'postgres',  # Change this to your PostgreSQL password
    'host': 'localhost',
    'port': '5432'
}

# Mock data configuration
LOCATIONS = ['Main Entrance', 'Office Area', 'Cafeteria', 'Meeting Room', 'Reception']
DEVICES = ['CAM001', 'CAM002', 'CAM003', 'CAM004', 'CAM005']
DEPARTMENTS = ['Engineering', 'HR', 'Sales', 'Marketing', 'Operations']
POSITIONS = ['Manager', 'Senior', 'Junior', 'Lead', 'Associate']

def generate_mock_emotion_prediction():
    """Generate mock emotion predictions that sum to 1.0"""
    emotions = {
        'neutral': random.random(),
        'happiness': random.random(),
        'surprise': random.random(),
        'sadness': random.random(),
        'anger': random.random(),
        'disgust': random.random(),
        'fear': random.random(),
        'contempt': random.random()
    }
    # Normalize to sum to 1.0
    total = sum(emotions.values())
    return {k: v/total for k, v in emotions.items()}

def generate_timestamps(num_records, start_date=None):
    """Generate a list of timestamps for the past week"""
    if start_date is None:
        start_date = datetime.now() - timedelta(days=7)
    
    timestamps = []
    for _ in range(num_records):
        # Random timestamp within the past week
        random_days = random.uniform(0, 7)
        random_time = start_date + timedelta(days=random_days)
        timestamps.append(random_time)
    
    return sorted(timestamps)

def get_file_size(file_path):
    """Get file size in bytes."""
    try:
        return os.path.getsize(file_path)
    except OSError:
        return 0

def insert_client(cursor, client_type):
    """Insert a client and return its ID."""
    cursor.execute(
        """
        INSERT INTO clients (client_type, name)
        VALUES (%s, %s)
        RETURNING client_id
        """,
        (client_type, f"Facial Recognition {client_type.title()}")
    )
    return cursor.fetchone()[0]

def insert_user(cursor, client_id, employee_id):
    """Insert a user and return its ID."""
    cursor.execute(
        """
        INSERT INTO users (client_id, external_id, department, position)
        VALUES (%s, %s, %s, %s)
        RETURNING user_id
        """,
        (
            client_id,
            employee_id,
            random.choice(DEPARTMENTS),
            random.choice(POSITIONS)
        )
    )
    return cursor.fetchone()[0]

def insert_facial_images(cursor, images_data):
    """Bulk insert facial images and return their IDs."""
    execute_values(
        cursor,
        """
        INSERT INTO facial_images 
        (user_id, client_id, filename, file_path, file_size)
        VALUES %s
        RETURNING image_id
        """,
        images_data,
        fetch=True
    )
    return cursor.fetchall()

def insert_emotion_records(cursor, records_data):
    """Bulk insert emotion records."""
    execute_values(
        cursor,
        """
        INSERT INTO emotion_records 
        (user_id, client_id, timestamp, neutral, happiness, surprise, sadness,
         anger, disgust, fear, contempt, confidence, location, device_id, session_id)
        VALUES %s
        """,
        records_data
    )

def process_directory(base_path, client_type):
    """Process a client directory and insert data into the database."""
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cursor = conn.cursor()
        
        print(f"Processing {client_type}...")
        
        # Insert client
        client_id = insert_client(cursor, client_type)
        
        # Process each employee directory
        client_path = Path(base_path) / client_type
        if not client_path.exists():
            print(f"Warning: Path {client_path} does not exist")
            return

        for employee_dir in client_path.iterdir():
            if employee_dir.is_dir():
                employee_id = employee_dir.name
                print(f"Processing employee {employee_id}")
                
                # Insert user
                user_id = insert_user(cursor, client_id, employee_id)
                
                # Generate emotion records for the past week
                records_data = []
                timestamps = generate_timestamps(random.randint(50, 100))  # 50-100 records per employee
                
                for timestamp in timestamps:
                    emotions = generate_mock_emotion_prediction()
                    session_id = uuid.uuid4()
                    
                    records_data.append((
                        user_id,
                        client_id,
                        timestamp,
                        emotions['neutral'],
                        emotions['happiness'],
                        emotions['surprise'],
                        emotions['sadness'],
                        emotions['anger'],
                        emotions['disgust'],
                        emotions['fear'],
                        emotions['contempt'],
                        random.uniform(0.7, 1.0),  # confidence
                        random.choice(LOCATIONS),
                        random.choice(DEVICES),
                        session_id
                    ))
                
                # Bulk insert emotion records
                if records_data:
                    insert_emotion_records(cursor, records_data)
        
        # Refresh materialized views
        cursor.execute("SELECT refresh_emotion_stats()")
        
        conn.commit()
        print(f"Successfully processed {client_type}")
        
    except Exception as e:
        print(f"Error processing {client_type}: {str(e)}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python insert_data.py <base_path>")
        sys.exit(1)
        
    base_path = sys.argv[1]
    
    # Process each client directory
    for client_type in ['client1', 'client2', 'server']:
        process_directory(base_path, client_type)

if __name__ == "__main__":
    main() 