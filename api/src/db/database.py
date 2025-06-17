"""
Database initialization and models for the biometric authentication system
"""

from sqlalchemy import create_engine, Column, String, Float, DateTime, Boolean, ForeignKey, JSON, LargeBinary
from sqlalchemy.orm import foreign
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy import text
import uuid
from datetime import datetime
import logging
from typing import Optional
import os
from fastapi import HTTPException, status, Depends
from sqlalchemy.orm import Session
from pathlib import Path

logger = logging.getLogger(__name__)

# Initialize SQLAlchemy
Base = declarative_base()

# Use SQLite for development, PostgreSQL for production
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/biometric.db")  # Updated to match docker-compose

logger.info(f"Using database at: {DATABASE_URL}")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    echo=True  # Enable SQL query logging
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def generate_uuid():
    """Generate UUID as string"""
    return str(uuid.uuid4())

class User(Base):
    """User model for storing biometric authentication data"""
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    department = Column(String, nullable=False)
    role = Column(String, nullable=False)
    password_hash = Column(String, nullable=False)
    face_encoding = Column(LargeBinary)  # Encrypted face recognition data
    created_at = Column(DateTime, default=datetime.utcnow)
    last_authenticated = Column(DateTime)
    
    # Relationships
    auth_logs = relationship(
        "AuthenticationLog",
        primaryjoin="User.id == foreign(AuthenticationLog.user_id)",
        backref="user",
        uselist=True
    )
    verification_requests = relationship("VerificationRequest", back_populates="user")

class AuthenticationLog(Base):
    """Model for storing authentication attempts"""
    __tablename__ = "authentication_logs"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, index=True)  # No foreign key constraint since users are in folders
    success = Column(Boolean, nullable=False)
    confidence = Column(Float, nullable=False)
    emotion_data = Column(JSON)  # Store full emotion analysis results
    created_at = Column(DateTime, default=datetime.utcnow)
    device_info = Column(String)  # Store device information
    captured_image = Column(LargeBinary, nullable=True)  # Optional captured image storage

class VerificationRequest(Base):
    """Model for verification requests"""
    __tablename__ = "verification_requests"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"))
    reason = Column(String, nullable=False)
    additional_notes = Column(String)
    captured_image = Column(LargeBinary)  # Encrypted captured image
    confidence = Column(Float)
    status = Column(String, default="pending")  # pending, approved, rejected
    created_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime)
    
    # Relationships
    user = relationship("User", back_populates="verification_requests")

class Client(Base):
    """Model for federated learning clients"""
    __tablename__ = "clients"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, nullable=False)
    client_type = Column(String, nullable=False)  # 'client1', 'client2', 'server'
    created_at = Column(DateTime, default=datetime.utcnow)
    last_update = Column(DateTime)
    model_version = Column(String)
    privacy_budget = Column(Float, default=1.0)
    metrics = Column(JSON)  # Stores training metrics

def test_db_connection():
    """Test database connection and log results"""
    try:
        with engine.connect() as conn:
            # Test basic connection
            result = conn.execute(text("SELECT 1"))
            logger.info("Database connection test successful")
            
            # Check if authentication_logs table exists
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='authentication_logs'"))
            tables = result.fetchall()
            if not tables:
                logger.error("Authentication logs table not found in database!")
            else:
                logger.info("Authentication logs table exists")
                
                # Count logs
                result = conn.execute(text("SELECT COUNT(*) FROM authentication_logs"))
                count = result.fetchone()[0]
                logger.info(f"Found {count} authentication logs in database")
                
            return True
    except Exception as e:
        logger.error(f"Database connection test failed: {str(e)}")
        return False

async def init_db():
    """Initialize database and create tables"""
    try:
        logger.info("Initializing database...")
        
        # Test connection first
        if not test_db_connection():
            logger.error("Database connection test failed during initialization")
            return
        
        # Create tables
        Base.metadata.create_all(bind=engine)
        logger.info("✓ Database tables created")
        
        # Run migrations
        from api.src.db.migrate_db import migrate_database
        if migrate_database():
            logger.info("✓ Database migrations completed")
        else:
            logger.error("Database migrations failed")
            
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

def get_db():
    """Get database session with connection testing"""
    db = SessionLocal()
    try:
        # Test connection with a simple query
        db.execute(text("SELECT 1"))
        logger.debug("Database session created successfully")
        return db
    except Exception as e:
        logger.error(f"Failed to create database session: {e}")
        db.close()
        raise

# Create database functions
def create_user(db, user_data: dict) -> Optional[User]:
    """Create a new user"""
    try:
        user = User(**user_data)
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to create user: {e}")
        return None

def log_authentication(db, log_data: dict) -> Optional[AuthenticationLog]:
    """Log an authentication attempt"""
    try:
        log = AuthenticationLog(**log_data)
        db.add(log)
        db.commit()
        db.refresh(log)
        return log
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to log authentication: {e}")
        return None

def create_verification_request(db, request_data: dict) -> Optional[VerificationRequest]:
    """Create a new verification request"""
    try:
        request = VerificationRequest(**request_data)
        db.add(request)
        db.commit()
        db.refresh(request)
        return request
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to create verification request: {e}")
        return None 