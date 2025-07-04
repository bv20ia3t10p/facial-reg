"""
Database initialization and models for the biometric authentication system
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, JSON, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy import text
import uuid
from datetime import datetime
import logging
from typing import Optional, Any, Dict, Generator
import os
import json
from ..utils.datetime_utils import get_current_time

from ..utils.common import generate_uuid, generate_auth_log_id

logger = logging.getLogger(__name__)

def json_safe_dumps(obj: Any) -> str:
    """Convert object to JSON string, handling datetime objects"""
    def datetime_handler(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    return json.dumps(obj, default=datetime_handler, indent=2)

def sanitize_log_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize data for logging by removing sensitive fields and converting datetime objects"""
    if not isinstance(data, dict):
        return data
        
    sanitized = {}
    sensitive_fields = {'password_hash', 'captured_image', 'face_encoding'}
    
    for k, v in data.items():
        if k in sensitive_fields:
            continue
        if isinstance(v, datetime):
            sanitized[k] = v.isoformat()
        elif isinstance(v, dict):
            sanitized[k] = sanitize_log_data(v)
        else:
            sanitized[k] = v
    
    return sanitized

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:////app/database/client1.db")
logger.info(f"Using database at: {DATABASE_URL}")

# Configure SQLite database
if DATABASE_URL.startswith("sqlite"):
    db_path = DATABASE_URL.replace('sqlite:///', '')
    db_dir = os.path.dirname(db_path)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)

    # Create a temporary engine to set up PRAGMA statements
    temp_engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False}
    )
    
    with temp_engine.connect() as conn:
        # Set SQLite PRAGMA statements
        pragmas = [
            "PRAGMA journal_mode = WAL",
            "PRAGMA synchronous = NORMAL",
            "PRAGMA cache_size = -2000",
            "PRAGMA busy_timeout = 30000",
            "PRAGMA temp_store = MEMORY",
            "PRAGMA page_size = 4096",
            "PRAGMA mmap_size = 268435456",
            "PRAGMA locking_mode = EXCLUSIVE",
            "PRAGMA cache_spill = FALSE"
        ]
        for pragma in pragmas:
            conn.execute(text(pragma))
            
    temp_engine.dispose()

# Create the main engine
engine = create_engine(
    DATABASE_URL,
    connect_args={
        "check_same_thread": False,
        "timeout": 30  # Keep a reasonable timeout
    } if DATABASE_URL.startswith("sqlite") else {},
    echo=False,      # Set to False to reduce log noise
    pool_pre_ping=True,
    pool_recycle=1800  # 30 minutes
)

# Create session factory
SessionLocal = sessionmaker(
    bind=engine,
    class_=Session,  # Use synchronous Session
    autocommit=False,
    autoflush=False,
    expire_on_commit=False
)

Base = declarative_base()

class User(Base):
    """User model for storing biometric authentication data"""
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False, index=True)
    department = Column(String, nullable=False)
    role = Column(String, nullable=False)
    password_hash = Column(String, nullable=False)
    face_encoding = Column(LargeBinary)
    created_at = Column(DateTime(timezone=True), nullable=False, default=get_current_time)
    last_authenticated = Column(DateTime(timezone=True))
    
    # Relationships
    auth_logs = relationship(
        "AuthenticationLog",
        primaryjoin="User.id == foreign(AuthenticationLog.user_id)",
        backref="user",
        uselist=True
    )
    verification_requests = relationship("VerificationRequest", back_populates="user")

class AuthenticationLog(Base):
    """Model for authentication logs"""
    __tablename__ = "authentication_logs"
    
    id = Column(String, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    confidence = Column(Float, nullable=False)
    success = Column(Boolean, nullable=False)
    threshold = Column(Float, nullable=True)  # Add threshold field
    emotion_data = Column(JSON)
    created_at = Column(DateTime(timezone=True), default=get_current_time)
    device_info = Column(String)
    captured_image = Column(String)  # Path to stored image
    authenticated_at = Column(DateTime, nullable=True)  # Add authenticated_at field

class VerificationRequest(Base):
    """Model for verification requests"""
    __tablename__ = "verification_requests"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    status = Column(String)  # pending, approved, rejected
    created_at = Column(DateTime(timezone=True), default=get_current_time)
    processed_at = Column(DateTime(timezone=True), nullable=True)
    request_data = Column(Text)  # JSON string of request data
    
    # Relationships
    user = relationship("User", back_populates="verification_requests")

class Client(Base):
    """Model for federated learning clients"""
    __tablename__ = "clients"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, nullable=False)
    client_type = Column(String, nullable=False)  # 'client1', 'client2', 'server'
    created_at = Column(DateTime(timezone=True), default=get_current_time)
    last_update = Column(DateTime)
    model_version = Column(String)
    privacy_budget = Column(Float, default=1.0)
    metrics = Column(JSON)  # Stores training metrics

def get_db() -> Generator[Session, None, None]:
    """Get database session with connection testing and proper cleanup"""
    session_id = str(uuid.uuid4())[:8]  # Generate short session ID for logging
    db = None
    try:
        logger.info(f"=== Creating new database session {session_id} ===")
        db = SessionLocal()
        
        # Test connection with a simple query
        try:
            result = db.execute(text("SELECT 1"))
            if not result.fetchone():
                logger.error(f"[Session {session_id}] Database connection test failed")
                raise Exception("Database connection test failed")
                logger.info(f"[Session {session_id}] Database connection test successful")
        except Exception as conn_err:
            # Log the specific connection error
            logger.error(f"[Session {session_id}] Database connection error: {conn_err}", exc_info=True)
            
            # Try to reconnect to the database with a fresh engine
            if DATABASE_URL.startswith("sqlite"):
                # For SQLite, check if database file exists and is accessible
                db_path = DATABASE_URL.replace('sqlite:///', '')
                if not os.path.exists(os.path.dirname(db_path)):
                    os.makedirs(os.path.dirname(db_path), exist_ok=True)
                logger.info(f"[Session {session_id}] Ensuring SQLite database path exists: {db_path}")
            
            # Close the session and try to create a new one
            if db:
                try:
                    db.close()
                except Exception:
                    pass
            db = SessionLocal()
            # Test again
            result = db.execute(text("SELECT 1"))
            if not result.fetchone():
                raise Exception("Database reconnection test failed")
            logger.info(f"[Session {session_id}] Database reconnection successful")
        
        yield db
        
    except Exception as e:
        logger.error(f"[Session {session_id}] Failed to create database session: {str(e)}", exc_info=True)
        if db:
            try:
                logger.debug(f"[Session {session_id}] Rolling back failed session")
                db.rollback()
            except Exception as close_error:
                logger.error(f"[Session {session_id}] Error rolling back failed session: {close_error}", exc_info=True)
        raise
        
    finally:
        if db:
            try:
                logger.debug(f"[Session {session_id}] Closing database session")
                db.close()
                logger.info(f"=== Closed database session {session_id} ===")
            except Exception as e:
                logger.error(f"[Session {session_id}] Error closing database session: {e}", exc_info=True)

def init_db():
    """Initialize database and create tables"""
    try:
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

def create_user(db, user_data: dict) -> Optional[User]:
    """Create a new user"""
    try:
        sanitized_data = sanitize_log_data(user_data)
        logger.info(f"Creating new user with data: {json_safe_dumps(sanitized_data)}")
        user = User(**user_data)
        db.add(user)
        db.commit()
        db.refresh(user)
        logger.info(f"Successfully created user: {user.id}")
        return user
    except Exception as e:
        logger.error(f"Failed to create user: {e}")
        db.rollback()
        return None

def log_authentication(db, log_data: dict) -> Optional[AuthenticationLog]:
    """Log an authentication attempt"""
    try:
        # Generate unique ID for the log entry if not provided
        if 'id' not in log_data:
            log_data['id'] = generate_auth_log_id()
            
        # Validate required fields
        required_fields = ['user_id', 'success', 'confidence', 'created_at']
        missing_fields = [field for field in required_fields if field not in log_data]
        if missing_fields:
            logger.error(f"Missing required fields for authentication log: {missing_fields}")
            return None

        # Sanitize data for logging
        sanitized_data = sanitize_log_data(log_data)
        logger.info(f"Creating authentication log: {json_safe_dumps(sanitized_data)}")

        # Create and save the log entry
        log_entry = AuthenticationLog(**log_data)
        db.add(log_entry)
        db.commit()
        db.refresh(log_entry)
        
        logger.info(f"Successfully created authentication log: {log_entry.id}")
        return log_entry
        
    except Exception as e:
        logger.error(f"Failed to create authentication log: {e}")
        db.rollback()
        return None

def create_verification_request(db, request_data: dict) -> Optional[VerificationRequest]:
    """Create a new verification request"""
    try:
        # Create a sanitized copy of the data for logging (without the image)
        log_data = request_data.copy()
        if 'captured_image' in log_data:
            log_data['captured_image'] = '[BASE64_IMAGE_DATA]'
        
        logger.info(f"Creating verification request: {json_safe_dumps(log_data)}")
        
        request = VerificationRequest(**request_data)
        db.add(request)
        db.commit()
        db.refresh(request)
        
        logger.info(f"Successfully created verification request: {request.id}")
        return request
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to create verification request: {e}")
        return None

def create_client(db, client_data: dict) -> Optional[Client]:
    """Create a new client"""
    try:
        sanitized_data = sanitize_log_data(client_data)
        logger.info(f"Creating new client with data: {json_safe_dumps(sanitized_data)}")
        client = Client(**client_data)
        db.add(client)
        db.commit()
        db.refresh(client)
        logger.info(f"Successfully created client: {client.id}")
        return client
    except Exception as e:
        logger.error(f"Failed to create client: {e}")
        db.rollback()
        return None 