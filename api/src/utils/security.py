"""
Security utilities for password hashing and verification
"""

from passlib.context import CryptContext
import secrets
import string
import logging
from typing import Tuple, Optional, Union
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from datetime import datetime, timedelta
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db.database import User, get_db
from .common import generate_uuid

logger = logging.getLogger(__name__)

# Initialize password context
pwd_context = CryptContext(
    schemes=["bcrypt"],
    default="bcrypt",
    bcrypt__rounds=12  # Adjust for security vs performance
)

# JWT settings
SECRET_KEY = "8Zj8XbfpyaD4m7Af9VQJKTbGQxUW3Zm9CHZRYNLuEXxNLFmHm6G2PeZhVhyQ7kGx"  # Replace with environment variable in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 hours

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class TokenData(BaseModel):
    user_id: Optional[str] = None

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """Get current user from JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        token_data = TokenData(user_id=user_id)
    except JWTError:
        raise credentials_exception
    
    # Get user from database
    user = db.query(User).filter(User.id == token_data.user_id).first()
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user"""
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_password_hash(password: str) -> str:
    """Hash a password using bcrypt"""
    try:
        return pwd_context.hash(password)
    except Exception as e:
        logger.error(f"Password hashing failed: {e}")
        raise

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        logger.error(f"Password verification failed: {e}")
        return False

def generate_secure_token(length: int = 32) -> str:
    """Generate a cryptographically secure random token"""
    try:
        alphabet = string.ascii_letters + string.digits
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    except Exception as e:
        logger.error(f"Token generation failed: {e}")
        raise

def generate_api_key() -> Tuple[str, str]:
    """Generate API key and secret"""
    try:
        api_key = generate_secure_token(32)
        api_secret = generate_secure_token(64)
        return api_key, api_secret
    except Exception as e:
        logger.error(f"API key generation failed: {e}")
        raise

def validate_password_strength(password: str) -> Tuple[bool, str]:
    """Validate password strength"""
    try:
        if len(password) < 12:
            return False, "Password must be at least 12 characters long"
        
        if not any(c.isupper() for c in password):
            return False, "Password must contain at least one uppercase letter"
            
        if not any(c.islower() for c in password):
            return False, "Password must contain at least one lowercase letter"
            
        if not any(c.isdigit() for c in password):
            return False, "Password must contain at least one number"
            
        if not any(c in string.punctuation for c in password):
            return False, "Password must contain at least one special character"
            
        return True, "Password meets strength requirements"
        
    except Exception as e:
        logger.error(f"Password validation failed: {e}")
        return False, str(e)

def sanitize_input(input_str: str) -> str:
    """Sanitize user input to prevent injection attacks"""
    try:
        # Remove common SQL injection patterns
        blacklist = ["--", ";", "/*", "*/", "@@", "@", "char", "nchar", 
                    "varchar", "nvarchar", "alter", "begin", "cast", 
                    "create", "cursor", "declare", "delete", "drop", 
                    "end", "exec", "execute", "fetch", "insert", "kill", 
                    "select", "sys", "sysobjects", "syscolumns", "table", 
                    "update"]
        
        sanitized = input_str
        for word in blacklist:
            sanitized = sanitized.replace(word, "")
        
        return sanitized
        
    except Exception as e:
        logger.error(f"Input sanitization failed: {e}")
        return ""

def validate_email(email: str) -> bool:
    """Validate email format"""
    try:
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    except Exception as e:
        logger.error(f"Email validation failed: {e}")
        return False

def hash_api_key(api_key: str) -> str:
    """Hash API key for storage"""
    try:
        return pwd_context.hash(api_key)
    except Exception as e:
        logger.error(f"API key hashing failed: {e}")
        raise

def verify_api_key(api_key: str, hashed_key: str) -> bool:
    """Verify API key against its hash"""
    try:
        return pwd_context.verify(api_key, hashed_key)
    except Exception as e:
        logger.error(f"API key verification failed: {e}")
        return False

def authenticate_user(db: Session, email: str, password: str) -> Union[User, bool]:
    """Authenticate a user with email and password"""
    try:
        # Find user by email
        user = db.query(User).filter(User.email == email).first()
        
        # If no user found or password doesn't match
        if not user or not verify_password(password, user.password_hash):
            return False
            
        # Return the user if authentication was successful
        return user
        
    except Exception as e:
        logger.error(f"User authentication failed: {e}")
        return False