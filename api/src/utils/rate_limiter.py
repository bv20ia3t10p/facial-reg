"""
Rate limiter utility for API rate limiting
"""

from datetime import datetime, timedelta
from collections import defaultdict
import logging
from typing import Dict, DefaultDict, List
import threading

logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiter implementation using sliding window"""
    
    def __init__(self, window_size: int = 60):
        self.window_size = window_size  # Window size in seconds
        self.requests: DefaultDict[str, List[datetime]] = defaultdict(list)
        self.lock = threading.Lock()
        
    def _clean_old_requests(self, key: str):
        """Remove requests outside the current window"""
        try:
            now = datetime.utcnow()
            window_start = now - timedelta(seconds=self.window_size)
            
            with self.lock:
                self.requests[key] = [
                    req_time for req_time in self.requests[key]
                    if req_time > window_start
                ]
                
        except Exception as e:
            logger.error(f"Failed to clean old requests: {e}")
    
    def check_rate_limit(
        self,
        key: str,
        limit_type: str = "general",
        max_requests: int = 10
    ) -> bool:
        """Check if request is within rate limit"""
        try:
            # Create composite key
            composite_key = f"{key}:{limit_type}"
            
            # Clean old requests
            self._clean_old_requests(composite_key)
            
            # Get current window requests
            with self.lock:
                current_requests = len(self.requests[composite_key])
                
                # Check if under limit
                if current_requests < max_requests:
                    # Add new request
                    self.requests[composite_key].append(datetime.utcnow())
                    return True
                    
                return False
                
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return False
    
    def get_remaining_requests(
        self,
        key: str,
        limit_type: str = "general",
        max_requests: int = 10
    ) -> Dict[str, int]:
        """Get remaining requests information"""
        try:
            composite_key = f"{key}:{limit_type}"
            self._clean_old_requests(composite_key)
            
            with self.lock:
                current_requests = len(self.requests[composite_key])
                remaining = max(0, max_requests - current_requests)
                
                return {
                    "remaining": remaining,
                    "limit": max_requests,
                    "current": current_requests
                }
                
        except Exception as e:
            logger.error(f"Failed to get remaining requests: {e}")
            return {"error": str(e)}
    
    def reset_limits(self, key: str = None):
        """Reset rate limits for a key or all keys"""
        try:
            with self.lock:
                if key:
                    # Reset specific key
                    self.requests.pop(key, None)
                else:
                    # Reset all
                    self.requests.clear()
                    
        except Exception as e:
            logger.error(f"Failed to reset rate limits: {e}")
    
    def is_blocked(
        self,
        key: str,
        limit_type: str = "general",
        max_requests: int = 10
    ) -> bool:
        """Check if key is currently blocked"""
        try:
            composite_key = f"{key}:{limit_type}"
            self._clean_old_requests(composite_key)
            
            with self.lock:
                return len(self.requests[composite_key]) >= max_requests
                
        except Exception as e:
            logger.error(f"Failed to check if blocked: {e}")
            return True  # Fail safe
    
    def get_block_duration(
        self,
        key: str,
        limit_type: str = "general"
    ) -> int:
        """Get remaining block duration in seconds"""
        try:
            composite_key = f"{key}:{limit_type}"
            
            with self.lock:
                if not self.requests[composite_key]:
                    return 0
                    
                oldest_request = min(self.requests[composite_key])
                now = datetime.utcnow()
                
                # Calculate when the oldest request will expire
                expires_at = oldest_request + timedelta(seconds=self.window_size)
                
                if expires_at > now:
                    return int((expires_at - now).total_seconds())
                return 0
                
        except Exception as e:
            logger.error(f"Failed to get block duration: {e}")
            return self.window_size  # Fail safe 