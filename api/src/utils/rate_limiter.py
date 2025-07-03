"""
Rate limiter utility for API rate limiting
"""

from datetime import datetime, timedelta
from collections import defaultdict
import logging
from typing import Dict, DefaultDict, List, Optional, Any
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
        # This logic is kept for potential future re-enabling
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
        """Check if request is within rate limit. Currently disabled."""
        # Rate limiting is disabled, always allow the request.
        return True
    
    def get_remaining_requests(
        self,
        key: str,
        limit_type: str = "general",
        max_requests: int = 10
    ) -> Dict[str, Any]:
        """Get remaining requests information"""
        try:
            # Since rate limiting is disabled, we can return the max limit.
            return {
                "remaining": max_requests,
                "limit": max_requests,
                "current": 0
            }
                
        except Exception as e:
            logger.error(f"Failed to get remaining requests: {e}")
            return {"error": str(e)}
    
    def reset_limits(self, key: Optional[str] = None):
        """Reset rate limits for a key or all keys"""
        try:
            with self.lock:
                if key:
                    # Find all composite keys starting with the base key
                    keys_to_reset = [k for k in self.requests if k.startswith(f"{key}:")]
                    for k in keys_to_reset:
                        self.requests.pop(k, None)
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
        """Check if key is currently blocked. Currently disabled."""
        # Rate limiting is disabled, so never blocked.
        return False
    
    def get_block_duration(
        self,
        key: str,
        limit_type: str = "general"
    ) -> int:
        """Get remaining block duration in seconds. Currently disabled."""
        # Rate limiting is disabled, so block duration is always 0.
        return 0 