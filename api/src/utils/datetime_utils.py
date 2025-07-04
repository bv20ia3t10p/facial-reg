"""
Datetime utilities for handling UTC+7 timestamps
"""

from datetime import datetime, timedelta, timezone

# Define UTC+7 timezone
UTC_PLUS_7 = timezone(timedelta(hours=7))

def get_current_time() -> datetime:
    """Get current time in UTC+7"""
    return datetime.now(UTC_PLUS_7)

def get_current_time_str() -> str:
    """Get current time in UTC+7 as ISO format string"""
    return get_current_time().isoformat()

def get_current_time_formatted(format_str: str = "%Y%m%d_%H%M%S") -> str:
    """Get current time in UTC+7 with custom format"""
    return get_current_time().strftime(format_str)

def from_utc(utc_dt: datetime) -> datetime:
    """Convert UTC datetime to UTC+7"""
    if utc_dt.tzinfo is None:
        utc_dt = utc_dt.replace(tzinfo=timezone.utc)
    return utc_dt.astimezone(UTC_PLUS_7)

def to_utc(local_dt: datetime) -> datetime:
    """Convert UTC+7 datetime to UTC"""
    if local_dt.tzinfo is None:
        local_dt = local_dt.replace(tzinfo=UTC_PLUS_7)
    return local_dt.astimezone(timezone.utc)

def get_default_expiry(minutes: int = 15) -> datetime:
    """Get expiry time in UTC+7, default 15 minutes from now"""
    return get_current_time() + timedelta(minutes=minutes) 