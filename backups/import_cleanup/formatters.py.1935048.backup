"""
Data Formatting Utilities - Extracted from monolith
Common formatting functions for UI display
"""

import humanize
from datetime import datetime, timedelta
from typing import Union, Optional

def format_bytes(bytes_value: Union[int, float], precision: int = 1) -> str:
    """
    Format bytes value into human-readable string
    
    Args:
        bytes_value: Size in bytes
        precision: Decimal places
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    if bytes_value is None:
        return "N/A"
    
    try:
        return humanize.naturalsize(bytes_value, binary=True, gnu=False)
    except:
        return f"{bytes_value} bytes"

def format_duration(seconds: Union[int, float]) -> str:
    """
    Format duration in seconds to human-readable string
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (e.g., "2 minutes", "1.5 hours")
    """
    if seconds is None:
        return "N/A"
        
    try:
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        elif seconds < 86400:
            hours = seconds / 3600
            return f"{hours:.1f}h"
        else:
            days = seconds / 86400
            return f"{days:.1f}d"
    except:
        return f"{seconds}s"

def format_timestamp(timestamp: Union[str, datetime], format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format timestamp string or datetime object
    
    Args:
        timestamp: Timestamp to format
        format_str: Output format string
        
    Returns:
        Formatted timestamp string
    """
    if timestamp is None:
        return "N/A"
        
    try:
        if isinstance(timestamp, str):
            # Try parsing common formats
            for fmt in ["%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"]:
                try:
                    dt = datetime.strptime(timestamp, fmt)
                    return dt.strftime(format_str)
                except ValueError:
                    continue
            return timestamp  # Return as-is if parsing fails
        elif isinstance(timestamp, datetime):
            return timestamp.strftime(format_str)
        else:
            return str(timestamp)
    except:
        return str(timestamp)

def format_percentage(value: Union[int, float], precision: int = 1) -> str:
    """
    Format decimal value as percentage
    
    Args:
        value: Decimal value (0.0 to 1.0)
        precision: Decimal places
        
    Returns:
        Formatted percentage string
    """
    if value is None:
        return "N/A"
        
    try:
        percentage = value * 100 if value <= 1.0 else value
        return f"{percentage:.{precision}f}%"
    except:
        return f"{value}%"

def format_number(value: Union[int, float], precision: int = 0) -> str:
    """
    Format large numbers with thousand separators
    
    Args:
        value: Numeric value
        precision: Decimal places
        
    Returns:
        Formatted number string
    """
    if value is None:
        return "N/A"
        
    try:
        if precision == 0:
            return f"{int(value):,}"
        else:
            return f"{float(value):,.{precision}f}"
    except:
        return str(value)

def format_metric_delta(current: Union[int, float], previous: Union[int, float]) -> tuple:
    """
    Calculate and format metric delta for Streamlit metrics
    
    Args:
        current: Current value
        previous: Previous value
        
    Returns:
        Tuple of (delta_value, delta_string)
    """
    if current is None or previous is None:
        return None, "N/A"
        
    try:
        delta = current - previous
        if delta == 0:
            return delta, "No change"
        elif delta > 0:
            return delta, f"+{format_number(delta)}"
        else:
            return delta, f"{format_number(delta)}"
    except:
        return None, "Error"

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to specified length with suffix
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix for truncated text
        
    Returns:
        Truncated text string
    """
    if not text:
        return ""
        
    if len(text) <= max_length:
        return text
        
    return text[:max_length - len(suffix)] + suffix

def format_status_badge(status: str) -> str:
    """
    Format status as colored badge
    
    Args:
        status: Status string
        
    Returns:
        HTML badge string
    """
    status_colors = {
        "healthy": "#28a745",
        "running": "#28a745", 
        "online": "#28a745",
        "active": "#28a745",
        "warning": "#ffc107",
        "degraded": "#ffc107",
        "error": "#dc3545",
        "offline": "#dc3545",
        "failed": "#dc3545",
        "inactive": "#6c757d",
        "unknown": "#6c757d"
    }
    
    color = status_colors.get(status.lower(), "#6c757d")
    
    return f"""
    <span style="
        background-color: {color};
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.8em;
        font-weight: bold;
        text-transform: uppercase;
    ">{status}</span>
    """