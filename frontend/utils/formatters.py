"""
Formatting Utilities for SutazAI Frontend
Common formatters for dates, status, sizes, etc.
"""

import time
from datetime import datetime, timezone
from typing import Any, Optional

def format_timestamp(timestamp: Any) -> str:
    """
    Format timestamp for display
    """
    if not timestamp:
        return "Unknown"
    
    try:
        if isinstance(timestamp, str):
            # Try parsing ISO format
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except:
                return timestamp  # Return as-is if can't parse
        elif isinstance(timestamp, (int, float)):
            dt = datetime.fromtimestamp(timestamp)
        else:
            return str(timestamp)
        
        return dt.strftime("%Y-%m-%d %H:%M:%S")
        
    except Exception:
        return str(timestamp)

def format_status_badge(status: str) -> str:
    """
    Format status as HTML badge
    """
    status_colors = {
        "healthy": "#28a745",
        "online": "#28a745", 
        "operational": "#28a745",
        "warning": "#ffc107",
        "degraded": "#ffc107",
        "unhealthy": "#dc3545",
        "offline": "#dc3545",
        "error": "#dc3545",
        "unknown": "#6c757d"
    }
    
    color = status_colors.get(status.lower(), "#6c757d")
    
    return f"""
    <span style="
        display: inline-block;
        padding: 4px 8px;
        background-color: {color};
        color: white;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: 500;
    ">
        {status.title()}
    </span>
    """

def format_bytes(bytes_value: Any) -> str:
    """
    Format bytes as human-readable string
    """
    if not isinstance(bytes_value, (int, float)) or bytes_value < 0:
        return "0 B"
    
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    size = float(bytes_value)
    unit_index = 0
    
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    if unit_index == 0:
        return f"{int(size)} {units[unit_index]}"
    else:
        return f"{size:.1f} {units[unit_index]}"

def format_duration(seconds: Any) -> str:
    """
    Format duration in seconds as human-readable string
    """
    if not isinstance(seconds, (int, float)) or seconds < 0:
        return "0s"
    
    units = [
        (86400, 'd'),
        (3600, 'h'), 
        (60, 'm'),
        (1, 's')
    ]
    
    result = []
    remaining = int(seconds)
    
    for unit_seconds, unit_name in units:
        if remaining >= unit_seconds:
            count = remaining // unit_seconds
            remaining = remaining % unit_seconds
            result.append(f"{count}{unit_name}")
    
    if not result:
        return "0s"
    
    return " ".join(result[:2])  # Show max 2 units

def format_percentage(value: Any, decimals: int = 1) -> str:
    """
    Format value as percentage
    """
    try:
        return f"{float(value):.{decimals}f}%"
    except (ValueError, TypeError):
        return "0.0%"

def format_number(value: Any, decimals: int = 0) -> str:
    """
    Format number with thousands separator
    """
    try:
        if decimals == 0:
            return f"{int(value):,}"
        else:
            return f"{float(value):,.{decimals}f}"
    except (ValueError, TypeError):
        return "0"