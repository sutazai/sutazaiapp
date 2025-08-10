#!/usr/bin/env python3
"""
Common Utilities for SutazAI
============================

Consolidated utility functions from across the codebase.
Replaces 363+ individual utility scripts with a single, well-organized module.
"""

import os
import sys
import json
import yaml
import logging
import hashlib
import subprocess
import psutil
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import socket
import time

class ConfigurationError(Exception):
    """Configuration-related errors"""
    pass

class ValidationError(Exception):
    """Validation-related errors"""
    pass

def setup_logging(
    name: str = 'sutazai',
    level: str = 'INFO',
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup standardized logging for SutazAI components.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        format_string: Optional custom format string
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    if not logger.handlers:  # Avoid duplicate handlers
        # Create formatter
        if format_string is None:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(format_string)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    return logger

def load_config(
    config_path: Union[str, Path], 
    required_keys: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Load configuration from JSON or YAML file.
    
    Args:
        config_path: Path to configuration file
        required_keys: List of required configuration keys
        
    Returns:
        Configuration dictionary
        
    Raises:
        ConfigurationError: If file not found or required keys missing
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise ConfigurationError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yml', '.yaml']:
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
    except Exception as e:
        raise ConfigurationError(f"Error loading config {config_path}: {e}")
    
    # Validate required keys
    if required_keys:
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ConfigurationError(f"Missing required config keys: {missing_keys}")
    
    return config

def validate_ports(ports: List[int], host: str = 'localhost') -> Dict[int, bool]:
    """
    Validate that ports are available and not in use.
    
    Args:
        ports: List of ports to check
        host: Host to check ports on
        
    Returns:
        Dictionary mapping port to availability status
    """
    results = {}
    
    for port in ports:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex((host, port))
                results[port] = result != 0  # True if available (connection failed)
        except Exception:
            results[port] = False  # Assume unavailable on error
    
    return results

def get_system_info() -> Dict[str, Any]:
    """
    Get comprehensive system information.
    
    Returns:
        System information dictionary
    """
    info = {
        'hostname': socket.gethostname(),
        'platform': sys.platform,
        'python_version': sys.version,
        'cpu_count': psutil.cpu_count(),
        'memory': {
            'total': psutil.virtual_memory().total,
            'available': psutil.virtual_memory().available,
            'percent': psutil.virtual_memory().percent
        },
        'disk': {
            'total': psutil.disk_usage('/').total,
            'free': psutil.disk_usage('/').free,
            'percent': psutil.disk_usage('/').percent
        },
        'timestamp': datetime.now().isoformat()
    }
    
    # Add network interfaces
    info['network'] = {}
    for interface, addresses in psutil.net_if_addrs().items():
        info['network'][interface] = [
            {'family': addr.family.name, 'address': addr.address}
            for addr in addresses
        ]
    
    return info

def check_dependencies(requirements: List[str]) -> Dict[str, bool]:
    """
    Check if required dependencies are installed.
    
    Args:
        requirements: List of package names to check
        
    Returns:
        Dictionary mapping package name to availability
    """
    results = {}
    
    for package in requirements:
        try:
            __import__(package)
            results[package] = True
        except ImportError:
            results[package] = False
    
    return results

def run_command(
    command: Union[str, List[str]], 
    cwd: Optional[str] = None,
    timeout: int = 30,
    capture_output: bool = True
) -> Tuple[int, str, str]:
    """
    Run shell command with proper error handling.
    
    Args:
        command: Command to run (string or list)
        cwd: Working directory
        timeout: Command timeout in seconds
        capture_output: Whether to capture stdout/stderr
        
    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    try:
        if isinstance(command, str):
            command = command.split()
            
        result = subprocess.run(
            command,
            cwd=cwd,
            timeout=timeout,
            capture_output=capture_output,
            text=True,
            check=False
        )
        
        return result.returncode, result.stdout or '', result.stderr or ''
        
    except subprocess.TimeoutExpired:
        return -1, '', f'Command timed out after {timeout} seconds'
    except Exception as e:
        return -1, '', str(e)

def format_size(size_bytes: int) -> str:
    """
    Format byte size into human readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string (e.g., "1.2 GB")
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"

def get_file_hash(file_path: Union[str, Path], algorithm: str = 'md5') -> str:
    """
    Calculate hash of a file.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm (md5, sha1, sha256)
        
    Returns:
        Hex digest of file hash
    """
    hash_obj = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()

def retry_operation(
    operation: callable,
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple = (Exception,)
) -> Any:
    """
    Retry an operation with exponential backoff.
    
    Args:
        operation: Function to retry
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts
        backoff: Backoff multiplier
        exceptions: Exceptions to catch and retry on
        
    Returns:
        Result of successful operation
        
    Raises:
        Last exception if all attempts fail
    """
    last_exception = None
    
    for attempt in range(max_attempts):
        try:
            return operation()
        except exceptions as e:
            last_exception = e
            if attempt < max_attempts - 1:
                time.sleep(delay)
                delay *= backoff
    
    raise last_exception

def health_check_url(
    url: str,
    timeout: int = 5,
    expected_status: int = 200
) -> Tuple[bool, str]:
    """
    Perform health check on HTTP endpoint.
    
    Args:
        url: URL to check
        timeout: Request timeout
        expected_status: Expected HTTP status code
        
    Returns:
        Tuple of (success, message)
    """
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == expected_status:
            return True, f"OK ({response.status_code})"
        else:
            return False, f"HTTP {response.status_code}"
    except requests.exceptions.RequestException as e:
        return False, str(e)

def ensure_directory(path: Union[str, Path], mode: int = 0o755) -> Path:
    """
    Ensure directory exists, create if necessary.
    
    Args:
        path: Directory path
        mode: Directory permissions
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True, mode=mode)
    return path

def cleanup_old_files(
    directory: Union[str, Path],
    pattern: str = "*",
    days_old: int = 7
) -> List[Path]:
    """
    Clean up old files in a directory.
    
    Args:
        directory: Directory to clean
        pattern: File pattern to match
        days_old: Files older than this many days will be deleted
        
    Returns:
        List of deleted files
    """
    directory = Path(directory)
    cutoff_time = datetime.now() - timedelta(days=days_old)
    deleted_files = []
    
    for file_path in directory.glob(pattern):
        if file_path.is_file():
            file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
            if file_time < cutoff_time:
                try:
                    file_path.unlink()
                    deleted_files.append(file_path)
                except OSError:
                    pass  # Ignore errors
    
    return deleted_files

# Environment and configuration helpers
def get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean environment variable."""
    value = os.environ.get(key, '').lower()
    return value in ('1', 'true', 'yes', 'on') if value else default

def get_env_int(key: str, default: int = 0) -> int:
    """Get integer environment variable."""
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return default

def get_env_list(key: str, default: List[str] = None, separator: str = ',') -> List[str]:
    """Get list environment variable."""
    value = os.environ.get(key, '')
    if not value:
        return default or []
    return [item.strip() for item in value.split(separator) if item.strip()]

# Docker and container utilities
def is_running_in_container() -> bool:
    """Check if running inside a container."""
    return (
        Path('/.dockerenv').exists() or
        Path('/proc/1/cgroup').exists() and
        any('docker' in line or 'containerd' in line 
            for line in Path('/proc/1/cgroup').read_text().splitlines())
    )

def get_container_id() -> Optional[str]:
    """Get container ID if running in container."""
    if not is_running_in_container():
        return None
    
    try:
        # Try Docker
        with open('/proc/self/cgroup', 'r') as f:
            for line in f:
                if 'docker' in line:
                    return line.split('/')[-1].strip()
        
        # Try containerd/k8s
        hostname = socket.gethostname()
        if len(hostname) == 12:  # Typical container hostname length
            return hostname
            
    except Exception:
        pass
    
    return None

if __name__ == "__main__":
    # Simple test/demo
    logger = setup_logging()
    logger.info("SutazAI Common Utils loaded successfully")
    
    # Test system info
    info = get_system_info()
    logger.info(f"System: {info['hostname']} - {info['cpu_count']} CPUs")
    logger.info(f"Memory: {format_size(info['memory']['available'])} available")
    logger.info(f"Disk: {format_size(info['disk']['free'])} free")