"""
SutazAI Utility Functions

Comprehensive collection of helper functions for common operations.
"""

import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

try:
    import jwt  # PyJWT package
    from cryptography.fernet import Fernet
except ImportError as e:
    raise ImportError(
        f"Required package not found. Please run 'pip install PyJWT cryptography': {e}"
    )

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
if not os.path.exists(_log_dir):
    os.makedirs(_log_dir, exist_ok=True)
handler = logging.FileHandler(os.path.join(_log_dir, "utils.log"))
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)

# Type variables for generic functions
T = TypeVar("T")
R = TypeVar("R")

# JWT Configuration
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION = 3600  # 1 hour
JWT_LEEWAY = 60  # 1 minute leeway for clock skew


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable:
    """
    Retry decorator with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch

    Returns:
        Decorated function
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            current_delay = delay
            last_exception: Optional[Exception] = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        if last_exception is not None:
                            raise last_exception
                        raise RuntimeError("Unexpected error in retry mechanism")

                    logger.warning(
                        f"Attempt {attempt + 1}/{max_attempts} failed: {str(e)}. "
                        f"Retrying in {current_delay:.1f} seconds..."
                    )

                    time.sleep(current_delay)
                    current_delay *= backoff

            if last_exception is not None:
                raise last_exception
            raise RuntimeError("Unexpected error in retry mechanism")

        return wrapper

    return decorator


def memoize(ttl: int = 300) -> Callable:
    """
    Memoization decorator with time-to-live.

    Args:
        ttl: Time to live for cached results in seconds

    Returns:
        Decorated function
    """
    cache: Dict[str, Tuple[Any, float]] = {}

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Create cache key from function name and arguments
            key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            key_hash = hashlib.md5(key.encode()).hexdigest()

            # Check if result is cached and not expired
            if key_hash in cache:
                result, timestamp = cache[key_hash]
                if time.time() - timestamp < ttl:
                    return result

                # Remove expired result
                del cache[key_hash]

            # Calculate and cache result
            result = func(*args, **kwargs)
            cache[key_hash] = (result, time.time())
            return result

        return wrapper

    return decorator


def validate_input(
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    pattern: Optional[str] = None,
    custom_validator: Optional[Callable[[str], bool]] = None,
) -> Callable:
    """
    Input validation decorator.

    Args:
        min_length: Minimum input length
        max_length: Maximum input length
        pattern: Regex pattern to match
        custom_validator: Custom validation function

    Returns:
        Decorated function
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(input_str: str, *args: Any, **kwargs: Any) -> T:
            if not isinstance(input_str, str):
                raise ValueError("Input must be a string")

            if min_length and len(input_str) < min_length:
                raise ValueError(
                    f"Input length must be at least {min_length} characters"
                )

            if max_length and len(input_str) > max_length:
                raise ValueError(
                    f"Input length must not exceed {max_length} characters"
                )

            if pattern and not re.match(pattern, input_str):
                raise ValueError(f"Input must match pattern: {pattern}")

            if custom_validator and not custom_validator(input_str):
                raise ValueError("Input failed custom validation")

            return func(input_str, *args, **kwargs)

        return wrapper

    return decorator


def safe_json_loads(data: str) -> Dict[str, Any]:
    """
    Safely parse JSON string with error handling.

    Args:
        data: JSON string to parse

    Returns:
        Parsed JSON data

    Raises:
        ValueError: If JSON is invalid
    """
    try:
        return json.loads(data)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {str(e)}")


def generate_token(
    payload: Dict[str, Any], secret_key: str, expiration: int = 3600
) -> str:
    """
    Generate JWT token with payload.

    Args:
        payload: Token payload
        secret_key: Secret key for signing
        expiration: Token expiration time in seconds

    Returns:
        Encoded JWT token
    """
    try:
        payload["exp"] = datetime.now(timezone.utc).timestamp() + expiration
        return jwt.encode(payload, secret_key, algorithm="HS256")
    except Exception as e:
        raise ValueError(f"Failed to generate token: {str(e)}")


def verify_token(token: str, secret_key: str) -> Dict[str, Any]:
    """
    Verify and decode JWT token.

    Args:
        token: JWT token to verify
        secret_key: Secret key for verification

    Returns:
        Decoded token payload

    Raises:
        ValueError: If token is invalid or expired
    """
    try:
        return jwt.decode(token, secret_key, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise ValueError("Token has expired")
    except jwt.InvalidTokenError as e:
        raise ValueError(f"Invalid token: {str(e)}")


def encrypt_data(data: str, key: Optional[bytes] = None) -> Tuple[str, bytes]:
    """
    Encrypt data using Fernet symmetric encryption.

    Args:
        data: Data to encrypt
        key: Encryption key (generated if not provided)

    Returns:
        Tuple of (encrypted data, encryption key)
    """
    try:
        if not key:
            key = Fernet.generate_key()

        f = Fernet(key)
        encrypted_data = f.encrypt(data.encode())
        return encrypted_data.decode(), key
    except Exception as e:
        raise ValueError(f"Encryption failed: {str(e)}")


def decrypt_data(encrypted_data: str, key: bytes) -> str:
    """
    Decrypt data using Fernet symmetric encryption.

    Args:
        encrypted_data: Data to decrypt
        key: Encryption key

    Returns:
        Decrypted data

    Raises:
        ValueError: If decryption fails
    """
    try:
        f = Fernet(key)
        decrypted_data = f.decrypt(encrypted_data.encode())
        return decrypted_data.decode()
    except Exception as e:
        raise ValueError(f"Decryption failed: {str(e)}")


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent directory traversal.

    Args:
        filename: Filename to sanitize

    Returns:
        Sanitized filename
    """
    # Remove path separators and null bytes
    filename = os.path.basename(filename)
    filename = filename.replace("\x00", "")

    # Remove potentially dangerous characters
    filename = re.sub(r"[^\w\-_\. ]", "", filename)

    # Ensure filename is not empty
    if not filename:
        filename = "unnamed_file"

    return filename


def format_timestamp(
    timestamp: Optional[float] = None, format_str: str = "%Y-%m-%d %H:%M:%S"
) -> str:
    """
    Format timestamp to human-readable string.

    Args:
        timestamp: Unix timestamp (uses current time if None)
        format_str: Datetime format string

    Returns:
        Formatted timestamp string
    """
    if timestamp is None:
        timestamp = time.time()
    return datetime.fromtimestamp(timestamp).strftime(format_str)


def parse_size_string(size_str: str) -> int:
    """
    Parse size string (e.g., '1.5GB') to bytes.

    Args:
        size_str: Size string to parse

    Returns:
        Size in bytes

    Raises:
        ValueError: If size string is invalid
    """
    units = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}

    match = re.match(r"^([\d.]+)\s*([A-Za-z]+)$", size_str.strip())
    if not match:
        raise ValueError(f"Invalid size string: {size_str}")

    size, unit = match.groups()
    unit = unit.upper()

    if unit not in units:
        raise ValueError(f"Invalid unit: {unit}")

    try:
        return int(float(size) * units[unit])
    except ValueError:
        raise ValueError(f"Invalid size value: {size}")


def chunk_list(lst: List[T], chunk_size: int) -> List[List[T]]:
    """
    Split list into chunks of specified size.

    Args:
        lst: List to split
        chunk_size: Size of each chunk

    Returns:
        List of chunks
    """
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def flatten_dict(
    d: Dict[str, Any], parent_key: str = "", separator: str = "."
) -> Dict[str, Any]:
    """
    Flatten nested dictionary with dot notation.

    Args:
        d: Dictionary to flatten
        parent_key: Parent key for nested items
        separator: Separator for keys

    Returns:
        Flattened dictionary
    """
    items: List[Tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{separator}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, separator).items())
        else:
            items.append((new_key, v))
    return dict(items)


@retry(max_attempts=3, delay=1.0)
def add(x: Union[int, float], y: Union[int, float]) -> Union[int, float]:
    """
    Add two numbers with retry on failure.

    Args:
        x: First number
        y: Second number

    Returns:
        Sum of numbers
    """
    return x + y


@validate_input(min_length=1)
def divide(x: Union[int, float], y: Union[int, float]) -> Union[int, float]:
    """
    Divide two numbers with input validation.

    Args:
        x: Dividend
        y: Divisor

    Returns:
        Division result

    Raises:
        ValueError: If divisor is zero
    """
    if y == 0:
        raise ValueError("Cannot divide by zero")
    return x / y
