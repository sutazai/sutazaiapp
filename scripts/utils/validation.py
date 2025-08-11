"""
Input validation utilities for SutazAI
"""
import re
from typing import Optional, Set, List
import logging

logger = logging.getLogger(__name__)

# Define allowed model patterns to prevent injection attacks
ALLOWED_MODEL_PATTERN = re.compile(r'^[a-zA-Z0-9\-_.:/@]+$')

# Maximum model name length to prevent buffer overflow attempts
MAX_MODEL_NAME_LENGTH = 256

# Known safe model names (will be expanded by model manager)
DEFAULT_ALLOWED_MODELS: Set[str] = {
    "tinyllama",
    "nomic-embed-text",
    "llama2", 
    "llama2:7b",
    "llama2:13b",
    "codellama",
    "codellama:7b",
    "codellama:13b",
    "mistral",
    "mistral:7b",
    "phi",
    "phi:2.7b",
    "gemma",
    "gemma:7b"
}

def validate_model_name(model_name: Optional[str]) -> Optional[str]:
    """
    Validate and sanitize model name for security and correctness.
    
    Args:
        model_name: The model name to validate (can be None for default)
        
    Returns:
        Validated model name or None for default
        
    Raises:
        ValueError: If model name is invalid or potentially malicious
    """
    # DEBUG: Log every validation call
    logger.info(f"🔍 VALIDATION CALLED: model_name={repr(model_name)}")
    
    # Allow None (will use default model)
    if model_name is None:
        logger.info("✅ VALIDATION: None model name allowed")
        return None
    
    # Convert to string and strip whitespace
    model_name = str(model_name).strip()
    
    # Allow empty string (will use default model)
    if not model_name:
        return None
    
    # Security: Check for maximum length to prevent buffer overflow
    if len(model_name) > MAX_MODEL_NAME_LENGTH:
        raise ValueError(f"Model name too long (max {MAX_MODEL_NAME_LENGTH} characters)")
    
    # Security: Validate character pattern to prevent injection attacks
    if not ALLOWED_MODEL_PATTERN.match(model_name):
        logger.warning(f"🚨 VALIDATION BLOCKING INVALID CHARS: model_name={repr(model_name)}")
        raise ValueError(
            "Model name contains invalid characters. "
            "Only letters, numbers, hyphens, underscores, dots, colons, slashes, and @ are allowed."
        )
    
    # Security: Check for dangerous patterns that could indicate injection attempts
    dangerous_patterns = [
        '../',           # Path traversal
        '..\\',          # Windows path traversal
        ';',             # Command injection
        '|',             # Command chaining
        '&',             # Command chaining
        '`',             # Command substitution
        '$(',            # Command substitution
        '${',            # Variable substitution
        '<',             # Input redirection
        '>',             # Output redirection
        '\\x',           # Hex encoding
        '\\u',           # Unicode encoding
        'javascript:',   # JavaScript URL
        'data:',         # Data URL
        'vbscript:',     # VBScript URL
    ]
    
    model_name_lower = model_name.lower()
    for pattern in dangerous_patterns:
        if pattern in model_name_lower:
            logger.warning(f"🚨 VALIDATION BLOCKING: model_name={repr(model_name)}, pattern={pattern}")
            raise ValueError(f"Model name contains potentially dangerous pattern: {pattern}")
    
    # Security: Prevent excessively long tag specifications
    if ':' in model_name:
        parts = model_name.split(':')
        if len(parts) > 2:
            raise ValueError("Model name can only contain one colon for version specification")
        
        model_base, version = parts
        if not model_base or not version:
            raise ValueError("Invalid model name format. Expected 'model:version'")
        
        # Additional validation for version part
        if not re.match(r'^[a-zA-Z0-9\-_.]+$', version):
            raise ValueError("Model version contains invalid characters")
    
    # Security: Prevent registry manipulation attempts
    if model_name.startswith(('http://', 'https://', 'ftp://', 'file://')):
        raise ValueError("Model name cannot be a URL")
    
    # Log validation for monitoring
    logger.debug(f"Validated model name: {model_name}")
    
    return model_name


def is_model_allowed(model_name: str, available_models: Optional[List[str]] = None) -> bool:
    """
    Check if a model name is in the list of allowed/available models.
    
    Args:
        model_name: The model name to check
        available_models: List of available models (optional)
        
    Returns:
        True if model is allowed, False otherwise
    """
    if not model_name:
        return True  # None/empty uses default model
    
    # Check against default allowed models
    if model_name in DEFAULT_ALLOWED_MODELS:
        return True
    
    # Check against provided available models list
    if available_models:
        return model_name in available_models
    
    # If no specific list provided, check if it matches common patterns
    # This is a fallback for when model manager isn't available
    common_patterns = [
        r'^tinyllama$',
        r'^llama2(:[0-9]+[bm])?$',
        r'^codellama(:[0-9]+[bm])?$',
        r'^mistral(:[0-9]+[bm])?$',
        r'^phi(:[0-9]+\.?[0-9]*[bm])?$',
        r'^gemma(:[0-9]+[bm])?$',
        r'^nomic-embed-text$',
    ]
    
    for pattern in common_patterns:
        if re.match(pattern, model_name):
            return True
    
    return False


def sanitize_string_input(input_str: Optional[str], max_length: int = 1000) -> str:
    """
    Sanitize general string input for XSS and injection prevention.
    
    Args:
        input_str: Input string to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized string
        
    Raises:
        ValueError: If input is invalid
    """
    if input_str is None:
        return ""
    
    # Convert to string and normalize whitespace
    sanitized = str(input_str).strip()
    
    # Check length
    if len(sanitized) > max_length:
        raise ValueError(f"Input too long (max {max_length} characters)")
    
    # Remove dangerous patterns
    dangerous_patterns = [
        r'<script[^>]*>.*?</script>',     # Script tags
        r'javascript:',                   # JavaScript URLs
        r'vbscript:',                    # VBScript URLs
        r'data:.*base64',                # Base64 data URLs
        r'on\w+\s*=',                    # Event handlers
    ]
    
    for pattern in dangerous_patterns:
        sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
    
    return sanitized