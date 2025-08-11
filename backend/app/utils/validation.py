"""
ULTRA SECURITY INPUT VALIDATION MODULE

This module provides comprehensive input validation and sanitization functions
to prevent injection attacks, XSS, path traversal, and other security vulnerabilities.

All input validation follows OWASP guidelines and implements defense-in-depth.
"""

import re
import os
import html
import uuid
from typing import Optional, List, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class SecurityValidationError(ValueError):
    """Raised when input fails security validation"""
    pass


# Security patterns for validation
ALPHANUMERIC_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')
MODEL_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9_.-]+$')
AGENT_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')
TASK_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')
UUID_PATTERN = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')

# Dangerous patterns to block
DANGEROUS_PATTERNS = [
    re.compile(r'<script[^>]*>', re.IGNORECASE),
    re.compile(r'javascript:', re.IGNORECASE),
    re.compile(r'vbscript:', re.IGNORECASE),
    re.compile(r'on\w+\s*=', re.IGNORECASE),
    re.compile(r'\.\./', re.IGNORECASE),
    re.compile(r'\.\.\\', re.IGNORECASE),
    re.compile(r'\\x[0-9a-f]{2}', re.IGNORECASE),
    re.compile(r'%[0-9a-f]{2}', re.IGNORECASE),
    re.compile(r'\x00'),  # Null bytes
    re.compile(r'[\r\n]'),  # Line breaks in inputs where not expected
]

# SQL injection patterns
SQL_INJECTION_PATTERNS = [
    re.compile(r'(\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b)', re.IGNORECASE),
    re.compile(r'(--|#|\*|/\*|\*/)', re.IGNORECASE),
    re.compile(r'(\b(or|and)\b\s*\d+\s*=\s*\d+)', re.IGNORECASE),
    re.compile(r"('|(\\'))", re.IGNORECASE),
]

# Command injection patterns
COMMAND_INJECTION_PATTERNS = [
    re.compile(r'[;&|`$()]'),
    re.compile(r'\b(cat|ls|pwd|whoami|id|ps|kill|rm|mv|cp|chmod|chown|sudo|su)\b', re.IGNORECASE),
]


def validate_model_name(model_name: str) -> str:
    """
    Validate and sanitize AI model name to prevent injection attacks
    
    Args:
        model_name: Model name to validate
        
    Returns:
        Validated and sanitized model name
        
    Raises:
        SecurityValidationError: If model name is invalid or dangerous
    """
    if not model_name or not isinstance(model_name, str):
        raise SecurityValidationError("Model name must be a non-empty string")
    
    # Length check
    if len(model_name) > 100:
        raise SecurityValidationError("Model name too long (max 100 characters)")
    
    # Strip whitespace and convert to lowercase
    clean_name = model_name.strip().lower()
    
    # Check against dangerous patterns
    for pattern in DANGEROUS_PATTERNS:
        if pattern.search(clean_name):
            raise SecurityValidationError(f"Model name contains dangerous characters: {model_name}")
    
    # Check against valid pattern
    if not MODEL_NAME_PATTERN.match(clean_name):
        raise SecurityValidationError(f"Model name contains invalid characters: {model_name}")
    
    # Whitelist of allowed models
    ALLOWED_MODELS = {
        "tinyllama", "tinyllama:latest", "llama2", "llama2:latest", 
        "mistral", "mistral:latest", "codellama", "codellama:latest",
        "phi", "phi:latest", "gemma", "gemma:latest"
    }
    
    if clean_name not in ALLOWED_MODELS:
        logger.warning(f"Model name not in whitelist, using default: {clean_name}")
        return "tinyllama"  # Default safe model
    
    return clean_name


def validate_agent_id(agent_id: str) -> str:
    """
    Validate agent ID to prevent directory traversal and injection
    
    Args:
        agent_id: Agent ID to validate
        
    Returns:
        Validated agent ID
        
    Raises:
        SecurityValidationError: If agent ID is invalid
    """
    if not agent_id or not isinstance(agent_id, str):
        raise SecurityValidationError("Agent ID must be a non-empty string")
    
    # Length check
    if len(agent_id) > 50:
        raise SecurityValidationError("Agent ID too long (max 50 characters)")
    
    clean_id = agent_id.strip().lower()
    
    # Check against dangerous patterns
    for pattern in DANGEROUS_PATTERNS:
        if pattern.search(clean_id):
            raise SecurityValidationError(f"Agent ID contains dangerous characters: {agent_id}")
    
    # Check against valid pattern
    if not AGENT_ID_PATTERN.match(clean_id):
        raise SecurityValidationError(f"Agent ID contains invalid characters: {agent_id}")
    
    # Whitelist of known agents
    ALLOWED_AGENTS = {
        "hardware-resource-optimizer", "ai-agent-orchestrator", "ollama-integration",
        "task-assignment-coordinator", "resource-arbitration-agent", "jarvis-automation-agent",
        "jarvis-hardware-optimizer", "text-analysis-agent", "knowledge-graph-builder"
    }
    
    if clean_id not in ALLOWED_AGENTS:
        raise SecurityValidationError(f"Unknown agent ID: {agent_id}")
    
    return clean_id


def validate_task_id(task_id: str) -> str:
    """
    Validate task ID (UUID format)
    
    Args:
        task_id: Task ID to validate
        
    Returns:
        Validated task ID
        
    Raises:
        SecurityValidationError: If task ID is invalid
    """
    if not task_id or not isinstance(task_id, str):
        raise SecurityValidationError("Task ID must be a non-empty string")
    
    clean_id = task_id.strip().lower()
    
    # Check if it's a valid UUID
    if not UUID_PATTERN.match(clean_id):
        raise SecurityValidationError(f"Task ID must be a valid UUID: {task_id}")
    
    # Additional validation - ensure it's a proper UUID
    try:
        uuid.UUID(clean_id)
    except ValueError:
        raise SecurityValidationError(f"Invalid UUID format: {task_id}")
    
    return clean_id


def validate_cache_pattern(pattern: Optional[str]) -> Optional[str]:
    """
    Validate cache pattern for cache operations
    
    Args:
        pattern: Cache pattern to validate
        
    Returns:
        Validated pattern or None
        
    Raises:
        SecurityValidationError: If pattern is dangerous
    """
    if pattern is None:
        return None
    
    if not isinstance(pattern, str):
        raise SecurityValidationError("Cache pattern must be a string")
    
    # Length check
    if len(pattern) > 200:
        raise SecurityValidationError("Cache pattern too long (max 200 characters)")
    
    clean_pattern = pattern.strip()
    
    # Check against dangerous patterns
    for dangerous_pattern in DANGEROUS_PATTERNS:
        if dangerous_pattern.search(clean_pattern):
            raise SecurityValidationError(f"Cache pattern contains dangerous characters: {pattern}")
    
    # Allow only safe characters for cache patterns
    safe_pattern = re.compile(r'^[a-zA-Z0-9_:*.-]+$')
    if not safe_pattern.match(clean_pattern):
        raise SecurityValidationError(f"Cache pattern contains invalid characters: {pattern}")
    
    return clean_pattern


def sanitize_user_input(user_input: str, max_length: int = 1000) -> str:
    """
    Sanitize user input to prevent XSS and injection attacks
    
    Args:
        user_input: User input to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized input
        
    Raises:
        SecurityValidationError: If input is dangerous or too long
    """
    if not isinstance(user_input, str):
        raise SecurityValidationError("Input must be a string")
    
    if len(user_input) > max_length:
        raise SecurityValidationError(f"Input too long (max {max_length} characters)")
    
    # HTML escape to prevent XSS
    sanitized = html.escape(user_input)
    
    # Check for SQL injection patterns
    for pattern in SQL_INJECTION_PATTERNS:
        if pattern.search(sanitized):
            raise SecurityValidationError("Input contains potential SQL injection patterns")
    
    # Check for command injection patterns
    for pattern in COMMAND_INJECTION_PATTERNS:
        if pattern.search(sanitized):
            raise SecurityValidationError("Input contains potential command injection patterns")
    
    # Check for dangerous patterns
    for pattern in DANGEROUS_PATTERNS:
        if pattern.search(sanitized):
            raise SecurityValidationError("Input contains dangerous patterns")
    
    return sanitized.strip()


def validate_file_path(file_path: str, allowed_dirs: List[str] = None) -> str:
    """
    Validate file path to prevent directory traversal attacks
    
    Args:
        file_path: File path to validate
        allowed_dirs: List of allowed base directories
        
    Returns:
        Validated absolute path
        
    Raises:
        SecurityValidationError: If path is dangerous or not allowed
    """
    if not file_path or not isinstance(file_path, str):
        raise SecurityValidationError("File path must be a non-empty string")
    
    if len(file_path) > 500:
        raise SecurityValidationError("File path too long (max 500 characters)")
    
    # Normalize the path to prevent traversal
    try:
        normalized_path = os.path.normpath(file_path)
        resolved_path = Path(normalized_path).resolve()
    except (ValueError, OSError) as e:
        raise SecurityValidationError(f"Invalid file path: {e}")
    
    # Check for directory traversal attempts
    if '..' in normalized_path or normalized_path.startswith('/'):
        raise SecurityValidationError("Directory traversal attempt detected")
    
    # Check against allowed directories if specified
    if allowed_dirs:
        path_str = str(resolved_path)
        if not any(path_str.startswith(allowed_dir) for allowed_dir in allowed_dirs):
            raise SecurityValidationError(f"File path not in allowed directories: {file_path}")
    
    return str(resolved_path)


def validate_json_payload(payload: Dict[str, Any], required_fields: List[str] = None) -> Dict[str, Any]:
    """
    Validate JSON payload structure and content
    
    Args:
        payload: JSON payload to validate
        required_fields: List of required field names
        
    Returns:
        Validated payload
        
    Raises:
        SecurityValidationError: If payload is invalid or dangerous
    """
    if not isinstance(payload, dict):
        raise SecurityValidationError("Payload must be a JSON object")
    
    # Check payload size
    if len(str(payload)) > 10000:  # 10KB limit
        raise SecurityValidationError("Payload too large (max 10KB)")
    
    # Check for required fields
    if required_fields:
        for field in required_fields:
            if field not in payload:
                raise SecurityValidationError(f"Missing required field: {field}")
    
    # Recursively sanitize string values
    def sanitize_dict(d):
        if isinstance(d, dict):
            return {k: sanitize_dict(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [sanitize_dict(item) for item in d]
        elif isinstance(d, str):
            return sanitize_user_input(d, max_length=1000)
        else:
            return d
    
    try:
        sanitized_payload = sanitize_dict(payload)
        return sanitized_payload
    except Exception as e:
        raise SecurityValidationError(f"Failed to sanitize payload: {e}")


def validate_pagination_params(offset: int = 0, limit: int = 100) -> tuple[int, int]:
    """
    Validate pagination parameters
    
    Args:
        offset: Pagination offset
        limit: Pagination limit
        
    Returns:
        Validated (offset, limit) tuple
        
    Raises:
        SecurityValidationError: If parameters are invalid
    """
    if not isinstance(offset, int) or offset < 0:
        raise SecurityValidationError("Offset must be a non-negative integer")
    
    if not isinstance(limit, int) or limit < 1:
        raise SecurityValidationError("Limit must be a positive integer")
    
    # Prevent excessive data requests
    if offset > 100000:
        raise SecurityValidationError("Offset too large (max 100,000)")
    
    if limit > 1000:
        raise SecurityValidationError("Limit too large (max 1,000)")
    
    return offset, limit