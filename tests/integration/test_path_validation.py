#!/usr/bin/env python3
"""
import logging

logger = logging.getLogger(__name__)
Test path validation function independently
"""

import os
from pathlib import Path

def validate_safe_path(requested_path: str, base_path: str = "/") -> str:
    """Validate path to prevent directory traversal attacks"""
    # Normalize and resolve the path
    requested = Path(requested_path).resolve()
    base = Path(base_path).resolve()
    
    # Check if the resolved path is within the base path
    try:
        requested.relative_to(base)
        return str(requested)
    except ValueError:
        raise ValueError(f"Path traversal attempt detected: {requested_path}")

def test_path_validation():
    """Test the path validation function"""
    logger.info("🔒 Testing path traversal protection...")
    
    # Test valid paths
    try:
        safe_path = validate_safe_path("/tmp", "/")
        logger.info(f"   ✅ Valid path accepted: {safe_path}")
    except Exception as e:
        logger.info(f"   ❌ Valid path rejected: {e}")
        return False
    
    # Test path traversal attempts
    dangerous_paths = [
        "../../etc/passwd",
        "/tmp/../../../etc/shadow",
        "../../../../root/.ssh/id_rsa",
        "/var/../../../home/user/.bashrc"
    ]
    
    blocked_count = 0
    for dangerous_path in dangerous_paths:
        try:
            result = validate_safe_path(dangerous_path, "/tmp")
            logger.info(f"   ❌ Dangerous path allowed: {dangerous_path} -> {result}")
        except ValueError as e:
            logger.info(f"   ✅ Blocked dangerous path: {dangerous_path}")
            blocked_count += 1
        except Exception as e:
            logger.error(f"   ❌ Unexpected error for {dangerous_path}: {e}")
    
    if blocked_count == len(dangerous_paths):
        logger.info("   ✅ All path traversal attempts blocked!")
        return True
    else:
        logger.info(f"   ❌ Only {blocked_count}/{len(dangerous_paths)} attacks blocked")
        return False

if __name__ == "__main__":
    success = test_path_validation()
    if success:
        logger.info("\n✅ PATH VALIDATION ULTRA-FIX VERIFIED!")
        exit(0)
    else:
        logger.info("\n❌ PATH VALIDATION NEEDS FIXING!")
        exit(1)