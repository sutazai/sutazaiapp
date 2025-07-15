#!/usr/bin/env python3.11
"""Tests for the security module."""

import pytest
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List
from pathlib import Path

    encrypt_data,
    decrypt_data,
    hash_password,
    verify_password,
    generate_token,
    verify_token,
    TokenManager,
    KeyManager,
    SecurityManager,
    SecurityError,
)


@pytest.fixture
def key_manager():
    """Create a test key manager."""
    return KeyManager()


@pytest.fixture
def token_manager():
    """Create a test token manager."""
    return TokenManager()


@pytest.fixture
def security_manager():
    """Create a test security manager."""
    return SecurityManager()


def test_encryption_decryption():
    """Test data encryption and decryption."""
    # Test string encryption
    data = "sensitive data"
    encrypted = encrypt_data(data)
    decrypted = decrypt_data(encrypted)
    assert decrypted == data

    # Test dictionary encryption
    data_dict = {"key": "value", "number": 42}
    encrypted = encrypt_data(data_dict)
    decrypted = decrypt_data(encrypted)
    assert decrypted == data_dict

    # Test empty data
    encrypted = encrypt_data("")
    decrypted = decrypt_data(encrypted)
    assert decrypted == ""

    # Test invalid encrypted data
    with pytest.raises(SecurityError):
        decrypt_data("invalid encrypted data")


def test_password_hashing():
    """Test password hashing and verification."""
    # Test password hashing
    password = "test_password"
    hashed = hash_password(password)
    assert hashed != password
    assert len(hashed) > 0

    # Test password verification
    assert verify_password(password, hashed) is True
    assert verify_password("wrong_password", hashed) is False

    # Test empty password
    hashed = hash_password("")
    assert verify_password("", hashed) is True
    assert verify_password("wrong", hashed) is False


def test_token_generation_verification():
    """Test token generation and verification."""
    # Test token generation
    payload = {"user_id": "123", "role": "admin"}
    token = generate_token(payload)
    assert len(token) > 0

    # Test token verification
    verified = verify_token(token)
    assert verified["user_id"] == "123"
    assert verified["role"] == "admin"

    # Test invalid token
    with pytest.raises(SecurityError):
        verify_token("invalid.token.here")

    # Test expired token
    expired_payload = {
        "user_id": "123",
        "exp": datetime.utcnow() - timedelta(hours=1),
    }
    expired_token = generate_token(expired_payload)
    with pytest.raises(SecurityError):
        verify_token(expired_token)


def test_key_manager(key_manager):
    """Test key manager functionality."""
    # Test key generation
    key = key_manager.generate_key()
    assert len(key) > 0

    # Test key storage
    key_manager.store_key("test_key", key)
    assert key_manager.get_key("test_key") == key

    # Test key rotation
    new_key = key_manager.rotate_key("test_key")
    assert new_key != key
    assert key_manager.get_key("test_key") == new_key

    # Test key deletion
    key_manager.delete_key("test_key")
    assert key_manager.get_key("test_key") is None


def test_token_manager(token_manager):
    """Test token manager functionality."""
    # Test token creation
    token = token_manager.create_token(
        user_id="123",
        roles=["admin"],
        expires_in=3600,
    )
    assert len(token) > 0

    # Test token validation
    payload = token_manager.validate_token(token)
    assert payload["user_id"] == "123"
    assert "admin" in payload["roles"]

    # Test token refresh
    new_token = token_manager.refresh_token(token)
    assert new_token != token
    assert token_manager.validate_token(new_token)["user_id"] == "123"

    # Test token revocation
    token_manager.revoke_token(token)
    with pytest.raises(SecurityError):
        token_manager.validate_token(token)


def test_security_manager(security_manager):
    """Test security manager functionality."""
    # Test secure file operations
    test_file = Path("test_secure.txt")
    data = "sensitive data"

    # Test secure write
    security_manager.secure_write(test_file, data)
    assert test_file.exists()

    # Test secure read
    read_data = security_manager.secure_read(test_file)
    assert read_data == data

    # Test secure delete
    security_manager.secure_delete(test_file)
    assert not test_file.exists()

    # Test secure environment variables
    security_manager.set_secure_env("TEST_KEY", "test_value")
    assert os.environ["TEST_KEY"] == "test_value"
    security_manager.remove_secure_env("TEST_KEY")
    assert "TEST_KEY" not in os.environ


def test_security_validation():
    """Test security validation functionality."""
    # Test password strength validation
    assert security_manager.validate_password_strength("StrongP@ss123") is True
    assert security_manager.validate_password_strength("weak") is False

    # Test input sanitization
    sanitized = security_manager.sanitize_input("<script>alert('xss')</script>")
    assert "<script>" not in sanitized

    # Test URL validation
    assert security_manager.validate_url("https://example.com") is True
    assert security_manager.validate_url("javascript:alert('xss')") is False

    # Test file path validation
    assert security_manager.validate_file_path("/safe/path/file.txt") is True
    assert security_manager.validate_file_path("../../../etc/passwd") is False


def test_security_audit():
    """Test security audit functionality."""
    # Test audit log creation
    security_manager.log_security_event(
        event_type="login",
        user_id="123",
        details={"ip": "192.168.1.1"},
    )

    # Test audit log retrieval
    events = security_manager.get_security_events(
        user_id="123",
        event_type="login",
    )
    assert len(events) > 0
    assert events[0]["user_id"] == "123"
    assert events[0]["event_type"] == "login"

    # Test audit log cleanup
    security_manager.cleanup_audit_logs(days=0)
    events = security_manager.get_security_events()
    assert len(events) == 0


def test_security_configuration():
    """Test security configuration functionality."""
    # Test security policy configuration
    policy = {
        "password_min_length": 12,
        "require_special_chars": True,
        "max_login_attempts": 5,
        "session_timeout": 3600,
    }
    security_manager.configure_security_policy(policy)

    # Test policy enforcement
    assert security_manager.validate_password_strength("Short") is False
    assert security_manager.validate_password_strength("StrongP@ss123") is True

    # Test session management
    session = security_manager.create_session("123")
    assert security_manager.validate_session(session) is True
    security_manager.invalidate_session(session)
    assert security_manager.validate_session(session) is False
