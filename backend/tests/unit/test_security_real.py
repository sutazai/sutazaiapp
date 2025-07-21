"""
Unit tests for Security Module - Real Implementation
"""
import pytest
import asyncio
import sys
import os
import jwt
import secrets
from datetime import datetime, timedelta, timezone
from cryptography.fernet import Fernet

# Add the backend directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.core.security import (
    EncryptionManager, AuthenticationManager, AuthorizationManager,
    InputValidator, AuditLogger, RateLimiter, ComplianceManager,
    SecurityManager, SecurityLevel, ComplianceStandard, AccessPolicy
)

@pytest.fixture
def encryption_manager():
    """Create real EncryptionManager instance"""
    return EncryptionManager()

@pytest.fixture
def auth_manager():
    """Create real AuthenticationManager instance"""
    return AuthenticationManager()

@pytest.fixture
def authz_manager():
    """Create real AuthorizationManager instance"""
    return AuthorizationManager()

@pytest.fixture
def input_validator():
    """Create real InputValidator instance"""
    return InputValidator()

@pytest.fixture
def audit_logger():
    """Create real AuditLogger instance"""
    return AuditLogger()

@pytest.fixture
def rate_limiter():
    """Create real RateLimiter instance"""
    return RateLimiter()

@pytest.fixture
def compliance_manager():
    """Create real ComplianceManager instance"""
    return ComplianceManager()

@pytest.fixture
def security_manager():
    """Create real SecurityManager instance"""
    return SecurityManager()

class TestEncryptionManager:
    """Test encryption functionality"""
    
    def test_encryption_key_generation(self, encryption_manager):
        """Test master key generation"""
        assert encryption_manager.master_key is not None
        assert len(encryption_manager.master_key) == 44  # Base64 encoded Fernet key length
        assert encryption_manager.fernet is not None
    
    def test_encrypt_decrypt_cycle(self, encryption_manager):
        """Test encryption and decryption of data"""
        test_data = "This is sensitive information"
        
        # Encrypt
        encrypted = encryption_manager.encrypt_data(test_data)
        assert encrypted != test_data
        assert len(encrypted) > len(test_data)
        
        # Decrypt
        decrypted = encryption_manager.decrypt_data(encrypted)
        assert decrypted == test_data
    
    def test_encrypt_empty_data(self, encryption_manager):
        """Test encryption of empty string"""
        encrypted = encryption_manager.encrypt_data("")
        assert encrypted == ""
        
        decrypted = encryption_manager.decrypt_data("")
        assert decrypted == ""
    
    def test_decrypt_invalid_data(self, encryption_manager):
        """Test decryption of invalid data"""
        with pytest.raises(ValueError, match="Invalid encrypted data"):
            encryption_manager.decrypt_data("invalid_base64_data")
    
    def test_password_hashing(self, encryption_manager):
        """Test password hashing functionality"""
        password = "MySecurePassword123!"
        
        # Hash password
        hashed, salt = encryption_manager.hash_password(password)
        
        assert hashed != password
        assert salt is not None
        assert len(hashed) == 44  # Base64 encoded hash length
        assert len(salt) == 44  # Base64 encoded salt length
        
        # Verify correct password
        assert encryption_manager.verify_password(password, hashed, salt)
        
        # Verify incorrect password
        assert not encryption_manager.verify_password("WrongPassword", hashed, salt)
    
    def test_password_hashing_with_custom_salt(self, encryption_manager):
        """Test password hashing with provided salt"""
        password = "TestPassword"
        custom_salt = secrets.token_bytes(32)
        
        hashed1, salt1 = encryption_manager.hash_password(password, custom_salt)
        hashed2, salt2 = encryption_manager.hash_password(password, custom_salt)
        
        # Same password and salt should produce same hash
        assert hashed1 == hashed2
        assert salt1 == salt2

class TestAuthenticationManager:
    """Test authentication functionality"""
    
    def test_jwt_token_creation(self, auth_manager):
        """Test JWT access token creation"""
        user_id = "test_user_123"
        scopes = ["read", "write", "admin"]
        
        token = auth_manager.create_access_token(user_id, scopes)
        
        assert isinstance(token, str)
        assert len(token) > 0
        
        # Decode and verify
        payload = jwt.decode(
            token,
            auth_manager.secret_key,
            algorithms=[auth_manager.algorithm]
        )
        
        assert payload["sub"] == user_id
        assert payload["scopes"] == scopes
        assert payload["type"] == "access"
        assert "exp" in payload
        assert "iat" in payload
    
    def test_refresh_token_creation(self, auth_manager):
        """Test JWT refresh token creation"""
        user_id = "test_user_456"
        
        token = auth_manager.create_refresh_token(user_id)
        
        payload = jwt.decode(
            token,
            auth_manager.secret_key,
            algorithms=[auth_manager.algorithm]
        )
        
        assert payload["sub"] == user_id
        assert payload["type"] == "refresh"
        assert "exp" in payload
    
    def test_token_verification(self, auth_manager):
        """Test token verification"""
        user_id = "verify_test"
        token = auth_manager.create_access_token(user_id)
        
        # Verify valid token
        payload = auth_manager.verify_token(token)
        assert payload["sub"] == user_id
        assert payload["type"] == "access"
        
        # Verify invalid token
        with pytest.raises(ValueError, match="Invalid token"):
            auth_manager.verify_token("invalid.token.here")
        
        # Verify wrong token type
        refresh_token = auth_manager.create_refresh_token(user_id)
        with pytest.raises(ValueError, match="Invalid token type"):
            auth_manager.verify_token(refresh_token, token_type="access")
    
    def test_token_expiration(self, auth_manager):
        """Test token expiration handling"""
        # Create expired token
        expired_payload = {
            "sub": "expired_user",
            "exp": datetime.now(timezone.utc) - timedelta(hours=1),
            "type": "access"
        }
        
        expired_token = jwt.encode(
            expired_payload,
            auth_manager.secret_key,
            algorithm=auth_manager.algorithm
        )
        
        with pytest.raises(ValueError, match="Token has expired"):
            auth_manager.verify_token(expired_token)
    
    @pytest.mark.asyncio
    async def test_user_authentication(self, auth_manager):
        """Test user authentication"""
        # Test valid credentials
        user = await auth_manager.authenticate_user("admin", "secure_password")
        assert user is not None
        assert user["user_id"] == "admin_001"
        assert user["role"] == "admin"
        assert "admin" in user["scopes"]
        
        # Test invalid credentials
        user = await auth_manager.authenticate_user("admin", "wrong_password")
        assert user is None
        
        user = await auth_manager.authenticate_user("unknown_user", "password")
        assert user is None

class TestAuthorizationManager:
    """Test authorization functionality"""
    
    def test_permission_check_admin(self, authz_manager):
        """Test admin permission checking"""
        admin_user = {
            "user_id": "admin_001",
            "role": "admin",
            "username": "admin"
        }
        
        # Admin should have access to everything
        assert authz_manager.check_permission(admin_user, "/api/v1/admin/users", "GET")
        assert authz_manager.check_permission(admin_user, "/api/v1/brain/think", "POST")
        assert authz_manager.check_permission(admin_user, "/api/v1/models/delete", "DELETE")
    
    def test_permission_check_user(self, authz_manager):
        """Test regular user permissions"""
        regular_user = {
            "user_id": "user_123",
            "role": "user",
            "username": "testuser"
        }
        
        # User should have read/write but not admin
        assert authz_manager.check_permission(regular_user, "/api/v1/brain/think", "GET")
        assert authz_manager.check_permission(regular_user, "/api/v1/brain/think", "POST")
        assert not authz_manager.check_permission(regular_user, "/api/v1/admin/users", "GET")
    
    def test_permission_check_viewer(self, authz_manager):
        """Test viewer permissions"""
        viewer = {
            "user_id": "viewer_001",
            "role": "viewer"
        }
        
        # Viewer should only have read permissions
        assert authz_manager.check_permission(viewer, "/api/v1/models", "GET")
        assert not authz_manager.check_permission(viewer, "/api/v1/models", "POST")
        assert not authz_manager.check_permission(viewer, "/api/v1/models", "DELETE")
    
    def test_resource_pattern_matching(self, authz_manager):
        """Test resource pattern matching"""
        # Test wildcard matching
        assert authz_manager._match_resource("/api/v1/brain/think", "/api/*")
        assert authz_manager._match_resource("/api/v1/admin/users", "/api/v1/admin/*")
        assert not authz_manager._match_resource("/health", "/api/*")
        
        # Test exact matching
        assert authz_manager._match_resource("/api/v1/brain/think", "/api/v1/brain/think")
        assert not authz_manager._match_resource("/api/v1/brain/status", "/api/v1/brain/think")

class TestInputValidator:
    """Test input validation functionality"""
    
    def test_valid_input(self, input_validator):
        """Test validation of clean input"""
        clean_inputs = [
            "This is a normal text input",
            "User123 with numbers",
            "Special chars: !@#$%^&*()",
            "Multi\nline\ntext"
        ]
        
        for input_text in clean_inputs:
            result = input_validator.validate_input(input_text)
            assert result == input_text
    
    def test_malicious_input_detection(self, input_validator):
        """Test detection of malicious input"""
        malicious_inputs = [
            "<script>alert('XSS')</script>",
            "javascript:void(0)",
            '<img src=x onerror="alert(1)">',
            "onclick=alert('XSS')",
            "eval(malicious_code)",
            "vbscript:msgbox('test')",
            "data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg=="
        ]
        
        for malicious_input in malicious_inputs:
            with pytest.raises(ValueError, match="Potentially malicious content detected"):
                input_validator.validate_input(malicious_input)
    
    def test_input_length_validation(self, input_validator):
        """Test input length validation"""
        # Valid length
        valid_input = "x" * 9999
        result = input_validator.validate_input(valid_input)
        assert result == valid_input
        
        # Exceeds max length
        long_input = "x" * 10001
        with pytest.raises(ValueError, match="Input exceeds maximum length"):
            input_validator.validate_input(long_input)
    
    def test_email_validation(self, input_validator):
        """Test email format validation"""
        # Valid emails
        valid_emails = [
            "user@example.com",
            "test.user+tag@domain.co.uk",
            "admin123@subdomain.example.org"
        ]
        
        for email in valid_emails:
            result = input_validator.validate_input(email, "email")
            assert result == email
        
        # Invalid emails
        invalid_emails = [
            "not-an-email",
            "@example.com",
            "user@",
            "user@.com",
            "user@domain",
            "user @domain.com"
        ]
        
        for email in invalid_emails:
            with pytest.raises(ValueError, match="Invalid email format"):
                input_validator.validate_input(email, "email")
    
    def test_url_validation(self, input_validator):
        """Test URL format validation"""
        # Valid URLs
        valid_urls = [
            "https://example.com",
            "http://subdomain.example.org/path",
            "https://example.com:8080/api/v1"
        ]
        
        for url in valid_urls:
            result = input_validator.validate_input(url, "url")
            assert result == url
        
        # Invalid URLs
        invalid_urls = [
            "not-a-url",
            "ftp://example.com",  # Only http/https allowed
            "https://",
            "//example.com",
            "javascript:alert(1)"
        ]
        
        for url in invalid_urls:
            with pytest.raises(ValueError, match="Invalid URL format"):
                input_validator.validate_input(url, "url")
    
    def test_json_validation(self, input_validator):
        """Test JSON format validation"""
        # Valid JSON
        valid_json = '{"key": "value", "number": 123, "array": [1, 2, 3]}'
        result = input_validator.validate_input(valid_json, "json")
        assert result == valid_json
        
        # Invalid JSON
        invalid_json = '{"key": "value", invalid}'
        with pytest.raises(ValueError, match="Invalid JSON format"):
            input_validator.validate_input(invalid_json, "json")

class TestRateLimiter:
    """Test rate limiting functionality"""
    
    @pytest.mark.asyncio
    async def test_rate_limit_basic(self, rate_limiter):
        """Test basic rate limiting"""
        identifier = "test_user_ip"
        
        # Should allow initial requests
        for _ in range(10):
            assert await rate_limiter.check_rate_limit(identifier, limit=10)
        
        # Should block after limit
        assert not await rate_limiter.check_rate_limit(identifier, limit=10)
    
    @pytest.mark.asyncio
    async def test_rate_limit_window(self, rate_limiter):
        """Test rate limit window behavior"""
        identifier = "window_test"
        rate_limiter.window_size = 1  # 1 second window for testing
        
        # Fill the limit
        for _ in range(5):
            await rate_limiter.check_rate_limit(identifier, limit=5)
        
        # Should be blocked
        assert not await rate_limiter.check_rate_limit(identifier, limit=5)
        
        # Wait for window to pass
        await asyncio.sleep(1.1)
        
        # Should be allowed again
        assert await rate_limiter.check_rate_limit(identifier, limit=5)
    
    @pytest.mark.asyncio
    async def test_multiple_identifiers(self, rate_limiter):
        """Test rate limiting with multiple identifiers"""
        identifier1 = "user1"
        identifier2 = "user2"
        
        # Fill limit for identifier1
        for _ in range(10):
            await rate_limiter.check_rate_limit(identifier1, limit=10)
        
        # identifier1 should be blocked
        assert not await rate_limiter.check_rate_limit(identifier1, limit=10)
        
        # identifier2 should still be allowed
        assert await rate_limiter.check_rate_limit(identifier2, limit=10)

class TestAuditLogger:
    """Test audit logging functionality"""
    
    @pytest.mark.asyncio
    async def test_log_event(self, audit_logger):
        """Test event logging"""
        await audit_logger.log_event(
            event_type="test_event",
            severity="info",
            source="test_suite",
            details={"test": "data"},
            user_id="test_user",
            ip_address="127.0.0.1"
        )
        
        assert len(audit_logger.events) == 1
        event = audit_logger.events[0]
        
        assert event.event_type == "test_event"
        assert event.severity == "info"
        assert event.source == "test_suite"
        assert event.details == {"test": "data"}
        assert event.user_id == "test_user"
        assert event.ip_address == "127.0.0.1"
        assert event.id.startswith("evt_")
    
    @pytest.mark.asyncio
    async def test_audit_trail_filtering(self, audit_logger):
        """Test audit trail filtering"""
        # Log various events
        await audit_logger.log_event("login", "info", "auth", {}, user_id="user1")
        await audit_logger.log_event("api_call", "info", "api", {}, user_id="user2")
        await audit_logger.log_event("error", "critical", "system", {}, user_id="user1")
        
        # Test filtering by user
        trail = await audit_logger.get_audit_trail({"user_id": "user1"})
        assert len(trail) == 2
        assert all(event["user"] == "user1" for event in trail)
        
        # Test filtering by event type
        trail = await audit_logger.get_audit_trail({"event_type": "login"})
        assert len(trail) == 1
        assert trail[0]["type"] == "login"
        
        # Test filtering by severity
        trail = await audit_logger.get_audit_trail({"severity": "critical"})
        assert len(trail) == 1
        assert trail[0]["severity"] == "critical"

class TestComplianceManager:
    """Test compliance functionality"""
    
    @pytest.mark.asyncio
    async def test_gdpr_compliance_check(self, compliance_manager):
        """Test GDPR compliance checking"""
        # Data without personal information - should pass
        non_personal_data = {
            "action": "calculate",
            "value": 123
        }
        assert await compliance_manager.check_gdpr_compliance("process", non_personal_data)
        
        # Data with personal information but with consent - should pass
        personal_data_with_consent = {
            "email": "user@example.com",
            "name": "John Doe",
            "user_consent": True
        }
        assert await compliance_manager.check_gdpr_compliance("store", personal_data_with_consent)
        
        # Data with personal information without consent - should fail
        personal_data_no_consent = {
            "email": "user@example.com",
            "name": "John Doe"
        }
        assert not await compliance_manager.check_gdpr_compliance("store", personal_data_no_consent)
    
    @pytest.mark.asyncio
    async def test_data_anonymization(self, compliance_manager):
        """Test data anonymization"""
        personal_data = {
            "email": "john.doe@example.com",
            "name": "John Doe",
            "phone": "+1234567890",
            "address": "123 Main St",
            "ip_address": "192.168.1.1",
            "user_id": "user_12345",
            "other_field": "non-personal data"
        }
        
        anonymized = await compliance_manager.anonymize_data(personal_data)
        
        # Personal fields should be hashed
        assert anonymized["email"] != personal_data["email"]
        assert anonymized["name"] != personal_data["name"]
        assert anonymized["phone"] != personal_data["phone"]
        assert anonymized["address"] != personal_data["address"]
        assert anonymized["ip_address"] != personal_data["ip_address"]
        assert anonymized["user_id"] != personal_data["user_id"]
        
        # Non-personal fields should remain unchanged
        assert anonymized["other_field"] == personal_data["other_field"]
        
        # Hashes should be consistent
        assert len(anonymized["email"]) == 12  # Truncated hash
    
    @pytest.mark.asyncio
    async def test_data_requests(self, compliance_manager):
        """Test GDPR data request handling"""
        user_id = "test_user_123"
        
        # Test access request
        access_result = await compliance_manager.handle_data_request("access", user_id)
        assert "user_id" in access_result
        assert access_result["user_id"] == user_id
        
        # Test portability request
        portability_result = await compliance_manager.handle_data_request("portability", user_id)
        assert "format" in portability_result
        assert "data" in portability_result
        
        # Test erasure request
        erasure_result = await compliance_manager.handle_data_request("erasure", user_id)
        assert erasure_result["status"] == "completed"
        assert erasure_result["user_id"] == user_id
        
        # Test invalid request type
        with pytest.raises(ValueError):
            await compliance_manager.handle_data_request("invalid_type", user_id)

class TestSecurityManager:
    """Test integrated security manager"""
    
    @pytest.mark.asyncio
    async def test_secure_request_processing(self, security_manager):
        """Test complete request security processing"""
        request_data = {
            "path": "/api/v1/brain/think",
            "method": "POST",
            "user": {
                "user_id": "test_user",
                "role": "user"
            },
            "ip": "192.168.1.100",
            "body": {
                "text": "Process this request",
                "data": "Some data"
            }
        }
        
        # Should process successfully
        result = await security_manager.secure_request(request_data)
        assert result is not None
        assert result["body"]["text"] == "Process this request"
    
    @pytest.mark.asyncio
    async def test_rate_limit_enforcement(self, security_manager):
        """Test rate limiting in security manager"""
        # Simulate many requests from same IP
        request_data = {
            "path": "/api/v1/test",
            "ip": "192.168.1.50"
        }
        
        # Fill up rate limit
        for _ in range(100):
            try:
                await security_manager.secure_request(request_data)
            except ValueError:
                break
        
        # Next request should be rate limited
        with pytest.raises(ValueError, match="Rate limit exceeded"):
            await security_manager.secure_request(request_data)
    
    def test_security_headers(self, security_manager):
        """Test security headers generation"""
        headers = security_manager.get_security_headers()
        
        required_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
            "Strict-Transport-Security",
            "Content-Security-Policy",
            "Referrer-Policy",
            "Permissions-Policy"
        ]
        
        for header in required_headers:
            assert header in headers
            assert headers[header] is not None
            assert len(headers[header]) > 0
    
    @pytest.mark.asyncio
    async def test_security_report_generation(self, security_manager):
        """Test security report generation"""
        # Log some events first
        await security_manager.audit.log_event("test1", "info", "test", {})
        await security_manager.audit.log_event("test2", "critical", "test", {})
        
        report = await security_manager.generate_security_report()
        
        assert "timestamp" in report
        assert "summary" in report
        assert report["summary"]["total_events"] >= 2
        assert report["summary"]["compliance_standards"] == ["gdpr", "soc2"]
        assert report["summary"]["encryption_enabled"] is True
        assert report["summary"]["rate_limiting_enabled"] is True
        assert "recent_alerts" in report
        assert "recommendations" in report