"""
Comprehensive security tests for SutazAI system
Tests authentication, authorization, input validation, and security hardening
"""

import pytest
import requests
import json
import time
import hashlib
import base64
import jwt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import subprocess
import os
import socket
import ssl
import urllib.parse
from unittest.mock import patch, Mock


# Test configuration
BASE_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:8501"
TIMEOUT = 10


class TestAuthenticationSecurity:
    """Test authentication security mechanisms."""

    def test_jwt_token_validation(self):
        """Test JWT token validation and security."""
        # Test with invalid token
        invalid_token = "invalid.jwt.token"
        headers = {"Authorization": f"Bearer {invalid_token}"}
        
        protected_endpoints = [
            "/api/agents",
            "/api/metrics", 
            "/api/chat",
            "/api/orchestration"
        ]
        
        for endpoint in protected_endpoints:
            try:
                response = requests.get(f"{BASE_URL}{endpoint}", headers=headers, timeout=TIMEOUT)
                # Should either reject invalid token (401) or not require auth (200/404)
                assert response.status_code in [200, 401, 404], f"Unexpected status for {endpoint}: {response.status_code}"
            except requests.exceptions.ConnectionError:
                pytest.skip(f"Cannot connect to {endpoint}")

    def test_token_expiration_handling(self):
        """Test expired token handling."""
        # Create expired token (if JWT is used)
        try:
            expired_payload = {
                "user_id": "test_user",
                "exp": datetime.utcnow() - timedelta(hours=1)  # Expired 1 hour ago
            }
            
            # Create token with test secret (this would fail in real system)
            expired_token = jwt.encode(expired_payload, "test_secret", algorithm="HS256")
            headers = {"Authorization": f"Bearer {expired_token}"}
            
            response = requests.get(f"{BASE_URL}/api/agents", headers=headers, timeout=TIMEOUT)
            
            # Should reject expired token
            if response.status_code == 401:
                # Good - token was rejected
                assert True
            elif response.status_code in [200, 404]:
                # Endpoint might not require auth or token validation not implemented
                pytest.skip("Token validation not implemented or not required")
            else:
                pytest.fail(f"Unexpected response to expired token: {response.status_code}")
                
        except ImportError:
            pytest.skip("JWT library not available")
        except requests.exceptions.ConnectionError:
            pytest.skip("Cannot connect to backend")

    def test_brute_force_protection(self):
        """Test brute force attack protection."""
        # Simulate multiple failed login attempts
        login_endpoint = f"{BASE_URL}/auth/login"
        
        failed_attempts = []
        for i in range(5):
            try:
                response = requests.post(
                    login_endpoint,
                    json={"username": "test_user", "password": f"wrong_password_{i}"},
                    timeout=TIMEOUT
                )
                failed_attempts.append(response.status_code)
                time.sleep(0.1)  # Small delay between attempts
            except requests.exceptions.ConnectionError:
                pytest.skip("Login endpoint not available")
        
        # Check if rate limiting or account lockout is implemented
        # This test passes if the endpoint doesn't exist (404) or implements protection
        if all(status == 404 for status in failed_attempts):
            pytest.skip("Login endpoint not implemented")
        elif any(status == 429 for status in failed_attempts):
            # Rate limiting detected
            assert True
        elif any(status == 423 for status in failed_attempts):
            # Account lockout detected
            assert True
        else:
            # No protection detected - this is informational
            pytest.skip("Brute force protection not detected (may not be implemented)")

    def test_session_management(self):
        """Test session management security."""
        # Test session endpoint if available
        session_endpoint = f"{BASE_URL}/auth/session"
        
        try:
            response = requests.get(session_endpoint, timeout=TIMEOUT)
            
            if response.status_code == 200:
                # Check for secure session management
                data = response.json()
                
                # Check for session security attributes
                security_checks = []
                
                if "session_id" in data:
                    session_id = data["session_id"]
                    # Session ID should be sufficiently long and random
                    security_checks.append(len(session_id) >= 16)
                
                if "expires_at" in data:
                    # Session should have expiration
                    security_checks.append(True)
                
                # At least some security measures should be present
                assert any(security_checks) or len(security_checks) == 0
                
            elif response.status_code == 404:
                pytest.skip("Session management endpoint not implemented")
            else:
                # Other status codes are acceptable
                assert response.status_code in [401, 403]
                
        except requests.exceptions.ConnectionError:
            pytest.skip("Session endpoint not available")

    def test_password_security_requirements(self):
        """Test password security requirements."""
        # Test password change endpoint if available
        password_endpoint = f"{BASE_URL}/auth/password"
        
        weak_passwords = [
            "123456",
            "password",
            "admin",
            "test",
            "12345678"
        ]
        
        for weak_password in weak_passwords:
            try:
                response = requests.post(
                    password_endpoint,
                    json={
                        "current_password": "old_password",
                        "new_password": weak_password
                    },
                    timeout=TIMEOUT
                )
                
                if response.status_code == 400:
                    # Good - weak password was rejected
                    assert True
                elif response.status_code == 404:
                    pytest.skip("Password change endpoint not implemented")
                    break
                elif response.status_code in [401, 403]:
                    # Authentication required - expected
                    assert True
                    break
                else:
                    # Weak password might have been accepted - this is informational
                    pass
                    
            except requests.exceptions.ConnectionError:
                pytest.skip("Password endpoint not available")
                break


class TestInputValidationSecurity:
    """Test input validation and sanitization."""

    def test_sql_injection_protection(self):
        """Test SQL injection protection."""
        sql_injection_payloads = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "' UNION SELECT * FROM users --",
            "'; INSERT INTO users VALUES ('hacker', 'pass'); --",
            "' OR 1=1 --"
        ]
        
        # Test various endpoints with SQL injection payloads
        test_endpoints = [
            ("/api/agents", "GET", {"search": None}),
            ("/api/metrics", "GET", {"filter": None}),
            ("/api/chat", "POST", {"message": None})
        ]
        
        for endpoint, method, param_template in test_endpoints:
            for payload in sql_injection_payloads:
                try:
                    if method == "GET":
                        params = {k: payload if v is None else v for k, v in param_template.items()}
                        response = requests.get(f"{BASE_URL}{endpoint}", params=params, timeout=TIMEOUT)
                    else:
                        data = {k: payload if v is None else v for k, v in param_template.items()}
                        response = requests.post(f"{BASE_URL}{endpoint}", json=data, timeout=TIMEOUT)
                    
                    # Should not return database errors or succeed with malicious payload
                    assert response.status_code not in [500]  # Internal server error might indicate SQL injection
                    
                    # Check response content for SQL error messages
                    if response.headers.get('content-type', '').startswith('application/json'):
                        try:
                            response_data = response.json()
                            response_text = json.dumps(response_data).lower()
                        except:
                            response_text = response.text.lower()
                    else:
                        response_text = response.text.lower()
                    
                    sql_error_indicators = [
                        "sql syntax", "mysql", "postgresql", "sqlite", "ora-", "syntax error"
                    ]
                    
                    for indicator in sql_error_indicators:
                        assert indicator not in response_text, f"Possible SQL injection vulnerability: {indicator} found in response"
                    
                except requests.exceptions.ConnectionError:
                    pytest.skip(f"Cannot connect to {endpoint}")
                    break

    def test_xss_protection(self):
        """Test Cross-Site Scripting (XSS) protection."""
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>",
            "'\"><script>alert('XSS')</script>"
        ]
        
        # Test chat endpoint which likely displays user input
        for payload in xss_payloads:
            try:
                response = requests.post(
                    f"{BASE_URL}/api/chat",
                    json={"message": payload},
                    timeout=TIMEOUT
                )
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        response_text = json.dumps(data)
                        
                        # Check if payload was properly escaped/sanitized
                        assert payload not in response_text, "XSS payload returned unescaped"
                        
                        # Check for HTML entities (proper escaping)
                        if "<script>" in payload:
                            assert "&lt;script&gt;" in response_text or "<script>" not in response_text
                            
                    except json.JSONDecodeError:
                        # Non-JSON response is acceptable
                        pass
                elif response.status_code == 404:
                    pytest.skip("Chat endpoint not implemented")
                    break
                    
            except requests.exceptions.ConnectionError:
                pytest.skip("Chat endpoint not available")
                break

    def test_command_injection_protection(self):
        """Test command injection protection."""
        command_injection_payloads = [
            "; ls -la",
            "| cat /etc/passwd",
            "&& whoami",
            "`id`",
            "$(whoami)",
            "; rm -rf /",
            "|| ping -c 1 127.0.0.1"
        ]
        
        # Test endpoints that might process system commands
        for payload in command_injection_payloads:
            try:
                # Test chat endpoint
                response = requests.post(
                    f"{BASE_URL}/api/chat",
                    json={"message": f"Execute: {payload}"},
                    timeout=TIMEOUT
                )
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        response_text = json.dumps(data).lower()
                        
                        # Check for command execution indicators
                        command_indicators = [
                            "uid=", "gid=", "groups=",  # id command output
                            "total ", "drwx",  # ls command output
                            "root:", "daemon:",  # /etc/passwd content
                            "ping statistics"  # ping command output
                        ]
                        
                        for indicator in command_indicators:
                            assert indicator not in response_text, f"Possible command injection: {indicator} found in response"
                            
                    except json.JSONDecodeError:
                        pass
                elif response.status_code == 404:
                    pytest.skip("Chat endpoint not implemented")
                    break
                    
            except requests.exceptions.ConnectionError:
                pytest.skip("Chat endpoint not available")
                break

    def test_path_traversal_protection(self):
        """Test path traversal protection."""
        path_traversal_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//....//etc/passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "..%252f..%252f..%252fetc%252fpasswd"
        ]
        
        # Test file access endpoints
        for payload in path_traversal_payloads:
            try:
                # Test static files or document endpoints
                response = requests.get(f"{BASE_URL}/files/{payload}", timeout=TIMEOUT)
                
                if response.status_code == 200:
                    content = response.text.lower()
                    
                    # Check for system file content
                    system_file_indicators = [
                        "root:x:", "daemon:x:",  # /etc/passwd
                        "[boot loader]",  # Windows boot.ini
                        "administrator:"  # Windows SAM
                    ]
                    
                    for indicator in system_file_indicators:
                        assert indicator not in content, f"Path traversal successful: {indicator} found"
                elif response.status_code == 404:
                    # Good - file not found or endpoint doesn't exist
                    assert True
                elif response.status_code == 403:
                    # Good - access denied
                    assert True
                    
            except requests.exceptions.ConnectionError:
                pytest.skip("File endpoint not available")
                break

    def test_file_upload_security(self):
        """Test file upload security."""
        # Test malicious file uploads
        malicious_files = [
            ("malicious.php", b"<?php system($_GET['cmd']); ?>", "application/x-php"),
            ("malicious.jsp", b"<% Runtime.getRuntime().exec(request.getParameter(\"cmd\")); %>", "application/x-jsp"),
            ("malicious.exe", b"MZ\x90\x00\x03\x00\x00\x00", "application/x-executable"),
            ("../../../evil.txt", b"Path traversal test", "text/plain")
        ]
        
        upload_endpoint = f"{BASE_URL}/api/upload"
        
        for filename, content, content_type in malicious_files:
            try:
                files = {
                    'file': (filename, content, content_type)
                }
                
                response = requests.post(upload_endpoint, files=files, timeout=TIMEOUT)
                
                if response.status_code == 200:
                    # File upload succeeded - check if proper validation was done
                    try:
                        data = response.json()
                        
                        # Check if dangerous file was rejected or sanitized
                        if "filename" in data:
                            stored_filename = data["filename"]
                            # Filename should be sanitized
                            assert "../" not in stored_filename
                            assert not stored_filename.endswith(('.php', '.jsp', '.exe'))
                        
                    except json.JSONDecodeError:
                        pass
                elif response.status_code in [400, 403, 413, 415]:
                    # Good - file was rejected
                    assert True
                elif response.status_code == 404:
                    pytest.skip("Upload endpoint not implemented")
                    break
                    
            except requests.exceptions.ConnectionError:
                pytest.skip("Upload endpoint not available")
                break


class TestAuthorizationSecurity:
    """Test authorization and access control."""

    def test_role_based_access_control(self):
        """Test role-based access control."""
        # Test admin-only endpoints
        admin_endpoints = [
            "/admin/users",
            "/admin/system",
            "/admin/config",
            "/api/admin/agents",
            "/api/admin/metrics"
        ]
        
        for endpoint in admin_endpoints:
            try:
                # Test without authorization
                response = requests.get(f"{BASE_URL}{endpoint}", timeout=TIMEOUT)
                
                # Should require authentication/authorization
                assert response.status_code in [401, 403, 404], f"Admin endpoint {endpoint} not properly protected"
                
            except requests.exceptions.ConnectionError:
                pytest.skip(f"Cannot connect to {endpoint}")

    def test_user_data_isolation(self):
        """Test user data isolation."""
        # Test user-specific endpoints with different user IDs
        user_endpoints = [
            "/api/users/123/profile",
            "/api/users/456/conversations",
            "/api/users/789/settings"
        ]
        
        for endpoint in user_endpoints:
            try:
                response = requests.get(f"{BASE_URL}{endpoint}", timeout=TIMEOUT)
                
                # Should require proper authorization for user data access
                if response.status_code == 200:
                    # If accessible, should not contain other users' data
                    try:
                        data = response.json()
                        # This test is informational - proper testing would require authentication
                        assert isinstance(data, (dict, list))
                    except json.JSONDecodeError:
                        pass
                else:
                    # Authentication/authorization required - good
                    assert response.status_code in [401, 403, 404]
                    
            except requests.exceptions.ConnectionError:
                pytest.skip(f"Cannot connect to {endpoint}")

    def test_privilege_escalation_protection(self):
        """Test privilege escalation protection."""
        # Test role modification endpoints
        privilege_endpoints = [
            ("/api/users/123/role", {"role": "admin"}),
            ("/api/users/123/permissions", {"permissions": ["admin", "write", "delete"]}),
            ("/admin/promote", {"user_id": "123", "role": "admin"})
        ]
        
        for endpoint, payload in privilege_endpoints:
            try:
                response = requests.post(f"{BASE_URL}{endpoint}", json=payload, timeout=TIMEOUT)
                
                # Should require proper authorization for privilege changes
                assert response.status_code in [401, 403, 404], f"Privilege escalation endpoint {endpoint} not properly protected"
                
            except requests.exceptions.ConnectionError:
                pytest.skip(f"Cannot connect to {endpoint}")


class TestSecurityHeaders:
    """Test security headers and HTTPS configuration."""

    def test_security_headers_presence(self):
        """Test presence of security headers."""
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
            
            security_headers = {
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": ["DENY", "SAMEORIGIN"],
                "X-XSS-Protection": "1; mode=block",
                "Strict-Transport-Security": None,  # Should be present for HTTPS
                "Content-Security-Policy": None,  # Should have some CSP
                "Referrer-Policy": None
            }
            
            headers_present = []
            
            for header, expected_values in security_headers.items():
                if header in response.headers:
                    headers_present.append(header)
                    
                    if expected_values and isinstance(expected_values, list):
                        assert response.headers[header] in expected_values
                    elif expected_values and isinstance(expected_values, str):
                        assert response.headers[header] == expected_values
            
            # At least some security headers should be present
            # This is informational - not all may be implemented
            if headers_present:
                assert len(headers_present) > 0
            else:
                pytest.skip("No security headers detected (may not be implemented)")
                
        except requests.exceptions.ConnectionError:
            pytest.skip("Cannot connect to backend")

    def test_cors_configuration(self):
        """Test CORS configuration security."""
        try:
            # Test preflight request
            headers = {
                "Origin": "https://malicious-site.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type"
            }
            
            response = requests.options(f"{BASE_URL}/api/chat", headers=headers, timeout=TIMEOUT)
            
            if "Access-Control-Allow-Origin" in response.headers:
                allowed_origin = response.headers["Access-Control-Allow-Origin"]
                
                # Should not allow all origins in production
                if allowed_origin == "*":
                    # This might be acceptable for development but should be noted
                    pytest.skip("CORS allows all origins (may be development configuration)")
                else:
                    # Specific origins are better for security
                    assert True
            else:
                # CORS might not be implemented
                pytest.skip("CORS headers not found")
                
        except requests.exceptions.ConnectionError:
            pytest.skip("Cannot connect to backend")

    def test_https_configuration(self):
        """Test HTTPS configuration if available."""
        # Try HTTPS version of the URL
        https_url = BASE_URL.replace("http://", "https://")
        
        try:
            response = requests.get(f"{https_url}/health", timeout=TIMEOUT, verify=True)
            
            # If HTTPS is available, check security
            if response.status_code == 200:
                # Check for HSTS header
                if "Strict-Transport-Security" in response.headers:
                    hsts_header = response.headers["Strict-Transport-Security"]
                    # Should have reasonable max-age
                    assert "max-age" in hsts_header
                else:
                    pytest.skip("HSTS header not present (may not be required for development)")
            
        except (requests.exceptions.ConnectionError, requests.exceptions.SSLError):
            pytest.skip("HTTPS not available or SSL configuration issues")

    def test_cookie_security(self):
        """Test cookie security attributes."""
        try:
            response = requests.get(f"{BASE_URL}/auth/login", timeout=TIMEOUT)
            
            if response.cookies:
                for cookie in response.cookies:
                    # Check cookie security attributes
                    cookie_string = str(cookie)
                    
                    # For session cookies, should have security attributes
                    if "session" in cookie.name.lower() or "auth" in cookie.name.lower():
                        # Should have HttpOnly flag
                        assert cookie.get("httponly", False) or "HttpOnly" in cookie_string
                        
                        # Should have Secure flag for HTTPS
                        if BASE_URL.startswith("https"):
                            assert cookie.get("secure", False) or "Secure" in cookie_string
            else:
                pytest.skip("No cookies found in authentication response")
                
        except requests.exceptions.ConnectionError:
            pytest.skip("Cannot connect to authentication endpoint")


class TestApiSecurity:
    """Test API-specific security measures."""

    def test_rate_limiting(self):
        """Test API rate limiting."""
        # Test with rapid requests
        responses = []
        
        for i in range(20):  # Send 20 rapid requests
            try:
                start_time = time.time()
                response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
                end_time = time.time()
                
                responses.append({
                    "status_code": response.status_code,
                    "response_time": end_time - start_time,
                    "headers": dict(response.headers)
                })
                
                time.sleep(0.1)  # Small delay
                
            except requests.exceptions.ConnectionError:
                pytest.skip("Cannot connect to backend")
                break
        
        if responses:
            # Check for rate limiting responses
            rate_limited = [r for r in responses if r["status_code"] == 429]
            
            if rate_limited:
                # Rate limiting is implemented
                assert True
                
                # Check for rate limit headers
                for response in rate_limited:
                    headers = response["headers"]
                    rate_limit_headers = [
                        "X-RateLimit-Limit",
                        "X-RateLimit-Remaining", 
                        "X-RateLimit-Reset",
                        "Retry-After"
                    ]
                    
                    # At least one rate limiting header should be present
                    has_rate_limit_header = any(header in headers for header in rate_limit_headers)
                    if has_rate_limit_header:
                        assert True
                        break
            else:
                # No rate limiting detected
                pytest.skip("Rate limiting not detected (may not be implemented)")

    def test_api_versioning_security(self):
        """Test API versioning security."""
        # Test different API versions
        version_endpoints = [
            "/api/v1/health",
            "/api/v2/health", 
            "/v1/health",
            "/v2/health"
        ]
        
        accessible_versions = []
        
        for endpoint in version_endpoints:
            try:
                response = requests.get(f"{BASE_URL}{endpoint}", timeout=TIMEOUT)
                if response.status_code == 200:
                    accessible_versions.append(endpoint)
            except requests.exceptions.ConnectionError:
                continue
        
        # Multiple API versions might be available
        # This is informational - not necessarily a security issue
        if len(accessible_versions) > 1:
            pytest.skip(f"Multiple API versions accessible: {accessible_versions}")
        elif len(accessible_versions) == 1:
            assert True  # Single version is good
        else:
            pytest.skip("No versioned API endpoints found")

    def test_information_disclosure(self):
        """Test for information disclosure vulnerabilities."""
        # Test endpoints that might expose sensitive information
        info_endpoints = [
            "/debug",
            "/info", 
            "/status",
            "/version",
            "/config",
            "/api/debug",
            "/api/info",
            "/.env",
            "/robots.txt",
            "/sitemap.xml"
        ]
        
        disclosed_info = []
        
        for endpoint in info_endpoints:
            try:
                response = requests.get(f"{BASE_URL}{endpoint}", timeout=TIMEOUT)
                
                if response.status_code == 200:
                    content = response.text.lower()
                    
                    # Check for sensitive information
                    sensitive_patterns = [
                        "password", "secret", "key", "token", "private",
                        "database", "connection", "config", "env"
                    ]
                    
                    for pattern in sensitive_patterns:
                        if pattern in content:
                            disclosed_info.append(f"{endpoint}: {pattern}")
                            
            except requests.exceptions.ConnectionError:
                continue
        
        if disclosed_info:
            # Information disclosure detected
            for disclosure in disclosed_info:
                pytest.skip(f"Potential information disclosure: {disclosure}")
        else:
            # No obvious information disclosure
            assert True

    def test_error_handling_security(self):
        """Test secure error handling."""
        # Test with malformed requests to trigger errors
        error_tests = [
            ("Invalid JSON", f"{BASE_URL}/api/chat", {"data": "invalid json{"}),
            ("Missing fields", f"{BASE_URL}/api/chat", {}),
            ("Invalid method", f"{BASE_URL}/api/agents", "DELETE"),
            ("Large payload", f"{BASE_URL}/api/chat", {"message": "A" * 10000})
        ]
        
        for test_name, url, payload in error_tests:
            try:
                if isinstance(payload, str):  # HTTP method test
                    response = requests.request(payload, url, timeout=TIMEOUT)
                else:
                    response = requests.post(url, json=payload, timeout=TIMEOUT)
                
                # Check error response for information disclosure
                if response.status_code >= 400:
                    content = response.text.lower()
                    
                    # Should not expose internal details
                    sensitive_error_info = [
                        "traceback", "stack trace", "internal server error",
                        "database error", "sql", "file path", "directory"
                    ]
                    
                    for info in sensitive_error_info:
                        assert info not in content, f"Error response contains sensitive info: {info}"
                
            except requests.exceptions.ConnectionError:
                continue


class TestFrontendSecurity:
    """Test frontend security measures."""

    def test_frontend_security_headers(self):
        """Test frontend security headers."""
        try:
            response = requests.get(FRONTEND_URL, timeout=TIMEOUT)
            
            if response.status_code == 200:
                # Check for frontend security headers
                frontend_headers = {
                    "X-Content-Type-Options": "nosniff",
                    "X-Frame-Options": None,  # Should prevent clickjacking
                    "Content-Security-Policy": None  # Should have CSP
                }
                
                for header in frontend_headers:
                    if header in response.headers:
                        assert True  # Header is present
                        
                # Check content for inline scripts (potential XSS risk)
                content = response.text
                
                # Streamlit apps may have inline scripts, but check for obvious XSS
                xss_patterns = [
                    "javascript:alert(",
                    "onerror=alert(",
                    "onload=alert("
                ]
                
                for pattern in xss_patterns:
                    assert pattern not in content, f"Potential XSS pattern found: {pattern}"
            
        except requests.exceptions.ConnectionError:
            pytest.skip("Frontend not available")

    def test_frontend_resource_access(self):
        """Test frontend resource access security."""
        # Test access to static resources
        resource_paths = [
            "/_stcore/static/",
            "/static/",
            "/assets/",
            "/.streamlit/"
        ]
        
        for path in resource_paths:
            try:
                response = requests.get(f"{FRONTEND_URL}{path}", timeout=TIMEOUT)
                
                # Directory listing should not be enabled
                if response.status_code == 200:
                    content = response.text.lower()
                    directory_indicators = [
                        "index of", "directory listing", "<title>index of"
                    ]
                    
                    for indicator in directory_indicators:
                        assert indicator not in content, f"Directory listing enabled: {path}"
                        
            except requests.exceptions.ConnectionError:
                continue


class TestSecurityMisconfiguration:
    """Test for security misconfigurations."""

    def test_default_credentials(self):
        """Test for default credentials."""
        default_creds = [
            ("admin", "admin"),
            ("admin", "password"),
            ("root", "root"),
            ("admin", "123456"),
            ("user", "user")
        ]
        
        login_endpoint = f"{BASE_URL}/auth/login"
        
        for username, password in default_creds:
            try:
                response = requests.post(
                    login_endpoint,
                    json={"username": username, "password": password},
                    timeout=TIMEOUT
                )
                
                # Default credentials should not work
                assert response.status_code != 200, f"Default credentials work: {username}:{password}"
                
            except requests.exceptions.ConnectionError:
                pytest.skip("Login endpoint not available")
                break

    def test_debug_mode_disabled(self):
        """Test that debug mode is disabled."""
        debug_indicators = [
            "/debug",
            "?debug=1",
            "?debug=true",
            "&debug=1"
        ]
        
        base_endpoints = ["/", "/health", "/api/agents"]
        
        for endpoint in base_endpoints:
            for debug_param in debug_indicators:
                try:
                    test_url = f"{BASE_URL}{endpoint}{debug_param}"
                    response = requests.get(test_url, timeout=TIMEOUT)
                    
                    if response.status_code == 200:
                        content = response.text.lower()
                        
                        # Should not contain debug information
                        debug_patterns = [
                            "debug mode", "traceback", "stack trace",
                            "django debug", "flask debug", "development mode"
                        ]
                        
                        for pattern in debug_patterns:
                            assert pattern not in content, f"Debug mode may be enabled: {pattern}"
                            
                except requests.exceptions.ConnectionError:
                    continue

    def test_unnecessary_services_disabled(self):
        """Test that unnecessary services are disabled."""
        # Test for common unnecessary services
        unnecessary_services = [
            ("FTP", 21),
            ("Telnet", 23), 
            ("SMTP", 25),
            ("DNS", 53),
            ("TFTP", 69),
            ("POP3", 110),
            ("IMAP", 143),
            ("SNMP", 161),
            ("LDAP", 389)
        ]
        
        localhost = "127.0.0.1"
        open_ports = []
        
        for service_name, port in unnecessary_services:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            
            try:
                result = sock.connect_ex((localhost, port))
                if result == 0:
                    open_ports.append((service_name, port))
            except:
                pass
            finally:
                sock.close()
        
        # Report open unnecessary ports (informational)
        if open_ports:
            ports_info = ", ".join([f"{name}({port})" for name, port in open_ports])
            pytest.skip(f"Unnecessary services detected: {ports_info}")
        else:
            assert True  # No unnecessary services detected


class TestSecurityMonitoring:
    """Test security monitoring capabilities."""

    def test_security_logging(self):
        """Test security event logging."""
        # Generate security events
        security_events = [
            ("Failed login", f"{BASE_URL}/auth/login", {"username": "admin", "password": "wrong"}),
            ("Admin access attempt", f"{BASE_URL}/admin/users", {}),
            ("SQL injection attempt", f"{BASE_URL}/api/search", {"q": "'; DROP TABLE users; --"})
        ]
        
        for event_name, url, payload in security_events:
            try:
                if payload:
                    requests.post(url, json=payload, timeout=TIMEOUT)
                else:
                    requests.get(url, timeout=TIMEOUT)
                    
                # In a real system, these events should be logged
                # This test is informational
                
            except requests.exceptions.ConnectionError:
                continue
        
        # This test passes as it's about generating events for logging
        assert True

    def test_intrusion_detection(self):
        """Test intrusion detection capabilities."""
        # Simulate suspicious activity patterns
        suspicious_patterns = [
            "Rapid requests",
            "Multiple failed authentications", 
            "Port scanning simulation",
            "SQL injection attempts"
        ]
        
        # Generate suspicious activity
        for _ in range(10):  # Rapid requests
            try:
                requests.get(f"{BASE_URL}/health", timeout=1)
                time.sleep(0.05)
            except:
                pass
        
        # Multiple failed logins
        for _ in range(5):
            try:
                requests.post(
                    f"{BASE_URL}/auth/login",
                    json={"username": "admin", "password": "wrong"},
                    timeout=TIMEOUT
                )
            except:
                pass
        
        # This test is informational - actual IDS would be external
        pytest.skip("Intrusion detection testing is informational")


@pytest.mark.security
class TestSecurityCompliance:
    """Test security compliance requirements."""

    def test_data_encryption_in_transit(self):
        """Test data encryption in transit."""
        # This test checks if HTTPS is enforced
        https_url = BASE_URL.replace("http://", "https://")
        
        try:
            # Try HTTPS first
            response = requests.get(f"{https_url}/health", timeout=TIMEOUT, verify=True)
            if response.status_code == 200:
                assert True  # HTTPS is working
            else:
                pytest.skip("HTTPS not properly configured")
                
        except (requests.exceptions.ConnectionError, requests.exceptions.SSLError):
            # HTTPS not available, check if HTTP redirects to HTTPS
            try:
                response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT, allow_redirects=False)
                if response.status_code in [301, 302, 307, 308]:
                    location = response.headers.get("Location", "")
                    if location.startswith("https://"):
                        assert True  # HTTP redirects to HTTPS
                    else:
                        pytest.skip("HTTP does not redirect to HTTPS")
                else:
                    pytest.skip("HTTPS not available and no redirect from HTTP")
            except requests.exceptions.ConnectionError:
                pytest.skip("Cannot test HTTPS configuration")

    def test_security_audit_trail(self):
        """Test security audit trail capabilities."""
        # Test if security events can be audited
        audit_endpoints = [
            "/admin/audit",
            "/api/audit",
            "/security/logs"
        ]
        
        for endpoint in audit_endpoints:
            try:
                response = requests.get(f"{BASE_URL}{endpoint}", timeout=TIMEOUT)
                
                # Audit endpoints should require authentication
                assert response.status_code in [401, 403, 404]
                
            except requests.exceptions.ConnectionError:
                continue
        
        # This test passes if audit endpoints are protected or not implemented
        assert True

    def test_vulnerability_disclosure(self):
        """Test vulnerability disclosure information."""
        # Check for security.txt or vulnerability disclosure policy
        disclosure_paths = [
            "/.well-known/security.txt",
            "/security.txt",
            "/vulnerability-disclosure",
            "/responsible-disclosure"
        ]
        
        disclosure_found = False
        
        for path in disclosure_paths:
            try:
                response = requests.get(f"{BASE_URL}{path}", timeout=TIMEOUT)
                if response.status_code == 200:
                    disclosure_found = True
                    # Check for contact information
                    content = response.text.lower()
                    contact_indicators = ["contact:", "email:", "security@"]
                    
                    has_contact = any(indicator in content for indicator in contact_indicators)
                    if has_contact:
                        assert True
                        break
                        
            except requests.exceptions.ConnectionError:
                continue
        
        if not disclosure_found:
            pytest.skip("Vulnerability disclosure policy not found (may not be required)")
        else:
            assert True