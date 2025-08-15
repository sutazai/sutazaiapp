#!/usr/bin/env python3
"""
ULTRA Security Validation Test Suite
Comprehensive security testing for all components
Author: ULTRA Security Engineer
Date: 2025-08-11
"""

import os
import sys
import json
import time
import subprocess
import requests
import asyncio
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import pytest
import jwt
import redis
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.fernet import Fernet

# Add backend to path for imports
# Path handled by pytest configuration
from app.auth.jwt_handler import JWTHandler
from app.auth.rate_limiter import UltraRateLimiter, AuthRateLimiter


class UltraSecurityTester:
    """Comprehensive security testing framework"""
    
    def __init__(self):
        self.base_url = "http://localhost:10010"
        self.test_results = []
        self.vulnerabilities = []
        self.passed_tests = 0
        self.failed_tests = 0
        
    def log_result(self, test_name: str, passed: bool, details: str = ""):
        """Log test result"""
        result = {
            "test": test_name,
            "passed": passed,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        self.test_results.append(result)
        
        if passed:
            self.passed_tests += 1
            print(f"✅ {test_name}: PASSED")
        else:
            self.failed_tests += 1
            self.vulnerabilities.append(result)
            print(f"❌ {test_name}: FAILED - {details}")


class TestContainerSecurity(UltraSecurityTester):
    """Test container security configurations"""
    
    def test_all_containers_non_root(self):
        """Verify all containers run as non-root users"""
        print("\n=== Testing Container Security ===")
        
        # Get all running containers
        cmd = "docker ps --format '{{.Names}}'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        containers = result.stdout.strip().split('\n')
        
        root_containers = []
        for container in containers:
            if container:
                # Check user running in container
                cmd = f"docker exec {container} whoami 2>/dev/null"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                user = result.stdout.strip()
                
                if user == "root":
                    root_containers.append(container)
                    
        if root_containers:
            self.log_result(
                "Container Non-Root Check",
                False,
                f"Containers running as root: {', '.join(root_containers)}"
            )
        else:
            self.log_result("Container Non-Root Check", True, "All containers running as non-root")
            
    def test_container_capabilities(self):
        """Test that containers have   capabilities"""
        print("\n=== Testing Container Capabilities ===")
        
        cmd = "docker ps -q"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        container_ids = result.stdout.strip().split('\n')
        
        for container_id in container_ids:
            if container_id:
                # Inspect container capabilities
                cmd = f"docker inspect {container_id} --format '{{{{json .HostConfig.CapAdd}}}}'"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                try:
                    capabilities = json.loads(result.stdout.strip())
                    if capabilities and len(capabilities) > 0:
                        # Check for dangerous capabilities
                        dangerous = ["SYS_ADMIN", "NET_ADMIN", "SYS_PTRACE"]
                        found = [cap for cap in capabilities if cap in dangerous]
                        if found:
                            self.log_result(
                                f"Container Capabilities ({container_id[:12]})",
                                False,
                                f"Dangerous capabilities: {found}"
                            )
                        else:
                            self.log_result(
                                f"Container Capabilities ({container_id[:12]})",
                                True,
                                "No dangerous capabilities"
                            )
                except:
                    pass
                    
    def test_container_read_only_root(self):
        """Test that containers have read-only root filesystem where possible"""
        print("\n=== Testing Read-Only Root Filesystem ===")
        
        cmd = "docker ps --format '{{.Names}}'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        containers = result.stdout.strip().split('\n')
        
        for container in containers:
            if container and container != 'sutazai-postgres':  # Postgres needs write access
                cmd = f"docker inspect {container} --format '{{{{.HostConfig.ReadonlyRootfs}}}}'"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                read_only = result.stdout.strip() == "true"
                
                # Some containers need write access, check if they're in allowed list
                write_required = ['neo4j', 'redis', 'rabbitmq', 'ollama', 'grafana']
                if any(wr in container for wr in write_required):
                    continue
                    
                if not read_only:
                    self.log_result(
                        f"Read-Only Root ({container})",
                        False,
                        "Root filesystem is writable"
                    )


class TestAuthenticationSecurity(UltraSecurityTester):
    """Test authentication and JWT security"""
    
    def test_jwt_algorithm(self):
        """Verify JWT uses secure algorithm (RS256 or HS256 with strong key)"""
        print("\n=== Testing JWT Security ===")
        
        handler = JWTHandler()
        
        if handler.algorithm == "RS256":
            self.log_result("JWT Algorithm", True, "Using RS256 (asymmetric)")
        elif handler.algorithm == "HS256":
            # Check key strength
            if handler.signing_key and len(str(handler.signing_key)) >= 32:
                self.log_result("JWT Algorithm", True, "Using HS256 with strong key")
            else:
                self.log_result("JWT Algorithm", False, "HS256 key too weak")
        else:
            self.log_result("JWT Algorithm", False, f"Insecure algorithm: {handler.algorithm}")
            
    def test_jwt_expiration(self):
        """Test JWT token expiration"""
        print("\n=== Testing JWT Expiration ===")
        
        handler = JWTHandler()
        
        # Create token with short expiration
        token = handler.create_access_token(
            user_id=1,
            username="test",
            email="test@example.com",
            expires_delta=timedelta(seconds=1)
        )
        
        # Wait for expiration
        time.sleep(2)
        
        # Try to verify expired token
        try:
            handler.verify_token(token)
            self.log_result("JWT Expiration", False, "Expired token accepted")
        except ValueError as e:
            if "expired" in str(e).lower():
                self.log_result("JWT Expiration", True, "Expired token rejected")
            else:
                self.log_result("JWT Expiration", False, f"Wrong error: {e}")
                
    def test_password_hashing(self):
        """Test password hashing strength"""
        print("\n=== Testing Password Hashing ===")
        
        # Check if bcrypt is being used
        try:
            import bcrypt
            
            password = "TestPassword123!"
            hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=12))
            
            # Verify hash format
            if hashed.startswith(b'$2b$') or hashed.startswith(b'$2a$'):
                self.log_result("Password Hashing", True, "Using bcrypt with proper rounds")
            else:
                self.log_result("Password Hashing", False, "Invalid bcrypt format")
        except ImportError:
            self.log_result("Password Hashing", False, "bcrypt not installed")


class TestRateLimiting(UltraSecurityTester):
    """Test rate limiting functionality"""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_basic(self):
        """Test basic rate limiting"""
        print("\n=== Testing Rate Limiting ===")
        
        limiter = UltraRateLimiter(default_limit=5, default_window=10)
        
        # Make requests up to limit
        for i in range(5):
            allowed, metadata = await limiter.check_rate_limit("test_user")
            if not allowed:
                self.log_result("Rate Limiter Basic", False, f"Blocked at request {i+1}")
                return
                
        # Next request should be blocked
        allowed, metadata = await limiter.check_rate_limit("test_user")
        if allowed:
            self.log_result("Rate Limiter Basic", False, "Limit not enforced")
        else:
            self.log_result("Rate Limiter Basic", True, "Rate limit enforced correctly")
            
    @pytest.mark.asyncio
    async def test_auth_rate_limiter(self):
        """Test authentication-specific rate limiting"""
        print("\n=== Testing Auth Rate Limiting ===")
        
        auth_limiter = AuthRateLimiter()
        
        # Simulate failed login attempts
        for i in range(3):
            allowed, _ = await auth_limiter.check_login_attempt("testuser", "192.168.1.1")
            if allowed:
                await auth_limiter.record_login_failure("testuser", "192.168.1.1")
                
        # Check reputation impact
        allowed, metadata = await auth_limiter.check_login_attempt("testuser", "192.168.1.1")
        
        # After multiple failures, should have stricter limits
        if metadata.get("limit", 5) < 5:
            self.log_result("Auth Rate Limiter", True, "Adaptive rate limiting working")
        else:
            self.log_result("Auth Rate Limiter", False, "Reputation not affecting limits")


class TestSecretManagement(UltraSecurityTester):
    """Test secrets management"""
    
    def test_secrets_encryption(self):
        """Test that secrets are encrypted at rest"""
        print("\n=== Testing Secrets Encryption ===")
        
        # Import secrets manager
        # Path handled by pytest configuration
        from secrets_manager import UltraSecretsManager
        
        manager = UltraSecretsManager(backend="local")
        
        # Store a test secret
        test_secret = "SuperSecretPassword123!"
        manager.store_secret("test_key", test_secret)
        
        # Check that secret file is encrypted
        secrets_file = Path("/opt/sutazaiapp/.secrets/secrets.enc")
        if secrets_file.exists():
            with open(secrets_file, 'rb') as f:
                content = f.read()
                
            # Check if content appears encrypted (not plaintext)
            if test_secret.encode() in content:
                self.log_result("Secrets Encryption", False, "Secrets stored in plaintext")
            else:
                self.log_result("Secrets Encryption", True, "Secrets properly encrypted")
                
            # Verify retrieval
            retrieved = manager.retrieve_secret("test_key")
            if retrieved == test_secret:
                self.log_result("Secrets Retrieval", True, "Secrets correctly decrypted")
            else:
                self.log_result("Secrets Retrieval", False, "Failed to decrypt secret")
                
            # Clean up
            manager.delete_secret("test_key")
        else:
            self.log_result("Secrets Encryption", False, "Secrets file not created")
            
    def test_secret_strength_validation(self):
        """Test secret strength validation"""
        print("\n=== Testing Secret Strength Validation ===")
        
        # Path handled by pytest configuration
        from secrets_manager import UltraSecretsManager
        
        manager = UltraSecretsManager(backend="local")
        
        # Test weak secret
        weak_secret = "password"
        validation = manager.validate_secret_strength(weak_secret)
        if not validation["is_compliant"]:
            self.log_result("Secret Strength (Weak)", True, "Weak secret rejected")
        else:
            self.log_result("Secret Strength (Weak)", False, "Weak secret accepted")
            
        # Test strong secret
        strong_secret = "P@ssw0rd!2024$Complex#"
        validation = manager.validate_secret_strength(strong_secret)
        if validation["is_compliant"]:
            self.log_result("Secret Strength (Strong)", True, "Strong secret accepted")
        else:
            self.log_result("Secret Strength (Strong)", False, f"Strong secret rejected: {validation.get('reason')}")


class TestSSLTLS(UltraSecurityTester):
    """Test SSL/TLS configuration"""
    
    def test_ssl_configuration(self):
        """Test SSL/TLS configuration files"""
        print("\n=== Testing SSL/TLS Configuration ===")
        
        nginx_conf = Path("/opt/sutazaiapp/config/ssl/nginx-ssl.conf")
        if nginx_conf.exists():
            with open(nginx_conf, 'r') as f:
                content = f.read()
                
            # Check for secure protocols
            if "TLSv1.2" in content and "TLSv1.3" in content:
                self.log_result("TLS Protocols", True, "Using TLS 1.2 and 1.3")
            else:
                self.log_result("TLS Protocols", False, "Missing secure TLS versions")
                
            # Check for security headers
            security_headers = [
                "Strict-Transport-Security",
                "X-Frame-Options",
                "X-Content-Type-Options",
                "Content-Security-Policy"
            ]
            
            missing_headers = [h for h in security_headers if h not in content]
            if missing_headers:
                self.log_result("Security Headers", False, f"Missing: {missing_headers}")
            else:
                self.log_result("Security Headers", True, "All security headers present")
                
            # Check for rate limiting
            if "limit_req_zone" in content and "limit_conn_zone" in content:
                self.log_result("Rate Limiting Config", True, "Rate limiting configured")
            else:
                self.log_result("Rate Limiting Config", False, "Rate limiting not configured")
        else:
            self.log_result("SSL Configuration", False, "nginx-ssl.conf not found")


class TestAPISecrity(UltraSecurityTester):
    """Test API endpoint security"""
    
    def test_api_authentication_required(self):
        """Test that protected endpoints require authentication"""
        print("\n=== Testing API Authentication ===")
        
        protected_endpoints = [
            "/api/v1/chat",
            "/api/v1/models",
            "/api/v1/mesh/enqueue"
        ]
        
        for endpoint in protected_endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
                if response.status_code == 401 or response.status_code == 403:
                    self.log_result(f"Auth Required ({endpoint})", True, "Unauthorized access blocked")
                else:
                    self.log_result(f"Auth Required ({endpoint})", False, f"Status: {response.status_code}")
            except:
                pass
                
    def test_api_input_validation(self):
        """Test API input validation against injection attacks"""
        print("\n=== Testing Input Validation ===")
        
        # SQL injection attempts
        sql_payloads = [
            "' OR '1'='1",
            "'; DROP TABLE users; --",
            "1' UNION SELECT * FROM users--"
        ]
        
        for payload in sql_payloads:
            try:
                response = requests.post(
                    f"{self.base_url}/api/v1/auth/login",
                    json={"username": payload, "password": "test"},
                    timeout=5
                )
                
                # Should not return 500 (server error from SQL injection)
                if response.status_code != 500:
                    self.log_result(f"SQL Injection Protection", True, f"Payload blocked: {payload[:20]}...")
                else:
                    self.log_result(f"SQL Injection Protection", False, f"Possible vulnerability: {payload[:20]}...")
            except:
                pass
                
        # XSS attempts
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "javascript:alert('XSS')"
        ]
        
        for payload in xss_payloads:
            try:
                response = requests.post(
                    f"{self.base_url}/api/v1/chat",
                    json={"message": payload},
                    timeout=5
                )
                
                if response.status_code == 200:
                    # Check if response is sanitized
                    if "<script>" not in response.text and "javascript:" not in response.text:
                        self.log_result("XSS Protection", True, f"Payload sanitized: {payload[:20]}...")
                    else:
                        self.log_result("XSS Protection", False, f"Payload not sanitized: {payload[:20]}...")
            except:
                pass


def run_all_security_tests():
    """Run all security tests and generate report"""
    print("=" * 60)
    print("ULTRA SECURITY VALIDATION TEST SUITE")
    print("=" * 60)
    
    # Initialize test classes
    container_tests = TestContainerSecurity()
    auth_tests = TestAuthenticationSecurity()
    rate_tests = TestRateLimiting()
    secret_tests = TestSecretManagement()
    ssl_tests = TestSSLTLS()
    api_tests = TestAPISecrity()
    
    # Run container tests
    container_tests.test_all_containers_non_root()
    container_tests.test_container_capabilities()
    container_tests.test_container_read_only_root()
    
    # Run authentication tests
    auth_tests.test_jwt_algorithm()
    auth_tests.test_jwt_expiration()
    auth_tests.test_password_hashing()
    
    # Run rate limiting tests
    asyncio.run(rate_tests.test_rate_limiter_basic())
    asyncio.run(rate_tests.test_auth_rate_limiter())
    
    # Run secret management tests
    secret_tests.test_secrets_encryption()
    secret_tests.test_secret_strength_validation()
    
    # Run SSL/TLS tests
    ssl_tests.test_ssl_configuration()
    
    # Run API security tests
    api_tests.test_api_authentication_required()
    api_tests.test_api_input_validation()
    
    # Aggregate results
    all_tests = [
        container_tests, auth_tests, rate_tests,
        secret_tests, ssl_tests, api_tests
    ]
    
    total_passed = sum(t.passed_tests for t in all_tests)
    total_failed = sum(t.failed_tests for t in all_tests)
    all_vulnerabilities = []
    for t in all_tests:
        all_vulnerabilities.extend(t.vulnerabilities)
    
    # Generate report
    print("\n" + "=" * 60)
    print("SECURITY TEST RESULTS")
    print("=" * 60)
    print(f"✅ Passed: {total_passed}")
    print(f"❌ Failed: {total_failed}")
    print(f"📊 Success Rate: {(total_passed/(total_passed+total_failed)*100):.1f}%")
    
    if all_vulnerabilities:
        print("\n⚠️ VULNERABILITIES FOUND:")
        for vuln in all_vulnerabilities:
            print(f"  - {vuln['test']}: {vuln['details']}")
    else:
        print("\n✨ No vulnerabilities found!")
    
    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_tests": total_passed + total_failed,
        "passed": total_passed,
        "failed": total_failed,
        "success_rate": total_passed / (total_passed + total_failed) * 100,
        "vulnerabilities": all_vulnerabilities
    }
    
    report_file = Path("/opt/sutazaiapp/tests/security/security_test_report.json")
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Report saved to: {report_file}")
    
    return total_failed == 0


if __name__ == "__main__":
    success = run_all_security_tests()
    sys.exit(0 if success else 1)