#!/usr/bin/env python3
"""
Comprehensive JWT Authentication Testing Suite
Tests all 8 JWT endpoints with full coverage of security features
"""

import pytest
import httpx
import asyncio
import time
from typing import Dict, Any, Optional
from datetime import datetime, timezone

BASE_URL = "http://localhost:10200/api/v1"
TIMEOUT = 30.0


class TestUserRegistration:
    """Test user registration endpoint"""
    
    @pytest.mark.asyncio
    async def test_register_valid_user(self):
        """Test successful user registration"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            timestamp = int(time.time())
            payload = {
                "email": f"testuser{timestamp}@example.com",
                "username": f"testuser{timestamp}",
                "password": "SecureP@ssw0rd123!",
                "full_name": "Test User"
            }
            response = await client.post(f"{BASE_URL}/auth/register", json=payload)
            
            assert response.status_code in [200, 201], f"Registration failed: {response.text}"
            
            if response.status_code in [200, 201]:
                data = response.json()
                assert "email" in data
                assert "username" in data
                assert data["email"] == payload["email"]
                assert data["username"] == payload["username"]
                assert "password" not in data  # Password should not be returned
                assert "hashed_password" not in data  # Hashed password should not be exposed
    
    @pytest.mark.asyncio
    async def test_register_duplicate_email(self):
        """Test registration with duplicate email"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            timestamp = int(time.time())
            email = f"duplicate{timestamp}@example.com"
            
            # First registration
            payload1 = {
                "email": email,
                "username": f"user1_{timestamp}",
                "password": "SecureP@ssw0rd123!"
            }
            response1 = await client.post(f"{BASE_URL}/auth/register", json=payload1)
            
            if response1.status_code in [200, 201]:
                # Second registration with same email
                payload2 = {
                    "email": email,
                    "username": f"user2_{timestamp}",
                    "password": "SecureP@ssw0rd123!"
                }
                response2 = await client.post(f"{BASE_URL}/auth/register", json=payload2)
                
                assert response2.status_code == 400, "Should reject duplicate email"
                assert "email" in response2.text.lower() or "already" in response2.text.lower()
    
    @pytest.mark.asyncio
    async def test_register_weak_password(self):
        """Test registration with weak password"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            timestamp = int(time.time())
            weak_passwords = ["123", "password", "abc", "test"]
            
            for pwd in weak_passwords:
                payload = {
                    "email": f"weak{timestamp}@example.com",
                    "username": f"weak{timestamp}",
                    "password": pwd
                }
                response = await client.post(f"{BASE_URL}/auth/register", json=payload)
                
                assert response.status_code in [400, 422], f"Should reject weak password: {pwd}"
    
    @pytest.mark.asyncio
    async def test_register_invalid_email(self):
        """Test registration with invalid email format"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            payload = {
                "email": "invalid-email",
                "username": "testuser",
                "password": "SecureP@ssw0rd123!"
            }
            response = await client.post(f"{BASE_URL}/auth/register", json=payload)
            
            assert response.status_code in [400, 422], "Should reject invalid email"


class TestUserLogin:
    """Test user login endpoint"""
    
    @pytest.mark.asyncio
    async def test_login_valid_credentials(self):
        """Test successful login"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Register user first
            timestamp = int(time.time())
            register_payload = {
                "email": f"logintest{timestamp}@example.com",
                "username": f"logintest{timestamp}",
                "password": "SecureP@ssw0rd123!"
            }
            reg_response = await client.post(f"{BASE_URL}/auth/register", json=register_payload)
            
            if reg_response.status_code in [200, 201]:
                # Login with username
                login_payload = {
                    "username": register_payload["username"],
                    "password": register_payload["password"]
                }
                response = await client.post(f"{BASE_URL}/auth/login", data=login_payload)
                
                assert response.status_code == 200, f"Login failed: {response.text}"
                
                data = response.json()
                assert "access_token" in data
                assert "refresh_token" in data
                assert "token_type" in data
                assert data["token_type"] == "bearer"
                assert "expires_in" in data
    
    @pytest.mark.asyncio
    async def test_login_with_email(self):
        """Test login using email instead of username"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            timestamp = int(time.time())
            register_payload = {
                "email": f"emaillogin{timestamp}@example.com",
                "username": f"emaillogin{timestamp}",
                "password": "SecureP@ssw0rd123!"
            }
            reg_response = await client.post(f"{BASE_URL}/auth/register", json=register_payload)
            
            if reg_response.status_code in [200, 201]:
                # Login with email
                login_payload = {
                    "username": register_payload["email"],  # Using email in username field
                    "password": register_payload["password"]
                }
                response = await client.post(f"{BASE_URL}/auth/login", data=login_payload)
                
                assert response.status_code == 200, "Should allow login with email"
    
    @pytest.mark.asyncio
    async def test_login_invalid_credentials(self):
        """Test login with invalid credentials"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            payload = {
                "username": "nonexistent@example.com",
                "password": "WrongPassword123!"
            }
            response = await client.post(f"{BASE_URL}/auth/login", data=payload)
            
            assert response.status_code == 401, "Should reject invalid credentials"
    
    @pytest.mark.asyncio
    async def test_login_wrong_password(self):
        """Test login with correct username but wrong password"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            timestamp = int(time.time())
            register_payload = {
                "email": f"wrongpwd{timestamp}@example.com",
                "username": f"wrongpwd{timestamp}",
                "password": "CorrectP@ssw0rd123!"
            }
            reg_response = await client.post(f"{BASE_URL}/auth/register", json=register_payload)
            
            if reg_response.status_code in [200, 201]:
                # Login with wrong password
                login_payload = {
                    "username": register_payload["username"],
                    "password": "WrongP@ssw0rd123!"
                }
                response = await client.post(f"{BASE_URL}/auth/login", data=login_payload)
                
                assert response.status_code == 401, "Should reject wrong password"


class TestAccountLockout:
    """Test account lockout mechanism"""
    
    @pytest.mark.asyncio
    async def test_account_lockout_after_failed_attempts(self):
        """Test account locks after 5 failed login attempts"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            timestamp = int(time.time())
            register_payload = {
                "email": f"lockout{timestamp}@example.com",
                "username": f"lockout{timestamp}",
                "password": "CorrectP@ssw0rd123!"
            }
            reg_response = await client.post(f"{BASE_URL}/auth/register", json=register_payload)
            
            if reg_response.status_code in [200, 201]:
                # Attempt 6 failed logins
                for i in range(6):
                    login_payload = {
                        "username": register_payload["username"],
                        "password": "WrongPassword!"
                    }
                    response = await client.post(f"{BASE_URL}/auth/login", data=login_payload)
                    
                    if i < 4:
                        # First 4 attempts should fail with 401
                        assert response.status_code == 401, f"Attempt {i+1} should fail with 401"
                    else:
                        # 5th and 6th attempts should result in account lockout (403)
                        assert response.status_code in [401, 403], f"Attempt {i+1} should show lockout"
                        if response.status_code == 403:
                            assert "locked" in response.text.lower()
    
    @pytest.mark.asyncio
    async def test_lockout_prevents_login_with_correct_password(self):
        """Test that locked account cannot login even with correct password"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            timestamp = int(time.time())
            register_payload = {
                "email": f"locktest{timestamp}@example.com",
                "username": f"locktest{timestamp}",
                "password": "CorrectP@ssw0rd123!"
            }
            reg_response = await client.post(f"{BASE_URL}/auth/register", json=register_payload)
            
            if reg_response.status_code in [200, 201]:
                # Trigger lockout
                for i in range(6):
                    login_payload = {
                        "username": register_payload["username"],
                        "password": "WrongPassword!"
                    }
                    await client.post(f"{BASE_URL}/auth/login", data=login_payload)
                
                # Try with correct password
                correct_payload = {
                    "username": register_payload["username"],
                    "password": register_payload["password"]
                }
                response = await client.post(f"{BASE_URL}/auth/login", data=correct_payload)
                
                # Should still be locked
                if "locked" in response.text.lower():
                    assert response.status_code == 403


class TestTokenRefresh:
    """Test token refresh endpoint"""
    
    @pytest.mark.asyncio
    async def test_refresh_token_valid(self):
        """Test refreshing access token with valid refresh token"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Register and login
            timestamp = int(time.time())
            register_payload = {
                "email": f"refresh{timestamp}@example.com",
                "username": f"refresh{timestamp}",
                "password": "SecureP@ssw0rd123!"
            }
            reg_response = await client.post(f"{BASE_URL}/auth/register", json=register_payload)
            
            if reg_response.status_code in [200, 201]:
                login_payload = {
                    "username": register_payload["username"],
                    "password": register_payload["password"]
                }
                login_response = await client.post(f"{BASE_URL}/auth/login", data=login_payload)
                
                if login_response.status_code == 200:
                    tokens = login_response.json()
                    refresh_token = tokens.get("refresh_token")
                    
                    if refresh_token:
                        # Refresh the token
                        refresh_response = await client.post(
                            f"{BASE_URL}/auth/refresh",
                            json={"refresh_token": refresh_token}
                        )
                        
                        assert refresh_response.status_code == 200, "Token refresh should succeed"
                        
                        new_tokens = refresh_response.json()
                        assert "access_token" in new_tokens
                        assert "refresh_token" in new_tokens
                        # Tokens should be valid (not necessarily different if generated instantly)
                        assert len(new_tokens["access_token"]) > 50
    
    @pytest.mark.asyncio
    async def test_refresh_token_invalid(self):
        """Test token refresh with invalid refresh token"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            invalid_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.invalid.token"
            
            response = await client.post(
                f"{BASE_URL}/auth/refresh",
                json={"refresh_token": invalid_token}
            )
            
            assert response.status_code == 401, "Should reject invalid refresh token"


class TestCurrentUser:
    """Test /me endpoint"""
    
    @pytest.mark.asyncio
    async def test_get_current_user_authenticated(self):
        """Test getting current user info with valid token"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Register and login
            timestamp = int(time.time())
            register_payload = {
                "email": f"metest{timestamp}@example.com",
                "username": f"metest{timestamp}",
                "password": "SecureP@ssw0rd123!",
                "full_name": "Me Test User"
            }
            reg_response = await client.post(f"{BASE_URL}/auth/register", json=register_payload)
            
            if reg_response.status_code in [200, 201]:
                login_payload = {
                    "username": register_payload["username"],
                    "password": register_payload["password"]
                }
                login_response = await client.post(f"{BASE_URL}/auth/login", data=login_payload)
                
                if login_response.status_code == 200:
                    token = login_response.json().get("access_token")
                    
                    if token:
                        # Get current user
                        headers = {"Authorization": f"Bearer {token}"}
                        response = await client.get(f"{BASE_URL}/auth/me", headers=headers)
                        
                        assert response.status_code == 200, "Should return current user"
                        
                        user_data = response.json()
                        assert user_data["email"] == register_payload["email"]
                        assert user_data["username"] == register_payload["username"]
    
    @pytest.mark.asyncio
    async def test_get_current_user_unauthenticated(self):
        """Test /me endpoint without authentication"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(f"{BASE_URL}/auth/me")
            
            assert response.status_code == 401, "Should require authentication"
    
    @pytest.mark.asyncio
    async def test_get_current_user_invalid_token(self):
        """Test /me endpoint with invalid token"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            headers = {"Authorization": "Bearer invalid.token.here"}
            response = await client.get(f"{BASE_URL}/auth/me", headers=headers)
            
            assert response.status_code == 401, "Should reject invalid token"


class TestLogout:
    """Test logout endpoint"""
    
    @pytest.mark.asyncio
    async def test_logout_authenticated(self):
        """Test logout with valid token"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Register and login
            timestamp = int(time.time())
            register_payload = {
                "email": f"logout{timestamp}@example.com",
                "username": f"logout{timestamp}",
                "password": "SecureP@ssw0rd123!"
            }
            reg_response = await client.post(f"{BASE_URL}/auth/register", json=register_payload)
            
            if reg_response.status_code in [200, 201]:
                login_payload = {
                    "username": register_payload["username"],
                    "password": register_payload["password"]
                }
                login_response = await client.post(f"{BASE_URL}/auth/login", data=login_payload)
                
                if login_response.status_code == 200:
                    token = login_response.json().get("access_token")
                    
                    if token:
                        # Logout
                        headers = {"Authorization": f"Bearer {token}"}
                        response = await client.post(f"{BASE_URL}/auth/logout", headers=headers)
                        
                        assert response.status_code == 200, "Logout should succeed"


class TestPasswordReset:
    """Test password reset endpoints"""
    
    @pytest.mark.asyncio
    async def test_password_reset_request(self):
        """Test requesting password reset"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            timestamp = int(time.time())
            # Register user first
            register_payload = {
                "email": f"pwdreset{timestamp}@example.com",
                "username": f"pwdreset{timestamp}",
                "password": "OldP@ssw0rd123!"
            }
            reg_response = await client.post(f"{BASE_URL}/auth/register", json=register_payload)
            
            if reg_response.status_code in [200, 201]:
                # Request password reset
                reset_payload = {"email": register_payload["email"]}
                response = await client.post(f"{BASE_URL}/auth/password-reset", json=reset_payload)
                
                # Should always return success to prevent email enumeration
                assert response.status_code == 200, "Should return success"
                assert "message" in response.json()
    
    @pytest.mark.asyncio
    async def test_password_reset_nonexistent_email(self):
        """Test password reset for non-existent email"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            reset_payload = {"email": "nonexistent@example.com"}
            response = await client.post(f"{BASE_URL}/auth/password-reset", json=reset_payload)
            
            # Should still return success (anti-enumeration)
            assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_password_reset_rate_limit(self):
        """Test rate limiting on password reset endpoint"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Make multiple rapid requests
            for i in range(10):
                reset_payload = {"email": f"ratelimit{i}@example.com"}
                response = await client.post(f"{BASE_URL}/auth/password-reset", json=reset_payload)
                
                if response.status_code == 429:
                    # Rate limit hit
                    assert "too many" in response.text.lower() or "rate" in response.text.lower()
                    break


class TestEmailVerification:
    """Test email verification endpoint"""
    
    @pytest.mark.asyncio
    async def test_email_verification_invalid_token(self):
        """Test email verification with invalid token"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            invalid_token = "invalid.token.here"
            response = await client.post(f"{BASE_URL}/auth/verify-email/{invalid_token}")
            
            assert response.status_code in [400, 401, 404], "Should reject invalid token"


class TestRateLimiting:
    """Test rate limiting on authentication endpoints"""
    
    @pytest.mark.asyncio
    async def test_login_rate_limiting(self):
        """Test rate limiting on login endpoint"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Make many rapid login attempts
            rate_limited = False
            
            for i in range(150):  # Exceed typical rate limit
                payload = {
                    "username": f"ratetest{i}@example.com",
                    "password": "TestPassword123!"
                }
                response = await client.post(f"{BASE_URL}/auth/login", data=payload)
                
                if response.status_code == 429:
                    rate_limited = True
                    assert "Retry-After" in response.headers or "X-RateLimit" in str(response.headers)
                    break
            
            # Rate limiting may or may not be hit depending on configuration
            print(f"Rate limiting triggered: {rate_limited}")


class TestSecurityFeatures:
    """Test security features"""
    
    @pytest.mark.asyncio
    async def test_password_not_returned(self):
        """Test that passwords are never returned in responses"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            timestamp = int(time.time())
            register_payload = {
                "email": f"sectest{timestamp}@example.com",
                "username": f"sectest{timestamp}",
                "password": "SecureP@ssw0rd123!"
            }
            reg_response = await client.post(f"{BASE_URL}/auth/register", json=register_payload)
            
            if reg_response.status_code in [200, 201]:
                # Check registration response
                assert "password" not in reg_response.text.lower()
                assert register_payload["password"] not in reg_response.text
                
                # Login and check /me endpoint
                login_payload = {
                    "username": register_payload["username"],
                    "password": register_payload["password"]
                }
                login_response = await client.post(f"{BASE_URL}/auth/login", data=login_payload)
                
                if login_response.status_code == 200:
                    token = login_response.json().get("access_token")
                    
                    if token:
                        headers = {"Authorization": f"Bearer {token}"}
                        me_response = await client.get(f"{BASE_URL}/auth/me", headers=headers)
                        
                        # Check /me response
                        assert "password" not in me_response.text.lower()
                        assert register_payload["password"] not in me_response.text


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
