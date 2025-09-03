#!/usr/bin/env python3
"""
Test authentication endpoints
"""
import requests
import json
import sys

BASE_URL = "http://localhost:10200"

def test_register():
    """Test user registration"""
    url = f"{BASE_URL}/api/v1/auth/register"
    data = {
        "email": "test@example.com",
        "username": "testuser",
        "full_name": "Test User",
        "password": "SecurePassword123!",
        "is_active": True,
        "is_superuser": False
    }
    
    try:
        response = requests.post(url, json=data)
        print(f"Registration Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 201
    except Exception as e:
        print(f"Registration failed: {e}")
        return False

def test_login():
    """Test user login"""
    url = f"{BASE_URL}/api/v1/auth/login"
    data = {
        "username": "testuser",
        "password": "SecurePassword123!",
        "grant_type": "password"
    }
    
    try:
        # OAuth2 expects form data, not JSON
        response = requests.post(url, data=data)
        print(f"\nLogin Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            token_data = response.json()
            return token_data.get("access_token")
        return None
    except Exception as e:
        print(f"Login failed: {e}")
        return None

def test_me(token):
    """Test getting current user info"""
    url = f"{BASE_URL}/api/v1/auth/me"
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.get(url, headers=headers)
        print(f"\nGet Me Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Get me failed: {e}")
        return False

def main():
    print("Testing Authentication Endpoints")
    print("=" * 50)
    
    # Test registration
    print("\n1. Testing Registration...")
    registered = test_register()
    
    # Test login
    print("\n2. Testing Login...")
    token = test_login()
    
    if token:
        print(f"\nReceived token: {token[:20]}...")
        
        # Test getting user info
        print("\n3. Testing Get Current User...")
        test_me(token)
    else:
        print("\nLogin failed, skipping authenticated endpoints")

if __name__ == "__main__":
    main()