#!/usr/bin/env python3
"""
Comprehensive authentication testing script
Tests all JWT authentication endpoints and functionality
"""

import asyncio
import httpx
import json
from datetime import datetime
from typing import Optional, Dict, Any
import os
import sys
from pathlib import Path

# Add backend directory to path for imports
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy import select, text

# Import our models and utilities
from app.models.user import User
from app.core.database import Base, engine, async_session_maker
from app.core.config import settings

# Test configuration
BASE_URL = f"http://localhost:{settings.PORT}/api/v1"
TEST_USER = {
    "email": "test@example.com",
    "username": "testuser",
    "full_name": "Test User",
    "password": "TestPassword123!"
}

class AuthTester:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.access_token = None
        self.refresh_token = None
        
    async def setup_database(self):
        """Initialize database and create tables"""
        print("\n🔧 Setting up database...")
        try:
            # Create all tables
            async with engine.begin() as conn:
                # Drop existing tables for clean test
                await conn.run_sync(Base.metadata.drop_all)
                await conn.run_sync(Base.metadata.create_all)
            print("✅ Database tables created successfully")
            
            # Verify tables exist
            async with async_session_maker() as session:
                result = await session.execute(text(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
                ))
                tables = [row[0] for row in result]
                print(f"📊 Tables created: {tables}")
                
        except Exception as e:
            print(f"❌ Database setup failed: {e}")
            raise
    
    async def test_health(self):
        """Test health endpoint"""
        print("\n🏥 Testing health endpoint...")
        try:
            response = await self.client.get(f"{BASE_URL.replace('/api/v1', '')}/health")
            assert response.status_code == 200
            data = response.json()
            print(f"✅ Health check passed: {data}")
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            raise
    
    async def test_register(self):
        """Test user registration"""
        print(f"\n👤 Testing registration...")
        try:
            response = await self.client.post(
                f"{BASE_URL}/auth/register",
                json=TEST_USER
            )
            
            if response.status_code == 201:
                data = response.json()
                print(f"✅ User registered successfully: {data['username']} (ID: {data['id']})")
                return data
            else:
                print(f"❌ Registration failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Registration error: {e}")
            raise
    
    async def test_login(self):
        """Test user login"""
        print(f"\n🔑 Testing login...")
        try:
            # OAuth2 compatible login (form data)
            response = await self.client.post(
                f"{BASE_URL}/auth/login",
                data={
                    "username": TEST_USER["username"],
                    "password": TEST_USER["password"]
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                self.access_token = data["access_token"]
                self.refresh_token = data["refresh_token"]
                print(f"✅ Login successful!")
                print(f"   - Access token: {self.access_token[:20]}...")
                print(f"   - Token type: {data['token_type']}")
                print(f"   - Expires in: {data['expires_in']} seconds")
                return data
            else:
                print(f"❌ Login failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Login error: {e}")
            raise
    
    async def test_get_current_user(self):
        """Test getting current user info"""
        print("\n👤 Testing get current user...")
        
        if not self.access_token:
            print("⚠️ No access token available, logging in first...")
            await self.test_login()
        
        try:
            response = await self.client.get(
                f"{BASE_URL}/auth/me",
                headers={"Authorization": f"Bearer {self.access_token}"}
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Current user retrieved successfully:")
                print(f"   - Username: {data['username']}")
                print(f"   - Email: {data['email']}")
                print(f"   - Active: {data['is_active']}")
                return data
            else:
                print(f"❌ Failed to get current user: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Get current user error: {e}")
    
    async def cleanup(self):
        """Clean up resources"""
        await self.client.aclose()
        await engine.dispose()
    
    async def run_all_tests(self):
        """Run all authentication tests"""
        print("\n" + "="*60)
        print("🚀 STARTING JWT AUTHENTICATION TESTS")
        print("="*60)
        
        try:
            # Setup
            await self.setup_database()
            
            # Health check
            await self.test_health()
            
            # Registration tests
            await self.test_register()
            
            # Login tests
            await self.test_login()
            
            # Protected endpoint tests
            await self.test_get_current_user()
            
            print("\n" + "="*60)
            print("✅ BASIC TESTS COMPLETED SUCCESSFULLY!")
            print("="*60)
            
        except Exception as e:
            print(f"\n❌ TEST SUITE FAILED: {e}")
            raise
        finally:
            await self.cleanup()


async def main():
    """Main entry point"""
    tester = AuthTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    print(f"🔧 Using database: {settings.DATABASE_URL}")
    print(f"🌐 Testing API at: {BASE_URL}")
    asyncio.run(main())
