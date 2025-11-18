#!/usr/bin/env python3
"""
Real Authentication Integration Tests
Tests actual JWT flows with database operations - NO MOCKS
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import select, text
import asyncio
from datetime import datetime, timezone

from app.main import app
from app.core.database import Base, get_db
from app.core.config import settings
from app.models.user import User

# Test database URL (separate from production)
TEST_DATABASE_URL = settings.DATABASE_URL.replace("/jarvis_ai", "/jarvis_ai_test")

# Create test engine
test_engine = create_async_engine(
    TEST_DATABASE_URL,
    echo=False,
    pool_pre_ping=True
)

test_session_maker = async_sessionmaker(
    test_engine,
    class_=AsyncSession,
    expire_on_commit=False
)


async def get_test_db():
    """Override database dependency for tests"""
    async with test_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()


# Note: db_session and client fixtures are provided by conftest.py
# No need to redefine them here


class TestRealAuthenticationFlow:
    """Test complete authentication flows with real database operations"""
    
    @pytest.mark.asyncio
    async def test_register_creates_user_in_database(self, client: AsyncClient, db_session: AsyncSession):
        """Test that registration actually creates user in PostgreSQL"""
        # Register user
        response = await client.post(
            "/api/v1/auth/register",
            json={
                "email": "real@test.com",
                "username": "realuser",
                "password": "RealPassword123!",
                "full_name": "Real Test User"
            }
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["email"] == "real@test.com"
        assert data["username"] == "realuser"
        assert "id" in data
        user_id = data["id"]
        
        # Verify user exists in database
        result = await db_session.execute(
            select(User).where(User.id == user_id)
        )
        db_user = result.scalar_one()
        
        assert db_user is not None
        assert db_user.email == "real@test.com"
        assert db_user.username == "realuser"
        assert db_user.hashed_password is not None
        assert db_user.hashed_password != "RealPassword123!"  # Should be hashed
        assert db_user.is_active is True
        assert db_user.is_verified is False
        assert db_user.created_at is not None
    
    @pytest.mark.asyncio
    async def test_login_with_real_password_verification(self, client: AsyncClient, db_session: AsyncSession):
        """Test login verifies password against actual database hash"""
        # Register user first
        await client.post(
            "/api/v1/auth/register",
            json={
                "email": "login@test.com",
                "username": "loginuser",
                "password": "LoginPass123!",
                "full_name": "Login User"
            }
        )
        
        # Login with correct credentials
        response = await client.post(
            "/api/v1/auth/login",
            data={
                "username": "loginuser",
                "password": "LoginPass123!"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] > 0
        
        # Verify failed login attempts are reset in database
        db_session.expire_all()  # Refresh session to see HTTP-committed changes
        result = await db_session.execute(
            select(User).where(User.username == "loginuser")
        )
        db_user = result.scalar_one()
        assert db_user.failed_login_attempts == 0
        assert db_user.last_login is not None
    
    @pytest.mark.asyncio
    async def test_login_with_wrong_password_increments_failures(self, client: AsyncClient, db_session: AsyncSession):
        """Test failed login attempts are tracked in database"""
        # Register user
        await client.post(
            "/api/v1/auth/register",
            json={
                "email": "fail@test.com",
                "username": "failuser",
                "password": "CorrectPass123!",
                "full_name": "Fail User"
            }
        )
        
        # Attempt login with wrong password
        response = await client.post(
            "/api/v1/auth/login",
            data={
                "username": "failuser",
                "password": "WrongPassword123!"
            }
        )
        
        assert response.status_code == 401
        
        # Verify failed attempts incremented in database
        db_session.expire_all()  # Refresh session
        result = await db_session.execute(
            select(User).where(User.username == "failuser")
        )
        db_user = result.scalar_one()
        assert db_user.failed_login_attempts == 1
    
    @pytest.mark.asyncio
    async def test_account_lockout_after_5_failed_attempts(self, client: AsyncClient, db_session: AsyncSession):
        """Test account is locked in database after 5 failed attempts"""
        # Register user
        await client.post(
            "/api/v1/auth/register",
            json={
                "email": "lock@test.com",
                "username": "lockuser",
                "password": "LockPass123!",
                "full_name": "Lock User"
            }
        )
        
        # Make 5 failed login attempts
        for i in range(5):
            response = await client.post(
                "/api/v1/auth/login",
                data={
                    "username": "lockuser",
                    "password": "WrongPassword!"
                }
            )
        
        # 5th attempt should lock account
        assert response.status_code == 403
        assert "locked" in response.json()["detail"].lower()
        
        # Verify account is locked in database
        db_session.expire_all()  # Refresh session
        result = await db_session.execute(
            select(User).where(User.username == "lockuser")
        )
        db_user = result.scalar_one()
        assert db_user.failed_login_attempts >= 5
        assert db_user.account_locked_until is not None
        assert db_user.account_locked_until > datetime.now(timezone.utc)
    
    @pytest.mark.asyncio
    async def test_get_me_returns_real_user_data(self, client: AsyncClient, db_session: AsyncSession):
        """Test /me endpoint returns actual user data from database"""
        # Register and login
        await client.post(
            "/api/v1/auth/register",
            json={
                "email": "me@test.com",
                "username": "meuser",
                "password": "MePass123!",
                "full_name": "Me User"
            }
        )
        
        login_response = await client.post(
            "/api/v1/auth/login",
            data={"username": "meuser", "password": "MePass123!"}
        )
        access_token = login_response.json()["access_token"]
        
        # Get current user
        response = await client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "me@test.com"
        assert data["username"] == "meuser"
        assert data["full_name"] == "Me User"
        assert data["is_active"] is True
        assert "created_at" in data
    
    @pytest.mark.asyncio
    async def test_refresh_token_generates_new_tokens(self, client: AsyncClient, db_session: AsyncSession):
        """Test refresh token generates new access and refresh tokens"""
        import asyncio
        
        # Register and login
        await client.post(
            "/api/v1/auth/register",
            json={
                "email": "refresh@test.com",
                "username": "refreshuser",
                "password": "RefreshPass123!",
                "full_name": "Refresh User"
            }
        )
        
        login_response = await client.post(
            "/api/v1/auth/login",
            data={"username": "refreshuser", "password": "RefreshPass123!"}
        )
        old_refresh_token = login_response.json()["refresh_token"]
        old_access_token = login_response.json()["access_token"]
        
        # Wait 2 seconds to ensure new tokens have different timestamps
        await asyncio.sleep(2)
        
        # Refresh tokens
        response = await client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": old_refresh_token}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["access_token"] != old_access_token
        assert data["refresh_token"] != old_refresh_token
    
    @pytest.mark.asyncio
    async def test_logout_invalidates_refresh_token(self, client: AsyncClient, db_session: AsyncSession):
        """Test logout clears refresh token in database"""
        # Register and login
        await client.post(
            "/api/v1/auth/register",
            json={
                "email": "logout@test.com",
                "username": "logoutuser",
                "password": "LogoutPass123!",
                "full_name": "Logout User"
            }
        )
        
        login_response = await client.post(
            "/api/v1/auth/login",
            data={"username": "logoutuser", "password": "LogoutPass123!"}
        )
        access_token = login_response.json()["access_token"]
        
        # Logout
        response = await client.post(
            "/api/v1/auth/logout",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        
        assert response.status_code == 200
        
        # Verify refresh token cleared in database
        result = await db_session.execute(
            select(User).where(User.username == "logoutuser")
        )
        db_user = result.scalar_one()
        assert db_user.refresh_token is None
    
    @pytest.mark.asyncio
    async def test_duplicate_email_registration_fails(self, client: AsyncClient, db_session: AsyncSession):
        """Test registration prevents duplicate emails"""
        # Register first user
        await client.post(
            "/api/v1/auth/register",
            json={
                "email": "duplicate@test.com",
                "username": "user1",
                "password": "Pass123!",
                "full_name": "User 1"
            }
        )
        
        # Attempt to register with same email
        response = await client.post(
            "/api/v1/auth/register",
            json={
                "email": "duplicate@test.com",
                "username": "user2",
                "password": "Pass123!",
                "full_name": "User 2"
            }
        )
        
        assert response.status_code == 400
        assert "already registered" in response.json()["detail"].lower()
    
    @pytest.mark.asyncio
    async def test_weak_password_rejected(self, client: AsyncClient, db_session: AsyncSession):
        """Test weak passwords are rejected"""
        response = await client.post(
            "/api/v1/auth/register",
            json={
                "email": "weak@test.com",
                "username": "weakuser",
                "password": "weak",
                "full_name": "Weak User"
            }
        )
        
        # FastAPI/Pydantic validation returns 422 for invalid request body
        assert response.status_code == 422
        error_msg = str(response.json()["detail"][0]).lower()
        assert "password" in error_msg or "string" in error_msg or "character" in error_msg


class TestDatabaseIntegrity:
    """Test database constraints and transaction handling"""
    
    @pytest.mark.asyncio
    async def test_transaction_rollback_on_error(self, client: AsyncClient, db_session: AsyncSession):
        """Test database transaction rollback on errors"""
        # This should fail and rollback
        try:
            await client.post(
                "/api/v1/auth/register",
                json={
                    "email": "invalid-email",  # Invalid email format
                    "username": "testuser",
                    "password": "Pass123!",
                    "full_name": "Test User"
                }
            )
        except Exception:
            pass
        
        # Verify no partial data in database
        db_session.expire_all()  # Refresh session
        result = await db_session.execute(
            select(User).where(User.username == "testuser")
        )
        user = result.scalar_one_or_none()
        assert user is None
    
    @pytest.mark.asyncio
    async def test_concurrent_registrations_handled(self, client: AsyncClient, db_session: AsyncSession):
        """Test concurrent registration attempts are handled correctly"""
        # Attempt concurrent registrations with same username
        tasks = [
            client.post(
                "/api/v1/auth/register",
                json={
                    "email": f"concurrent{i}@test.com",
                    "username": "sameuser",
                    "password": "Pass123!",
                    "full_name": f"User {i}"
                }
            )
            for i in range(5)
        ]
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Only one should succeed
        success_count = sum(1 for r in responses if not isinstance(r, Exception) and r.status_code == 201)
        assert success_count == 1
        
        # Verify only one user in database
        result = await db_session.execute(
            select(User).where(User.username == "sameuser")
        )
        users = result.scalars().all()
        assert len(users) == 1


@pytest.mark.asyncio
async def test_database_connection_pool_health():
    """Test database connection pool is properly configured"""
    # conftest.py creates test_engine with pool_size=5
    assert test_engine.pool.size() == 5
    
    # Test multiple concurrent connections
    tasks = []
    for _ in range(10):
        async def query():
            async with test_session_maker() as session:
                result = await session.execute(text("SELECT 1"))
                return result.scalar()
        tasks.append(query())
    
    results = await asyncio.gather(*tasks)
    assert all(r == 1 for r in results)
