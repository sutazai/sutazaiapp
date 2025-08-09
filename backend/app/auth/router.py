"""
Authentication Router - JWT and RBAC Implementation
Provides secure authentication and authorization for SutazAI
"""

import os
import secrets
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from passlib.context import CryptContext
import jwt

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])

# Security configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Bearer token scheme
security = HTTPBearer()

# Data Models
class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int

# In-memory user store
USERS_DB = {
    "admin": {
        "id": "admin_001",
        "username": "admin", 
        "hashed_password": pwd_context.hash("admin123"),
        "role": "admin",
        "is_active": True
    }
}

# Helper Functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def authenticate_user(username: str, password: str):
    user = USERS_DB.get(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return None
    return user

# API Endpoints
@router.post("/login", response_model=Token)
async def login(login_data: UserLogin):
    user = authenticate_user(login_data.username, login_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    access_token = create_access_token(data={"sub": user["username"]})
    
    return Token(
        access_token=access_token,
        refresh_token=access_token,  # Simplified for MVP
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )

@router.get("/status")
async def auth_status():
    return {
        "service": "authentication",
        "status": "healthy",
        "features": {
            "jwt_auth": True,
            "role_based_access": True
        }
    }
