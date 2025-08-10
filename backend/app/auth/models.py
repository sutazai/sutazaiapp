"""
Authentication models and schemas for SUTAZAI
Pydantic models for request/response validation
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, EmailStr, Field, field_validator
from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Table, Text, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
# UUID imports removed - using Integer IDs to match database schema

from app.core.database import Base


# SQLAlchemy User Model
class User(Base):
    """User database model"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Additional fields for enhanced security
    last_login = Column(DateTime(timezone=True))
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime(timezone=True))
    
    # Note: permissions column removed to match actual database schema
    
    def __repr__(self):
        return f"<User(username='{self.username}', email='{self.email}')>"


# Pydantic Models for API
class UserBase(BaseModel):
    """Base user schema"""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    
    @field_validator('username')
    @classmethod
    def validate_username(cls, v):
        """Validate username format"""
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Username must contain only alphanumeric characters, underscores, and hyphens')
        return v.lower()


class UserCreate(UserBase):
    """Schema for user registration"""
    password: str = Field(..., min_length=8)
    
    @field_validator('password')
    @classmethod
    def validate_password(cls, v):
        """Basic password validation"""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v


class UserLogin(BaseModel):
    """Schema for user login"""
    username: str
    password: str


class UserInDB(UserBase):
    """User schema with database fields"""
    id: int  # Changed from str to int to match database schema
    is_active: bool = True
    is_admin: bool = False
    created_at: datetime
    updated_at: Optional[datetime] = None
    last_login: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class UserResponse(UserBase):
    """User response schema (public)"""
    id: int  # Changed from str to int to match database schema
    is_active: bool = True
    is_admin: bool = False
    created_at: datetime
    
    class Config:
        from_attributes = True


class Token(BaseModel):
    """Token response schema"""
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int = 1800  # 30 minutes default
    user: Optional[UserResponse] = None


class TokenData(BaseModel):
    """Token payload data"""
    user_id: Optional[int] = None  # Changed from str to int to match database schema
    username: Optional[str] = None
    email: Optional[str] = None
    is_admin: bool = False
    scopes: List[str] = []


class PasswordReset(BaseModel):
    """Password reset request"""
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    """Password reset confirmation"""
    token: str
    new_password: str = Field(..., min_length=8)


class PasswordChange(BaseModel):
    """Password change request"""
    current_password: str
    new_password: str = Field(..., min_length=8)