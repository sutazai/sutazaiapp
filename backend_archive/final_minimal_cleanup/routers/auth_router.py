"""
Authentication Router

This module provides API routes for user authentication and authorization.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from pydantic import BaseModel
from sqlalchemy.orm import Session
from passlib.context import CryptContext

from backend.core.database import get_db
from backend.models.user_model import User
from backend.core.config import get_settings
from backend.core.security import create_access_token, authenticate_user
from backend.dependencies import get_current_active_user
from backend.schemas import Token, TokenData, UserRead, UserCreate, UserUpdate
from backend.crud import user_crud

# Set up logging
logger = logging.getLogger("auth_router")

# Create router
router = APIRouter()

# Get application settings
settings = get_settings()

# Security
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# Models
class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None


class UserUpdate(BaseModel):
    """Schema for updating user data."""
    email: Optional[str] = None
    password: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    is_active: Optional[bool] = None
    is_admin: Optional[bool] = None # Allow updating admin status


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    is_active: bool
    is_admin: bool


# Alias UserResponse as UserRead for consistency if used elsewhere
UserRead = UserResponse


# Helper functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def authenticate_user(db: Session, username: str, password: str):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return False
    if not verify_password(password, user.password_hash):
        return False
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM
    )
    return encoded_jwt


# User endpoints
@router.post(
    "/register", 
    response_model=UserRead, # Use UserRead from schemas
    status_code=status.HTTP_201_CREATED,
    summary="Register new user"
)
async def register_user(user_in: UserCreate, db: Session = Depends(get_db)):
    """
    Register a new user.
    - **username**: Unique username.
    - **email**: Unique email.
    - **password**: User password.
    """
    # Check if username already exists
    existing_user = user_crud.get_user_by_username(db, username=user_in.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered",
        )

    # Check if email already exists
    # TODO: Add crud function get_user_by_email
    # existing_email_user = user_crud.get_user_by_email(db, email=user_in.email)
    # if existing_email_user:
    #     raise HTTPException(
    #         status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered"
    #     )

    # Create new user using CRUD function
    try:
        db_user = user_crud.create_user(db=db, user_in=user_in)
    except Exception as e:
        # Catch potential DB errors during creation
        logger.error(f"Error creating user in DB: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error occurred during user registration."
        )
    
    return db_user # UserRead conversion happens automatically via response_model


@router.post("/token", summary="Get access token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)) -> Token:
    """Authenticate user and return access token."""
    user = await authenticate_user(db=db, username=form_data.username, password=form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return Token(access_token=access_token, token_type="bearer")


# Dependencies for protected routes
async def get_current_user(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
):
    """
    Get the current authenticated user from the token.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        subject = payload.get("sub")
        if not isinstance(subject, str):
            raise credentials_exception
        username: str = subject
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.username == token_data.username).first()
    if user is None:
        raise credentials_exception

    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)):
    """
    Check if the current user is active.
    """
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


# Protected routes
@router.get("/users/me", response_model=UserRead, summary="Get current user")
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """
    Get information about the currently authenticated user.
    """
    return current_user


@router.get("/users/{user_id}", response_model=UserRead, summary="Get user by ID")
async def read_user(
    user_id: int,
    current_user: User = Depends(get_current_active_superuser), # Require superuser to read arbitrary users
    db: Session = Depends(get_db),
):
    """
    Get information about a specific user by ID.
    Requires superuser privileges.
    """
    user = user_crud.get_user_by_id(db, user_id=user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@router.patch("/users/{user_id}", response_model=UserRead, summary="Update user")
async def update_user_endpoint(
    user_id: int,
    user_in: UserUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_superuser), # Require superuser to update
):
    """Update user details. Requires superuser privileges."""
    db_user = user_crud.get_user_by_id(db, user_id=user_id)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    updated_user = user_crud.update_user(db=db, db_user=db_user, user_in=user_in)
    return updated_user

# Add delete endpoint if needed
# @router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
# async def delete_user_endpoint(...):
#    ...
