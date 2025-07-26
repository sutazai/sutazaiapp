"""
User Service Module

Handles business logic related to user authentication, registration, and management.
Uses user_crud for database interactions.
"""

import logging
from typing import Optional, Tuple

from sqlmodel import Session

from backend.models.user_model import User
from backend.crud import user_crud
from backend.core.security import verify_password, get_password_hash
from backend.routers.auth_router import UserCreate # For register_user type hint

logger = logging.getLogger(__name__)

class UserService:
    """
    Service class for handling user-related business logic.
    Uses dependency injection for the database session.
    """

    # Remove __init__ if db session is injected per-method
    # def __init__(self, db: Session = Depends(get_db)):
    #     self.db = db

    async def authenticate(self, db: Session, *, username: str, password: str) -> Optional[User]:
        """
        Authenticate a user by username and password using crud and security utils.
        """
        user = user_crud.get_user_by_username(db, username=username)
        if not user:
            return None
        if not verify_password(password, user.hashed_password):
            return None
        return user

    async def register_user(self, db: Session, *, user_in: UserCreate) -> Tuple[Optional[User], Optional[str]]:
        """
        Register a new user using crud, checking for existing users.
        Returns (User, None) on success, (None, error_message) on failure.
        """
        # Check if username already exists
        if user_crud.get_user_by_username(db, username=user_in.username):
            return None, "Username already taken"

        # Check if email already exists (assuming email is unique)
        # Note: user_crud needs get_user_by_email if email uniqueness is enforced
        # if user_crud.get_user_by_email(db, email=user_in.email):
        #     return None, "Email already registered"

        try:
            # user_crud.create_user handles hashing internally now
            user = user_crud.create_user(db=db, user_in=user_in)
            return user, None
        except Exception as e:
            logger.error(f"Error during user creation in service: {e}", exc_info=True)
            # Rollback might be needed if crud doesn't handle it
            # db.rollback()
            return None, f"Failed to create user: {e}"

    async def change_password(self, db: Session, *, user_id: int, current_password: str, new_password: str) -> Tuple[bool, Optional[str]]:
        """
        Change a user's password after verifying the current password.
        Returns (True, None) on success, (False, error_message) on failure.
        """
        user = user_crud.get_user_by_id(db, id=user_id)
        if not user:
            return False, "User not found"

        if not verify_password(current_password, user.hashed_password):
            return False, "Current password is incorrect"

        try:
            # Update the user object directly - SQLModel handles the update on commit
            user.hashed_password = get_password_hash(new_password)
            db.add(user)
            db.commit()
            db.refresh(user)
            return True, None
        except Exception as e:
            logger.error(f"Error changing password for user {user_id}: {e}", exc_info=True)
            db.rollback()
            return False, f"Failed to change password: {e}"
