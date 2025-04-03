from typing import Optional

from core import settings

from backend.models.base_models import User # Assuming User model is defined here
from backend.core.security import get_password_hash
from sqlmodel import Session, select # Import Session and select

def get_user_by_username(db: Session, username: str) -> Optional[User]:
    statement = select(User).where(User.username == username)
    user = db.exec(statement).first()
    return user

def create_user(db: Session, user_in: User) -> User:
    hashed_password = get_password_hash(user_in.password)
    # Create a new User instance using the input user data
    db_user = User(
        username=user_in.username,
        email=user_in.email, # Add email if present in input
        hashed_password=hashed_password,
        # Add other fields as necessary from user_in or defaults
        is_active=user_in.is_active if hasattr(user_in, 'is_active') else True, # Default to active if not provided
        is_superuser=user_in.is_superuser if hasattr(user_in, 'is_superuser') else False # Default to not superuser
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user 