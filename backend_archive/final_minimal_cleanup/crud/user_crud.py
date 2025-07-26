from typing import List, Optional


from sqlmodel import Session, select # Import Session and select from sqlmodel

from backend.models.base_models import User
# Import schemas from the central location
from backend.schemas import UserCreate, UserUpdate
from backend.core.security import get_password_hash

# Get settings (optional, only if settings are directly used in this file)
# settings = get_settings()

def get_user_by_username(db: Session, username: str) -> Optional[User]: # db is sqlmodel.Session
    statement = select(User).where(User.username == username)
    user = db.exec(statement).first()
    return user

def get_user_by_id(db: Session, id: int) -> Optional[User]:
    """Get a user by their ID."""
    user = db.get(User, id)
    return user

def get_users(db: Session, skip: int = 0, limit: int = 100) -> List[User]:
    """Retrieve a list of users with optional pagination."""
    statement = select(User).offset(skip).limit(limit)
    users = db.exec(statement).all()
    return list(users) # Cast Sequence to list

def create_user(db: Session, user_in: UserCreate) -> User: # Use UserCreate schema
    # REMOVED Example usage of imported settings:
    # if settings.SOME_SETTING:
    #     pass

    hashed_password = get_password_hash(user_in.password)
    # Create a new User instance using the input user data
    db_user = User(
        username=user_in.username,
        email=user_in.email, # Add email if present in input
        hashed_password=hashed_password,
        # Add other fields as necessary from user_in or defaults
        is_active=True, # Default to active if not provided
        is_superuser=False # Default to not superuser
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user 

# Add update_user function using UserUpdate schema
def update_user(db: Session, db_user: User, user_in: UserUpdate) -> User:
    """Update an existing user."""
    user_data = user_in.model_dump(exclude_unset=True) # Use model_dump in Pydantic v2
    if "password" in user_data:
        hashed_password = get_password_hash(user_data["password"])
        db_user.hashed_password = hashed_password
        del user_data["password"] # Don't set plain password

    for key, value in user_data.items():
        setattr(db_user, key, value)

    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def delete_user(db: Session, id: int) -> Optional[User]:
    """Delete a user by ID."""
    user = db.get(User, id)
    if user:
        db.delete(user)
        db.commit()
        # Return the deleted user object (or just True/None)
        return user
    return None

# Add update_user function using UserUpdate schema (assuming it exists)
# def update_user(db: Session, user: User, user_in: UserUpdate) -> User:
#     ... 