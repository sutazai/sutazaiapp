from sqlmodel import Session
from core.database import engine # Import engine
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt # Import jwt
from pydantic import ValidationError
from typing import Optional

from core import settings
from backend.crud import user_crud # Import user_crud
from backend.models.base_models import User # Import User model
from backend.schemas import TokenData # Import TokenData schema
from backend.core.exceptions import AuthenticationError # Import custom exception

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def get_db() -> Session: # Correct return type hint
    with Session(engine) as session:
        yield session 

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY, # Use settings.SECRET_KEY
            algorithms=[settings.ALGORITHM] # Use settings.ALGORITHM
        )
        username: Optional[str] = payload.get("sub") # Make username Optional initially
        if username is None:
            raise credentials_exception
        # Validate username format if needed before creating TokenData
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    except ValidationError: # Catch Pydantic validation errors for TokenData
         raise credentials_exception

    user = user_crud.get_user_by_username(db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    # Ensure current_user has is_active attribute
    if not getattr(current_user, 'is_active', False):
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user 