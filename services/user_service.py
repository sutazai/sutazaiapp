from .models import User
from werkzeug.security import generate_password_hash, check_password_hash


class UserService:
    """
    Service class for handling user-related business logic
    """

    @staticmethod
    def authenticate(username, password):
        """
        Authenticate a user with username and password
        """
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            return user
        return None

    @staticmethod
    def register_user(username, email, password, **kwargs):
        """
        Register a new user
        """
        # Check if user already exists
        if User.query.filter_by(username=username).first():
            return None, "Username already taken"

        if User.query.filter_by(email=email).first():
            return None, "Email already registered"

        # Create new user
        hashed_password = generate_password_hash(password)

        user = User(
            username=username, email=email, password_hash=hashed_password, **kwargs
        )

        # Add to database
        from .base import db

        db.session.add(user)
        db.session.commit()

        return user, None  # Return user and no error

    @staticmethod
    def change_password(user_id, current_password, new_password):
        """
        Change a user's password
        """
        user = User.query.get(user_id)
        if not user:
            return False, "User not found"

        if not check_password_hash(user.password_hash, current_password):
            return False, "Current password is incorrect"

        user.password_hash = generate_password_hash(new_password)

        from .base import db

        db.session.commit()

        return True, None
