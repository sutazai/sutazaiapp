from .models import User, db


class UserController:
    """
    Controller class for handling user-related operations
    """

    @staticmethod
    def get_user_by_id(user_id):
        """Get a user by ID"""
        return db.session.query(User).filter(User.id == user_id).first()

    @staticmethod
    def get_user_by_username(username):
        """Get a user by username"""
        return db.session.query(User).filter(User.username == username).first()

    @staticmethod
    def get_user_by_email(email):
        """Get a user by email"""
        return db.session.query(User).filter(User.email == email).first()

    @staticmethod
    def create_user(user_data):
        """Create a new user"""
        user = User(**user_data)
        db.session.add(user)
        db.session.commit()
        return user

    @staticmethod
    def update_user(user_id, user_data):
        """Update an existing user"""
        user = UserController.get_user_by_id(user_id)
        if user:
            for key, value in user_data.items():
                setattr(user, key, value)
            db.session.commit()
        return user

    @staticmethod
    def delete_user(user_id):
        """Delete a user"""
        user = UserController.get_user_by_id(user_id)
        if user:
            db.session.delete(user)
            db.session.commit()
            return True
        return False
