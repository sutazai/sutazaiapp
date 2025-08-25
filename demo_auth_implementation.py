"""
Demo: User Authentication Implementation
Generated via /sc:implement command
Backend-architect and security-engineer agents activated
"""

from typing import Optional, Dict
import hashlib
import secrets
import jwt
from datetime import datetime, timedelta

class AuthenticationSystem:
    """
    Secure authentication implementation following best practices
    - Password hashing with salt
    - JWT token generation
    - Session management
    """
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.users_db = {}  # In production, use proper database
        
    def hash_password(self, password: str, salt: Optional[str] = None) -> tuple[str, str]:
        """Hash password with salt for secure storage"""
        if not salt:
            salt = secrets.token_hex(32)
        
        # Use PBKDF2 for password hashing
        pwd_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # iterations
        )
        
        return pwd_hash.hex(), salt
    
    def register_user(self, username: str, password: str, email: str) -> Dict:
        """Register new user with secure password storage"""
        # Validation
        if username in self.users_db:
            raise ValueError("Username already exists")
        
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters")
        
        # Hash password
        pwd_hash, salt = self.hash_password(password)
        
        # Store user
        user = {
            'username': username,
            'email': email,
            'password_hash': pwd_hash,
            'salt': salt,
            'created_at': datetime.utcnow().isoformat()
        }
        
        self.users_db[username] = user
        return {'username': username, 'email': email}
    
    def authenticate(self, username: str, password: str) -> Optional[str]:
        """Authenticate user and return JWT token"""
        user = self.users_db.get(username)
        if not user:
            return None
        
        # Verify password
        pwd_hash, _ = self.hash_password(password, user['salt'])
        if pwd_hash != user['password_hash']:
            return None
        
        # Generate JWT token
        payload = {
            'username': username,
            'exp': datetime.utcnow() + timedelta(hours=24)
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm='HS256')
        return token
    
    def verify_token(self, token: str) -> Optional[Dict]:
        """Verify JWT token and return user data"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None


# Example usage with error handling
if __name__ == "__main__":
    # Initialize system
    auth = AuthenticationSystem(secret_key="your-secret-key-here")
    
    try:
        # Register user
        result = auth.register_user("john_doe", "SecurePass123!", "john@example.com")
        print(f"✅ User registered: {result}")
        
        # Authenticate
        token = auth.authenticate("john_doe", "SecurePass123!")
        if token:
            print(f"✅ Authentication successful")
            print(f"Token: {token[:20]}...")
            
            # Verify token
            user_data = auth.verify_token(token)
            print(f"✅ Token verified: {user_data}")
        
    except ValueError as e:
        print(f"❌ Error: {e}")