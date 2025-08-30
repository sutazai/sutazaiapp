"""
Security Remediation Module for JARVIS Frontend
Provides secure implementations and utilities to fix identified vulnerabilities
"""

import os
import re
import html
import secrets
import hashlib
import hmac
import json
import time
from typing import Dict, Optional, List, Any, Tuple
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
import tempfile

# Security dependencies (add to requirements.txt):
# pip install python-jose[cryptography] passlib[bcrypt] bleach python-magic cryptography pyotp

try:
    from jose import jwt, JWTError
    from passlib.context import CryptContext
    import bleach
    import magic
    from cryptography.fernet import Fernet
    import pyotp
except ImportError as e:
    print(f"Security dependencies missing: {e}")
    print("Install with: pip install python-jose[cryptography] passlib[bcrypt] bleach python-magic cryptography pyotp")


# ============================================================================
# AUTHENTICATION & AUTHORIZATION
# ============================================================================

class SecureAuthenticationManager:
    """Secure authentication system with JWT tokens and password hashing"""
    
    def __init__(self):
        self.SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
        self.ALGORITHM = "HS256"
        self.ACCESS_TOKEN_EXPIRE_MINUTES = 30
        self.REFRESH_TOKEN_EXPIRE_DAYS = 7
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
    def hash_password(self, password: str) -> str:
        """Securely hash password using bcrypt"""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + (expires_delta or timedelta(minutes=self.ACCESS_TOKEN_EXPIRE_MINUTES))
        to_encode.update({
            "exp": expire,
            "type": "access",
            "iat": datetime.utcnow()
        })
        return jwt.encode(to_encode, self.SECRET_KEY, algorithm=self.ALGORITHM)
    
    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create JWT refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=self.REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode.update({
            "exp": expire,
            "type": "refresh",
            "iat": datetime.utcnow()
        })
        return jwt.encode(to_encode, self.SECRET_KEY, algorithm=self.ALGORITHM)
    
    def verify_token(self, token: str, token_type: str = "access") -> Optional[Dict]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.SECRET_KEY, algorithms=[self.ALGORITHM])
            if payload.get("type") != token_type:
                return None
            return payload
        except JWTError:
            return None
    
    def generate_2fa_secret(self) -> str:
        """Generate TOTP secret for 2FA"""
        return pyotp.random_base32()
    
    def verify_2fa_token(self, secret: str, token: str) -> bool:
        """Verify TOTP 2FA token"""
        totp = pyotp.TOTP(secret)
        return totp.verify(token, valid_window=1)


# ============================================================================
# INPUT VALIDATION & SANITIZATION
# ============================================================================

class InputValidator:
    """Comprehensive input validation and sanitization"""
    
    # Allowed HTML tags for rich text (minimal set for security)
    ALLOWED_TAGS = ['b', 'i', 'u', 'strong', 'em', 'code', 'pre', 'br']
    ALLOWED_ATTRIBUTES = {}
    
    # Dangerous patterns that might indicate attacks
    DANGEROUS_PATTERNS = [
        r'<script[^>]*>.*?</script>',  # Script tags
        r'javascript:',                 # JavaScript protocol
        r'on\w+\s*=',                   # Event handlers
        r';\s*rm\s+-rf',                # Shell command injection
        r'&&\s*curl',                   # Command chaining
        r'\|\s*sh',                     # Pipe to shell
        r'`.*`',                        # Command substitution
        r'\$\(.*\)',                    # Command substitution
        r'\.\./',                       # Path traversal
        r'<iframe',                     # Iframe injection
        r'data:text/html',              # Data URI XSS
    ]
    
    @classmethod
    def sanitize_html(cls, text: str) -> str:
        """Sanitize HTML content to prevent XSS"""
        if not text:
            return ""
        
        # Use bleach to clean HTML
        cleaned = bleach.clean(
            text,
            tags=cls.ALLOWED_TAGS,
            attributes=cls.ALLOWED_ATTRIBUTES,
            strip=True
        )
        
        return cleaned
    
    @classmethod
    def escape_html(cls, text: str) -> str:
        """Escape HTML entities for safe display"""
        return html.escape(text)
    
    @classmethod
    def validate_input(cls, text: str, max_length: int = 5000, 
                      allow_html: bool = False) -> Tuple[bool, str]:
        """
        Validate user input for security threats
        Returns: (is_valid, sanitized_text or error_message)
        """
        if not text:
            return True, ""
        
        # Check length
        if len(text) > max_length:
            return False, f"Input exceeds maximum length of {max_length} characters"
        
        # Check for dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                return False, "Input contains potentially dangerous content"
        
        # Sanitize based on HTML allowance
        if allow_html:
            sanitized = cls.sanitize_html(text)
        else:
            sanitized = cls.escape_html(text)
        
        return True, sanitized
    
    @classmethod
    def validate_email(cls, email: str) -> bool:
        """Validate email address format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @classmethod
    def validate_username(cls, username: str) -> bool:
        """Validate username (alphanumeric + underscore, 3-20 chars)"""
        pattern = r'^[a-zA-Z0-9_]{3,20}$'
        return bool(re.match(pattern, username))
    
    @classmethod
    def validate_sql_input(cls, text: str) -> str:
        """Prevent SQL injection by escaping special characters"""
        # This is a basic example - use parameterized queries in production
        sql_escape_chars = ["'", '"', ";", "--", "/*", "*/", "xp_", "sp_", "0x"]
        for char in sql_escape_chars:
            text = text.replace(char, "")
        return text


# ============================================================================
# SESSION MANAGEMENT
# ============================================================================

class SecureSessionManager:
    """Server-side session management with security features"""
    
    def __init__(self, redis_client=None):
        self.sessions = {}  # In-memory storage (use Redis in production)
        self.redis = redis_client
        self.SESSION_TIMEOUT = 3600  # 1 hour
        self.SESSION_ID_LENGTH = 32
        
    def create_session(self, user_id: str, metadata: Dict = None) -> str:
        """Create secure session with server-side storage"""
        session_id = secrets.token_urlsafe(self.SESSION_ID_LENGTH)
        
        session_data = {
            'session_id': session_id,
            'user_id': user_id,
            'created_at': datetime.utcnow().isoformat(),
            'last_activity': datetime.utcnow().isoformat(),
            'ip_address': metadata.get('ip_address') if metadata else None,
            'user_agent': metadata.get('user_agent') if metadata else None,
            'csrf_token': secrets.token_urlsafe(32),
            'metadata': metadata or {}
        }
        
        if self.redis:
            self.redis.setex(
                f"session:{session_id}",
                self.SESSION_TIMEOUT,
                json.dumps(session_data)
            )
        else:
            self.sessions[session_id] = session_data
        
        return session_id
    
    def validate_session(self, session_id: str, validate_csrf: bool = True,
                        csrf_token: str = None) -> Optional[Dict]:
        """Validate session and update last activity"""
        if self.redis:
            session_data = self.redis.get(f"session:{session_id}")
            if not session_data:
                return None
            session_data = json.loads(session_data)
        else:
            session_data = self.sessions.get(session_id)
            if not session_data:
                return None
        
        # Check session timeout
        last_activity = datetime.fromisoformat(session_data['last_activity'])
        if datetime.utcnow() - last_activity > timedelta(seconds=self.SESSION_TIMEOUT):
            self.destroy_session(session_id)
            return None
        
        # Validate CSRF token if required
        if validate_csrf and csrf_token != session_data.get('csrf_token'):
            return None
        
        # Update last activity
        session_data['last_activity'] = datetime.utcnow().isoformat()
        
        if self.redis:
            self.redis.setex(
                f"session:{session_id}",
                self.SESSION_TIMEOUT,
                json.dumps(session_data)
            )
        else:
            self.sessions[session_id] = session_data
        
        return session_data
    
    def destroy_session(self, session_id: str):
        """Destroy session"""
        if self.redis:
            self.redis.delete(f"session:{session_id}")
        else:
            self.sessions.pop(session_id, None)
    
    def rotate_session(self, old_session_id: str) -> Optional[str]:
        """Rotate session ID for security"""
        session_data = self.validate_session(old_session_id, validate_csrf=False)
        if not session_data:
            return None
        
        # Destroy old session
        self.destroy_session(old_session_id)
        
        # Create new session with same data
        return self.create_session(
            session_data['user_id'],
            session_data.get('metadata')
        )


# ============================================================================
# RATE LIMITING
# ============================================================================

class RateLimiter:
    """Rate limiting to prevent abuse and DoS attacks"""
    
    def __init__(self, redis_client=None):
        self.limits = {}  # In-memory storage (use Redis in production)
        self.redis = redis_client
        
    def check_rate_limit(self, key: str, max_requests: int = 60,
                        window_seconds: int = 60) -> bool:
        """
        Check if rate limit is exceeded
        Returns True if within limit, False if exceeded
        """
        current_time = time.time()
        window_start = current_time - window_seconds
        
        if self.redis:
            # Redis implementation
            pipe = self.redis.pipeline()
            pipe.zremrangebyscore(f"rate:{key}", 0, window_start)
            pipe.zadd(f"rate:{key}", {str(current_time): current_time})
            pipe.zcard(f"rate:{key}")
            pipe.expire(f"rate:{key}", window_seconds + 1)
            results = pipe.execute()
            request_count = results[2]
        else:
            # In-memory implementation
            if key not in self.limits:
                self.limits[key] = []
            
            # Remove old entries
            self.limits[key] = [
                t for t in self.limits[key] if t > window_start
            ]
            
            request_count = len(self.limits[key])
            
            if request_count < max_requests:
                self.limits[key].append(current_time)
        
        return request_count <= max_requests
    
    def get_remaining_requests(self, key: str, max_requests: int = 60,
                              window_seconds: int = 60) -> int:
        """Get number of remaining requests in current window"""
        current_time = time.time()
        window_start = current_time - window_seconds
        
        if self.redis:
            count = self.redis.zcount(f"rate:{key}", window_start, current_time)
        else:
            if key not in self.limits:
                return max_requests
            valid_requests = [t for t in self.limits[key] if t > window_start]
            count = len(valid_requests)
        
        return max(0, max_requests - count)


# ============================================================================
# FILE UPLOAD SECURITY
# ============================================================================

class SecureFileHandler:
    """Secure file upload and processing"""
    
    ALLOWED_AUDIO_EXTENSIONS = {'.wav', '.mp3', '.ogg', '.m4a', '.flac'}
    ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
    ALLOWED_DOCUMENT_EXTENSIONS = {'.pdf', '.txt', '.md', '.csv'}
    
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB default
    
    @classmethod
    def validate_file(cls, file_data: bytes, filename: str,
                     allowed_extensions: set = None,
                     max_size: int = None) -> Tuple[bool, str]:
        """
        Validate uploaded file for security
        Returns: (is_valid, error_message or safe_filename)
        """
        if not file_data or not filename:
            return False, "No file provided"
        
        # Check file size
        max_size = max_size or cls.MAX_FILE_SIZE
        if len(file_data) > max_size:
            return False, f"File exceeds maximum size of {max_size / 1024 / 1024:.1f}MB"
        
        # Check file extension
        file_ext = Path(filename).suffix.lower()
        allowed = allowed_extensions or cls.ALLOWED_AUDIO_EXTENSIONS
        
        if file_ext not in allowed:
            return False, f"File type {file_ext} not allowed"
        
        # Verify MIME type using python-magic
        try:
            mime = magic.from_buffer(file_data, mime=True)
            
            # Map extensions to expected MIME types
            mime_map = {
                '.wav': ['audio/wav', 'audio/x-wav'],
                '.mp3': ['audio/mpeg', 'audio/mp3'],
                '.ogg': ['audio/ogg', 'application/ogg'],
                '.m4a': ['audio/mp4', 'audio/x-m4a'],
                '.jpg': ['image/jpeg'],
                '.jpeg': ['image/jpeg'],
                '.png': ['image/png'],
                '.gif': ['image/gif'],
                '.pdf': ['application/pdf'],
                '.txt': ['text/plain'],
            }
            
            expected_mimes = mime_map.get(file_ext, [])
            if expected_mimes and mime not in expected_mimes:
                return False, f"File content does not match extension (detected: {mime})"
        except Exception:
            pass  # Continue if magic is not available
        
        # Generate safe filename
        safe_filename = cls.generate_safe_filename(filename)
        
        return True, safe_filename
    
    @classmethod
    def generate_safe_filename(cls, original_filename: str) -> str:
        """Generate safe filename with hash"""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        file_hash = hashlib.sha256(original_filename.encode()).hexdigest()[:8]
        extension = Path(original_filename).suffix.lower()
        
        # Remove dangerous characters from name
        base_name = Path(original_filename).stem
        safe_base = re.sub(r'[^a-zA-Z0-9_-]', '_', base_name)[:20]
        
        return f"{timestamp}_{safe_base}_{file_hash}{extension}"
    
    @classmethod
    def save_uploaded_file(cls, file_data: bytes, filename: str,
                          upload_dir: str = "/tmp/uploads") -> Optional[Path]:
        """Securely save uploaded file"""
        # Validate file first
        is_valid, result = cls.validate_file(file_data, filename)
        if not is_valid:
            raise ValueError(result)
        
        safe_filename = result
        
        # Create upload directory if it doesn't exist
        upload_path = Path(upload_dir)
        upload_path.mkdir(parents=True, exist_ok=True)
        
        # Save file with restricted permissions
        file_path = upload_path / safe_filename
        file_path.write_bytes(file_data)
        file_path.chmod(0o644)  # Read for all, write for owner only
        
        return file_path


# ============================================================================
# CSRF PROTECTION
# ============================================================================

class CSRFProtection:
    """CSRF token generation and validation"""
    
    @staticmethod
    def generate_csrf_token() -> str:
        """Generate CSRF token"""
        return secrets.token_urlsafe(32)
    
    @staticmethod
    def validate_csrf_token(session_token: str, submitted_token: str) -> bool:
        """Validate CSRF token using constant-time comparison"""
        if not session_token or not submitted_token:
            return False
        return hmac.compare_digest(session_token, submitted_token)
    
    @staticmethod
    def create_csrf_field(token: str) -> str:
        """Create hidden CSRF field for forms"""
        return f'<input type="hidden" name="csrf_token" value="{html.escape(token)}">'


# ============================================================================
# DATA ENCRYPTION
# ============================================================================

class DataEncryption:
    """Encrypt sensitive data at rest and in transit"""
    
    def __init__(self, key: str = None):
        if key:
            self.key = key.encode()
        else:
            self.key = Fernet.generate_key()
        self.cipher = Fernet(self.key)
    
    def encrypt(self, data: str) -> str:
        """Encrypt string data"""
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt string data"""
        return self.cipher.decrypt(encrypted_data.encode()).decode()
    
    def encrypt_dict(self, data: Dict) -> str:
        """Encrypt dictionary data"""
        json_str = json.dumps(data)
        return self.encrypt(json_str)
    
    def decrypt_dict(self, encrypted_data: str) -> Dict:
        """Decrypt dictionary data"""
        json_str = self.decrypt(encrypted_data)
        return json.loads(json_str)


# ============================================================================
# SECURITY HEADERS
# ============================================================================

def get_security_headers() -> Dict[str, str]:
    """Get recommended security headers"""
    return {
        "Content-Security-Policy": (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' data:; "
            "connect-src 'self' wss: https:; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self';"
        ),
        "X-Frame-Options": "SAMEORIGIN",
        "X-Content-Type-Options": "nosniff",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "geolocation=(), microphone=(self), camera=()"
    }


# ============================================================================
# STREAMLIT INTEGRATION HELPERS
# ============================================================================

def secure_streamlit_setup():
    """Setup secure Streamlit configuration"""
    import streamlit as st
    
    # Initialize security managers in session state
    if 'auth_manager' not in st.session_state:
        st.session_state.auth_manager = SecureAuthenticationManager()
    
    if 'session_manager' not in st.session_state:
        st.session_state.session_manager = SecureSessionManager()
    
    if 'rate_limiter' not in st.session_state:
        st.session_state.rate_limiter = RateLimiter()
    
    if 'input_validator' not in st.session_state:
        st.session_state.input_validator = InputValidator()
    
    if 'csrf_token' not in st.session_state:
        st.session_state.csrf_token = CSRFProtection.generate_csrf_token()
    
    return st.session_state


def require_authentication(func):
    """Decorator to require authentication for Streamlit pages"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        import streamlit as st
        
        if 'authenticated' not in st.session_state or not st.session_state.authenticated:
            st.error("🔒 Authentication required")
            st.stop()
        
        # Validate session
        if 'session_id' in st.session_state:
            session_data = st.session_state.session_manager.validate_session(
                st.session_state.session_id,
                validate_csrf=False
            )
            if not session_data:
                st.session_state.authenticated = False
                st.error("Session expired. Please login again.")
                st.stop()
        
        return func(*args, **kwargs)
    return wrapper


def sanitize_user_input(text: str, max_length: int = 5000) -> str:
    """Sanitize user input for Streamlit display"""
    validator = InputValidator()
    is_valid, result = validator.validate_input(text, max_length, allow_html=False)
    
    if not is_valid:
        raise ValueError(result)
    
    return result


# ============================================================================
# EXAMPLE SECURE IMPLEMENTATIONS
# ============================================================================

def secure_chat_handler(message: str, session_id: str, csrf_token: str) -> Dict:
    """Example of secure chat message handler"""
    
    # Initialize managers
    session_manager = SecureSessionManager()
    rate_limiter = RateLimiter()
    validator = InputValidator()
    
    # Validate session and CSRF
    session_data = session_manager.validate_session(session_id, True, csrf_token)
    if not session_data:
        raise PermissionError("Invalid session or CSRF token")
    
    # Check rate limit
    user_id = session_data['user_id']
    if not rate_limiter.check_rate_limit(f"chat:{user_id}", max_requests=30, window_seconds=60):
        raise Exception("Rate limit exceeded. Please wait before sending more messages.")
    
    # Validate and sanitize input
    is_valid, sanitized_message = validator.validate_input(message, max_length=5000)
    if not is_valid:
        raise ValueError(sanitized_message)
    
    # Process message (your backend logic here)
    response = {
        "success": True,
        "message": sanitized_message,
        "response": "Processed securely",
        "timestamp": datetime.utcnow().isoformat()
    }
    
    return response


def secure_login_handler(username: str, password: str, totp_code: str = None) -> Dict:
    """Example of secure login handler"""
    
    auth_manager = SecureAuthenticationManager()
    session_manager = SecureSessionManager()
    rate_limiter = RateLimiter()
    validator = InputValidator()
    
    # Rate limit login attempts
    if not rate_limiter.check_rate_limit(f"login:{username}", max_requests=5, window_seconds=300):
        raise Exception("Too many login attempts. Please try again later.")
    
    # Validate username format
    if not validator.validate_username(username):
        raise ValueError("Invalid username format")
    
    # Verify credentials (implement your user lookup here)
    # Example: user = db.get_user(username)
    # if not user or not auth_manager.verify_password(password, user.password_hash):
    #     raise ValueError("Invalid credentials")
    
    # Verify 2FA if enabled
    # if user.totp_secret and not auth_manager.verify_2fa_token(user.totp_secret, totp_code):
    #     raise ValueError("Invalid 2FA code")
    
    # Create session and tokens
    session_id = session_manager.create_session(username)
    access_token = auth_manager.create_access_token({"sub": username})
    refresh_token = auth_manager.create_refresh_token({"sub": username})
    
    return {
        "success": True,
        "session_id": session_id,
        "access_token": access_token,
        "refresh_token": refresh_token,
        "csrf_token": session_manager.sessions[session_id]['csrf_token']
    }


if __name__ == "__main__":
    print("Security Remediation Module")
    print("-" * 50)
    print("This module provides secure implementations to fix vulnerabilities")
    print("\nExample usage:")
    print("  from security_remediation import SecureAuthenticationManager, InputValidator")
    print("  auth = SecureAuthenticationManager()")
    print("  validator = InputValidator()")
    print("\nRun security tests:")
    print("  python -m pytest test_security.py")