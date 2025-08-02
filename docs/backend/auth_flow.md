# Authentication and Session Management

## Overview

The SutazAI backend implements a comprehensive authentication and session management system using JWT tokens, role-based access control (RBAC), and secure session handling. The system supports both API key authentication for external services and user-based authentication for interactive sessions.

## Authentication Architecture

### Authentication Flow Diagram

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Client    │    │     API     │    │    Auth     │    │  Database   │
│             │    │   Gateway   │    │  Service    │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │                   │
       │ 1. Login Request  │                   │                   │
       ├──────────────────►│                   │                   │
       │                   │ 2. Validate Creds│                   │
       │                   ├──────────────────►│                   │
       │                   │                   │ 3. Check User    │
       │                   │                   ├──────────────────►│
       │                   │                   │ 4. User Data     │
       │                   │                   │◄──────────────────┤
       │                   │ 5. Generate JWT  │                   │
       │                   │◄──────────────────┤                   │
       │ 6. JWT Token      │                   │                   │
       │◄──────────────────┤                   │                   │
       │                   │                   │                   │
       │ 7. API Request    │                   │                   │
       ├──────────────────►│                   │                   │
       │                   │ 8. Validate JWT  │                   │
       │                   ├──────────────────►│                   │
       │                   │ 9. Token Valid   │                   │
       │                   │◄──────────────────┤                   │
       │                   │ 10. Process Req  │                   │
       │                   │──────────────────►│                   │
       │ 11. Response      │                   │                   │
       │◄──────────────────┤                   │                   │
```

## Authentication Methods

### 1. JWT Token Authentication

#### Token Structure
```json
{
  "header": {
    "alg": "HS256",
    "typ": "JWT"
  },
  "payload": {
    "sub": "user_id",
    "username": "john_doe",
    "roles": ["user", "agent_manager"],
    "permissions": ["read:agents", "write:tasks"],
    "iat": 1640995200,
    "exp": 1641001200,
    "jti": "unique_token_id"
  },
  "signature": "HMACSHA256(...)"
}
```

#### Implementation
```python
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional

class JWTTokenHandler:
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = 30
        self.refresh_token_expire_days = 7
    
    def create_access_token(self, data: dict) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        to_encode.update({"exp": expire, "type": "access"})
        
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(self, data: dict) -> str:
        """Create JWT refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        to_encode.update({"exp": expire, "type": "refresh"})
        
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[dict]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except JWTError:
            return None
```

### 2. API Key Authentication

#### API Key Management
```python
class APIKeyManager:
    def __init__(self, db_session):
        self.db = db_session
    
    async def create_api_key(self, user_id: str, name: str, scopes: List[str]) -> APIKey:
        """Create new API key for user"""
        api_key = APIKey(
            key=self.generate_api_key(),
            user_id=user_id,
            name=name,
            scopes=scopes,
            created_at=datetime.utcnow(),
            last_used_at=None,
            is_active=True
        )
        
        self.db.add(api_key)
        await self.db.commit()
        return api_key
    
    async def validate_api_key(self, key: str) -> Optional[APIKey]:
        """Validate API key and update last_used_at"""
        api_key = await self.db.query(APIKey).filter(
            APIKey.key == key,
            APIKey.is_active == True
        ).first()
        
        if api_key:
            api_key.last_used_at = datetime.utcnow()
            await self.db.commit()
        
        return api_key
    
    def generate_api_key(self) -> str:
        """Generate secure API key"""
        return f"sai_{secrets.token_urlsafe(32)}"
```

## Role-Based Access Control (RBAC)

### Permission System
```python
from enum import Enum
from typing import Set

class Permission(str, Enum):
    # Agent management
    READ_AGENTS = "read:agents"
    CREATE_AGENTS = "create:agents"
    UPDATE_AGENTS = "update:agents"
    DELETE_AGENTS = "delete:agents"
    
    # Task management
    READ_TASKS = "read:tasks"
    CREATE_TASKS = "create:tasks"
    UPDATE_TASKS = "update:tasks"
    DELETE_TASKS = "delete:tasks"
    
    # System administration
    READ_SYSTEM = "read:system"
    UPDATE_SYSTEM = "update:system"
    ADMIN_SYSTEM = "admin:system"

class Role(str, Enum):
    USER = "user"
    AGENT_MANAGER = "agent_manager"
    TASK_COORDINATOR = "task_coordinator"
    SYSTEM_ADMIN = "system_admin"
    SUPER_ADMIN = "super_admin"

# Role-permission mappings
ROLE_PERMISSIONS = {
    Role.USER: {
        Permission.READ_AGENTS,
        Permission.READ_TASKS,
        Permission.CREATE_TASKS
    },
    Role.AGENT_MANAGER: {
        Permission.READ_AGENTS,
        Permission.CREATE_AGENTS,
        Permission.UPDATE_AGENTS,
        Permission.READ_TASKS,
        Permission.CREATE_TASKS
    },
    Role.TASK_COORDINATOR: {
        Permission.READ_AGENTS,
        Permission.READ_TASKS,
        Permission.CREATE_TASKS,
        Permission.UPDATE_TASKS,
        Permission.DELETE_TASKS
    },
    Role.SYSTEM_ADMIN: {
        Permission.READ_SYSTEM,
        Permission.UPDATE_SYSTEM,
        *ROLE_PERMISSIONS[Role.TASK_COORDINATOR]
    },
    Role.SUPER_ADMIN: {
        Permission.ADMIN_SYSTEM,
        *ROLE_PERMISSIONS[Role.SYSTEM_ADMIN]
    }
}
```

### Permission Checking
```python
class PermissionChecker:
    def __init__(self):
        self.role_permissions = ROLE_PERMISSIONS
    
    def user_has_permission(self, user_roles: List[Role], required_permission: Permission) -> bool:
        """Check if user has required permission"""
        user_permissions = set()
        
        for role in user_roles:
            user_permissions.update(self.role_permissions.get(role, set()))
        
        return required_permission in user_permissions
    
    def require_permission(self, permission: Permission):
        """Decorator to require specific permission"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                current_user = get_current_user()
                
                if not self.user_has_permission(current_user.roles, permission):
                    raise HTTPException(
                        status_code=403,
                        detail=f"Permission required: {permission}"
                    )
                
                return await func(*args, **kwargs)
            return wrapper
        return decorator
```

## Authentication Middleware

### JWT Authentication Middleware
```python
from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

class JWTAuthMiddleware:
    def __init__(self, jwt_handler: JWTTokenHandler):
        self.jwt_handler = jwt_handler
        self.security = HTTPBearer()
    
    async def __call__(self, request: Request, call_next):
        # Skip authentication for public endpoints
        if self.is_public_endpoint(request.url.path):
            return await call_next(request)
        
        # Extract and validate token
        try:
            credentials = await self.security(request)
            payload = self.jwt_handler.verify_token(credentials.credentials)
            
            if not payload:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid authentication credentials"
                )
            
            # Add user info to request state
            request.state.user = await self.get_user_from_payload(payload)
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=401,
                detail="Authentication failed"
            )
        
        return await call_next(request)
    
    def is_public_endpoint(self, path: str) -> bool:
        """Check if endpoint is public (no auth required)"""
        public_paths = ["/health", "/docs", "/redoc", "/openapi.json", "/auth/login"]
        return any(path.startswith(public_path) for public_path in public_paths)
```

## Session Management

### Session Store
```python
import redis
import json
from datetime import datetime, timedelta

class SessionManager:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.session_timeout = timedelta(hours=24)
    
    async def create_session(self, user_id: str, session_data: dict) -> str:
        """Create new user session"""
        session_id = f"session:{secrets.token_urlsafe(32)}"
        
        session_info = {
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat(),
            "data": session_data
        }
        
        await self.redis.setex(
            session_id,
            self.session_timeout,
            json.dumps(session_info)
        )
        
        return session_id
    
    async def get_session(self, session_id: str) -> Optional[dict]:
        """Retrieve session information"""
        session_data = await self.redis.get(session_id)
        
        if not session_data:
            return None
        
        session_info = json.loads(session_data)
        
        # Update last activity
        session_info["last_activity"] = datetime.utcnow().isoformat()
        await self.redis.setex(
            session_id,
            self.session_timeout,
            json.dumps(session_info)
        )
        
        return session_info
    
    async def delete_session(self, session_id: str):
        """Delete user session"""
        await self.redis.delete(session_id)
    
    async def cleanup_expired_sessions(self):
        """Clean up expired sessions (background task)"""
        # Redis handles expiration automatically
        pass
```

## Login and Authentication Endpoints

### Authentication Routes
```python
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

router = APIRouter(prefix="/auth", tags=["authentication"])

@router.post("/login", response_model=TokenResponse)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    user_service: UserService = Depends(get_user_service),
    jwt_handler: JWTTokenHandler = Depends(get_jwt_handler)
):
    """User login with username/password"""
    
    # Authenticate user
    user = await user_service.authenticate_user(
        form_data.username,
        form_data.password
    )
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )
    
    # Create tokens
    access_token = jwt_handler.create_access_token(
        data={"sub": user.id, "username": user.username, "roles": user.roles}
    )
    refresh_token = jwt_handler.create_refresh_token(
        data={"sub": user.id}
    )
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer"
    )

@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    refresh_token: str,
    jwt_handler: JWTTokenHandler = Depends(get_jwt_handler),
    user_service: UserService = Depends(get_user_service)
):
    """Refresh access token using refresh token"""
    
    payload = jwt_handler.verify_token(refresh_token)
    
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    user = await user_service.get_user(payload["sub"])
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    # Create new access token
    access_token = jwt_handler.create_access_token(
        data={"sub": user.id, "username": user.username, "roles": user.roles}
    )
    
    return TokenResponse(
        access_token=access_token,
        token_type="bearer"
    )

@router.post("/logout")
async def logout(
    current_user: User = Depends(get_current_user),
    session_manager: SessionManager = Depends(get_session_manager)
):
    """User logout"""
    
    # Add token to blacklist (optional)
    # await token_blacklist.add_token(current_user.token_jti)
    
    # Clear session if exists
    if hasattr(current_user, 'session_id'):
        await session_manager.delete_session(current_user.session_id)
    
    return {"message": "Successfully logged out"}
```

## User Management

### User Model
```python
from sqlalchemy import Column, String, Boolean, DateTime, JSON
from sqlalchemy.dialects.postgresql import UUID
import uuid

class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(100), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    roles = Column(JSON, nullable=False, default=["user"])
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    
    # Profile information
    full_name = Column(String(255))
    preferences = Column(JSON, default={})
```

### Password Hashing
```python
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class PasswordHandler:
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using bcrypt"""
        return pwd_context.hash(password)
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return pwd_context.verify(plain_password, hashed_password)
```

## Security Best Practices

### Token Security
- **Short-lived Access Tokens**: 30-minute expiration
- **Secure Refresh Tokens**: 7-day expiration with rotation
- **Token Blacklisting**: Invalidate compromised tokens
- **Rate Limiting**: Prevent brute force attacks

### Password Security
- **Strong Hashing**: bcrypt with appropriate cost factor
- **Password Policies**: Minimum length and complexity requirements
- **Password History**: Prevent password reuse
- **Account Lockout**: Temporary lockout after failed attempts

### Session Security
```python
class SecurityConfig:
    # JWT settings
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
    JWT_ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    REFRESH_TOKEN_EXPIRE_DAYS = 7
    
    # Session settings
    SESSION_TIMEOUT_HOURS = 24
    MAX_SESSIONS_PER_USER = 5
    
    # Security headers
    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains"
    }
    
    # Rate limiting
    RATE_LIMIT_REQUESTS = 100
    RATE_LIMIT_WINDOW = 300  # 5 minutes
```

## Monitoring and Auditing

### Authentication Logging
```python
import logging
from typing import Optional

class AuthAuditLogger:
    def __init__(self):
        self.logger = logging.getLogger("auth_audit")
    
    def log_login_attempt(self, username: str, success: bool, ip_address: str):
        """Log user login attempt"""
        self.logger.info(
            "Login attempt",
            extra={
                "username": username,
                "success": success,
                "ip_address": ip_address,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    def log_permission_check(self, user_id: str, permission: str, granted: bool):
        """Log permission check"""
        self.logger.info(
            "Permission check",
            extra={
                "user_id": user_id,
                "permission": permission,
                "granted": granted,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
```

## Integration with Frontend

### Token Storage
```javascript
// Frontend token management
class TokenManager {
    constructor() {
        this.accessToken = localStorage.getItem('access_token');
        this.refreshToken = localStorage.getItem('refresh_token');
    }
    
    setTokens(accessToken, refreshToken) {
        localStorage.setItem('access_token', accessToken);
        localStorage.setItem('refresh_token', refreshToken);
        this.accessToken = accessToken;
        this.refreshToken = refreshToken;
    }
    
    clearTokens() {
        localStorage.removeItem('access_token');
        localStorage.removeItem('refresh_token');
        this.accessToken = null;
        this.refreshToken = null;
    }
    
    async refreshAccessToken() {
        // Implement token refresh logic
    }
}
```

This authentication system provides comprehensive security for the SutazAI backend while maintaining flexibility and ease of use for different deployment scenarios.