#!/usr/bin/env python3
"""
Enterprise Security and Authentication Service - SECURE VERSION
"""

from fastapi import FastAPI, HTTPException, Depends, Security, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
from datetime import datetime, timedelta
import json
import jwt
import hashlib
import secrets
import bcrypt
from typing import Dict, Optional
import time
from collections import defaultdict
from backend.core.secure_config import load_config

app = FastAPI(title="SutazAI Security Service", version="2.0")
security = HTTPBearer()

# Load secure configuration
try:
    config = load_config()
    SECRET_KEY = config.JWT_SECRET
    ALGORITHM = "HS256"
except Exception as e:
    # Fallback to secure random key (will invalidate existing tokens)
    SECRET_KEY = secrets.token_urlsafe(64)
    ALGORITHM = "HS256"
    print(f"Warning: Using fallback JWT secret: {e}")

# Rate limiting
login_attempts = defaultdict(list)
MAX_LOGIN_ATTEMPTS = 5
RATE_LIMIT_WINDOW = 300  # 5 minutes

def hash_password(password: str) -> str:
    """Securely hash password using bcrypt"""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    """Verify password against bcrypt hash"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

# Secure user database with properly hashed passwords
USERS_DB = {
    "admin": {
        "username": "admin",
        "password_hash": hash_password("secure_admin_pass_2024!"),
        "role": "administrator",
        "permissions": ["read", "write", "delete", "admin"],
        "created_at": datetime.now().isoformat(),
        "last_login": None
    }
}

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=24)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid authentication")
        return username
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication")

@app.get("/")
async def root():
    return {"service": "Security & Auth", "status": "active", "timestamp": datetime.now().isoformat()}

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "auth_service", "port": 8094}

def check_rate_limit(ip: str) -> bool:
    """Check if IP is rate limited"""
    now = time.time()
    attempts = login_attempts[ip]
    
    # Remove old attempts outside the window
    login_attempts[ip] = [attempt for attempt in attempts if now - attempt < RATE_LIMIT_WINDOW]
    
    return len(login_attempts[ip]) < MAX_LOGIN_ATTEMPTS

def record_login_attempt(ip: str):
    """Record a login attempt"""
    login_attempts[ip].append(time.time())

@app.post("/login")
async def login(credentials: dict, request: Request):
    try:
        # Get client IP
        client_ip = request.client.host
        
        # Check rate limiting
        if not check_rate_limit(client_ip):
            raise HTTPException(
                status_code=429, 
                detail="Too many login attempts. Please try again later."
            )
        
        username = credentials.get("username")
        password = credentials.get("password")
        
        if not username or not password:
            record_login_attempt(client_ip)
            raise HTTPException(status_code=400, detail="Username and password required")
        
        user = USERS_DB.get(username)
        if not user:
            record_login_attempt(client_ip)
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Use secure password verification
        if not verify_password(password, user["password_hash"]):
            record_login_attempt(client_ip)
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Update last login
        user["last_login"] = datetime.now().isoformat()
        
        access_token = create_access_token(data={
            "sub": username,
            "role": user["role"],
            "permissions": user["permissions"],
            "iat": datetime.utcnow().timestamp()
        })
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "username": user["username"],
                "role": user["role"], 
                "permissions": user["permissions"],
                "last_login": user["last_login"]
            },
            "expires_in": 86400  # 24 hours
        }
    except HTTPException:
        raise
    except Exception as e:
        record_login_attempt(client_ip)
        raise HTTPException(status_code=500, detail="Authentication service error")

@app.get("/validate")
async def validate_token(current_user: str = Depends(verify_token)):
    user = USERS_DB.get(current_user)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "valid": True,
        "user": {
            "username": user["username"],
            "role": user["role"],
            "permissions": user["permissions"]
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/permissions")
async def check_permissions(current_user: str = Depends(verify_token)):
    user = USERS_DB.get(current_user)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "user": current_user,
        "permissions": user["permissions"],
        "role": user["role"],
        "can_read": "read" in user["permissions"],
        "can_write": "write" in user["permissions"], 
        "can_delete": "delete" in user["permissions"],
        "is_admin": "admin" in user["permissions"]
    }

@app.post("/logout")
async def logout(current_user: str = Depends(verify_token)):
    # In a real implementation, you'd blacklist the token
    return {
        "message": f"User {current_user} logged out successfully",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/security_status")
async def security_status():
    return {
        "service": "Security & Auth",
        "version": "2.0 - SECURE",
        "features": [
            "JWT Authentication with Secure Secrets",
            "Role-based Access Control",
            "Permission Management", 
            "Token Validation",
            "Bcrypt Password Hashing",
            "Rate Limiting Protection",
            "IP-based Attack Prevention",
            "Secure Configuration Management"
        ],
        "security_level": "Enterprise+",
        "active_features": 8,
        "security_enhancements": [
            "Replaced SHA256 with bcrypt",
            "Implemented rate limiting",
            "Added secure configuration loading",
            "Enhanced token validation",
            "Improved error handling"
        ],
        "rate_limiting": {
            "max_attempts": MAX_LOGIN_ATTEMPTS,
            "window_seconds": RATE_LIMIT_WINDOW,
            "active_blocks": len([ip for ip, attempts in login_attempts.items() if len(attempts) >= MAX_LOGIN_ATTEMPTS])
        },
        "timestamp": datetime.now().isoformat()
    }

@app.post("/admin/create_user")
async def create_user(user_data: dict, current_user: str = Depends(verify_token)):
    """Create new user - admin only"""
    admin_user = USERS_DB.get(current_user)
    if not admin_user or "admin" not in admin_user.get("permissions", []):
        raise HTTPException(status_code=403, detail="Admin privileges required")
    
    username = user_data.get("username")
    password = user_data.get("password")
    role = user_data.get("role", "user")
    
    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password required")
    
    if username in USERS_DB:
        raise HTTPException(status_code=409, detail="User already exists")
    
    # Set permissions based on role
    permissions = ["read", "write"] if role == "user" else ["read", "write", "delete", "admin"]
    
    USERS_DB[username] = {
        "username": username,
        "password_hash": hash_password(password),
        "role": role,
        "permissions": permissions,
        "created_at": datetime.now().isoformat(),
        "last_login": None,
        "created_by": current_user
    }
    
    return {
        "message": f"User {username} created successfully",
        "user": {
            "username": username,
            "role": role,
            "permissions": permissions
        }
    }

@app.get("/admin/users")
async def list_users(current_user: str = Depends(verify_token)):
    """List all users - admin only"""
    admin_user = USERS_DB.get(current_user)
    if not admin_user or "admin" not in admin_user.get("permissions", []):
        raise HTTPException(status_code=403, detail="Admin privileges required")
    
    users = []
    for username, user_data in USERS_DB.items():
        users.append({
            "username": username,
            "role": user_data["role"],
            "permissions": user_data["permissions"],
            "created_at": user_data.get("created_at"),
            "last_login": user_data.get("last_login")
        })
    
    return {"users": users, "total": len(users)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8094)