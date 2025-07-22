#!/usr/bin/env python3
"""
Enterprise Security and Authentication Service
"""

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
from datetime import datetime, timedelta
import json
import jwt
import hashlib
from typing import Dict, Optional

app = FastAPI(title="SutazAI Security Service", version="1.0")
security = HTTPBearer()

# In production, use environment variables
SECRET_KEY = "sutazai_secret_key_2024"
ALGORITHM = "HS256"

# Mock user database
USERS_DB = {
    "admin": {
        "username": "admin",
        "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
        "role": "administrator",
        "permissions": ["read", "write", "delete", "admin"]
    },
    "user": {
        "username": "user", 
        "password_hash": hashlib.sha256("user123".encode()).hexdigest(),
        "role": "user",
        "permissions": ["read", "write"]
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

@app.post("/login")
async def login(credentials: dict):
    try:
        username = credentials.get("username")
        password = credentials.get("password")
        
        if not username or not password:
            raise HTTPException(status_code=400, detail="Username and password required")
        
        user = USERS_DB.get(username)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        if password_hash != user["password_hash"]:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        access_token = create_access_token(data={"sub": username})
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "username": user["username"],
                "role": user["role"], 
                "permissions": user["permissions"]
            },
            "expires_in": 86400  # 24 hours
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
        "features": [
            "JWT Authentication",
            "Role-based Access Control",
            "Permission Management", 
            "Token Validation",
            "Secure Password Hashing"
        ],
        "security_level": "Enterprise",
        "active_features": 5,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8094)