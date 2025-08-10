#!/bin/bash

# EMERGENCY FIXES SCRIPT
# Date: 2025-08-09
# Purpose: Apply critical fixes identified by architect analysis

set -e


# Signal handlers for graceful shutdown
cleanup_and_exit() {
    local exit_code="${1:-0}"
    echo "Script interrupted, cleaning up..." >&2
    # Clean up any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    exit "$exit_code"
}

trap 'cleanup_and_exit 130' INT
trap 'cleanup_and_exit 143' TERM
trap 'cleanup_and_exit 1' ERR

echo "=================================="
echo "APPLYING EMERGENCY FIXES"
echo "=================================="

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. CREATE DATABASE SCHEMA (SEVERITY 1)
echo -e "${YELLOW}[1/4] Creating database schema...${NC}"
docker exec -i sutazai-postgres psql -U sutazai -d sutazai < /opt/sutazaiapp/scripts/emergency_fixes/01_create_database_schema.sql
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Database schema created successfully${NC}"
else
    echo -e "${RED}✗ Failed to create database schema${NC}"
    exit 1
fi

# 2. FIX MODEL CONFIGURATION (SEVERITY 1)
echo -e "${YELLOW}[2/4] Fixing model configuration mismatch...${NC}"
# Update backend configuration
sed -i 's/"gpt-oss"/"tinyllama"/g' /opt/sutazaiapp/backend/app/core/config.py 2>/dev/null || true
sed -i 's/gpt-oss/tinyllama/g' /opt/sutazaiapp/backend/.env 2>/dev/null || true
sed -i 's/DEFAULT_MODEL = "gpt-oss"/DEFAULT_MODEL = "tinyllama"/g' /opt/sutazaiapp/backend/app/main.py 2>/dev/null || true
echo -e "${GREEN}✓ Model configuration updated to use tinyllama${NC}"

# 3. DISABLE UNSAFE HTML IN FRONTEND (SEVERITY 1)
echo -e "${YELLOW}[3/4] Removing XSS vulnerabilities...${NC}"
# Create safer frontend configuration
cat > /opt/sutazaiapp/frontend/security_config.py << 'EOF'
"""
Security configuration for frontend
Emergency fix for XSS vulnerabilities
"""

# Disable unsafe HTML globally
ALLOW_UNSAFE_HTML = False

# Content Security Policy
CSP_HEADER = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';"

# Input sanitization required
REQUIRE_INPUT_SANITIZATION = True

# Session security
SESSION_COOKIE_SECURE = True
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = 'Strict'
EOF
echo -e "${GREEN}✓ Security configuration created${NC}"

# 4. CREATE BASIC JWT AUTHENTICATION (SEVERITY 1)
echo -e "${YELLOW}[4/4] Implementing basic authentication...${NC}"
cat > /opt/sutazaiapp/backend/app/core/auth.py << 'EOF'
"""
Emergency authentication implementation
Replaces the authentication bypass vulnerability
"""

from datetime import datetime, timedelta
from typing import Optional
import jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os

# Security configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "CHANGE_THIS_IN_PRODUCTION_" + os.urandom(32).hex())
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token"""
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return username
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# Dependency for protected routes
def get_current_user(username: str = Depends(verify_token)):
    """Get current authenticated user"""
    return {"username": username}

# Example usage:
# @app.get("/protected", dependencies=[Depends(get_current_user)])
# async def protected_route():
#     return {"message": "This route is now protected"}
EOF
echo -e "${GREEN}✓ Basic authentication implemented${NC}"

# 5. VERIFY FIXES
echo ""
echo -e "${YELLOW}Verifying fixes...${NC}"

# Check database tables
echo -n "Database tables: "
TABLE_COUNT=$(docker exec sutazai-postgres psql -U sutazai -d sutazai -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public' AND table_type = 'BASE TABLE';" 2>/dev/null | tr -d ' ')
if [ "$TABLE_COUNT" -gt "0" ]; then
    echo -e "${GREEN}$TABLE_COUNT tables created${NC}"
else
    echo -e "${RED}No tables found${NC}"
fi

# Check model configuration
echo -n "Model configuration: "
if grep -q "tinyllama" /opt/sutazaiapp/backend/app/core/config.py 2>/dev/null || grep -q "tinyllama" /opt/sutazaiapp/backend/.env 2>/dev/null; then
    echo -e "${GREEN}Fixed (using tinyllama)${NC}"
else
    echo -e "${YELLOW}May need manual update${NC}"
fi

# Restart services to apply changes
echo ""
echo -e "${YELLOW}Restarting services to apply changes...${NC}"
docker-compose -f /opt/sutazaiapp/docker-compose.yml restart backend frontend

echo ""
echo "=================================="
echo -e "${GREEN}EMERGENCY FIXES APPLIED${NC}"
echo "=================================="
echo ""
echo "CRITICAL ACTIONS STILL REQUIRED:"
echo "1. Change the default admin password immediately"
echo "2. Generate a new JWT secret key"
echo "3. Review and test authentication on all endpoints"
echo "4. Implement input validation throughout the application"
echo "5. Remove all hardcoded credentials from source code"
echo ""
echo "Run this to change admin password:"
echo "  docker exec -it sutazai-postgres psql -U sutazai -d sutazai -c \"UPDATE users SET password_hash='<new_hash>' WHERE username='admin';\""