#!/usr/bin/env python3
"""
SutazAI Security Hardening Implementation Script

This script implements critical security fixes identified in the security audit.
Run with appropriate permissions and backup your system before execution.
"""

import os
import sys
import json
import yaml
import secrets
import hashlib
from pathlib import Path
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SecurityHardening:
    """Implements security hardening measures for SutazAI system"""
    
    def __init__(self, base_path: str = "/opt/sutazaiapp"):
        self.base_path = Path(base_path)
        self.backup_path = self.base_path / "security_backups"
        self.backup_path.mkdir(exist_ok=True)
        
    def create_backup(self, file_path: Path) -> None:
        """Create backup of file before modification"""
        if file_path.exists():
            backup_file = self.backup_path / f"{file_path.name}.backup"
            backup_file.write_text(file_path.read_text())
            logger.info(f"Backup created: {backup_file}")
    
    def generate_secure_secret(self, length: int = 64) -> str:
        """Generate cryptographically secure secret"""
        return secrets.token_urlsafe(length)
    
    def fix_authentication_system(self) -> None:
        """Replace mock authentication with secure implementation"""
        logger.info("Fixing authentication system...")
        
        auth_file = self.base_path / "backend/app/core/security.py"
        self.create_backup(auth_file)
        
        secure_auth_code = '''"""
Secure Authentication System for SutazAI
"""
import jwt
import bcrypt
import secrets
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)

class SecureAuth:
    """Secure authentication implementation"""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        if len(secret_key) < 32:
            raise ValueError("Secret key must be at least 32 characters")
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.token_expiry = timedelta(minutes=30)
        self.refresh_expiry = timedelta(days=7)
        
        # In production, use database or secure storage
        self.users_db = {
            "admin": {
                "user_id": "admin_001",
                "password_hash": self._hash_password("CHANGE_IN_PRODUCTION_2025!"),
                "role": "admin",
                "scopes": ["read", "write", "admin"]
            }
        }
    
    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    
    async def authenticate_user(self, username: str, password: str) -> Optional[Dict]:
        """Authenticate user with secure password verification"""
        user = self.users_db.get(username)
        if not user:
            # Prevent timing attacks
            bcrypt.hashpw(b"fake_password", bcrypt.gensalt())
            return None
            
        if not self._verify_password(password, user["password_hash"]):
            return None
            
        return {
            "user_id": user["user_id"],
            "username": username,
            "role": user["role"],
            "scopes": user["scopes"]
        }
    
    def create_access_token(self, user_id: str, scopes: List[str] = None) -> str:
        """Create secure JWT access token"""
        payload = {
            "sub": user_id,
            "scopes": scopes or [],
            "exp": datetime.utcnow() + self.token_expiry,
            "iat": datetime.utcnow(),
            "jti": secrets.token_urlsafe(16)  # JWT ID for revocation
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(self, user_id: str) -> str:
        """Create secure refresh token"""
        payload = {
            "sub": user_id,
            "type": "refresh",
            "exp": datetime.utcnow() + self.refresh_expiry,
            "iat": datetime.utcnow(),
            "jti": secrets.token_urlsafe(16)
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Dict:
        """Verify JWT token with proper validation"""
        try:
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm],
                options={"require": ["exp", "iat", "sub"]}
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")

class SecurityManager:
    """Main security manager"""
    
    def __init__(self):
        secret_key = os.getenv("SECRET_KEY")
        if not secret_key or len(secret_key) < 32:
            logger.warning("Weak or missing SECRET_KEY, generating secure one")
            secret_key = secrets.token_urlsafe(64)
        
        self.auth = SecureAuth(secret_key)
    
    async def generate_security_report(self) -> Dict:
        """Generate comprehensive security report"""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "authentication": {
                "secure_jwt": True,
                "password_hashing": "bcrypt",
                "token_expiry": self.auth.token_expiry.total_seconds(),
                "algorithm": self.auth.algorithm
            },
            "recommendations": [
                "Change default admin password immediately",
                "Implement user database with proper access controls",
                "Add rate limiting for authentication endpoints",
                "Enable MFA for admin accounts"
            ]
        }

# Global security manager instance
security_manager = SecurityManager()
'''
        
        auth_file.write_text(secure_auth_code)
        logger.info("Secure authentication system implemented")
    
    def fix_docker_security(self) -> None:
        """Fix Docker security configurations"""
        logger.info("Fixing Docker security configurations...")
        
        docker_compose_file = self.base_path / "docker-compose.yml"
        self.create_backup(docker_compose_file)
        
        # Read and modify docker-compose.yml
        with open(docker_compose_file, 'r') as f:
            content = f.read()
        
        # Remove Docker socket mounts (comment them out)
        content = content.replace(
            '- /var/run/docker.sock:/var/run/docker.sock:ro',
            '# SECURITY FIX: Docker socket mount removed\n      # - /var/run/docker.sock:/var/run/docker.sock:ro'
        )
        content = content.replace(
            '- /var/run/docker.sock:/var/run/docker.sock',
            '# SECURITY FIX: Docker socket mount removed\n      # - /var/run/docker.sock:/var/run/docker.sock'
        )
        
        # Remove exposed database ports
        content = content.replace(
            '- "5432:5432"',
            '# SECURITY FIX: Database port exposure removed\n      # - "5432:5432"'
        )
        content = content.replace(
            '- "6379:6379"',
            '# SECURITY FIX: Redis port exposure removed\n      # - "6379:6379"'
        )
        
        with open(docker_compose_file, 'w') as f:
            f.write(content)
        
        logger.info("Docker security configurations fixed")
    
    def fix_cors_configuration(self) -> None:
        """Fix CORS configuration to be more restrictive"""
        logger.info("Fixing CORS configuration...")
        
        main_file = self.base_path / "backend/app/working_main.py"
        self.create_backup(main_file)
        
        with open(main_file, 'r') as f:
            content = f.read()
        
        # Replace wildcard CORS with more secure configuration
        old_cors = '''app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)'''
        
        new_cors = '''# SECURITY FIX: Restricted CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "https://sutazai.yourdomain.com"  # Replace with your actual domain
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Requested-With"],
    expose_headers=["X-Total-Count"],
)'''
        
        content = content.replace(old_cors, new_cors)
        
        with open(main_file, 'w') as f:
            f.write(content)
        
        logger.info("CORS configuration secured")
    
    def implement_input_validation(self) -> None:
        """Add input validation to API endpoints"""
        logger.info("Implementing input validation...")
        
        validation_utils = '''"""
Input validation utilities for SutazAI
"""
import re
import html
from typing import Any, Dict, List
from fastapi import HTTPException

class InputValidator:
    """Secure input validation utilities"""
    
    @staticmethod
    def sanitize_string(input_str: str, max_length: int = 1000) -> str:
        """Sanitize string input"""
        if not isinstance(input_str, str):
            raise HTTPException(status_code=400, detail="Input must be a string")
        
        if len(input_str) > max_length:
            raise HTTPException(status_code=400, detail=f"Input too long (max {max_length} chars)")
        
        # HTML escape to prevent XSS
        sanitized = html.escape(input_str)
        
        # Remove potentially dangerous patterns
        dangerous_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
            r'eval\s*\(',
            r'exec\s*\(',
        ]
        
        for pattern in dangerous_patterns:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
        
        return sanitized.strip()
    
    @staticmethod
    def validate_agent_task(task: Dict[str, Any]) -> Dict[str, Any]:
        """Validate agent task input"""
        if not isinstance(task, dict):
            raise HTTPException(status_code=400, detail="Task must be a dictionary")
        
        # Required fields
        required_fields = ['type', 'description']
        for field in required_fields:
            if field not in task:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        # Validate task type
        allowed_task_types = [
            'analysis', 'generation', 'processing', 'query', 'computation'
        ]
        if task['type'] not in allowed_task_types:
            raise HTTPException(status_code=400, detail="Invalid task type")
        
        # Sanitize string fields
        for key, value in task.items():
            if isinstance(value, str):
                task[key] = InputValidator.sanitize_string(value)
        
        return task
    
    @staticmethod
    def validate_model_name(model_name: str) -> str:
        """Validate model name to prevent path traversal"""
        if not re.match(r'^[a-zA-Z0-9._-]+$', model_name):
            raise HTTPException(status_code=400, detail="Invalid model name format")
        
        if '..' in model_name or '/' in model_name:
            raise HTTPException(status_code=400, detail="Model name contains invalid characters")
        
        return model_name

# Global validator instance
validator = InputValidator()
'''
        
        validation_file = self.base_path / "backend/app/core/validation.py"
        validation_file.write_text(validation_utils)
        logger.info("Input validation implemented")
    
    def create_environment_template(self) -> None:
        """Create secure environment template"""
        logger.info("Creating secure environment template...")
        
        env_template = f'''# SutazAI Secure Environment Configuration
# Generated by Security Hardening Script

# Security Configuration
SECRET_KEY={self.generate_secure_secret(64)}
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Database Configuration (Change these!)
POSTGRES_USER=sutazai_secure
POSTGRES_PASSWORD={self.generate_secure_secret(32)}
POSTGRES_DB=sutazai_production
POSTGRES_HOST=postgres
POSTGRES_PORT=5432

# Redis Configuration (Change these!)
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD={self.generate_secure_secret(32)}

# Vector Database API Keys (Generate new ones!)
CHROMADB_API_KEY={self.generate_secure_secret(48)}
QDRANT_API_KEY={self.generate_secure_secret(48)}

# Neo4j Configuration (Change these!)
NEO4J_PASSWORD={self.generate_secure_secret(32)}

# Model Configuration
OLLAMA_HOST=http://ollama:11434
DEFAULT_MODEL=qwen2.5:3b
EMBEDDING_MODEL=nomic-embed-text

# Environment
SUTAZAI_ENV=production
DEBUG=false
LOG_LEVEL=INFO

# Monitoring
METRICS_ENABLED=true
PROMETHEUS_PORT=9090

# Security Headers
ENABLE_SECURITY_HEADERS=true
HSTS_MAX_AGE=31536000
CSP_ENABLED=true
'''
        
        env_file = self.base_path / ".env.secure.template"
        env_file.write_text(env_template)
        logger.info(f"Secure environment template created: {env_file}")
        logger.warning("IMPORTANT: Copy .env.secure.template to .env and customize the values!")
    
    def create_security_middleware(self) -> None:
        """Create security middleware for enhanced protection"""
        logger.info("Creating security middleware...")
        
        middleware_code = '''"""
Security middleware for SutazAI
"""
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import time
import logging
from collections import defaultdict, deque
from typing import Dict, Deque
import ipaddress

logger = logging.getLogger(__name__)

class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware with rate limiting and protection features"""
    
    def __init__(self, app, rate_limit: int = 100, time_window: int = 60):
        super().__init__(app)
        self.rate_limit = rate_limit
        self.time_window = time_window
        self.request_counts: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=rate_limit))
        
        # Blocked IPs (in production, use Redis or database)
        self.blocked_ips = set()
        
        # Suspicious patterns
        self.suspicious_patterns = [
            'union select', 'drop table', '<script>', 'javascript:',
            '../', '..\\\\', 'eval(', 'exec(', '__import__'
        ]
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        client_ip = self._get_client_ip(request)
        
        # Check if IP is blocked
        if client_ip in self.blocked_ips:
            return JSONResponse(
                status_code=403,
                content={"detail": "IP address blocked"}
            )
        
        # Rate limiting
        if not self._check_rate_limit(client_ip):
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"}
            )
        
        # Check for suspicious patterns
        if await self._contains_suspicious_content(request):
            logger.warning(f"Suspicious request from IP: {client_ip}")
            self._track_suspicious_activity(client_ip)
            return JSONResponse(
                status_code=400,
                content={"detail": "Request blocked by security filter"}
            )
        
        # Process request
        response = await call_next(request)
        
        # Add security headers
        self._add_security_headers(response)
        
        # Log processing time
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        # Check for forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    def _check_rate_limit(self, client_ip: str) -> bool:
        """Check if client has exceeded rate limit"""
        now = time.time()
        
        # Clean old entries
        while (self.request_counts[client_ip] and 
               now - self.request_counts[client_ip][0] > self.time_window):
            self.request_counts[client_ip].popleft()
        
        # Check current count
        if len(self.request_counts[client_ip]) >= self.rate_limit:
            return False
        
        # Add current request
        self.request_counts[client_ip].append(now)
        return True
    
    async def _contains_suspicious_content(self, request: Request) -> bool:
        """Check request for suspicious content"""
        # Check URL path
        path = str(request.url.path).lower()
        for pattern in self.suspicious_patterns:
            if pattern in path:
                return True
        
        # Check query parameters
        query = str(request.url.query).lower()
        for pattern in self.suspicious_patterns:
            if pattern in query:
                return True
        
        # Check headers
        for header_name, header_value in request.headers.items():
            if any(pattern in header_value.lower() for pattern in self.suspicious_patterns):
                return True
        
        return False
    
    def _track_suspicious_activity(self, client_ip: str):
        """Track suspicious activity (implement blocking logic as needed)"""
        # In production, implement proper tracking and blocking
        logger.warning(f"Suspicious activity detected from {client_ip}")
    
    def _add_security_headers(self, response: Response):
        """Add security headers to response"""
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
        }
        
        for header, value in security_headers.items():
            response.headers[header] = value
'''
        
        middleware_file = self.base_path / "backend/app/middleware/security.py"
        middleware_file.parent.mkdir(exist_ok=True)
        middleware_file.write_text(middleware_code)
        logger.info("Security middleware created")
    
    def run_security_hardening(self) -> None:
        """Run all security hardening measures"""
        logger.info("Starting SutazAI Security Hardening...")
        
        try:
            self.fix_authentication_system()
            self.fix_docker_security()
            self.fix_cors_configuration()
            self.implement_input_validation()
            self.create_environment_template()
            self.create_security_middleware()
            
            logger.info("✅ Security hardening completed successfully!")
            logger.info("📋 Next steps:")
            logger.info("1. Copy .env.secure.template to .env and customize values")
            logger.info("2. Change default admin password immediately")
            logger.info("3. Review and test all security fixes")
            logger.info("4. Restart all services with new configuration")
            logger.info("5. Run security validation tests")
            
        except Exception as e:
            logger.error(f"❌ Security hardening failed: {e}")
            sys.exit(1)

def main():
    """Main execution function"""
    if len(sys.argv) > 1:
        base_path = sys.argv[1]
    else:
        base_path = "/opt/sutazaiapp"
    
    if not os.path.exists(base_path):
        logger.error(f"Base path does not exist: {base_path}")
        sys.exit(1)
    
    hardening = SecurityHardening(base_path)
    hardening.run_security_hardening()

if __name__ == "__main__":
    main()