"""
Enterprise Security and Compliance System
Implements comprehensive security measures for AGI/ASI system
"""
import asyncio
import logging
import hashlib
import secrets
import jwt
import re
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from enum import Enum
import os
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Security levels for different operations"""
    PUBLIC = "public"
    AUTHENTICATED = "authenticated"
    PRIVILEGED = "privileged"
    ADMIN = "admin"
    SYSTEM = "system"

class ComplianceStandard(Enum):
    """Compliance standards"""
    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    PCI_DSS = "pci_dss"

@dataclass
class SecurityEvent:
    """Security event record"""
    id: str
    timestamp: datetime
    event_type: str
    severity: str
    source: str
    details: Dict[str, Any]
    user_id: Optional[str] = None
    ip_address: Optional[str] = None

@dataclass
class AccessPolicy:
    """Access control policy"""
    id: str
    name: str
    resource: str
    actions: List[str]
    conditions: Dict[str, Any]
    required_level: SecurityLevel

class EncryptionManager:
    """Manages data encryption and key management"""
    
    def __init__(self):
        self.master_key = self._get_or_create_master_key()
        self.fernet = Fernet(self.master_key)
        self.key_rotation_interval = timedelta(days=90)
        
    def _get_or_create_master_key(self) -> bytes:
        """Get or create master encryption key"""
        key_file = os.getenv("ENCRYPTION_KEY_FILE", "/secure/keys/master.key")
        
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            # Generate new key
            key = Fernet.generate_key()
            os.makedirs(os.path.dirname(key_file), exist_ok=True)
            with open(key_file, 'wb') as f:
                f.write(key)
            os.chmod(key_file, 0o600)  # Restrict access
            return key
            
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        if not data:
            return data
        encrypted = self.fernet.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted).decode()
        
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        if not encrypted_data:
            return encrypted_data
        try:
            decoded = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted = self.fernet.decrypt(decoded)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise ValueError("Invalid encrypted data")
            
    def hash_password(self, password: str, salt: Optional[bytes] = None) -> tuple[str, str]:
        """Hash password with salt"""
        if salt is None:
            salt = secrets.token_bytes(32)
            
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        
        return key.decode(), base64.urlsafe_b64encode(salt).decode()
        
    def verify_password(self, password: str, hashed: str, salt: str) -> bool:
        """Verify password against hash"""
        salt_bytes = base64.urlsafe_b64decode(salt.encode())
        check_hash, _ = self.hash_password(password, salt_bytes)
        return secrets.compare_digest(check_hash, hashed)
        
    async def rotate_keys(self):
        """Rotate encryption keys"""
        logger.info("Starting key rotation...")
        # Implementation would re-encrypt data with new keys
        pass

class AuthenticationManager:
    """Manages user authentication and JWT tokens"""
    
    def __init__(self):
        self.secret_key = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
        self.algorithm = "HS256"
        self.access_token_expire = timedelta(hours=1)
        self.refresh_token_expire = timedelta(days=7)
        self.encryption_manager = EncryptionManager()
        
    def create_access_token(self, user_id: str, scopes: List[str] = None) -> str:
        """Create JWT access token"""
        expire = datetime.now(timezone.utc) + self.access_token_expire
        payload = {
            "sub": user_id,
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "type": "access",
            "scopes": scopes or []
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
    def create_refresh_token(self, user_id: str) -> str:
        """Create JWT refresh token"""
        expire = datetime.now(timezone.utc) + self.refresh_token_expire
        payload = {
            "sub": user_id,
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "type": "refresh"
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
    def verify_token(self, token: str, token_type: str = "access") -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            if payload.get("type") != token_type:
                raise ValueError("Invalid token type")
                
            return payload
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise ValueError(f"Invalid token: {str(e)}")
            
    async def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user credentials"""
        # In production, this would check against database
        # For now, return mock user
        if username == "admin" and password == "secure_password":
            return {
                "user_id": "admin_001",
                "username": username,
                "role": "admin",
                "scopes": ["read", "write", "admin"]
            }
        return None

class AuthorizationManager:
    """Manages access control and permissions"""
    
    def __init__(self):
        self.policies: Dict[str, AccessPolicy] = {}
        self.role_permissions: Dict[str, Set[str]] = {
            "admin": {"*"},  # All permissions
            "user": {"read", "write"},
            "viewer": {"read"},
            "agent": {"execute", "read"}
        }
        self._load_policies()
        
    def _load_policies(self):
        """Load access control policies"""
        # Define default policies
        policies = [
            AccessPolicy(
                id="pol_001",
                name="API Access",
                resource="/api/*",
                actions=["GET", "POST"],
                conditions={"authenticated": True},
                required_level=SecurityLevel.AUTHENTICATED
            ),
            AccessPolicy(
                id="pol_002",
                name="Admin Access",
                resource="/api/v1/admin/*",
                actions=["*"],
                conditions={"role": "admin"},
                required_level=SecurityLevel.ADMIN
            ),
            AccessPolicy(
                id="pol_003",
                name="Brain Access",
                resource="/api/v1/brain/*",
                actions=["POST"],
                conditions={"authenticated": True, "rate_limit": 100},
                required_level=SecurityLevel.AUTHENTICATED
            )
        ]
        
        for policy in policies:
            self.policies[policy.id] = policy
            
    def check_permission(self, user: Dict[str, Any], resource: str, action: str) -> bool:
        """Check if user has permission for resource/action"""
        user_role = user.get("role", "viewer")
        user_permissions = self.role_permissions.get(user_role, set())
        
        # Check if user has wildcard permission
        if "*" in user_permissions:
            return True
            
        # Check specific permissions
        for policy in self.policies.values():
            if self._match_resource(resource, policy.resource):
                if action in policy.actions or "*" in policy.actions:
                    # Check conditions
                    if self._check_conditions(user, policy.conditions):
                        return True
                        
        return False
        
    def _match_resource(self, resource: str, pattern: str) -> bool:
        """Match resource against pattern"""
        # Convert pattern to regex
        regex_pattern = pattern.replace("*", ".*")
        return bool(re.match(f"^{regex_pattern}$", resource))
        
    def _check_conditions(self, user: Dict[str, Any], conditions: Dict[str, Any]) -> bool:
        """Check if user meets policy conditions"""
        for key, value in conditions.items():
            if key == "authenticated" and value:
                if not user.get("user_id"):
                    return False
            elif key == "role":
                if user.get("role") != value:
                    return False
            # Add more condition checks as needed
            
        return True

class InputValidator:
    """Validates and sanitizes user inputs"""
    
    def __init__(self):
        self.max_input_length = 10000
        self.blocked_patterns = [
            r'<script[^>]*>.*?</script>',  # Script tags
            r'javascript:',  # JavaScript protocol
            r'on\w+\s*=',  # Event handlers
            r'eval\s*\(',  # Eval function
            r'expression\s*\(',  # CSS expressions
            r'vbscript:',  # VBScript protocol
            r'data:.*base64',  # Data URLs with base64
        ]
        
    def validate_input(self, input_data: str, input_type: str = "text") -> str:
        """Validate and sanitize input"""
        if not input_data:
            return input_data
            
        # Check length
        if len(input_data) > self.max_input_length:
            raise ValueError(f"Input exceeds maximum length of {self.max_input_length}")
            
        # Check for malicious patterns
        for pattern in self.blocked_patterns:
            if re.search(pattern, input_data, re.IGNORECASE | re.DOTALL):
                raise ValueError("Potentially malicious content detected")
                
        # Type-specific validation
        if input_type == "email":
            if not self._validate_email(input_data):
                raise ValueError("Invalid email format")
        elif input_type == "url":
            if not self._validate_url(input_data):
                raise ValueError("Invalid URL format")
        elif input_type == "json":
            if not self._validate_json(input_data):
                raise ValueError("Invalid JSON format")
                
        # Sanitize input
        sanitized = self._sanitize_input(input_data, input_type)
        
        return sanitized
        
    def _validate_email(self, email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
        
    def _validate_url(self, url: str) -> bool:
        """Validate URL format"""
        pattern = r'^https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/.*)?$'
        return bool(re.match(pattern, url))
        
    def _validate_json(self, json_str: str) -> bool:
        """Validate JSON format"""
        try:
            import json
            json.loads(json_str)
            return True
        except:
            return False
            
    def _sanitize_input(self, input_data: str, input_type: str) -> str:
        """Sanitize input based on type"""
        # Remove null bytes
        sanitized = input_data.replace('\x00', '')
        
        # HTML encode special characters for text
        if input_type == "text":
            sanitized = sanitized.replace('<', '&lt;').replace('>', '&gt;')
            
        return sanitized.strip()

class AuditLogger:
    """Logs security events for audit trail"""
    
    def __init__(self):
        self.events: List[SecurityEvent] = []
        self.retention_days = 90
        self.log_file = os.getenv("AUDIT_LOG_FILE", "/logs/security_audit.log")
        
    async def log_event(self, event_type: str, severity: str, 
                       source: str, details: Dict[str, Any],
                       user_id: Optional[str] = None,
                       ip_address: Optional[str] = None):
        """Log security event"""
        event = SecurityEvent(
            id=f"evt_{datetime.utcnow().timestamp()}_{secrets.token_hex(4)}",
            timestamp=datetime.utcnow(),
            event_type=event_type,
            severity=severity,
            source=source,
            details=details,
            user_id=user_id,
            ip_address=ip_address
        )
        
        self.events.append(event)
        
        # Write to file
        await self._write_to_file(event)
        
        # Alert on critical events
        if severity == "critical":
            await self._send_alert(event)
            
    async def _write_to_file(self, event: SecurityEvent):
        """Write event to audit log file"""
        log_entry = {
            "id": event.id,
            "timestamp": event.timestamp.isoformat(),
            "type": event.event_type,
            "severity": event.severity,
            "source": event.source,
            "user_id": event.user_id,
            "ip_address": event.ip_address,
            "details": event.details
        }
        
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        with open(self.log_file, 'a') as f:
            import json
            f.write(json.dumps(log_entry) + '\n')
            
    async def _send_alert(self, event: SecurityEvent):
        """Send alert for critical events"""
        logger.critical(f"SECURITY ALERT: {event.event_type} - {event.details}")
        # In production, this would send notifications
        
    async def get_audit_trail(self, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Get audit trail with optional filters"""
        trail = []
        
        for event in self.events:
            if filters:
                # Apply filters
                if filters.get("user_id") and event.user_id != filters["user_id"]:
                    continue
                if filters.get("event_type") and event.event_type != filters["event_type"]:
                    continue
                if filters.get("severity") and event.severity != filters["severity"]:
                    continue
                    
            trail.append({
                "id": event.id,
                "timestamp": event.timestamp.isoformat(),
                "type": event.event_type,
                "severity": event.severity,
                "source": event.source,
                "user": event.user_id,
                "ip": event.ip_address,
                "details": event.details
            })
            
        return trail

class RateLimiter:
    """Implements rate limiting for API protection"""
    
    def __init__(self):
        self.limits: Dict[str, Dict[str, Any]] = {}
        self.default_limit = 100  # requests per minute
        self.window_size = 60  # seconds
        
    async def check_rate_limit(self, identifier: str, limit: Optional[int] = None) -> bool:
        """Check if request is within rate limit"""
        current_time = time.time()
        limit = limit or self.default_limit
        
        if identifier not in self.limits:
            self.limits[identifier] = {
                "requests": [],
                "blocked_until": 0
            }
            
        # Check if currently blocked
        if current_time < self.limits[identifier]["blocked_until"]:
            return False
            
        # Remove old requests outside window
        self.limits[identifier]["requests"] = [
            req_time for req_time in self.limits[identifier]["requests"]
            if current_time - req_time < self.window_size
        ]
        
        # Check limit
        if len(self.limits[identifier]["requests"]) >= limit:
            # Block for window size
            self.limits[identifier]["blocked_until"] = current_time + self.window_size
            return False
            
        # Add current request
        self.limits[identifier]["requests"].append(current_time)
        return True

class ComplianceManager:
    """Manages compliance with various standards"""
    
    def __init__(self):
        self.enabled_standards: Set[ComplianceStandard] = {
            ComplianceStandard.GDPR,
            ComplianceStandard.SOC2
        }
        self.data_retention_policies = {
            "user_data": 365,  # days
            "logs": 90,
            "analytics": 180,
            "security_events": 730
        }
        
    async def check_gdpr_compliance(self, operation: str, data: Dict[str, Any]) -> bool:
        """Check GDPR compliance for operation"""
        if ComplianceStandard.GDPR not in self.enabled_standards:
            return True
            
        # Check for personal data
        if self._contains_personal_data(data):
            # Verify consent
            if not data.get("user_consent"):
                logger.warning(f"GDPR: Operation {operation} requires user consent")
                return False
                
            # Check data minimization
            if not self._check_data_minimization(data):
                logger.warning(f"GDPR: Operation {operation} violates data minimization")
                return False
                
        return True
        
    async def anonymize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize personal data"""
        anonymized = data.copy()
        
        # Anonymize common PII fields
        pii_fields = ["email", "name", "phone", "address", "ip_address", "user_id"]
        
        for field in pii_fields:
            if field in anonymized:
                anonymized[field] = self._hash_value(str(anonymized[field]))
                
        return anonymized
        
    async def handle_data_request(self, request_type: str, user_id: str) -> Dict[str, Any]:
        """Handle GDPR data requests"""
        if request_type == "access":
            # Right to access
            return await self._get_user_data(user_id)
        elif request_type == "portability":
            # Right to data portability
            return await self._export_user_data(user_id)
        elif request_type == "erasure":
            # Right to be forgotten
            return await self._delete_user_data(user_id)
        elif request_type == "rectification":
            # Right to rectification
            return {"status": "pending", "message": "Please provide corrected data"}
        else:
            raise ValueError(f"Unknown request type: {request_type}")
            
    def _contains_personal_data(self, data: Dict[str, Any]) -> bool:
        """Check if data contains personal information"""
        pii_indicators = ["email", "name", "phone", "ssn", "address", "ip_address"]
        return any(indicator in str(data).lower() for indicator in pii_indicators)
        
    def _check_data_minimization(self, data: Dict[str, Any]) -> bool:
        """Check if data collection follows minimization principle"""
        # Implementation would check against defined schemas
        return True
        
    def _hash_value(self, value: str) -> str:
        """Hash value for anonymization"""
        return hashlib.sha256(value.encode()).hexdigest()[:12]
        
    async def _get_user_data(self, user_id: str) -> Dict[str, Any]:
        """Get all user data for access request"""
        # Implementation would gather data from all sources
        return {"user_id": user_id, "data": "User data placeholder"}
        
    async def _export_user_data(self, user_id: str) -> Dict[str, Any]:
        """Export user data in portable format"""
        # Implementation would create portable data package
        return {"format": "json", "data": {"user_id": user_id}}
        
    async def _delete_user_data(self, user_id: str) -> Dict[str, Any]:
        """Delete user data for erasure request"""
        # Implementation would delete/anonymize user data
        return {"status": "completed", "user_id": user_id}

class SecurityManager:
    """Main security coordinator"""
    
    def __init__(self):
        self.encryption = EncryptionManager()
        self.auth = AuthenticationManager()
        self.authz = AuthorizationManager()
        self.validator = InputValidator()
        self.audit = AuditLogger()
        self.rate_limiter = RateLimiter()
        self.compliance = ComplianceManager()
        
    async def secure_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process request with full security checks"""
        # Log request
        await self.audit.log_event(
            "api_request",
            "info",
            "security_manager",
            {"endpoint": request_data.get("path")},
            user_id=request_data.get("user_id"),
            ip_address=request_data.get("ip")
        )
        
        # Rate limiting
        if not await self.rate_limiter.check_rate_limit(
            request_data.get("ip", "unknown")
        ):
            await self.audit.log_event(
                "rate_limit_exceeded",
                "warning",
                "security_manager",
                {"ip": request_data.get("ip")},
                ip_address=request_data.get("ip")
            )
            raise ValueError("Rate limit exceeded")
            
        # Input validation
        if "body" in request_data:
            for key, value in request_data["body"].items():
                if isinstance(value, str):
                    request_data["body"][key] = self.validator.validate_input(value)
                    
        # Authorization
        if request_data.get("user"):
            if not self.authz.check_permission(
                request_data["user"],
                request_data.get("path", ""),
                request_data.get("method", "GET")
            ):
                await self.audit.log_event(
                    "authorization_failed",
                    "warning",
                    "security_manager",
                    {
                        "user": request_data["user"]["user_id"],
                        "resource": request_data.get("path")
                    },
                    user_id=request_data["user"]["user_id"]
                )
                raise ValueError("Access denied")
                
        # Compliance checks
        if request_data.get("body"):
            if not await self.compliance.check_gdpr_compliance(
                request_data.get("path", ""),
                request_data["body"]
            ):
                raise ValueError("Compliance check failed")
                
        return request_data
        
    def get_security_headers(self) -> Dict[str, str]:
        """Get security headers for responses"""
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline';",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
        }
        
    async def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report"""
        # Get recent security events
        recent_events = await self.audit.get_audit_trail()
        
        # Count by severity
        severity_counts = {}
        for event in recent_events:
            severity = event.get("severity", "info")
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "summary": {
                "total_events": len(recent_events),
                "severity_breakdown": severity_counts,
                "compliance_standards": [s.value for s in self.compliance.enabled_standards],
                "encryption_enabled": True,
                "rate_limiting_enabled": True
            },
            "recent_alerts": [
                e for e in recent_events
                if e.get("severity") in ["critical", "high"]
            ][:10],
            "recommendations": [
                "Enable two-factor authentication for all admin accounts",
                "Review and update access policies quarterly",
                "Conduct regular security audits",
                "Implement automated vulnerability scanning"
            ]
        }

# Global security manager instance
security_manager = SecurityManager()

# FastAPI integration
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import time

router = APIRouter()
security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from JWT token"""
    try:
        payload = security_manager.auth.verify_token(credentials.credentials)
        return {
            "user_id": payload["sub"],
            "scopes": payload.get("scopes", [])
        }
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

@router.post("/auth/login")
async def login(username: str, password: str):
    """Authenticate user and return tokens"""
    user = await security_manager.auth.authenticate_user(username, password)
    
    if not user:
        await security_manager.audit.log_event(
            "login_failed",
            "warning",
            "auth",
            {"username": username}
        )
        raise HTTPException(status_code=401, detail="Invalid credentials")
        
    access_token = security_manager.auth.create_access_token(
        user["user_id"],
        user.get("scopes", [])
    )
    refresh_token = security_manager.auth.create_refresh_token(user["user_id"])
    
    await security_manager.audit.log_event(
        "login_success",
        "info",
        "auth",
        {"username": username},
        user_id=user["user_id"]
    )
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "user": user
    }

@router.post("/auth/refresh")
async def refresh_token(refresh_token: str):
    """Refresh access token"""
    try:
        payload = security_manager.auth.verify_token(refresh_token, "refresh")
        new_access_token = security_manager.auth.create_access_token(payload["sub"])
        
        return {
            "access_token": new_access_token,
            "token_type": "bearer"
        }
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

@router.get("/security/report", dependencies=[Depends(get_current_user)])
async def get_security_report():
    """Get security report (admin only)"""
    # Check admin permission
    # ... permission check ...
    
    report = await security_manager.generate_security_report()
    return report

@router.get("/audit/trail", dependencies=[Depends(get_current_user)])
async def get_audit_trail(
    user_id: Optional[str] = None,
    event_type: Optional[str] = None,
    severity: Optional[str] = None
):
    """Get audit trail with filters"""
    filters = {}
    if user_id:
        filters["user_id"] = user_id
    if event_type:
        filters["event_type"] = event_type
    if severity:
        filters["severity"] = severity
        
    trail = await security_manager.audit.get_audit_trail(filters)
    return {"count": len(trail), "events": trail}

@router.post("/compliance/gdpr/{request_type}")
async def handle_gdpr_request(
    request_type: str,
    current_user: Dict = Depends(get_current_user)
):
    """Handle GDPR data requests"""
    if request_type not in ["access", "portability", "erasure", "rectification"]:
        raise HTTPException(status_code=400, detail="Invalid request type")
        
    result = await security_manager.compliance.handle_data_request(
        request_type,
        current_user["user_id"]
    )
    
    await security_manager.audit.log_event(
        f"gdpr_{request_type}_request",
        "info",
        "compliance",
        {"request_type": request_type},
        user_id=current_user["user_id"]
    )
    
    return result

@router.post("/encrypt")
async def encrypt_data(data: str, current_user: Dict = Depends(get_current_user)):
    """Encrypt sensitive data"""
    encrypted = security_manager.encryption.encrypt_data(data)
    
    await security_manager.audit.log_event(
        "data_encrypted",
        "info",
        "encryption",
        {"data_length": len(data)},
        user_id=current_user["user_id"]
    )
    
    return {"encrypted": encrypted}

@router.post("/decrypt")
async def decrypt_data(encrypted: str, current_user: Dict = Depends(get_current_user)):
    """Decrypt sensitive data"""
    try:
        decrypted = security_manager.encryption.decrypt_data(encrypted)
        
        await security_manager.audit.log_event(
            "data_decrypted",
            "info",
            "encryption",
            {"success": True},
            user_id=current_user["user_id"]
        )
        
        return {"decrypted": decrypted}
    except ValueError as e:
        await security_manager.audit.log_event(
            "decryption_failed",
            "warning",
            "encryption",
            {"error": str(e)},
            user_id=current_user["user_id"]
        )
        raise HTTPException(status_code=400, detail=str(e))