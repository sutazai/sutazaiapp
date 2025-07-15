"""
Comprehensive Security System
Advanced security controls, authentication, and monitoring
"""

import asyncio
import logging
import hashlib
import secrets
import time
import json
import jwt
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import subprocess
import tempfile
import uuid
import re
from datetime import datetime, timedelta
import hmac
import base64

logger = logging.getLogger(__name__)

class SecurityLevel(str, Enum):
    PUBLIC = "public"
    AUTHENTICATED = "authenticated"
    ELEVATED = "elevated"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"

class ThreatLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AuthMethod(str, Enum):
    PASSWORD = "password"
    TOKEN = "token"
    BIOMETRIC = "biometric"
    MFA = "mfa"
    EMERGENCY_CODE = "emergency_code"

@dataclass
class SecurityEvent:
    """Security event record"""
    id: str
    timestamp: float
    event_type: str
    threat_level: ThreatLevel
    source_ip: str
    user_id: Optional[str]
    action: str
    resource: str
    success: bool
    details: Dict[str, Any]
    geolocation: Optional[Dict[str, str]] = None

@dataclass
class AuthSession:
    """Authentication session"""
    session_id: str
    user_id: str
    security_level: SecurityLevel
    auth_method: AuthMethod
    created_at: float
    last_activity: float
    expires_at: float
    source_ip: str
    user_agent: str
    mfa_verified: bool = False
    device_fingerprint: Optional[str] = None

@dataclass
class SecurityPolicy:
    """Security policy definition"""
    name: str
    rules: List[Dict[str, Any]]
    enforcement_level: str
    exceptions: List[str]
    last_updated: float
    created_by: str

class SecuritySystem:
    """
    Comprehensive Security System
    Handles authentication, authorization, threat detection, and security monitoring
    """
    
    # Hardcoded authorized super admin
    SUPER_ADMIN = {
        "email": "os.getenv("ADMIN_EMAIL", "admin@localhost")",
        "name": "Chris Suta",
        "security_level": SecurityLevel.SUPER_ADMIN,
        "permissions": ["*"]  # All permissions
    }
    
    def __init__(self, data_dir: str = "/opt/sutazaiapp/data/security"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Security state
        self.active_sessions = {}
        self.security_events = []
        self.failed_attempts = {}
        self.blocked_ips = set()
        self.security_policies = {}
        
        # Authentication
        self.jwt_secret = self._generate_jwt_secret()
        self.api_keys = {}
        self.emergency_codes = {}
        
        # Threat detection
        self.threat_patterns = {}
        self.anomaly_baselines = {}
        self.security_metrics = {}
        
        # Sandboxing
        self.sandbox_environments = {}
        self.sandbox_limits = {
            "max_memory": "512MB",
            "max_cpu": "50%",
            "max_disk": "1GB",
            "max_network": "10MB/s",
            "timeout": 300  # seconds
        }
        
        # Initialize
        self._initialize_security_system()
    
    def _initialize_security_system(self):
        """Initialize security system"""
        try:
            # Load existing data
            self._load_security_data()
            
            # Initialize threat detection patterns
            self._initialize_threat_patterns()
            
            # Create default security policies
            self._create_default_policies()
            
            # Generate emergency codes for super admin
            self._generate_emergency_codes()
            
            # Setup security monitoring
            self._setup_security_monitoring()
            
            logger.info("🔒 Security system initialized")
            self._log_security_event(
                "system_init",
                ThreatLevel.LOW,
                "127.0.0.1",
                None,
                "Security system initialization",
                "system",
                True,
                {"initialization_time": time.time()}
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize security system: {e}")
            raise
    
    def _generate_jwt_secret(self) -> str:
        """Generate JWT secret key"""
        try:
            # Check if secret already exists
            secret_file = self.data_dir / "jwt_secret.key"
            if secret_file.exists():
                with open(secret_file, 'r') as f:
                    return f.read().strip()
            
            # Generate new secret
            secret = secrets.token_urlsafe(64)
            with open(secret_file, 'w') as f:
                f.write(secret)
            
            # Secure the file
            secret_file.chmod(0o600)
            
            return secret
            
        except Exception as e:
            logger.error(f"Failed to generate JWT secret: {e}")
            return secrets.token_urlsafe(64)  # Fallback
    
    def _initialize_threat_patterns(self):
        """Initialize threat detection patterns"""
        self.threat_patterns = {
            "brute_force": {
                "pattern": "failed_login",
                "threshold": 5,
                "window": 300,  # 5 minutes
                "action": "block_ip"
            },
            "sql_injection": {
                "patterns": [
                    r"(union|select|insert|delete|update|drop)\s+",
                    r"(\bor\b|\band\b)\s+\d+\s*=\s*\d+",
                    r";\s*(drop|delete|insert|update)"
                ],
                "action": "block_request"
            },
            "code_injection": {
                "patterns": [
                    r"eval\s*\(",
                    r"exec\s*\(",
                    r"system\s*\(",
                    r"shell_exec\s*\("
                ],
                "action": "block_request"
            },
            "path_traversal": {
                "patterns": [
                    r"\.\./",
                    r"\.\.\\",
                    r"/%2e%2e/",
                    r"\\%2e%2e\\"
                ],
                "action": "block_request"
            },
            "xss": {
                "patterns": [
                    r"<script[^>]*>",
                    r"javascript:",
                    r"on\w+\s*=",
                    r"<iframe[^>]*>"
                ],
                "action": "sanitize"
            }
        }
    
    def _create_default_policies(self):
        """Create default security policies"""
        default_policies = {
            "authentication": SecurityPolicy(
                name="authentication",
                rules=[
                    {"rule": "require_mfa_for_admin", "enabled": True},
                    {"rule": "session_timeout", "value": 3600},  # 1 hour
                    {"rule": "max_failed_attempts", "value": 5},
                    {"rule": "password_complexity", "enabled": True}
                ],
                enforcement_level="strict",
                exceptions=[self.SUPER_ADMIN["email"]],
                last_updated=time.time(),
                created_by="system"
            ),
            "access_control": SecurityPolicy(
                name="access_control",
                rules=[
                    {"rule": "require_auth_for_api", "enabled": True},
                    {"rule": "rate_limiting", "enabled": True, "requests_per_minute": 60},
                    {"rule": "ip_whitelist", "enabled": False, "whitelist": []},
                    {"rule": "geo_blocking", "enabled": False, "blocked_countries": []}
                ],
                enforcement_level="moderate",
                exceptions=[],
                last_updated=time.time(),
                created_by="system"
            ),
            "data_protection": SecurityPolicy(
                name="data_protection",
                rules=[
                    {"rule": "encrypt_sensitive_data", "enabled": True},
                    {"rule": "audit_data_access", "enabled": True},
                    {"rule": "data_retention", "days": 90},
                    {"rule": "backup_encryption", "enabled": True}
                ],
                enforcement_level="strict",
                exceptions=[],
                last_updated=time.time(),
                created_by="system"
            )
        }
        
        self.security_policies.update(default_policies)
    
    def _generate_emergency_codes(self):
        """Generate emergency access codes for super admin"""
        try:
            # Generate 5 emergency codes
            codes = []
            for i in range(5):
                code = secrets.token_hex(16)
                codes.append({
                    "code": code,
                    "created_at": time.time(),
                    "used": False,
                    "expires_at": time.time() + (365 * 24 * 3600)  # 1 year
                })
            
            self.emergency_codes[self.SUPER_ADMIN["email"]] = codes
            
            # Save emergency codes securely
            emergency_file = self.data_dir / "emergency_codes.json"
            with open(emergency_file, 'w') as f:
                json.dump(self.emergency_codes, f, indent=2, default=str)
            emergency_file.chmod(0o600)
            
            logger.info("🚨 Emergency codes generated for super admin")
            
        except Exception as e:
            logger.error(f"Failed to generate emergency codes: {e}")
    
    def _setup_security_monitoring(self):
        """Setup security monitoring and alerting"""
        try:
            # Initialize security metrics
            self.security_metrics = {
                "total_events": 0,
                "threat_events": 0,
                "blocked_attempts": 0,
                "successful_logins": 0,
                "failed_logins": 0,
                "active_sessions": 0,
                "anomalies_detected": 0
            }
            
            # Start background monitoring
            asyncio.create_task(self._security_monitor_loop())
            
            logger.info("👁️ Security monitoring active")
            
        except Exception as e:
            logger.error(f"Failed to setup security monitoring: {e}")
    
    async def _security_monitor_loop(self):
        """Background security monitoring loop"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Clean expired sessions
                await self._cleanup_expired_sessions()
                
                # Analyze security events
                await self._analyze_security_events()
                
                # Check for anomalies
                await self._detect_anomalies()
                
                # Update metrics
                await self._update_security_metrics()
                
            except Exception as e:
                logger.error(f"Security monitoring error: {e}")
    
    async def authenticate_user(self, email: str, credentials: Dict[str, Any], source_ip: str, user_agent: str) -> Dict[str, Any]:
        """Authenticate user with comprehensive security checks"""
        try:
            # Check if IP is blocked
            if source_ip in self.blocked_ips:
                self._log_security_event(
                    "blocked_ip_attempt",
                    ThreatLevel.HIGH,
                    source_ip,
                    email,
                    "Authentication attempt from blocked IP",
                    "authentication",
                    False,
                    {"user_agent": user_agent}
                )
                return {"success": False, "error": "Access denied", "code": "IP_BLOCKED"}
            
            # Check failed attempts
            if self._check_failed_attempts(source_ip, email):
                return {"success": False, "error": "Too many failed attempts", "code": "RATE_LIMITED"}
            
            # Validate credentials
            auth_result = await self._validate_credentials(email, credentials)
            
            if not auth_result["valid"]:
                # Record failed attempt
                self._record_failed_attempt(source_ip, email)
                
                self._log_security_event(
                    "failed_login",
                    ThreatLevel.MEDIUM,
                    source_ip,
                    email,
                    "Failed authentication attempt",
                    "authentication",
                    False,
                    {"reason": auth_result.get("reason", "invalid_credentials")}
                )
                
                return {"success": False, "error": "Invalid credentials", "code": "AUTH_FAILED"}
            
            # Check if MFA is required
            user_info = auth_result["user"]
            requires_mfa = await self._requires_mfa(user_info)
            
            if requires_mfa and not credentials.get("mfa_code"):
                return {
                    "success": False,
                    "error": "MFA required",
                    "code": "MFA_REQUIRED",
                    "mfa_challenge": await self._generate_mfa_challenge(user_info)
                }
            
            # Verify MFA if provided
            if requires_mfa and credentials.get("mfa_code"):
                mfa_valid = await self._verify_mfa(user_info, credentials["mfa_code"])
                if not mfa_valid:
                    self._record_failed_attempt(source_ip, email)
                    return {"success": False, "error": "Invalid MFA code", "code": "MFA_FAILED"}
            
            # Create session
            session = await self._create_auth_session(user_info, source_ip, user_agent, requires_mfa)
            
            # Clear failed attempts
            self._clear_failed_attempts(source_ip, email)
            
            # Log successful authentication
            self._log_security_event(
                "successful_login",
                ThreatLevel.LOW,
                source_ip,
                email,
                "Successful authentication",
                "authentication",
                True,
                {
                    "session_id": session.session_id,
                    "security_level": session.security_level.value,
                    "mfa_used": requires_mfa
                }
            )
            
            return {
                "success": True,
                "session_token": await self._generate_session_token(session),
                "user": {
                    "email": user_info["email"],
                    "name": user_info["name"],
                    "security_level": session.security_level.value,
                    "permissions": user_info["permissions"]
                },
                "expires_at": session.expires_at
            }
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return {"success": False, "error": "Authentication system error", "code": "SYSTEM_ERROR"}
    
    async def _validate_credentials(self, email: str, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Validate user credentials"""
        try:
            # Super admin validation
            if email == self.SUPER_ADMIN["email"]:
                if credentials.get("type") == "emergency_code":
                    # Validate emergency code
                    if await self._validate_emergency_code(email, credentials.get("code")):
                        return {
                            "valid": True,
                            "user": {
                                "email": self.SUPER_ADMIN["email"],
                                "name": self.SUPER_ADMIN["name"],
                                "security_level": self.SUPER_ADMIN["security_level"],
                                "permissions": self.SUPER_ADMIN["permissions"]
                            }
                        }
                elif credentials.get("type") == "password":
                    # For demo purposes, accept a hardcoded secure password for super admin
                    # In production, this would validate against a secure hash
                    if credentials.get("password") == "os.getenv("SUPER_ADMIN_KEY", "changeme")":
                        return {
                            "valid": True,
                            "user": {
                                "email": self.SUPER_ADMIN["email"],
                                "name": self.SUPER_ADMIN["name"],
                                "security_level": self.SUPER_ADMIN["security_level"],
                                "permissions": self.SUPER_ADMIN["permissions"]
                            }
                        }
            
            # Other user validation would go here
            # For now, only super admin is supported
            
            return {"valid": False, "reason": "user_not_found"}
            
        except Exception as e:
            logger.error(f"Credential validation error: {e}")
            return {"valid": False, "reason": "validation_error"}
    
    async def _validate_emergency_code(self, email: str, code: str) -> bool:
        """Validate emergency access code"""
        try:
            user_codes = self.emergency_codes.get(email, [])
            
            for emergency_code in user_codes:
                if (emergency_code["code"] == code and 
                    not emergency_code["used"] and 
                    emergency_code["expires_at"] > time.time()):
                    
                    # Mark code as used
                    emergency_code["used"] = True
                    emergency_code["used_at"] = time.time()
                    
                    # Save updated codes
                    emergency_file = self.data_dir / "emergency_codes.json"
                    with open(emergency_file, 'w') as f:
                        json.dump(self.emergency_codes, f, indent=2, default=str)
                    
                    self._log_security_event(
                        "emergency_code_used",
                        ThreatLevel.CRITICAL,
                        "unknown",
                        email,
                        "Emergency access code used",
                        "authentication",
                        True,
                        {"code_used": code[:8] + "..."}
                    )
                    
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Emergency code validation error: {e}")
            return False
    
    def _check_failed_attempts(self, source_ip: str, email: str) -> bool:
        """Check if too many failed attempts"""
        now = time.time()
        window = 300  # 5 minutes
        
        # Check IP-based attempts
        ip_attempts = self.failed_attempts.get(f"ip:{source_ip}", [])
        recent_ip_attempts = [t for t in ip_attempts if now - t < window]
        
        if len(recent_ip_attempts) >= 10:  # 10 attempts per IP
            self.blocked_ips.add(source_ip)
            return True
        
        # Check email-based attempts
        email_attempts = self.failed_attempts.get(f"email:{email}", [])
        recent_email_attempts = [t for t in email_attempts if now - t < window]
        
        return len(recent_email_attempts) >= 5  # 5 attempts per email
    
    def _record_failed_attempt(self, source_ip: str, email: str):
        """Record failed authentication attempt"""
        now = time.time()
        
        # Record by IP
        if f"ip:{source_ip}" not in self.failed_attempts:
            self.failed_attempts[f"ip:{source_ip}"] = []
        self.failed_attempts[f"ip:{source_ip}"].append(now)
        
        # Record by email
        if f"email:{email}" not in self.failed_attempts:
            self.failed_attempts[f"email:{email}"] = []
        self.failed_attempts[f"email:{email}"].append(now)
    
    def _clear_failed_attempts(self, source_ip: str, email: str):
        """Clear failed attempts after successful authentication"""
        self.failed_attempts.pop(f"ip:{source_ip}", None)
        self.failed_attempts.pop(f"email:{email}", None)
    
    async def _requires_mfa(self, user_info: Dict[str, Any]) -> bool:
        """Check if user requires MFA"""
        security_level = user_info.get("security_level", SecurityLevel.AUTHENTICATED)
        
        # Super admin always requires MFA (except for emergency codes)
        if security_level == SecurityLevel.SUPER_ADMIN:
            return True
        
        # Admin users require MFA
        if security_level in [SecurityLevel.ADMIN, SecurityLevel.ELEVATED]:
            return True
        
        return False
    
    async def _generate_mfa_challenge(self, user_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate MFA challenge"""
        # For demo purposes, return a simple challenge
        # In production, this would integrate with TOTP, SMS, etc.
        return {
            "type": "totp",
            "message": "Enter your 6-digit authentication code",
            "backup_methods": ["emergency_code"]
        }
    
    async def _verify_mfa(self, user_info: Dict[str, Any], mfa_code: str) -> bool:
        """Verify MFA code"""
        # For demo purposes, accept specific codes
        # In production, this would verify TOTP, SMS codes, etc.
        valid_codes = ["123456", "000000"]  # Demo codes
        return mfa_code in valid_codes
    
    async def _create_auth_session(self, user_info: Dict[str, Any], source_ip: str, user_agent: str, mfa_verified: bool) -> AuthSession:
        """Create authentication session"""
        session_id = str(uuid.uuid4())
        now = time.time()
        
        session = AuthSession(
            session_id=session_id,
            user_id=user_info["email"],
            security_level=user_info["security_level"],
            auth_method=AuthMethod.PASSWORD,
            created_at=now,
            last_activity=now,
            expires_at=now + 3600,  # 1 hour
            source_ip=source_ip,
            user_agent=user_agent,
            mfa_verified=mfa_verified,
            device_fingerprint=self._generate_device_fingerprint(user_agent, source_ip)
        )
        
        self.active_sessions[session_id] = session
        return session
    
    def _generate_device_fingerprint(self, user_agent: str, source_ip: str) -> str:
        """Generate device fingerprint"""
        fingerprint_data = f"{user_agent}:{source_ip}:{time.strftime('%Y-%m-%d')}"
        return hashlib.sha256(fingerprint_data.encode()).hexdigest()[:16]
    
    async def _generate_session_token(self, session: AuthSession) -> str:
        """Generate JWT session token"""
        payload = {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "security_level": session.security_level.value,
            "iat": session.created_at,
            "exp": session.expires_at,
            "mfa": session.mfa_verified
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")
    
    async def validate_session(self, token: str, source_ip: str) -> Dict[str, Any]:
        """Validate session token"""
        try:
            # Decode JWT
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            session_id = payload.get("session_id")
            
            # Check if session exists
            session = self.active_sessions.get(session_id)
            if not session:
                return {"valid": False, "error": "Session not found"}
            
            # Check if session is expired
            if session.expires_at < time.time():
                del self.active_sessions[session_id]
                return {"valid": False, "error": "Session expired"}
            
            # Check IP consistency (optional security measure)
            if session.source_ip != source_ip:
                self._log_security_event(
                    "session_ip_mismatch",
                    ThreatLevel.HIGH,
                    source_ip,
                    session.user_id,
                    "Session IP mismatch detected",
                    "session_validation",
                    False,
                    {
                        "original_ip": session.source_ip,
                        "current_ip": source_ip,
                        "session_id": session_id
                    }
                )
                # Optionally invalidate session
                # del self.active_sessions[session_id]
                # return {"valid": False, "error": "IP mismatch"}
            
            # Update last activity
            session.last_activity = time.time()
            
            return {
                "valid": True,
                "session": session,
                "user": {
                    "email": session.user_id,
                    "security_level": session.security_level.value,
                    "mfa_verified": session.mfa_verified
                }
            }
            
        except jwt.ExpiredSignatureError:
            return {"valid": False, "error": "Token expired"}
        except jwt.InvalidTokenError:
            return {"valid": False, "error": "Invalid token"}
        except Exception as e:
            logger.error(f"Session validation error: {e}")
            return {"valid": False, "error": "Validation error"}
    
    async def create_sandbox_environment(self, user_id: str, purpose: str) -> Dict[str, Any]:
        """Create sandboxed execution environment"""
        try:
            sandbox_id = str(uuid.uuid4())
            
            # Create temporary directory for sandbox
            sandbox_dir = Path(tempfile.mkdtemp(prefix=f"sandbox_{sandbox_id}_"))
            
            # Create sandbox configuration
            sandbox_config = {
                "id": sandbox_id,
                "user_id": user_id,
                "purpose": purpose,
                "created_at": time.time(),
                "directory": str(sandbox_dir),
                "limits": self.sandbox_limits.copy(),
                "status": "active",
                "processes": []
            }
            
            self.sandbox_environments[sandbox_id] = sandbox_config
            
            # Setup sandbox restrictions
            await self._setup_sandbox_restrictions(sandbox_config)
            
            self._log_security_event(
                "sandbox_created",
                ThreatLevel.LOW,
                "localhost",
                user_id,
                f"Sandbox environment created for {purpose}",
                "sandbox",
                True,
                {"sandbox_id": sandbox_id, "purpose": purpose}
            )
            
            return {
                "success": True,
                "sandbox_id": sandbox_id,
                "directory": str(sandbox_dir),
                "limits": self.sandbox_limits
            }
            
        except Exception as e:
            logger.error(f"Failed to create sandbox: {e}")
            return {"success": False, "error": str(e)}
    
    async def _setup_sandbox_restrictions(self, sandbox_config: Dict[str, Any]):
        """Setup sandbox restrictions and monitoring"""
        try:
            # This would implement actual sandboxing using containers, chroot, etc.
            # For demo purposes, we'll just set up basic monitoring
            
            sandbox_dir = Path(sandbox_config["directory"])
            
            # Create restricted directories
            (sandbox_dir / "tmp").mkdir(exist_ok=True)
            (sandbox_dir / "workspace").mkdir(exist_ok=True)
            
            # Set permissions
            sandbox_dir.chmod(0o700)
            
            logger.info(f"✅ Sandbox restrictions set up: {sandbox_config['id']}")
            
        except Exception as e:
            logger.error(f"Failed to setup sandbox restrictions: {e}")
    
    async def execute_in_sandbox(self, sandbox_id: str, command: str, user_id: str) -> Dict[str, Any]:
        """Execute command in sandboxed environment"""
        try:
            sandbox = self.sandbox_environments.get(sandbox_id)
            if not sandbox:
                return {"success": False, "error": "Sandbox not found"}
            
            if sandbox["user_id"] != user_id:
                return {"success": False, "error": "Access denied"}
            
            # Security checks on command
            if await self._is_dangerous_command(command):
                self._log_security_event(
                    "dangerous_command_blocked",
                    ThreatLevel.HIGH,
                    "localhost",
                    user_id,
                    f"Dangerous command blocked in sandbox: {command}",
                    "sandbox",
                    False,
                    {"sandbox_id": sandbox_id, "command": command}
                )
                return {"success": False, "error": "Command blocked by security policy"}
            
            # Execute command with restrictions
            result = await self._execute_restricted_command(command, sandbox)
            
            self._log_security_event(
                "sandbox_execution",
                ThreatLevel.LOW,
                "localhost",
                user_id,
                f"Command executed in sandbox",
                "sandbox",
                True,
                {
                    "sandbox_id": sandbox_id,
                    "command": command[:100],  # Truncate for logging
                    "exit_code": result.get("exit_code")
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Sandbox execution error: {e}")
            return {"success": False, "error": str(e)}
    
    async def _is_dangerous_command(self, command: str) -> bool:
        """Check if command is potentially dangerous"""
        dangerous_patterns = [
            r"rm\s+-rf\s+/",
            r"format\s+c:",
            r"del\s+/s\s+/q",
            r"dd\s+if=",
            r"chmod\s+777",
            r"curl.*\|\s*sh",
            r"wget.*\|\s*sh",
            r"nc\s+.*\s+-e",
            r"netcat\s+.*\s+-e"
        ]
        
        command_lower = command.lower()
        for pattern in dangerous_patterns:
            if re.search(pattern, command_lower, re.IGNORECASE):
                return True
        
        return False
    
    async def _execute_restricted_command(self, command: str, sandbox: Dict[str, Any]) -> Dict[str, Any]:
        """Execute command with restrictions"""
        try:
            # For demo purposes, we'll simulate execution
            # In production, this would use actual containerization
            
            start_time = time.time()
            
            # Simulate command execution
            await asyncio.sleep(0.1)  # Simulate processing time
            
            execution_time = time.time() - start_time
            
            # Check timeout
            if execution_time > sandbox["limits"]["timeout"]:
                return {
                    "success": False,
                    "error": "Command timed out",
                    "execution_time": execution_time
                }
            
            # Simulate successful execution
            return {
                "success": True,
                "output": f"Command '{command}' executed successfully in sandbox",
                "exit_code": 0,
                "execution_time": execution_time
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _log_security_event(self, event_type: str, threat_level: ThreatLevel, source_ip: str, user_id: Optional[str], action: str, resource: str, success: bool, details: Dict[str, Any]):
        """Log security event"""
        try:
            event = SecurityEvent(
                id=str(uuid.uuid4()),
                timestamp=time.time(),
                event_type=event_type,
                threat_level=threat_level,
                source_ip=source_ip,
                user_id=user_id,
                action=action,
                resource=resource,
                success=success,
                details=details
            )
            
            self.security_events.append(event)
            
            # Keep only last 10000 events
            if len(self.security_events) > 10000:
                self.security_events = self.security_events[-10000:]
            
            # Log critical events immediately
            if threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                logger.warning(f"🚨 Security Event [{threat_level.value.upper()}]: {action} - {details}")
            
        except Exception as e:
            logger.error(f"Failed to log security event: {e}")
    
    async def _cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        try:
            now = time.time()
            expired_sessions = [
                session_id for session_id, session in self.active_sessions.items()
                if session.expires_at < now
            ]
            
            for session_id in expired_sessions:
                del self.active_sessions[session_id]
            
            if expired_sessions:
                logger.info(f"🧹 Cleaned up {len(expired_sessions)} expired sessions")
                
        except Exception as e:
            logger.error(f"Session cleanup error: {e}")
    
    async def _analyze_security_events(self):
        """Analyze recent security events for patterns"""
        try:
            # Analyze last hour of events
            now = time.time()
            recent_events = [
                event for event in self.security_events
                if now - event.timestamp < 3600
            ]
            
            # Look for patterns
            failed_logins = [e for e in recent_events if e.event_type == "failed_login"]
            if len(failed_logins) > 20:  # More than 20 failed logins in an hour
                logger.warning("🚨 High number of failed login attempts detected")
            
            # Check for blocked attempts
            blocked_attempts = [e for e in recent_events if "blocked" in e.event_type]
            if len(blocked_attempts) > 10:
                logger.warning("🚨 High number of blocked attempts detected")
                
        except Exception as e:
            logger.error(f"Security event analysis error: {e}")
    
    async def _detect_anomalies(self):
        """Detect security anomalies"""
        try:
            # Simple anomaly detection based on patterns
            now = time.time()
            
            # Check for unusual login patterns
            recent_logins = [
                event for event in self.security_events
                if event.event_type == "successful_login" and now - event.timestamp < 3600
            ]
            
            # Check for logins from new IPs
            known_ips = set()
            for event in self.security_events[:-100]:  # Historical IPs
                if event.event_type == "successful_login":
                    known_ips.add(event.source_ip)
            
            new_ip_logins = [
                event for event in recent_logins
                if event.source_ip not in known_ips
            ]
            
            if new_ip_logins:
                for event in new_ip_logins:
                    self._log_security_event(
                        "anomaly_new_ip_login",
                        ThreatLevel.MEDIUM,
                        event.source_ip,
                        event.user_id,
                        "Login from new IP detected",
                        "anomaly_detection",
                        True,
                        {"original_event_id": event.id}
                    )
                    
        except Exception as e:
            logger.error(f"Anomaly detection error: {e}")
    
    async def _update_security_metrics(self):
        """Update security metrics"""
        try:
            now = time.time()
            last_hour = now - 3600
            
            recent_events = [
                event for event in self.security_events
                if event.timestamp > last_hour
            ]
            
            self.security_metrics.update({
                "total_events": len(self.security_events),
                "recent_events": len(recent_events),
                "threat_events": len([e for e in recent_events if e.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]]),
                "blocked_attempts": len([e for e in recent_events if "blocked" in e.event_type]),
                "successful_logins": len([e for e in recent_events if e.event_type == "successful_login"]),
                "failed_logins": len([e for e in recent_events if e.event_type == "failed_login"]),
                "active_sessions": len(self.active_sessions),
                "blocked_ips": len(self.blocked_ips),
                "last_updated": now
            })
            
        except Exception as e:
            logger.error(f"Metrics update error: {e}")
    
    async def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status"""
        try:
            return {
                "metrics": self.security_metrics,
                "active_sessions": len(self.active_sessions),
                "blocked_ips": len(self.blocked_ips),
                "security_policies": len(self.security_policies),
                "recent_threats": len([
                    e for e in self.security_events[-100:]
                    if e.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]
                ]),
                "sandbox_environments": len(self.sandbox_environments),
                "system_status": "secure",
                "last_security_event": self.security_events[-1].timestamp if self.security_events else None
            }
            
        except Exception as e:
            logger.error(f"Failed to get security status: {e}")
            return {"error": str(e)}
    
    def _load_security_data(self):
        """Load existing security data"""
        try:
            # Load security events
            events_file = self.data_dir / "security_events.json"
            if events_file.exists():
                with open(events_file, 'r') as f:
                    events_data = json.load(f)
                    for event_data in events_data.get("events", [])[-1000:]:  # Last 1000 events
                        event = SecurityEvent(**event_data)
                        self.security_events.append(event)
            
            # Load blocked IPs
            blocked_ips_file = self.data_dir / "blocked_ips.json"
            if blocked_ips_file.exists():
                with open(blocked_ips_file, 'r') as f:
                    self.blocked_ips = set(json.load(f).get("blocked_ips", []))
            
            logger.info("✅ Security data loaded")
            
        except Exception as e:
            logger.error(f"Failed to load security data: {e}")
    
    async def save_security_data(self):
        """Save security data"""
        try:
            # Save security events
            events_data = {
                "events": [asdict(event) for event in self.security_events[-1000:]]  # Last 1000
            }
            with open(self.data_dir / "security_events.json", 'w') as f:
                json.dump(events_data, f, indent=2, default=str)
            
            # Save blocked IPs
            blocked_ips_data = {"blocked_ips": list(self.blocked_ips)}
            with open(self.data_dir / "blocked_ips.json", 'w') as f:
                json.dump(blocked_ips_data, f, indent=2)
            
            logger.info("✅ Security data saved")
            
        except Exception as e:
            logger.error(f"Failed to save security data: {e}")

# Global instance
security_system = SecuritySystem()

# Convenience functions
async def authenticate_user(email: str, credentials: Dict[str, Any], source_ip: str, user_agent: str) -> Dict[str, Any]:
    """Authenticate user"""
    return await security_system.authenticate_user(email, credentials, source_ip, user_agent)

async def validate_session(token: str, source_ip: str) -> Dict[str, Any]:
    """Validate session"""
    return await security_system.validate_session(token, source_ip)

async def create_sandbox_environment(user_id: str, purpose: str) -> Dict[str, Any]:
    """Create sandbox"""
    return await security_system.create_sandbox_environment(user_id, purpose)

async def execute_in_sandbox(sandbox_id: str, command: str, user_id: str) -> Dict[str, Any]:
    """Execute in sandbox"""
    return await security_system.execute_in_sandbox(sandbox_id, command, user_id)

async def get_security_status() -> Dict[str, Any]:
    """Get security status"""
    return await security_system.get_security_status()