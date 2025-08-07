#!/usr/bin/env python3
"""
Zero-Trust Architecture Implementation for SutazAI
Implements comprehensive identity verification and least-privilege access
"""

import asyncio
import hashlib
import hmac
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import redis
import psycopg2
from psycopg2.extras import RealDictCursor
import bcrypt
import secrets
import ssl
import os

class TrustLevel(Enum):
    UNTRUSTED = 0
    BASIC = 1
    VERIFIED = 2
    HIGH = 3
    CRITICAL = 4

class AccessType(Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"
    DELETE = "delete"

@dataclass
class SecurityContext:
    """Security context for each request/operation"""
    user_id: str
    session_id: str
    trust_level: TrustLevel
    permissions: Set[str]
    client_ip: str
    user_agent: str
    timestamp: datetime
    mfa_verified: bool = False
    device_fingerprint: Optional[str] = None
    risk_score: float = 0.0
    
class ZeroTrustEngine:
    """Core Zero-Trust architecture engine"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.redis_client = None
        self.db_connection = None
        self.encryption_key = None
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize security components"""
        try:
            # Initialize Redis for session management
            self.redis_client = redis.Redis(
                host=self.config.get('redis_host', 'redis'),
                port=self.config.get('redis_port', 6379),
                password=self.config.get('redis_password'),
                ssl=True,
                ssl_cert_reqs=ssl.CERT_REQUIRED,
                decode_responses=True
            )
            
            # Initialize PostgreSQL for audit logging
            self.db_connection = psycopg2.connect(
                host=self.config.get('postgres_host', 'postgres'),
                port=self.config.get('postgres_port', 5432),
                database=self.config.get('postgres_db', 'sutazai'),
                user=self.config.get('postgres_user', 'sutazai'),
                password=self.config.get('postgres_password'),
                sslmode='require'
            )
            
            # Initialize encryption
            self._setup_encryption()
            
            self.logger.info("Zero-Trust Engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Zero-Trust Engine: {e}")
            raise
    
    def _setup_encryption(self):
        """Setup encryption for sensitive data"""
        salt = self.config.get('encryption_salt', os.urandom(16))
        password = self.config.get('master_key', 'default-key').encode()
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        self.encryption_key = Fernet(key)
    
    async def authenticate_user(self, credentials: Dict[str, str], client_info: Dict[str, str]) -> Optional[SecurityContext]:
        """Authenticate user with multi-factor verification"""
        try:
            username = credentials.get('username')
            password = credentials.get('password')
            mfa_token = credentials.get('mfa_token')
            
            if not all([username, password]):
                return None
            
            # Verify primary credentials
            user_data = await self._verify_credentials(username, password)
            if not user_data:
                await self._log_security_event('auth_failure', {'username': username, 'client_ip': client_info.get('ip')})
                return None
            
            # Device fingerprinting
            device_fingerprint = self._generate_device_fingerprint(client_info)
            
            # Risk assessment
            risk_score = await self._calculate_risk_score(user_data, client_info, device_fingerprint)
            
            # Determine required trust level
            required_trust_level = self._determine_trust_level(risk_score)
            
            # MFA verification for higher trust levels
            mfa_verified = False
            if required_trust_level.value >= TrustLevel.VERIFIED.value:
                if not mfa_token:
                    raise SecurityException("MFA token required")
                mfa_verified = await self._verify_mfa(user_data['user_id'], mfa_token)
                if not mfa_verified:
                    await self._log_security_event('mfa_failure', {'user_id': user_data['user_id']})
                    return None
            
            # Create security context
            session_id = self._generate_session_id()
            context = SecurityContext(
                user_id=user_data['user_id'],
                session_id=session_id,
                trust_level=required_trust_level,
                permissions=set(user_data.get('permissions', [])),
                client_ip=client_info.get('ip', ''),
                user_agent=client_info.get('user_agent', ''),
                timestamp=datetime.utcnow(),
                mfa_verified=mfa_verified,
                device_fingerprint=device_fingerprint,
                risk_score=risk_score
            )
            
            # Store session
            await self._store_session(context)
            
            await self._log_security_event('auth_success', {
                'user_id': user_data['user_id'],
                'trust_level': required_trust_level.name,
                'risk_score': risk_score
            })
            
            return context
            
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            await self._log_security_event('auth_error', {'error': str(e)})
            return None
    
    async def authorize_request(self, context: SecurityContext, resource: str, action: AccessType) -> bool:
        """Authorize request based on zero-trust principles"""
        try:
            # Validate session
            if not await self._validate_session(context):
                return False
            
            # Check permissions
            if not self._has_permission(context, resource, action):
                await self._log_security_event('authorization_denied', {
                    'user_id': context.user_id,
                    'resource': resource,
                    'action': action.value
                })
                return False
            
            # Continuous risk assessment
            current_risk = await self._reassess_risk(context)
            if current_risk > self.config.get('max_risk_threshold', 0.8):
                await self._log_security_event('high_risk_detected', {
                    'user_id': context.user_id,
                    'risk_score': current_risk
                })
                return False
            
            # Update session activity
            await self._update_session_activity(context)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Authorization error: {e}")
            return False
    
    async def _verify_credentials(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Verify user credentials against secure storage"""
        try:
            cursor = self.db_connection.cursor(cursor_factory=RealDictCursor)
            cursor.execute(
                "SELECT user_id, password_hash, salt, permissions, is_active FROM users WHERE username = %s",
                (username,)
            )
            user_data = cursor.fetchone()
            cursor.close()
            
            if not user_data or not user_data['is_active']:
                return None
            
            # Verify password using bcrypt
            if bcrypt.checkpw(password.encode('utf-8'), user_data['password_hash'].encode('utf-8')):
                return dict(user_data)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Credential verification error: {e}")
            return None
    
    def _generate_device_fingerprint(self, client_info: Dict[str, str]) -> str:
        """Generate unique device fingerprint"""
        fingerprint_data = {
            'user_agent': client_info.get('user_agent', ''),
            'screen_resolution': client_info.get('screen_resolution', ''),
            'timezone': client_info.get('timezone', ''),
            'language': client_info.get('language', ''),
            'plugins': client_info.get('plugins', ''),
        }
        
        fingerprint_string = json.dumps(fingerprint_data, sort_keys=True)
        return hashlib.sha256(fingerprint_string.encode()).hexdigest()
    
    async def _calculate_risk_score(self, user_data: Dict[str, Any], client_info: Dict[str, str], device_fingerprint: str) -> float:
        """Calculate risk score based on multiple factors"""
        risk_factors = []
        
        # IP reputation check
        ip_risk = await self._check_ip_reputation(client_info.get('ip', ''))
        risk_factors.append(ip_risk)
        
        # Device familiarity
        device_risk = await self._check_device_familiarity(user_data['user_id'], device_fingerprint)
        risk_factors.append(device_risk)
        
        # Time-based patterns
        time_risk = await self._check_time_patterns(user_data['user_id'])
        risk_factors.append(time_risk)
        
        # Geographic anomalies
        geo_risk = await self._check_geographic_anomalies(user_data['user_id'], client_info.get('ip', ''))
        risk_factors.append(geo_risk)
        
        # Calculate weighted average
        weights = [0.3, 0.3, 0.2, 0.2]  # Configurable weights
        risk_score = sum(risk * weight for risk, weight in zip(risk_factors, weights))
        
        return min(risk_score, 1.0)  # Cap at 1.0
    
    def _determine_trust_level(self, risk_score: float) -> TrustLevel:
        """Determine trust level based on risk score"""
        if risk_score < 0.2:
            return TrustLevel.HIGH
        elif risk_score < 0.4:
            return TrustLevel.VERIFIED
        elif risk_score < 0.6:
            return TrustLevel.BASIC
        else:
            return TrustLevel.UNTRUSTED
    
    async def _verify_mfa(self, user_id: str, mfa_token: str) -> bool:
        """Verify multi-factor authentication token"""
        try:
            # Get user's MFA secret
            cursor = self.db_connection.cursor()
            cursor.execute("SELECT mfa_secret FROM user_mfa WHERE user_id = %s", (user_id,))
            result = cursor.fetchone()
            cursor.close()
            
            if not result:
                return False
            
            mfa_secret = result[0]
            
            # Verify TOTP token (implement TOTP verification logic)
            # This is a simplified example - implement proper TOTP verification
            import pyotp
            totp = pyotp.TOTP(mfa_secret)
            return totp.verify(mfa_token, valid_window=1)
            
        except Exception as e:
            self.logger.error(f"MFA verification error: {e}")
            return False
    
    def _generate_session_id(self) -> str:
        """Generate cryptographically secure session ID"""
        return secrets.token_urlsafe(32)
    
    async def _store_session(self, context: SecurityContext):
        """Store session in Redis with encryption"""
        try:
            session_data = asdict(context)
            session_data['timestamp'] = session_data['timestamp'].isoformat()
            session_data['trust_level'] = session_data['trust_level'].name
            session_data['permissions'] = list(session_data['permissions'])
            
            # Encrypt session data
            encrypted_data = self.encryption_key.encrypt(json.dumps(session_data).encode())
            
            # Store with TTL
            ttl = self.config.get('session_ttl', 3600)  # 1 hour default
            await self.redis_client.setex(
                f"session:{context.session_id}",
                ttl,
                encrypted_data
            )
            
        except Exception as e:
            self.logger.error(f"Session storage error: {e}")
            raise
    
    async def _validate_session(self, context: SecurityContext) -> bool:
        """Validate session exists and is not expired"""
        try:
            session_data = await self.redis_client.get(f"session:{context.session_id}")
            return session_data is not None
            
        except Exception as e:
            self.logger.error(f"Session validation error: {e}")
            return False
    
    def _has_permission(self, context: SecurityContext, resource: str, action: AccessType) -> bool:
        """Check if user has permission for resource and action"""
        required_permission = f"{resource}:{action.value}"
        return required_permission in context.permissions or "admin:all" in context.permissions
    
    async def _log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security event for audit trail"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute(
                """INSERT INTO security_audit_log 
                   (event_type, details, timestamp) 
                   VALUES (%s, %s, %s)""",
                (event_type, json.dumps(details), datetime.utcnow())
            )
            self.db_connection.commit()
            cursor.close()
            
        except Exception as e:
            self.logger.error(f"Security logging error: {e}")
    
    async def _check_ip_reputation(self, ip_address: str) -> float:
        """Check IP reputation against threat intelligence"""
        # Implement IP reputation checking
        # This could integrate with services like VirusTotal, AbuseIPDB, etc.
        return 0.1  # Placeholder
    
    async def _check_device_familiarity(self, user_id: str, device_fingerprint: str) -> float:
        """Check if device is familiar to user"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM user_devices WHERE user_id = %s AND device_fingerprint = %s",
                (user_id, device_fingerprint)
            )
            count = cursor.fetchone()[0]
            cursor.close()
            
            return 0.0 if count > 0 else 0.5  # Known device = low risk
            
        except Exception as e:
            self.logger.error(f"Device familiarity check error: {e}")
            return 0.5
    
    async def _check_time_patterns(self, user_id: str) -> float:
        """Check if login time matches user patterns"""
        # Implement behavioral analysis for time patterns
        return 0.1  # Placeholder
    
    async def _check_geographic_anomalies(self, user_id: str, ip_address: str) -> float:
        """Check for geographic anomalies"""
        # Implement geolocation-based risk assessment
        return 0.1  # Placeholder
    
    async def _reassess_risk(self, context: SecurityContext) -> float:
        """Continuously reassess risk during session"""
        # Implement continuous risk assessment
        return context.risk_score
    
    async def _update_session_activity(self, context: SecurityContext):
        """Update session last activity time"""
        try:
            await self.redis_client.expire(f"session:{context.session_id}", self.config.get('session_ttl', 3600))
        except Exception as e:
            self.logger.error(f"Session update error: {e}")

class SecurityException(Exception):
    """Custom exception for security-related errors"""
    pass

# Initialize database schema
INIT_SQL = """
-- Users table
CREATE TABLE IF NOT EXISTS users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(255) UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    salt TEXT NOT NULL,
    permissions JSONB DEFAULT '[]',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User MFA table
CREATE TABLE IF NOT EXISTS user_mfa (
    user_id UUID REFERENCES users(user_id),
    mfa_secret TEXT NOT NULL,
    backup_codes JSONB,
    enabled BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User devices table
CREATE TABLE IF NOT EXISTS user_devices (
    user_id UUID REFERENCES users(user_id),
    device_fingerprint TEXT NOT NULL,
    device_name TEXT,
    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_trusted BOOLEAN DEFAULT false
);

-- Security audit log
CREATE TABLE IF NOT EXISTS security_audit_log (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(255) NOT NULL,
    details JSONB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX(event_type, timestamp)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_security_audit_timestamp ON security_audit_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_user_devices_fingerprint ON user_devices(device_fingerprint);
"""

if __name__ == "__main__":
    # Example usage
    config = {
        'redis_host': 'redis',
        'redis_port': 6379,
        'postgres_host': 'postgres',
        'postgres_port': 5432,
        'postgres_db': 'sutazai',
        'postgres_user': 'sutazai',
        'session_ttl': 3600,
        'max_risk_threshold': 0.8
    }
    
    zt_engine = ZeroTrustEngine(config)
    print("Zero-Trust Architecture initialized successfully")