"""
Enhanced JWT Security Implementation with Key Rotation
Implements enterprise-grade JWT security with RS256, key rotation, and audit logging
"""

import os
import json
import secrets
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import logging
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
import jwt
from jwt import PyJWTError

logger = logging.getLogger(__name__)


class JWTKeyManager:
    """
    Manages JWT signing keys with rotation support
    """
    
    def __init__(self, keys_dir: str = "/opt/sutazaiapp/secrets/jwt"):
        self.keys_dir = Path(keys_dir)
        self.keys_dir.mkdir(parents=True, exist_ok=True)
        self.current_key_id = None
        self.keys = {}
        self.rotation_interval_days = 30
        
    def generate_key_pair(self) -> Tuple[str, bytes, bytes]:
        """Generate new RSA key pair for JWT signing"""
        # Generate key ID
        key_id = secrets.token_urlsafe(16)
        
        # Generate RSA key pair
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096,  # Use 4096 for enhanced security
            backend=default_backend()
        )
        
        # Serialize private key
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        # Serialize public key
        public_key = private_key.public_key()
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        return key_id, private_pem, public_pem
    
    def rotate_keys(self) -> str:
        """Rotate JWT signing keys"""
        logger.info("Starting JWT key rotation")
        
        # Generate new key pair
        key_id, private_pem, public_pem = self.generate_key_pair()
        
        # Save keys to files
        private_key_path = self.keys_dir / f"private_key_{key_id}.pem"
        public_key_path = self.keys_dir / f"public_key_{key_id}.pem"
        
        with open(private_key_path, 'wb') as f:
            f.write(private_pem)
        os.chmod(private_key_path, 0o600)  # Restrict access to private key
        
        with open(public_key_path, 'wb') as f:
            f.write(public_pem)
        
        # Update metadata
        metadata_path = self.keys_dir / "keys_metadata.json"
        metadata = self.load_metadata()
        
        metadata["current_key_id"] = key_id
        metadata["keys"][key_id] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "active",
            "private_key_path": str(private_key_path),
            "public_key_path": str(public_key_path)
        }
        
        # Mark old keys for deprecation (keep for verification)
        for old_key_id, key_info in metadata["keys"].items():
            if old_key_id != key_id and key_info["status"] == "active":
                key_info["status"] = "deprecated"
                key_info["deprecated_at"] = datetime.now(timezone.utc).isoformat()
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"JWT key rotation completed. New key ID: {key_id}")
        return key_id
    
    def load_metadata(self) -> Dict:
        """Load keys metadata"""
        metadata_path = self.keys_dir / "keys_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {"current_key_id": None, "keys": {}}
    
    def get_current_key(self) -> Tuple[str, bytes]:
        """Get current signing key"""
        metadata = self.load_metadata()
        
        if not metadata["current_key_id"]:
            # No keys exist, generate first key pair
            key_id = self.rotate_keys()
            metadata = self.load_metadata()
        
        key_id = metadata["current_key_id"]
        key_info = metadata["keys"][key_id]
        
        with open(key_info["private_key_path"], 'rb') as f:
            private_key = f.read()
        
        return key_id, private_key
    
    def get_public_keys(self) -> Dict[str, bytes]:
        """Get all public keys for verification"""
        metadata = self.load_metadata()
        public_keys = {}
        
        for key_id, key_info in metadata["keys"].items():
            if key_info["status"] in ["active", "deprecated"]:
                with open(key_info["public_key_path"], 'rb') as f:
                    public_keys[key_id] = f.read()
        
        return public_keys


class EnhancedJWTHandler:
    """
    Enhanced JWT handler with enterprise security features
    """
    
    def __init__(self):
        self.key_manager = JWTKeyManager()
        self.algorithm = "RS256"
        self.issuer = "sutazai-auth"
        self.audience = "sutazai-api"
        self.access_token_expire_minutes = 15  # Shorter for better security
        self.refresh_token_expire_days = 7
        self.audit_logger = self._setup_audit_logger()
        
    def _setup_audit_logger(self) -> logging.Logger:
        """Setup audit logger for JWT operations"""
        audit_logger = logging.getLogger("jwt_audit")
        audit_logger.setLevel(logging.INFO)
        
        # Create audit log file handler
        audit_path = Path("/opt/sutazaiapp/logs/jwt_audit.log")
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        
        handler = logging.FileHandler(audit_path)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        audit_logger.addHandler(handler)
        
        return audit_logger
    
    def create_access_token(
        self,
        user_id: str,
        username: str,
        email: str,
        roles: List[str] = None,
        permissions: List[str] = None,
        additional_claims: Dict = None
    ) -> str:
        """Create secure access token with comprehensive claims"""
        key_id, private_key = self.key_manager.get_current_key()
        
        now = datetime.now(timezone.utc)
        expire = now + timedelta(minutes=self.access_token_expire_minutes)
        
        # Build token claims
        claims = {
            # Standard claims
            "iss": self.issuer,
            "aud": self.audience,
            "sub": str(user_id),
            "exp": expire,
            "iat": now,
            "nbf": now,  # Not before
            "jti": secrets.token_urlsafe(16),  # JWT ID for tracking
            
            # Custom claims
            "typ": "access",
            "kid": key_id,  # Key ID for rotation
            "username": username,
            "email": email,
            "roles": roles or [],
            "permissions": permissions or [],
            
            # Security claims
            "token_version": "2.0",
            "security_level": "high",
            "mfa_verified": False,  # Track MFA status
        }
        
        # Add additional claims if provided
        if additional_claims:
            claims.update(additional_claims)
        
        # Create token
        token = jwt.encode(claims, private_key, algorithm=self.algorithm)
        
        # Audit log
        self.audit_logger.info(
            f"Access token created - User: {username}, JTI: {claims['jti']}, "
            f"Expires: {expire.isoformat()}"
        )
        
        return token
    
    def create_refresh_token(
        self,
        user_id: str,
        token_family: str = None
    ) -> str:
        """Create refresh token with token family for rotation detection"""
        key_id, private_key = self.key_manager.get_current_key()
        
        now = datetime.now(timezone.utc)
        expire = now + timedelta(days=self.refresh_token_expire_days)
        
        # Generate token family ID for refresh token rotation
        if not token_family:
            token_family = secrets.token_urlsafe(16)
        
        claims = {
            "iss": self.issuer,
            "aud": self.audience,
            "sub": str(user_id),
            "exp": expire,
            "iat": now,
            "nbf": now,
            "jti": secrets.token_urlsafe(16),
            "typ": "refresh",
            "kid": key_id,
            "family": token_family,  # Token family for rotation detection
            "generation": 1  # Token generation in family
        }
        
        token = jwt.encode(claims, private_key, algorithm=self.algorithm)
        
        # Audit log
        self.audit_logger.info(
            f"Refresh token created - User: {user_id}, Family: {token_family}, "
            f"JTI: {claims['jti']}"
        )
        
        return token
    
    def verify_token(
        self,
        token: str,
        token_type: str = "access",
        verify_exp: bool = True
    ) -> Dict[str, Any]:
        """Verify token with comprehensive validation"""
        public_keys = self.key_manager.get_public_keys()
        
        # Try to decode without verification first to get the kid
        try:
            unverified = jwt.decode(
                token,
                options={"verify_signature": False}
            )
            kid = unverified.get("kid")
        except Exception as e:
            self.audit_logger.warning(f"Token decode failed: {str(e)}")
            raise ValueError("Invalid token format")
        
        # Get the appropriate public key
        if kid not in public_keys:
            self.audit_logger.warning(f"Unknown key ID: {kid}")
            raise ValueError("Invalid token key")
        
        public_key = public_keys[kid]
        
        try:
            # Verify token with all security checks
            claims = jwt.decode(
                token,
                public_key,
                algorithms=[self.algorithm],
                issuer=self.issuer,
                audience=self.audience,
                options={
                    "verify_exp": verify_exp,
                    "verify_nbf": True,
                    "verify_iat": True,
                    "verify_aud": True,
                    "verify_iss": True,
                }
            )
            
            # Verify token type
            if claims.get("typ") != token_type:
                raise ValueError(f"Invalid token type: expected {token_type}")
            
            # Audit successful verification
            self.audit_logger.info(
                f"Token verified - Type: {token_type}, "
                f"User: {claims.get('sub')}, JTI: {claims.get('jti')}"
            )
            
            return claims
            
        except jwt.ExpiredSignatureError:
            self.audit_logger.warning(f"Token expired - JTI: {unverified.get('jti')}")
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError as e:
            self.audit_logger.warning(f"Token validation failed: {str(e)}")
            raise ValueError(f"Invalid token: {str(e)}")
    
    def refresh_access_token(
        self,
        refresh_token: str,
        user_data: Dict
    ) -> Tuple[str, str]:
        """
        Refresh access token using refresh token
        Returns new access token and rotated refresh token
        """
        # Verify refresh token
        claims = self.verify_token(refresh_token, token_type="refresh")
        
        # Check if refresh token is in blacklist (implement blacklist check)
        # This would typically check a Redis cache or database
        
        # Create new access token
        new_access_token = self.create_access_token(
            user_id=claims["sub"],
            username=user_data.get("username"),
            email=user_data.get("email"),
            roles=user_data.get("roles"),
            permissions=user_data.get("permissions")
        )
        
        # Rotate refresh token (create new one in same family)
        new_refresh_token = self.create_refresh_token(
            user_id=claims["sub"],
            token_family=claims.get("family")
        )
        
        # Audit token refresh
        self.audit_logger.info(
            f"Tokens refreshed - User: {claims['sub']}, "
            f"Old JTI: {claims['jti']}, Family: {claims.get('family')}"
        )
        
        return new_access_token, new_refresh_token
    
    def revoke_token(self, token: str, reason: str = "User requested"):
        """Revoke a token (add to blacklist)"""
        try:
            claims = self.verify_token(token, verify_exp=False)
            
            # Add to blacklist (implement actual blacklist storage)
            # This would typically add to Redis with expiration
            
            self.audit_logger.info(
                f"Token revoked - JTI: {claims['jti']}, "
                f"User: {claims.get('sub')}, Reason: {reason}"
            )
            
            return True
        except Exception as e:
            self.audit_logger.error(f"Token revocation failed: {str(e)}")
            return False
    
    def get_jwks(self) -> Dict:
        """Get JSON Web Key Set for public key distribution"""
        public_keys = self.key_manager.get_public_keys()
        jwks = {"keys": []}
        
        for key_id, public_key_pem in public_keys.items():
            # Load the public key
            from cryptography.hazmat.primitives.serialization import load_pem_public_key
            public_key = load_pem_public_key(public_key_pem, backend=default_backend())
            
            # Extract key parameters for JWKS
            numbers = public_key.public_numbers()
            
            # Convert to base64url encoding
            import base64
            
            def int_to_base64url(n):
                b = n.to_bytes((n.bit_length() + 7) // 8, 'big')
                return base64.urlsafe_b64encode(b).rstrip(b'=').decode('ascii')
            
            jwk = {
                "kty": "RSA",
                "use": "sig",
                "kid": key_id,
                "alg": self.algorithm,
                "n": int_to_base64url(numbers.n),
                "e": int_to_base64url(numbers.e)
            }
            
            jwks["keys"].append(jwk)
        
        return jwks


# Global instance
enhanced_jwt_handler = EnhancedJWTHandler()