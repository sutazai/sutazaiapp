"""
Enterprise-grade secrets management system
Compliant with Professional Project Standards Rule 5
"""
import os
import json
import logging
from typing import Optional, Dict, Any
from pathlib import Path
from cryptography.fernet import Fernet
from functools import lru_cache

logger = logging.getLogger(__name__)

class SecretsManager:
    """
    Centralized secrets management with encryption at rest
    Supports environment variables, encrypted files, and external vaults
    """
    
    def __init__(self):
        # Use /tmp for development or container environments
        if os.path.exists("/opt/sutazaiapp"):
            self.secrets_dir = Path("/opt/sutazaiapp/.secrets")
        else:
            # Fallback to tmp directory in containers
            self.secrets_dir = Path("/tmp/.secrets")
        
        try:
            self.secrets_dir.mkdir(exist_ok=True, mode=0o700)
        except (PermissionError, FileNotFoundError):
            # Fallback to temp directory if cannot create
            self.secrets_dir = Path("/tmp/.secrets")
            self.secrets_dir.mkdir(exist_ok=True, mode=0o700)
        self._cipher = self._initialize_encryption()
        self._cache: Dict[str, Any] = {}
    
    def _initialize_encryption(self) -> Optional[Fernet]:
        """Initialize encryption cipher with master key from environment"""
        master_key = os.getenv("SUTAZAI_MASTER_KEY")
        
        if not master_key:
            # Generate new master key if not exists
            master_key = Fernet.generate_key()
            key_file = self.secrets_dir / "master.key"
            
            if not key_file.exists():
                # Store master key securely (should be in external KMS in production)
                key_file.write_bytes(master_key)
                key_file.chmod(0o600)
                logger.warning("Generated new master key - store this securely!")
            else:
                master_key = key_file.read_bytes()
        
        return Fernet(master_key) if isinstance(master_key, bytes) else Fernet(master_key.encode())
    
    @lru_cache(maxsize=128)
    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Retrieve secret from multiple sources with fallback hierarchy:
        1. Environment variable (highest priority)
        2. Encrypted secrets file
        3. Default value
        """
        # Check environment first
        env_value = os.getenv(key)
        if env_value:
            return env_value
        
        # Check encrypted secrets file
        secrets_file = self.secrets_dir / "secrets.enc"
        if secrets_file.exists():
            try:
                encrypted_data = secrets_file.read_bytes()
                decrypted = self._cipher.decrypt(encrypted_data)
                secrets = json.loads(decrypted.decode())
                
                if key in secrets:
                    return secrets[key]
            except Exception as e:
                logger.error(f"Failed to read encrypted secrets: {e}")
        
        # Return default
        return default
    
    def set_secret(self, key: str, value: str) -> None:
        """Store secret in encrypted file"""
        secrets_file = self.secrets_dir / "secrets.enc"
        
        # Load existing secrets
        secrets = {}
        if secrets_file.exists():
            try:
                encrypted_data = secrets_file.read_bytes()
                decrypted = self._cipher.decrypt(encrypted_data)
                secrets = json.loads(decrypted.decode())
            except Exception as e:
                logger.error(f"Failed to load existing secrets: {e}")
        
        # Update secret
        secrets[key] = value
        
        # Encrypt and save
        encrypted = self._cipher.encrypt(json.dumps(secrets).encode())
        secrets_file.write_bytes(encrypted)
        secrets_file.chmod(0o600)
        
        # Clear cache
        self.get_secret.cache_clear()
    
    def rotate_secret(self, key: str, new_value: str) -> None:
        """
        Rotate a secret with audit logging
        """
        old_value_exists = bool(self.get_secret(key))
        self.set_secret(key, new_value)
        
        # Audit log (should go to secure audit system)
        logger.info(f"Secret rotated: {key} (existed: {old_value_exists})")
    
    def validate_secrets(self) -> Dict[str, bool]:
        """Validate all required secrets are present"""
        required_secrets = [
            "POSTGRES_PASSWORD",
            "REDIS_PASSWORD", 
            "NEO4J_PASSWORD",
            "RABBITMQ_PASSWORD",
            "JWT_SECRET_KEY",
            "API_KEY",
            "ENCRYPTION_KEY"
        ]
        
        validation = {}
        for secret in required_secrets:
            validation[secret] = bool(self.get_secret(secret))
        
        return validation

# Singleton instance
_secrets_manager = None

def get_secrets_manager() -> SecretsManager:
    """Get singleton secrets manager instance"""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager()
    return _secrets_manager