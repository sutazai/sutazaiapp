#!/usr/bin/env python3
"""
ULTRA Secrets Management System
Enterprise-grade secrets management with encryption, rotation, and audit logging
Author: ULTRA Security Engineer
Date: 2025-08-11
"""

import os
import json
import base64
import hashlib
import secrets
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Any, List
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
import hvac  # HashiCorp Vault client
import boto3  # AWS Secrets Manager client
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UltraSecretsManager:
    """Enterprise-grade secrets management with multiple backend support"""
    
    def __init__(self, backend: str = "local", config: Dict[str, Any] = None):
        """
        Initialize secrets manager with specified backend
        
        Args:
            backend: Backend type - local, vault, aws, azure
            config: Backend-specific configuration
        """
        self.backend = backend
        self.config = config or {}
        self.encryption_key = self._get_or_create_master_key()
        self.cipher = Fernet(self.encryption_key)
        self.audit_log = []
        self.secrets_cache = {}
        self.rotation_schedule = {}
        
        # Initialize backend client
        self.client = self._initialize_backend()
        
    def _get_or_create_master_key(self) -> bytes:
        """Get or create master encryption key"""
        key_file = Path("/opt/sutazaiapp/.secrets/master.key")
        key_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        
        if key_file.exists():
            with open(key_file, 'rb') as f:
                key = f.read()
        else:
            # Generate new master key
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            os.chmod(key_file, 0o600)
            logger.info("Generated new master encryption key")
            
        return key
        
    def _initialize_backend(self):
        """Initialize the secrets backend client"""
        if self.backend == "vault":
            # HashiCorp Vault
            client = hvac.Client(
                url=self.config.get('vault_url', 'http://localhost:8200'),
                token=self.config.get('vault_token')
            )
            if not client.is_authenticated():
                raise ValueError("Vault authentication failed")
            return client
            
        elif self.backend == "aws":
            # AWS Secrets Manager
            return boto3.client(
                'secretsmanager',
                region_name=self.config.get('aws_region', 'us-east-1')
            )
            
        elif self.backend == "azure":
            # Azure Key Vault
            credential = DefaultAzureCredential()
            vault_url = self.config.get('azure_vault_url')
            return SecretClient(vault_url=vault_url, credential=credential)
            
        else:
            # Local encrypted file storage
            self.secrets_file = Path("/opt/sutazaiapp/.secrets/secrets.enc")
            self.secrets_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            return None
            
    def generate_secret(self, length: int = 32, include_special: bool = True) -> str:
        """Generate cryptographically secure random secret"""
        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        if include_special:
            alphabet += "!@#$%^&*()_+-=[]{}|;:,.<>?"
        return ''.join(secrets.choice(alphabet) for _ in range(length))
        
    def store_secret(self, key: str, value: str, metadata: Dict = None) -> bool:
        """
        Store a secret securely with optional metadata
        
        Args:
            key: Secret identifier
            value: Secret value
            metadata: Optional metadata (tags, expiry, etc.)
        """
        try:
            # Audit log entry
            self._audit_log("STORE", key, metadata)
            
            if self.backend == "vault":
                self.client.secrets.kv.v2.create_or_update_secret(
                    path=key,
                    secret={"value": value, "metadata": metadata or {}}
                )
                
            elif self.backend == "aws":
                self.client.create_secret(
                    Name=key,
                    SecretString=json.dumps({"value": value, "metadata": metadata or {}})
                )
                
            elif self.backend == "azure":
                self.client.set_secret(key, value)
                
            else:
                # Local storage with encryption
                encrypted_value = self.cipher.encrypt(value.encode())
                secrets_data = self._load_local_secrets()
                secrets_data[key] = {
                    "value": base64.b64encode(encrypted_value).decode(),
                    "metadata": metadata or {},
                    "created": datetime.now().isoformat(),
                    "rotated": None
                }
                self._save_local_secrets(secrets_data)
                
            # Update cache
            self.secrets_cache[key] = value
            
            # Set rotation schedule if specified
            if metadata and "rotation_days" in metadata:
                self._schedule_rotation(key, metadata["rotation_days"])
                
            logger.info(f"Secret '{key}' stored successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store secret '{key}': {e}")
            return False
            
    def retrieve_secret(self, key: str) -> Optional[str]:
        """
        Retrieve a secret value
        
        Args:
            key: Secret identifier
            
        Returns:
            Secret value or None if not found
        """
        try:
            # Check cache first
            if key in self.secrets_cache:
                self._audit_log("RETRIEVE_CACHED", key)
                return self.secrets_cache[key]
                
            self._audit_log("RETRIEVE", key)
            
            if self.backend == "vault":
                response = self.client.secrets.kv.v2.read_secret_version(path=key)
                value = response["data"]["data"]["value"]
                
            elif self.backend == "aws":
                response = self.client.get_secret_value(SecretId=key)
                secret_data = json.loads(response["SecretString"])
                value = secret_data["value"]
                
            elif self.backend == "azure":
                secret = self.client.get_secret(key)
                value = secret.value
                
            else:
                # Local storage
                secrets_data = self._load_local_secrets()
                if key not in secrets_data:
                    return None
                encrypted_value = base64.b64decode(secrets_data[key]["value"])
                value = self.cipher.decrypt(encrypted_value).decode()
                
            # Update cache
            self.secrets_cache[key] = value
            return value
            
        except Exception as e:
            logger.error(f"Failed to retrieve secret '{key}': {e}")
            return None
            
    def rotate_secret(self, key: str, new_value: str = None) -> bool:
        """
        Rotate a secret to a new value
        
        Args:
            key: Secret identifier
            new_value: New secret value (auto-generated if not provided)
        """
        try:
            self._audit_log("ROTATE", key)
            
            # Generate new value if not provided
            if new_value is None:
                new_value = self.generate_secret()
                
            # Store old value for rollback
            old_value = self.retrieve_secret(key)
            
            # Update secret
            if self.store_secret(key, new_value, {"rotated": datetime.now().isoformat()}):
                logger.info(f"Secret '{key}' rotated successfully")
                
                # Clear from cache to force reload
                if key in self.secrets_cache:
                    del self.secrets_cache[key]
                    
                return True
            else:
                # Rollback on failure
                if old_value:
                    self.store_secret(key, old_value)
                return False
                
        except Exception as e:
            logger.error(f"Failed to rotate secret '{key}': {e}")
            return False
            
    def delete_secret(self, key: str) -> bool:
        """Delete a secret"""
        try:
            self._audit_log("DELETE", key)
            
            if self.backend == "vault":
                self.client.secrets.kv.v2.delete_metadata_and_all_versions(path=key)
                
            elif self.backend == "aws":
                self.client.delete_secret(SecretId=key, ForceDeleteWithoutRecovery=True)
                
            elif self.backend == "azure":
                self.client.begin_delete_secret(key)
                
            else:
                secrets_data = self._load_local_secrets()
                if key in secrets_data:
                    del secrets_data[key]
                    self._save_local_secrets(secrets_data)
                    
            # Remove from cache
            if key in self.secrets_cache:
                del self.secrets_cache[key]
                
            logger.info(f"Secret '{key}' deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete secret '{key}': {e}")
            return False
            
    def list_secrets(self) -> List[str]:
        """List all secret keys"""
        try:
            if self.backend == "vault":
                response = self.client.secrets.kv.v2.list_secrets(path="")
                return response["data"]["keys"]
                
            elif self.backend == "aws":
                response = self.client.list_secrets()
                return [s["Name"] for s in response["SecretList"]]
                
            elif self.backend == "azure":
                return [s.name for s in self.client.list_properties_of_secrets()]
                
            else:
                secrets_data = self._load_local_secrets()
                return list(secrets_data.keys())
                
        except Exception as e:
            logger.error(f"Failed to list secrets: {e}")
            return []
            
    def _load_local_secrets(self) -> Dict:
        """Load secrets from local encrypted file"""
        if not self.secrets_file.exists():
            return {}
            
        try:
            with open(self.secrets_file, 'rb') as f:
                encrypted_data = f.read()
            decrypted_data = self.cipher.decrypt(encrypted_data)
            return json.loads(decrypted_data)
        except:
            return {}
            
    def _save_local_secrets(self, secrets_data: Dict):
        """Save secrets to local encrypted file"""
        encrypted_data = self.cipher.encrypt(json.dumps(secrets_data).encode())
        with open(self.secrets_file, 'wb') as f:
            f.write(encrypted_data)
        os.chmod(self.secrets_file, 0o600)
        
    def _schedule_rotation(self, key: str, days: int):
        """Schedule automatic secret rotation"""
        rotation_date = datetime.now() + timedelta(days=days)
        self.rotation_schedule[key] = rotation_date
        logger.info(f"Scheduled rotation for '{key}' on {rotation_date.isoformat()}")
        
    def _audit_log(self, action: str, key: str, metadata: Dict = None):
        """Add entry to audit log"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "key": key,
            "metadata": metadata or {},
            "user": os.environ.get("USER", "unknown")
        }
        self.audit_log.append(entry)
        
        # Write to audit file
        audit_file = Path("/opt/sutazaiapp/.secrets/audit.log")
        with open(audit_file, 'a') as f:
            f.write(json.dumps(entry) + "\n")
            
    def get_audit_log(self, key: str = None) -> List[Dict]:
        """Get audit log entries"""
        if key:
            return [e for e in self.audit_log if e["key"] == key]
        return self.audit_log
        
    def validate_secret_strength(self, secret: str) -> Dict[str, Any]:
        """Validate secret strength and compliance"""
        result = {
            "length": len(secret),
            "has_uppercase": any(c.isupper() for c in secret),
            "has_lowercase": any(c.islower() for c in secret),
            "has_digits": any(c.isdigit() for c in secret),
            "has_special": any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in secret),
            "entropy": self._calculate_entropy(secret),
            "is_compliant": True
        }
        
        # Check compliance with policy
        if result["length"] < 12:
            result["is_compliant"] = False
            result["reason"] = "Secret too short (minimum 12 characters)"
        elif not (result["has_uppercase"] and result["has_lowercase"] and 
                  result["has_digits"] and result["has_special"]):
            result["is_compliant"] = False
            result["reason"] = "Secret must contain uppercase, lowercase, digits, and special characters"
        elif result["entropy"] < 60:
            result["is_compliant"] = False
            result["reason"] = "Secret entropy too low (minimum 60 bits)"
            
        return result
        
    def _calculate_entropy(self, secret: str) -> float:
        """Calculate Shannon entropy of a secret"""
        if not secret:
            return 0
        entropy = 0
        for i in range(256):
            char = chr(i)
            freq = secret.count(char) / len(secret)
            if freq > 0:
                entropy += -freq * (freq if freq == 0 else freq * (1 if freq == 0 else 1/freq))
        return entropy * len(secret)


def initialize_production_secrets():
    """Initialize all production secrets with secure values"""
    manager = UltraSecretsManager(backend="local")
    
    # Define required secrets
    required_secrets = {
        "JWT_SECRET_KEY": {"length": 64, "rotation_days": 90},
        "POSTGRES_PASSWORD": {"length": 32, "rotation_days": 30},
        "REDIS_PASSWORD": {"length": 32, "rotation_days": 30},
        "NEO4J_PASSWORD": {"length": 32, "rotation_days": 30},
        "RABBITMQ_PASSWORD": {"length": 32, "rotation_days": 30},
        "GRAFANA_ADMIN_PASSWORD": {"length": 24, "rotation_days": 60},
        "OLLAMA_API_KEY": {"length": 48, "rotation_days": 90},
        "ENCRYPTION_KEY": {"length": 32, "rotation_days": 180},
        "API_RATE_LIMIT_SECRET": {"length": 32, "rotation_days": 60},
        "WEBHOOK_SECRET": {"length": 48, "rotation_days": 90}
    }
    
    logger.info("Initializing production secrets...")
    
    for key, config in required_secrets.items():
        if not manager.retrieve_secret(key):
            secret = manager.generate_secret(length=config["length"])
            manager.store_secret(key, secret, config)
            logger.info(f"Generated secret: {key}")
        else:
            logger.info(f"Secret already exists: {key}")
            
    # Generate .env file
    env_file = Path("/opt/sutazaiapp/.env.production")
    with open(env_file, 'w') as f:
        f.write("# Auto-generated production environment file\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n\n")
        
        for key in required_secrets.keys():
            value = manager.retrieve_secret(key)
            f.write(f"{key}={value}\n")
            
    os.chmod(env_file, 0o600)
    logger.info(f"Production environment file generated: {env_file}")
    
    return manager


if __name__ == "__main__":
    # Initialize production secrets
    manager = initialize_production_secrets()
    
    # Validate all secrets
    print("\n=== Secret Validation Report ===")
    for key in manager.list_secrets():
        secret = manager.retrieve_secret(key)
        validation = manager.validate_secret_strength(secret)
        status = "✅ COMPLIANT" if validation["is_compliant"] else "❌ NON-COMPLIANT"
        print(f"{key}: {status}")
        if not validation["is_compliant"]:
            print(f"  Reason: {validation.get('reason', 'Unknown')}")
            
    # Show audit log
    print("\n=== Audit Log Summary ===")
    audit_entries = manager.get_audit_log()
    for entry in audit_entries[-10:]:  # Last 10 entries
        print(f"{entry['timestamp']}: {entry['action']} - {entry['key']}")