"""
Secure Data Storage and Backup System
Tamper-evident storage with encryption and backup capabilities
"""

import asyncio
import logging
import json
import time
import uuid
import hashlib
import hmac
import pickle
import gzip
import shutil
import os
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

logger = logging.getLogger(__name__)

class StorageLevel(str, Enum):
    TEMPORARY = "temporary"
    PERSISTENT = "persistent"
    CRITICAL = "critical"
    BACKUP = "backup"

class DataType(str, Enum):
    SYSTEM_STATE = "system_state"
    KNOWLEDGE_DATA = "knowledge_data"
    CODE_GENERATION = "code_generation"
    USER_DATA = "user_data"
    AUDIT_LOG = "audit_log"
    CONFIGURATION = "configuration"

@dataclass
class SecureDataRecord:
    """Secure data record with integrity verification"""
    id: str
    data_type: DataType
    storage_level: StorageLevel
    content_hash: str
    encrypted_content: bytes
    metadata: Dict[str, Any]
    created_at: float
    updated_at: float
    access_count: int = 0
    tamper_evidence: str = ""
    backup_copies: List[str] = None
    
    def __post_init__(self):
        if self.backup_copies is None:
            self.backup_copies = []

@dataclass
class BackupManifest:
    """Backup manifest with integrity information"""
    backup_id: str
    created_at: float
    records_count: int
    total_size: int
    integrity_hash: str
    backup_location: str
    compression_used: bool
    encryption_used: bool

class SecureStorageSystem:
    """
    Secure Storage and Backup System
    Provides tamper-evident storage with encryption and automated backups
    """
    
    # Hardcoded authorization
    AUTHORIZED_USER = "os.getenv("ADMIN_EMAIL", "admin@localhost")"
    
    def __init__(self, storage_dir: str = "/opt/sutazaiapp/data/secure_storage"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage organization
        self.primary_storage = self.storage_dir / "primary"
        self.backup_storage = self.storage_dir / "backups"
        self.temp_storage = self.storage_dir / "temp"
        
        # Create storage directories
        for storage_path in [self.primary_storage, self.backup_storage, self.temp_storage]:
            storage_path.mkdir(parents=True, exist_ok=True)
        
        # Security components
        self.encryption_key = self._initialize_encryption()
        self.cipher = Fernet(self.encryption_key)
        self.integrity_key = self._initialize_integrity_key()
        
        # Data management
        self.data_registry = {}  # record_id -> SecureDataRecord
        self.backup_manifests = {}  # backup_id -> BackupManifest
        
        # Storage policies
        self.auto_backup_enabled = True
        self.backup_interval = 3600  # 1 hour
        self.retention_policy = {
            StorageLevel.TEMPORARY: 86400,      # 1 day
            StorageLevel.PERSISTENT: 2592000,   # 30 days
            StorageLevel.CRITICAL: 31536000,    # 1 year
            StorageLevel.BACKUP: 94608000       # 3 years
        }
        
        # Monitoring
        self.storage_metrics = {
            "total_records": 0,
            "total_size": 0,
            "backup_count": 0,
            "last_backup": None,
            "integrity_checks": 0,
            "tamper_attempts": 0
        }
        
        # Initialize
        self._load_existing_data()
        self._start_background_tasks()
        
        logger.info("✅ Secure Storage System initialized")
    
    def _initialize_encryption(self) -> bytes:
        """Initialize encryption key"""
        key_file = self.storage_dir / ".encryption.key"
        
        if key_file.exists():
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            # Generate new key
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            os.chmod(key_file, 0o600)  # Restrict permissions
            return key
    
    def _initialize_integrity_key(self) -> bytes:
        """Initialize integrity verification key"""
        key_file = self.storage_dir / ".integrity.key"
        
        if key_file.exists():
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            # Generate new key
            key = os.urandom(32)
            with open(key_file, 'wb') as f:
                f.write(key)
            os.chmod(key_file, 0o600)
            return key
    
    def _load_existing_data(self):
        """Load existing data registry"""
        try:
            registry_file = self.storage_dir / "data_registry.json"
            if registry_file.exists():
                with open(registry_file, 'r') as f:
                    data = json.load(f)
                    for record_data in data.get("records", []):
                        # Convert bytes fields back from base64
                        if "encrypted_content" in record_data:
                            record_data["encrypted_content"] = base64.b64decode(record_data["encrypted_content"])
                        
                        record = SecureDataRecord(**record_data)
                        self.data_registry[record.id] = record
            
            # Load backup manifests
            manifests_file = self.storage_dir / "backup_manifests.json"
            if manifests_file.exists():
                with open(manifests_file, 'r') as f:
                    data = json.load(f)
                    for manifest_data in data.get("manifests", []):
                        manifest = BackupManifest(**manifest_data)
                        self.backup_manifests[manifest.backup_id] = manifest
            
            self._update_storage_metrics()
            logger.info(f"✅ Loaded {len(self.data_registry)} data records and {len(self.backup_manifests)} backup manifests")
            
        except Exception as e:
            logger.error(f"Failed to load existing data: {e}")
    
    def _start_background_tasks(self):
        """Start background tasks"""
        async def background_tasks():
            while True:
                try:
                    # Periodic backup
                    if self.auto_backup_enabled:
                        await self._periodic_backup()
                    
                    # Integrity checks
                    await self._periodic_integrity_check()
                    
                    # Cleanup old data
                    await self._cleanup_expired_data()
                    
                    # Wait before next cycle
                    await asyncio.sleep(self.backup_interval)
                    
                except Exception as e:
                    logger.error(f"Background task error: {e}")
                    await asyncio.sleep(60)
        
        asyncio.create_task(background_tasks())
    
    async def store_data(
        self,
        data: Any,
        data_type: DataType,
        storage_level: StorageLevel,
        user_id: str,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Store data securely with encryption and integrity verification"""
        try:
            # Authorization check
            if storage_level == StorageLevel.CRITICAL and user_id != self.AUTHORIZED_USER:
                raise PermissionError("Only authorized user can store critical data")
            
            record_id = str(uuid.uuid4())
            current_time = time.time()
            
            # Serialize data
            if isinstance(data, (dict, list)):
                serialized_data = json.dumps(data, default=str).encode()
            else:
                serialized_data = pickle.dumps(data)
            
            # Compress if large
            if len(serialized_data) > 1024:  # 1KB threshold
                serialized_data = gzip.compress(serialized_data)
                is_compressed = True
            else:
                is_compressed = False
            
            # Encrypt data
            encrypted_content = self.cipher.encrypt(serialized_data)
            
            # Calculate content hash
            content_hash = hashlib.sha256(serialized_data).hexdigest()
            
            # Generate tamper evidence
            tamper_evidence = self._generate_tamper_evidence(record_id, content_hash, current_time)
            
            # Create record
            record = SecureDataRecord(
                id=record_id,
                data_type=data_type,
                storage_level=storage_level,
                content_hash=content_hash,
                encrypted_content=encrypted_content,
                metadata={
                    **(metadata or {}),
                    "user_id": user_id,
                    "compressed": is_compressed,
                    "original_size": len(serialized_data)
                },
                created_at=current_time,
                updated_at=current_time,
                tamper_evidence=tamper_evidence
            )
            
            # Store record
            self.data_registry[record_id] = record
            
            # Write to disk
            await self._write_record_to_disk(record)
            
            # Update metrics
            self._update_storage_metrics()
            
            logger.info(f"✅ Data stored securely: {record_id} ({data_type.value})")
            return record_id
            
        except Exception as e:
            logger.error(f"Failed to store data: {e}")
            raise
    
    async def retrieve_data(self, record_id: str, user_id: str) -> Optional[Any]:
        """Retrieve and decrypt data with integrity verification"""
        try:
            record = self.data_registry.get(record_id)
            if not record:
                return None
            
            # Authorization check
            if record.storage_level == StorageLevel.CRITICAL and user_id != self.AUTHORIZED_USER:
                logger.warning(f"Unauthorized access attempt to critical data: {record_id}")
                self.storage_metrics["tamper_attempts"] += 1
                return None
            
            # Verify tamper evidence
            if not self._verify_tamper_evidence(record):
                logger.error(f"Tamper evidence verification failed for record: {record_id}")
                self.storage_metrics["tamper_attempts"] += 1
                return None
            
            # Decrypt data
            try:
                decrypted_data = self.cipher.decrypt(record.encrypted_content)
            except Exception as e:
                logger.error(f"Decryption failed for record {record_id}: {e}")
                return None
            
            # Decompress if needed
            if record.metadata.get("compressed", False):
                try:
                    decrypted_data = gzip.decompress(decrypted_data)
                except Exception as e:
                    logger.error(f"Decompression failed for record {record_id}: {e}")
                    return None
            
            # Verify content integrity
            content_hash = hashlib.sha256(decrypted_data).hexdigest()
            if content_hash != record.content_hash:
                logger.error(f"Content integrity verification failed for record: {record_id}")
                self.storage_metrics["tamper_attempts"] += 1
                return None
            
            # Deserialize data
            try:
                if record.data_type in [DataType.SYSTEM_STATE, DataType.CONFIGURATION]:
                    data = json.loads(decrypted_data.decode())
                else:
                    data = pickle.loads(decrypted_data)
            except json.JSONDecodeError:
                # Try pickle if JSON fails
                data = pickle.loads(decrypted_data)
            except Exception as e:
                logger.error(f"Deserialization failed for record {record_id}: {e}")
                return None
            
            # Update access metrics
            record.access_count += 1
            record.metadata["last_accessed"] = time.time()
            record.metadata["last_accessed_by"] = user_id
            
            logger.info(f"✅ Data retrieved successfully: {record_id}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to retrieve data: {e}")
            return None
    
    def _generate_tamper_evidence(self, record_id: str, content_hash: str, timestamp: float) -> str:
        """Generate tamper evidence signature"""
        evidence_data = f"{record_id}:{content_hash}:{timestamp}:{self.AUTHORIZED_USER}"
        return hmac.new(
            self.integrity_key,
            evidence_data.encode(),
            hashlib.sha256
        ).hexdigest()
    
    def _verify_tamper_evidence(self, record: SecureDataRecord) -> bool:
        """Verify tamper evidence signature"""
        try:
            expected_evidence = self._generate_tamper_evidence(
                record.id,
                record.content_hash,
                record.created_at
            )
            return hmac.compare_digest(record.tamper_evidence, expected_evidence)
        except Exception as e:
            logger.error(f"Tamper evidence verification error: {e}")
            return False
    
    async def _write_record_to_disk(self, record: SecureDataRecord):
        """Write record to appropriate storage location"""
        try:
            if record.storage_level == StorageLevel.TEMPORARY:
                storage_path = self.temp_storage
            else:
                storage_path = self.primary_storage
            
            # Create subdirectory for data type
            type_dir = storage_path / record.data_type.value
            type_dir.mkdir(exist_ok=True)
            
            # Write record metadata
            record_file = type_dir / f"{record.id}.json"
            record_data = asdict(record)
            
            # Convert bytes to base64 for JSON serialization
            record_data["encrypted_content"] = base64.b64encode(record.encrypted_content).decode()
            
            with open(record_file, 'w') as f:
                json.dump(record_data, f, indent=2, default=str)
            
            # Set appropriate permissions
            os.chmod(record_file, 0o600)
            
        except Exception as e:
            logger.error(f"Failed to write record to disk: {e}")
            raise
    
    async def create_backup(self, backup_location: Optional[str] = None, user_id: str = "system") -> str:
        """Create comprehensive system backup"""
        try:
            # Authorization check
            if user_id != self.AUTHORIZED_USER and user_id != "system":
                raise PermissionError("Only authorized user can create backups")
            
            backup_id = str(uuid.uuid4())
            current_time = time.time()
            
            if backup_location is None:
                backup_location = str(self.backup_storage / f"backup_{backup_id}")
            
            backup_path = Path(backup_location)
            backup_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"📦 Creating backup: {backup_id}")
            
            # Collect all data records
            backup_data = {
                "backup_id": backup_id,
                "created_at": current_time,
                "created_by": user_id,
                "records": []
            }
            
            total_size = 0
            records_count = 0
            
            for record in self.data_registry.values():
                # Skip temporary data in backups
                if record.storage_level == StorageLevel.TEMPORARY:
                    continue
                
                record_data = asdict(record)
                record_data["encrypted_content"] = base64.b64encode(record.encrypted_content).decode()
                backup_data["records"].append(record_data)
                
                total_size += len(record.encrypted_content)
                records_count += 1
            
            # Compress backup data
            backup_json = json.dumps(backup_data, default=str)
            compressed_backup = gzip.compress(backup_json.encode())
            
            # Encrypt backup
            encrypted_backup = self.cipher.encrypt(compressed_backup)
            
            # Write backup file
            backup_file = backup_path / f"backup_{backup_id}.dat"
            with open(backup_file, 'wb') as f:
                f.write(encrypted_backup)
            
            # Calculate integrity hash
            integrity_hash = hashlib.sha256(encrypted_backup).hexdigest()
            
            # Create manifest
            manifest = BackupManifest(
                backup_id=backup_id,
                created_at=current_time,
                records_count=records_count,
                total_size=total_size,
                integrity_hash=integrity_hash,
                backup_location=str(backup_file),
                compression_used=True,
                encryption_used=True
            )
            
            self.backup_manifests[backup_id] = manifest
            
            # Write manifest
            manifest_file = backup_path / f"manifest_{backup_id}.json"
            with open(manifest_file, 'w') as f:
                json.dump(asdict(manifest), f, indent=2, default=str)
            
            # Update metrics
            self.storage_metrics["backup_count"] += 1
            self.storage_metrics["last_backup"] = current_time
            
            # Save updated manifests
            await self._save_backup_manifests()
            
            logger.info(f"✅ Backup created successfully: {backup_id}")
            logger.info(f"   Records: {records_count}, Size: {total_size:,} bytes")
            
            return backup_id
            
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            raise
    
    async def restore_backup(self, backup_id: str, user_id: str) -> bool:
        """Restore from backup"""
        try:
            # Authorization check
            if user_id != self.AUTHORIZED_USER:
                raise PermissionError("Only authorized user can restore backups")
            
            manifest = self.backup_manifests.get(backup_id)
            if not manifest:
                logger.error(f"Backup manifest not found: {backup_id}")
                return False
            
            backup_file = Path(manifest.backup_location)
            if not backup_file.exists():
                logger.error(f"Backup file not found: {backup_file}")
                return False
            
            logger.info(f"🔄 Restoring backup: {backup_id}")
            
            # Read and verify backup
            with open(backup_file, 'rb') as f:
                encrypted_backup = f.read()
            
            # Verify integrity
            current_hash = hashlib.sha256(encrypted_backup).hexdigest()
            if current_hash != manifest.integrity_hash:
                logger.error(f"Backup integrity verification failed: {backup_id}")
                return False
            
            # Decrypt backup
            compressed_backup = self.cipher.decrypt(encrypted_backup)
            
            # Decompress backup
            backup_json = gzip.decompress(compressed_backup).decode()
            backup_data = json.loads(backup_json)
            
            # Clear current registry (with confirmation)
            logger.warning("⚠️ Clearing current data registry for restore")
            self.data_registry.clear()
            
            # Restore records
            restored_count = 0
            for record_data in backup_data["records"]:
                try:
                    # Convert base64 back to bytes
                    record_data["encrypted_content"] = base64.b64decode(record_data["encrypted_content"])
                    
                    record = SecureDataRecord(**record_data)
                    self.data_registry[record.id] = record
                    
                    # Write to disk
                    await self._write_record_to_disk(record)
                    restored_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to restore record: {e}")
                    continue
            
            # Update metrics
            self._update_storage_metrics()
            
            logger.info(f"✅ Backup restored successfully: {backup_id}")
            logger.info(f"   Restored {restored_count} records")
            
            return True
            
        except Exception as e:
            logger.error(f"Backup restoration failed: {e}")
            return False
    
    async def _periodic_backup(self):
        """Perform periodic backup"""
        try:
            current_time = time.time()
            last_backup = self.storage_metrics.get("last_backup", 0)
            
            if current_time - last_backup > self.backup_interval:
                await self.create_backup(user_id="system")
                
        except Exception as e:
            logger.error(f"Periodic backup failed: {e}")
    
    async def _periodic_integrity_check(self):
        """Perform periodic integrity checks"""
        try:
            checked_count = 0
            failed_count = 0
            
            # Check random sample of records
            import random
            sample_size = min(10, len(self.data_registry))
            sample_records = random.sample(list(self.data_registry.values()), sample_size)
            
            for record in sample_records:
                if self._verify_tamper_evidence(record):
                    checked_count += 1
                else:
                    failed_count += 1
                    logger.warning(f"Integrity check failed for record: {record.id}")
            
            self.storage_metrics["integrity_checks"] += checked_count
            
            if failed_count > 0:
                logger.error(f"Integrity check failures: {failed_count}/{sample_size}")
                
        except Exception as e:
            logger.error(f"Integrity check failed: {e}")
    
    async def _cleanup_expired_data(self):
        """Clean up expired data based on retention policy"""
        try:
            current_time = time.time()
            expired_records = []
            
            for record in self.data_registry.values():
                retention_period = self.retention_policy.get(record.storage_level, 86400)
                if current_time - record.created_at > retention_period:
                    expired_records.append(record.id)
            
            for record_id in expired_records:
                await self.delete_data(record_id, self.AUTHORIZED_USER)
                
            if expired_records:
                logger.info(f"🗑️ Cleaned up {len(expired_records)} expired records")
                
        except Exception as e:
            logger.error(f"Data cleanup failed: {e}")
    
    async def delete_data(self, record_id: str, user_id: str) -> bool:
        """Securely delete data"""
        try:
            record = self.data_registry.get(record_id)
            if not record:
                return False
            
            # Authorization check
            if record.storage_level == StorageLevel.CRITICAL and user_id != self.AUTHORIZED_USER:
                raise PermissionError("Only authorized user can delete critical data")
            
            # Remove from registry
            del self.data_registry[record_id]
            
            # Remove file from disk
            storage_path = self.temp_storage if record.storage_level == StorageLevel.TEMPORARY else self.primary_storage
            record_file = storage_path / record.data_type.value / f"{record_id}.json"
            
            if record_file.exists():
                record_file.unlink()
            
            # Update metrics
            self._update_storage_metrics()
            
            logger.info(f"🗑️ Data deleted: {record_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete data: {e}")
            return False
    
    def _update_storage_metrics(self):
        """Update storage metrics"""
        self.storage_metrics.update({
            "total_records": len(self.data_registry),
            "total_size": sum(len(record.encrypted_content) for record in self.data_registry.values()),
            "backup_count": len(self.backup_manifests)
        })
    
    async def _save_data_registry(self):
        """Save data registry to disk"""
        try:
            registry_data = {
                "records": [],
                "saved_at": time.time()
            }
            
            for record in self.data_registry.values():
                record_data = asdict(record)
                record_data["encrypted_content"] = base64.b64encode(record.encrypted_content).decode()
                registry_data["records"].append(record_data)
            
            with open(self.storage_dir / "data_registry.json", 'w') as f:
                json.dump(registry_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save data registry: {e}")
    
    async def _save_backup_manifests(self):
        """Save backup manifests to disk"""
        try:
            manifests_data = {
                "manifests": [asdict(manifest) for manifest in self.backup_manifests.values()],
                "saved_at": time.time()
            }
            
            with open(self.storage_dir / "backup_manifests.json", 'w') as f:
                json.dump(manifests_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save backup manifests: {e}")
    
    async def get_storage_status(self) -> Dict[str, Any]:
        """Get comprehensive storage status"""
        try:
            # Calculate storage by type
            storage_by_type = {}
            storage_by_level = {}
            
            for record in self.data_registry.values():
                # By type
                data_type = record.data_type.value
                if data_type not in storage_by_type:
                    storage_by_type[data_type] = {"count": 0, "size": 0}
                storage_by_type[data_type]["count"] += 1
                storage_by_type[data_type]["size"] += len(record.encrypted_content)
                
                # By level
                storage_level = record.storage_level.value
                if storage_level not in storage_by_level:
                    storage_by_level[storage_level] = {"count": 0, "size": 0}
                storage_by_level[storage_level]["count"] += 1
                storage_by_level[storage_level]["size"] += len(record.encrypted_content)
            
            return {
                "metrics": self.storage_metrics.copy(),
                "storage_by_type": storage_by_type,
                "storage_by_level": storage_by_level,
                "recent_backups": [
                    {
                        "backup_id": backup_id,
                        "created_at": manifest.created_at,
                        "records_count": manifest.records_count,
                        "total_size": manifest.total_size
                    }
                    for backup_id, manifest in sorted(
                        self.backup_manifests.items(),
                        key=lambda x: x[1].created_at,
                        reverse=True
                    )[:5]
                ],
                "retention_policy": self.retention_policy,
                "auto_backup_enabled": self.auto_backup_enabled
            }
            
        except Exception as e:
            logger.error(f"Failed to get storage status: {e}")
            return {"error": str(e)}
    
    async def cleanup(self):
        """Cleanup storage system"""
        try:
            # Save all data
            await self._save_data_registry()
            await self._save_backup_manifests()
            
            logger.info("✅ Secure storage system cleaned up")
            
        except Exception as e:
            logger.error(f"Storage cleanup failed: {e}")

# Global instance
secure_storage_system = SecureStorageSystem()

# Convenience functions
async def store_secure_data(data: Any, data_type: DataType, storage_level: StorageLevel, user_id: str, metadata: Dict[str, Any] = None) -> str:
    """Store data securely"""
    return await secure_storage_system.store_data(data, data_type, storage_level, user_id, metadata)

async def retrieve_secure_data(record_id: str, user_id: str) -> Optional[Any]:
    """Retrieve secure data"""
    return await secure_storage_system.retrieve_data(record_id, user_id)

async def create_system_backup(user_id: str) -> str:
    """Create system backup"""
    return await secure_storage_system.create_backup(user_id=user_id)

async def restore_system_backup(backup_id: str, user_id: str) -> bool:
    """Restore system backup"""
    return await secure_storage_system.restore_backup(backup_id, user_id)

async def get_storage_status() -> Dict[str, Any]:
    """Get storage status"""
    return await secure_storage_system.get_storage_status()