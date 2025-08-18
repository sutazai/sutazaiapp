"""
Memory Migration Service
Clean architecture implementation for migrating data between memory services
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json
import sqlite3
import httpx
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class MigrationConfig:
    """Configuration for memory service migration"""
    source_service: str
    target_service: str
    batch_size: int = 100
    timeout: float = 30.0
    backup_enabled: bool = True
    dry_run: bool = False

@dataclass
class MigrationResult:
    """Result of migration operation"""
    total_records: int
    migrated_count: int
    failed_count: int
    errors: List[str]
    duration_seconds: float
    backup_path: Optional[str] = None

class MemoryMigrationService:
    """Service for migrating memory data between MCP services"""
    
    def __init__(self, config: MigrationConfig):
        self.config = config
        self.unified_memory_url = "http://localhost:3009"
        self.backup_dir = Path("/opt/sutazaiapp/backups/mcp-migration")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    async def migrate_extended_memory(self) -> MigrationResult:
        """Migrate data from extended-memory to unified-memory"""
        start_time = datetime.now()
        errors = []
        migrated_count = 0
        failed_count = 0
        
        logger.info("Starting extended-memory to unified-memory migration")
        
        try:
            # Create backup if enabled
            backup_path = None
            if self.config.backup_enabled:
                backup_path = await self._create_backup("extended-memory")
            
            # Load extended-memory data
            source_data = await self._load_extended_memory_data()
            total_records = len(source_data)
            
            logger.info(f"Found {total_records} records to migrate")
            
            if self.config.dry_run:
                logger.info("DRY RUN: Would migrate {total_records} records")
                return MigrationResult(
                    total_records=total_records,
                    migrated_count=0,
                    failed_count=0,
                    errors=[],
                    duration_seconds=0.0,
                    backup_path=str(backup_path) if backup_path else None
                )
            
            # Migrate in batches
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                for i in range(0, total_records, self.config.batch_size):
                    batch = source_data[i:i + self.config.batch_size]
                    
                    for record in batch:
                        try:
                            # Transform to unified memory format
                            unified_data = self._transform_extended_memory_record(record)
                            
                            # Store in unified memory
                            response = await client.post(
                                f"{self.unified_memory_url}/memory/store",
                                json=unified_data
                            )
                            
                            if response.status_code == 200:
                                migrated_count += 1
                            else:
                                failed_count += 1
                                errors.append(f"Failed to migrate record {record.get('id', 'unknown')}: {response.text}")
                        
                        except Exception as e:
                            failed_count += 1
                            errors.append(f"Error migrating record: {str(e)}")
                    
                    # Log progress
                    logger.info(f"Migrated batch {i//self.config.batch_size + 1}: {migrated_count} success, {failed_count} failed")
            
            duration = (datetime.now() - start_time).total_seconds()
            
            result = MigrationResult(
                total_records=total_records,
                migrated_count=migrated_count,
                failed_count=failed_count,
                errors=errors,
                duration_seconds=duration,
                backup_path=str(backup_path) if backup_path else None
            )
            
            logger.info(f"Migration completed: {migrated_count}/{total_records} migrated in {duration:.2f}s")
            return result
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"Migration failed: {str(e)}")
            return MigrationResult(
                total_records=0,
                migrated_count=0,
                failed_count=1,
                errors=[str(e)],
                duration_seconds=duration
            )
    
    async def migrate_memory_bank(self) -> MigrationResult:
        """Migrate data from memory-bank-mcp to unified-memory"""
        start_time = datetime.now()
        errors = []
        migrated_count = 0
        failed_count = 0
        
        logger.info("Starting memory-bank-mcp to unified-memory migration")
        
        try:
            # Create backup if enabled
            backup_path = None
            if self.config.backup_enabled:
                backup_path = await self._create_backup("memory-bank-mcp")
            
            # Load memory-bank data
            source_data = await self._load_memory_bank_data()
            total_records = len(source_data)
            
            logger.info(f"Found {total_records} records to migrate")
            
            if self.config.dry_run:
                logger.info(f"DRY RUN: Would migrate {total_records} records")
                return MigrationResult(
                    total_records=total_records,
                    migrated_count=0,
                    failed_count=0,
                    errors=[],
                    duration_seconds=0.0,
                    backup_path=str(backup_path) if backup_path else None
                )
            
            # Migrate in batches
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                for i in range(0, total_records, self.config.batch_size):
                    batch = source_data[i:i + self.config.batch_size]
                    
                    for record in batch:
                        try:
                            # Transform to unified memory format
                            unified_data = self._transform_memory_bank_record(record)
                            
                            # Store in unified memory
                            response = await client.post(
                                f"{self.unified_memory_url}/memory/store",
                                json=unified_data
                            )
                            
                            if response.status_code == 200:
                                migrated_count += 1
                            else:
                                failed_count += 1
                                errors.append(f"Failed to migrate record {record.get('id', 'unknown')}: {response.text}")
                        
                        except Exception as e:
                            failed_count += 1
                            errors.append(f"Error migrating record: {str(e)}")
                    
                    # Log progress
                    logger.info(f"Migrated batch {i//self.config.batch_size + 1}: {migrated_count} success, {failed_count} failed")
            
            duration = (datetime.now() - start_time).total_seconds()
            
            result = MigrationResult(
                total_records=total_records,
                migrated_count=migrated_count,
                failed_count=failed_count,
                errors=errors,
                duration_seconds=duration,
                backup_path=str(backup_path) if backup_path else None
            )
            
            logger.info(f"Migration completed: {migrated_count}/{total_records} migrated in {duration:.2f}s")
            return result
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"Migration failed: {str(e)}")
            return MigrationResult(
                total_records=0,
                migrated_count=0,
                failed_count=1,
                errors=[str(e)],
                duration_seconds=duration
            )
    
    async def _create_backup(self, service_name: str) -> Optional[Path]:
        """Create backup of source service data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_dir / f"{service_name}_backup_{timestamp}.json"
        
        try:
            if service_name == "extended-memory":
                data = await self._load_extended_memory_data()
            elif service_name == "memory-bank-mcp":
                data = await self._load_memory_bank_data()
            else:
                logger.warning(f"Unknown service for backup: {service_name}")
                return None
            
            with open(backup_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Created backup: {backup_file}")
            return backup_file
            
        except Exception as e:
            logger.error(f"Failed to create backup for {service_name}: {str(e)}")
            return None
    
    async def _load_extended_memory_data(self) -> List[Dict[str, Any]]:
        """Load data from extended-memory service"""
        # Simulate loading from extended-memory database/storage
        # In real implementation, this would connect to the actual extended-memory storage
        return [
            {
                "id": f"extended_{i}",
                "key": f"extended_key_{i}",
                "content": f"Extended memory content {i}",
                "namespace": "migration_test",
                "tags": ["extended", "migration"],
                "importance_level": 5,
                "created_at": datetime.now().isoformat()
            }
            for i in range(5)  # Sample data for demonstration
        ]
    
    async def _load_memory_bank_data(self) -> List[Dict[str, Any]]:
        """Load data from memory-bank-mcp service"""
        # Simulate loading from memory-bank-mcp database/storage
        # In real implementation, this would connect to the actual memory-bank storage
        return [
            {
                "id": f"bank_{i}",
                "key": f"bank_key_{i}",
                "content": f"Memory bank content {i}",
                "namespace": "migration_test",
                "tags": ["bank", "migration"],
                "importance_level": 7,
                "created_at": datetime.now().isoformat()
            }
            for i in range(3)  # Sample data for demonstration
        ]
    
    def _transform_extended_memory_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Transform extended-memory record to unified memory format"""
        return {
            "key": record.get("key", record.get("id")),
            "content": record.get("content", ""),
            "namespace": record.get("namespace", "migrated_extended"),
            "tags": record.get("tags", []) + ["migrated_from_extended"],
            "importance_level": record.get("importance_level", 5),
            "ttl": 86400 * 30  # 30 days default TTL
        }
    
    def _transform_memory_bank_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Transform memory-bank record to unified memory format"""
        return {
            "key": record.get("key", record.get("id")),
            "content": record.get("content", ""),
            "namespace": record.get("namespace", "migrated_bank"),
            "tags": record.get("tags", []) + ["migrated_from_bank"],
            "importance_level": record.get("importance_level", 5),
            "ttl": 86400 * 30  # 30 days default TTL
        }

# Factory function for easy service creation
def create_migration_service(
    source_service: str,
    target_service: str = "unified-memory",
    **kwargs
) -> MemoryMigrationService:
    """Create migration service with configuration"""
    config = MigrationConfig(
        source_service=source_service,
        target_service=target_service,
        **kwargs
    )
    return MemoryMigrationService(config)