#!/usr/bin/env python3
"""
MCP Version Manager

Comprehensive version tracking, staging, and rollback management for MCP servers.
Provides safe update operations with comprehensive audit trails and recovery
capabilities following Enforcement Rules compliance.

Author: Claude AI Assistant (python-architect.md)
Created: 2025-08-15 11:25:00 UTC
Version: 1.0.0
"""

import asyncio
import json
import shutil
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging

from config import get_config, MCPAutomationConfig, UpdateMode


class VersionState(Enum):
    """Version states for MCP server tracking."""
    UNKNOWN = "unknown"
    INSTALLED = "installed"
    STAGED = "staged"
    TESTING = "testing"
    ACTIVE = "active"
    ROLLBACK = "rollback"
    ARCHIVED = "archived"
    FAILED = "failed"


class OperationType(Enum):
    """Types of version operations."""
    INSTALL = "install"
    UPDATE = "update"
    STAGE = "stage"
    ACTIVATE = "activate"
    ROLLBACK = "rollback"
    HEALTH_CHECK = "health_check"
    CLEANUP = "cleanup"


@dataclass
class VersionInfo:
    """Version information for MCP servers."""
    server_name: str
    package_name: str
    version: str
    installed_at: datetime
    state: VersionState
    source_url: str
    checksum: str
    size_bytes: int
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['installed_at'] = self.installed_at.isoformat()
        result['state'] = self.state.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VersionInfo':
        """Create from dictionary."""
        data = data.copy()
        data['installed_at'] = datetime.fromisoformat(data['installed_at'])
        data['state'] = VersionState(data['state'])
        return cls(**data)


@dataclass
class OperationRecord:
    """Record of version operations for audit trail."""
    operation_id: str
    operation_type: OperationType
    server_name: str
    from_version: Optional[str]
    to_version: str
    timestamp: datetime
    success: bool
    duration_seconds: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['operation_type'] = self.operation_type.value
        result['timestamp'] = self.timestamp.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OperationRecord':
        """Create from dictionary."""
        data = data.copy()
        data['operation_type'] = OperationType(data['operation_type'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class VersionManager:
    """
    Comprehensive version management for MCP servers.
    
    Provides version tracking, staging, activation, and rollback capabilities
    with comprehensive audit trails and safety mechanisms.
    """
    
    def __init__(self, config: Optional[MCPAutomationConfig] = None):
        """
        Initialize version manager.
        
        Args:
            config: Optional configuration override
        """
        self.config = config or get_config()
        self.logger = self._setup_logging()
        
        # Initialize storage paths
        self.versions_file = self.config.paths.automation_root / "versions.json"
        self.operations_file = self.config.paths.automation_root / "operations.json"
        
        # Initialize data structures
        self.versions: Dict[str, List[VersionInfo]] = {}
        self.operations: List[OperationRecord] = []
        
        # Load existing data
        self._load_versions()
        self._load_operations()
        
        self.logger.info("Version manager initialized successfully")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging for version manager."""
        logger = logging.getLogger(f"{__name__}.VersionManager")
        logger.setLevel(getattr(logging, self.config.log_level.value))
        
        if not logger.handlers:
            # Create logs directory
            self.config.paths.logs_root.mkdir(parents=True, exist_ok=True)
            
            # File handler for detailed logs
            log_file = self.config.paths.logs_root / f"version_manager_{datetime.now().strftime('%Y%m%d')}.log"
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
            # Console handler for immediate feedback
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def _load_versions(self):
        """Load version information from storage."""
        try:
            if self.versions_file.exists():
                with open(self.versions_file, 'r') as f:
                    data = json.load(f)
                
                self.versions = {}
                for server_name, version_list in data.items():
                    self.versions[server_name] = [
                        VersionInfo.from_dict(v) for v in version_list
                    ]
                
                self.logger.debug(f"Loaded versions for {len(self.versions)} servers")
            else:
                self.logger.info("No existing version file found, starting fresh")
        
        except Exception as e:
            self.logger.error(f"Failed to load versions: {e}")
            self.versions = {}
    
    def _load_operations(self):
        """Load operation history from storage."""
        try:
            if self.operations_file.exists():
                with open(self.operations_file, 'r') as f:
                    data = json.load(f)
                
                self.operations = [OperationRecord.from_dict(op) for op in data]
                self.logger.debug(f"Loaded {len(self.operations)} operation records")
            else:
                self.logger.info("No existing operations file found, starting fresh")
        
        except Exception as e:
            self.logger.error(f"Failed to load operations: {e}")
            self.operations = []
    
    def _save_versions(self):
        """Save version information to storage."""
        try:
            data = {}
            for server_name, version_list in self.versions.items():
                data[server_name] = [v.to_dict() for v in version_list]
            
            # Atomic write with backup
            temp_file = self.versions_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Backup existing file if it exists
            if self.versions_file.exists():
                backup_file = self.versions_file.with_suffix('.backup')
                shutil.move(self.versions_file, backup_file)
            
            # Move temp file to final location
            shutil.move(temp_file, self.versions_file)
            
            self.logger.debug("Version information saved successfully")
        
        except Exception as e:
            self.logger.error(f"Failed to save versions: {e}")
            raise
    
    def _save_operations(self):
        """Save operation history to storage."""
        try:
            data = [op.to_dict() for op in self.operations]
            
            # Atomic write with backup
            temp_file = self.operations_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Backup existing file if it exists
            if self.operations_file.exists():
                backup_file = self.operations_file.with_suffix('.backup')
                shutil.move(self.operations_file, backup_file)
            
            # Move temp file to final location
            shutil.move(temp_file, self.operations_file)
            
            self.logger.debug("Operation history saved successfully")
        
        except Exception as e:
            self.logger.error(f"Failed to save operations: {e}")
            raise
    
    async def get_current_version(self, server_name: str) -> Optional[str]:
        """
        Get current active version for a server.
        
        Args:
            server_name: Name of the MCP server
            
        Returns:
            Current version string or None if not found
        """
        try:
            # Check NPM global installation
            result = await asyncio.create_subprocess_exec(
                'npm', 'list', '-g', '--depth=0', '--json',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                npm_data = json.loads(stdout.decode())
                server_config = self.config.get_server_config(server_name)
                
                if server_config and 'package' in server_config:
                    package_name = server_config['package']
                    dependencies = npm_data.get('dependencies', {})
                    
                    if package_name in dependencies:
                        return dependencies[package_name]['version']
            
            # Fallback to version tracking
            if server_name in self.versions:
                active_versions = [
                    v for v in self.versions[server_name] 
                    if v.state == VersionState.ACTIVE
                ]
                if active_versions:
                    return active_versions[-1].version
            
            return None
        
        except Exception as e:
            self.logger.error(f"Failed to get current version for {server_name}: {e}")
            return None
    
    async def get_available_version(self, server_name: str) -> Optional[str]:
        """
        Get latest available version for a server from NPM registry.
        
        Args:
            server_name: Name of the MCP server
            
        Returns:
            Latest available version or None if not found
        """
        try:
            server_config = self.config.get_server_config(server_name)
            if not server_config or 'package' not in server_config:
                self.logger.error(f"No package configuration for server: {server_name}")
                return None
            
            package_name = server_config['package']
            
            # Query NPM registry for latest version
            result = await asyncio.create_subprocess_exec(
                'npm', 'view', package_name, 'version',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                version = stdout.decode().strip()
                self.logger.debug(f"Latest version for {server_name}: {version}")
                return version
            else:
                self.logger.error(f"Failed to query NPM for {package_name}: {stderr.decode()}")
                return None
        
        except Exception as e:
            self.logger.error(f"Failed to get available version for {server_name}: {e}")
            return None
    
    async def stage_version(self, server_name: str, version: str, 
                           source_path: Path) -> bool:
        """
        Stage a new version for testing.
        
        Args:
            server_name: Name of the MCP server
            version: Version to stage
            source_path: Path to downloaded package
            
        Returns:
            True if staging successful, False otherwise
        """
        operation_id = f"stage_{server_name}_{version}_{int(datetime.now().timestamp())}"
        start_time = datetime.now(timezone.utc)
        
        try:
            self.logger.info(f"Staging {server_name} version {version}")
            
            # Create staging directory
            staging_path = self.config.get_staging_path(server_name)
            staging_path.mkdir(parents=True, exist_ok=True)
            
            # Calculate checksum
            checksum = await self._calculate_checksum(source_path)
            
            # Copy to staging area
            staged_file = staging_path / f"{server_name}-{version}.tgz"
            shutil.copy2(source_path, staged_file)
            
            # Create version info
            version_info = VersionInfo(
                server_name=server_name,
                package_name=self.config.get_server_config(server_name)['package'],
                version=version,
                installed_at=start_time,
                state=VersionState.STAGED,
                source_url=str(source_path),
                checksum=checksum,
                size_bytes=source_path.stat().st_size,
                metadata={
                    'staging_path': str(staged_file),
                    'operation_id': operation_id
                }
            )
            
            # Add to version tracking
            if server_name not in self.versions:
                self.versions[server_name] = []
            
            self.versions[server_name].append(version_info)
            
            # Record operation
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            operation = OperationRecord(
                operation_id=operation_id,
                operation_type=OperationType.STAGE,
                server_name=server_name,
                from_version=await self.get_current_version(server_name),
                to_version=version,
                timestamp=start_time,
                success=True,
                duration_seconds=duration,
                metadata={'staging_path': str(staged_file)}
            )
            
            self.operations.append(operation)
            
            # Save state
            self._save_versions()
            self._save_operations()
            
            self.logger.info(f"Successfully staged {server_name} version {version}")
            return True
        
        except Exception as e:
            # Record failed operation
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            operation = OperationRecord(
                operation_id=operation_id,
                operation_type=OperationType.STAGE,
                server_name=server_name,
                from_version=await self.get_current_version(server_name),
                to_version=version,
                timestamp=start_time,
                success=False,
                duration_seconds=duration,
                error_message=str(e)
            )
            
            self.operations.append(operation)
            self._save_operations()
            
            self.logger.error(f"Failed to stage {server_name} version {version}: {e}")
            return False
    
    async def activate_version(self, server_name: str, version: str) -> bool:
        """
        Activate a staged version.
        
        Args:
            server_name: Name of the MCP server
            version: Version to activate
            
        Returns:
            True if activation successful, False otherwise
        """
        operation_id = f"activate_{server_name}_{version}_{int(datetime.now().timestamp())}"
        start_time = datetime.now(timezone.utc)
        
        try:
            self.logger.info(f"Activating {server_name} version {version}")
            
            # Find staged version
            staged_version = None
            if server_name in self.versions:
                for v in self.versions[server_name]:
                    if v.version == version and v.state == VersionState.STAGED:
                        staged_version = v
                        break
            
            if not staged_version:
                raise ValueError(f"No staged version {version} found for {server_name}")
            
            # Backup current version
            current_version = await self.get_current_version(server_name)
            if current_version:
                await self._backup_current_version(server_name, current_version)
            
            # Install staged version
            staging_path = Path(staged_version.metadata['staging_path'])
            server_config = self.config.get_server_config(server_name)
            package_name = server_config['package']
            
            # Install via NPM
            result = await asyncio.create_subprocess_exec(
                'npm', 'install', '-g', str(staging_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                raise RuntimeError(f"NPM install failed: {stderr.decode()}")
            
            # Update version states
            # Mark previous active versions as archived
            for v in self.versions[server_name]:
                if v.state == VersionState.ACTIVE:
                    v.state = VersionState.ARCHIVED
            
            # Mark staged version as active
            staged_version.state = VersionState.ACTIVE
            
            # Record operation
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            operation = OperationRecord(
                operation_id=operation_id,
                operation_type=OperationType.ACTIVATE,
                server_name=server_name,
                from_version=current_version,
                to_version=version,
                timestamp=start_time,
                success=True,
                duration_seconds=duration,
                metadata={'npm_output': stdout.decode()}
            )
            
            self.operations.append(operation)
            
            # Save state
            self._save_versions()
            self._save_operations()
            
            self.logger.info(f"Successfully activated {server_name} version {version}")
            return True
        
        except Exception as e:
            # Record failed operation
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            operation = OperationRecord(
                operation_id=operation_id,
                operation_type=OperationType.ACTIVATE,
                server_name=server_name,
                from_version=await self.get_current_version(server_name),
                to_version=version,
                timestamp=start_time,
                success=False,
                duration_seconds=duration,
                error_message=str(e)
            )
            
            self.operations.append(operation)
            self._save_operations()
            
            self.logger.error(f"Failed to activate {server_name} version {version}: {e}")
            return False
    
    async def rollback_version(self, server_name: str, target_version: Optional[str] = None) -> bool:
        """
        Rollback to a previous version.
        
        Args:
            server_name: Name of the MCP server
            target_version: Specific version to rollback to (latest archived if None)
            
        Returns:
            True if rollback successful, False otherwise
        """
        operation_id = f"rollback_{server_name}_{int(datetime.now().timestamp())}"
        start_time = datetime.now(timezone.utc)
        current_version = await self.get_current_version(server_name)
        
        try:
            self.logger.info(f"Rolling back {server_name} from version {current_version}")
            
            # Find target version
            rollback_version = None
            if server_name in self.versions:
                if target_version:
                    # Find specific version
                    for v in self.versions[server_name]:
                        if v.version == target_version and v.state in [VersionState.ARCHIVED, VersionState.ACTIVE]:
                            rollback_version = v
                            break
                else:
                    # Find latest archived version
                    archived_versions = [
                        v for v in self.versions[server_name] 
                        if v.state == VersionState.ARCHIVED
                    ]
                    if archived_versions:
                        rollback_version = sorted(archived_versions, key=lambda x: x.installed_at)[-1]
            
            if not rollback_version:
                raise ValueError(f"No suitable rollback version found for {server_name}")
            
            # Install rollback version
            server_config = self.config.get_server_config(server_name)
            package_name = server_config['package']
            package_with_version = f"{package_name}@{rollback_version.version}"
            
            result = await asyncio.create_subprocess_exec(
                'npm', 'install', '-g', package_with_version,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                raise RuntimeError(f"NPM install failed: {stderr.decode()}")
            
            # Update version states
            for v in self.versions[server_name]:
                if v.state == VersionState.ACTIVE:
                    v.state = VersionState.ROLLBACK
            
            rollback_version.state = VersionState.ACTIVE
            
            # Record operation
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            operation = OperationRecord(
                operation_id=operation_id,
                operation_type=OperationType.ROLLBACK,
                server_name=server_name,
                from_version=current_version,
                to_version=rollback_version.version,
                timestamp=start_time,
                success=True,
                duration_seconds=duration,
                metadata={'npm_output': stdout.decode()}
            )
            
            self.operations.append(operation)
            
            # Save state
            self._save_versions()
            self._save_operations()
            
            self.logger.info(f"Successfully rolled back {server_name} to version {rollback_version.version}")
            return True
        
        except Exception as e:
            # Record failed operation
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            operation = OperationRecord(
                operation_id=operation_id,
                operation_type=OperationType.ROLLBACK,
                server_name=server_name,
                from_version=current_version,
                to_version=target_version or "unknown",
                timestamp=start_time,
                success=False,
                duration_seconds=duration,
                error_message=str(e)
            )
            
            self.operations.append(operation)
            self._save_operations()
            
            self.logger.error(f"Failed to rollback {server_name}: {e}")
            return False
    
    async def _backup_current_version(self, server_name: str, version: str):
        """Create backup of current version."""
        try:
            backup_path = self.config.get_backup_path(server_name)
            backup_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = backup_path / f"{server_name}-{version}-{timestamp}.backup"
            
            # Create backup metadata
            backup_metadata = {
                'server_name': server_name,
                'version': version,
                'backed_up_at': datetime.now(timezone.utc).isoformat(),
                'original_installation': 'npm_global'
            }
            
            with open(backup_file, 'w') as f:
                json.dump(backup_metadata, f, indent=2)
            
            self.logger.debug(f"Created backup for {server_name} version {version}")
        
        except Exception as e:
            self.logger.warning(f"Failed to create backup for {server_name}: {e}")
    
    async def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            
            return hash_sha256.hexdigest()
        
        except Exception as e:
            self.logger.error(f"Failed to calculate checksum for {file_path}: {e}")
            return ""
    
    def get_version_history(self, server_name: str) -> List[VersionInfo]:
        """Get version history for a server."""
        return self.versions.get(server_name, [])
    
    def get_operation_history(self, server_name: Optional[str] = None, 
                            limit: Optional[int] = None) -> List[OperationRecord]:
        """Get operation history."""
        operations = self.operations
        
        if server_name:
            operations = [op for op in operations if op.server_name == server_name]
        
        # Sort by timestamp (newest first)
        operations = sorted(operations, key=lambda x: x.timestamp, reverse=True)
        
        if limit:
            operations = operations[:limit]
        
        return operations
    
    async def cleanup_old_versions(self, server_name: str, keep_count: int = 3) -> int:
        """
        Clean up old versions, keeping only the specified number.
        
        Args:
            server_name: Name of the MCP server
            keep_count: Number of versions to keep
            
        Returns:
            Number of versions cleaned up
        """
        try:
            if server_name not in self.versions:
                return 0
            
            versions = self.versions[server_name]
            
            # Keep active and recent archived versions
            active_versions = [v for v in versions if v.state == VersionState.ACTIVE]
            archived_versions = [v for v in versions if v.state == VersionState.ARCHIVED]
            
            # Sort archived versions by installation date
            archived_versions.sort(key=lambda x: x.installed_at, reverse=True)
            
            # Determine versions to remove
            keep_archived = archived_versions[:max(0, keep_count - len(active_versions))]
            remove_versions = archived_versions[len(keep_archived):]
            
            cleaned_count = 0
            for version in remove_versions:
                # Remove staging files if they exist
                if 'staging_path' in version.metadata:
                    staging_path = Path(version.metadata['staging_path'])
                    if staging_path.exists():
                        staging_path.unlink()
                
                # Remove from tracking
                versions.remove(version)
                cleaned_count += 1
            
            if cleaned_count > 0:
                self._save_versions()
                self.logger.info(f"Cleaned up {cleaned_count} old versions for {server_name}")
            
            return cleaned_count
        
        except Exception as e:
            self.logger.error(f"Failed to cleanup old versions for {server_name}: {e}")
            return 0


if __name__ == "__main__":
    # Version manager testing
    async def main():
        version_manager = VersionManager()
        
        # Test getting current versions
        for server_name in ["files", "postgres", "playwright-mcp"]:
            current = await version_manager.get_current_version(server_name)
            available = await version_manager.get_available_version(server_name)
            
            logger.info(f"{server_name}:")
            logger.info(f"  Current: {current}")
            logger.info(f"  Available: {available}")
            logger.info()
        
        # Show operation history
        operations = version_manager.get_operation_history(limit=5)
        logger.info(f"Recent operations: {len(operations)}")
        for op in operations:
            logger.info(f"  {op.timestamp}: {op.operation_type.value} {op.server_name} -> {op.to_version}")
    
    asyncio.run(main())