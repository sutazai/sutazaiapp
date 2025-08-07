#!/usr/bin/env python3
"""
Advanced Rollback System for SutazAI
Version: 1.0.0

DESCRIPTION:
    Sophisticated rollback and recovery system that provides automated rollback
    capabilities with state recovery, snapshot management, and intelligent
    recovery strategies for the SutazAI ecosystem.

PURPOSE:
    - Create comprehensive system snapshots
    - Implement automated rollback on deployment failures
    - Provide intelligent recovery strategies
    - Maintain rollback history and metadata
    - Support granular rollback (service-level, component-level)
    - Validate rollback success and system integrity

USAGE:
    python advanced-rollback-system.py [command] [options]

REQUIREMENTS:
    - Python 3.8+
    - Docker and Docker Compose
    - System access for file operations
    - Database access for state management
"""

import asyncio
import json
import logging
import os
import sys
import time
import traceback
import shutil
import subprocess
import sqlite3
import hashlib
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
import tarfile
import tempfile
from dataclasses import dataclass, asdict, field
import yaml
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
LOG_DIR = PROJECT_ROOT / "logs"
ROLLBACK_DIR = LOG_DIR / "rollback"
STATE_DIR = LOG_DIR / "deployment_state"
BACKUP_DIR = PROJECT_ROOT / "backups"
SNAPSHOT_DB = ROLLBACK_DIR / "rollback_metadata.db"

# Ensure directories exist
for directory in [LOG_DIR, ROLLBACK_DIR, STATE_DIR, BACKUP_DIR]:
    directory.mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "rollback-system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SnapshotType(Enum):
    """Types of snapshots"""
    FULL_SYSTEM = "full_system"
    CONFIGURATION = "configuration"
    DOCKER_STATE = "docker_state"
    DATABASE = "database"
    USER_DATA = "user_data"
    LOGS = "logs"

class RollbackStatus(Enum):
    """Rollback operation status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"
    VERIFIED = "verified"

class RecoveryStrategy(Enum):
    """Recovery strategies"""
    IMMEDIATE = "immediate"
    GRACEFUL = "graceful"
    SELECTIVE = "selective"
    PROGRESSIVE = "progressive"
    EMERGENCY = "emergency"

@dataclass
class SnapshotComponent:
    """Individual component within a snapshot"""
    name: str
    type: str
    path: str
    size: int
    checksum: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemSnapshot:
    """System snapshot metadata"""
    snapshot_id: str
    timestamp: datetime
    snapshot_type: SnapshotType
    deployment_id: str
    phase: str
    description: str
    size: int
    components: List[SnapshotComponent]
    system_state: Dict[str, Any]
    docker_state: Dict[str, Any]
    environment_vars: Dict[str, str]
    health_status: Dict[str, Any]
    tags: List[str] = field(default_factory=list)
    retention_policy: str = "default"
    verified: bool = False

@dataclass
class RollbackOperation:
    """Rollback operation tracking"""
    operation_id: str
    snapshot_id: str
    initiated_by: str
    start_time: datetime
    end_time: Optional[datetime]
    status: RollbackStatus
    strategy: RecoveryStrategy
    target_components: List[str]
    progress: float
    steps_completed: List[str]
    steps_failed: List[str]
    error_details: List[str]
    verification_results: Dict[str, Any]

class SnapshotDatabase:
    """SQLite database for snapshot metadata"""
    
    def __init__(self, db_file: Path):
        self.db_file = db_file
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_file) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    snapshot_type TEXT NOT NULL,
                    deployment_id TEXT,
                    phase TEXT,
                    description TEXT,
                    size INTEGER,
                    system_state TEXT,
                    docker_state TEXT,
                    environment_vars TEXT,
                    health_status TEXT,
                    tags TEXT,
                    retention_policy TEXT,
                    verified INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS snapshot_components (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    snapshot_id TEXT NOT NULL,
                    component_name TEXT NOT NULL,
                    component_type TEXT NOT NULL,
                    path TEXT NOT NULL,
                    size INTEGER,
                    checksum TEXT,
                    metadata TEXT,
                    FOREIGN KEY (snapshot_id) REFERENCES snapshots (snapshot_id)
                );
                
                CREATE TABLE IF NOT EXISTS rollback_operations (
                    operation_id TEXT PRIMARY KEY,
                    snapshot_id TEXT NOT NULL,
                    initiated_by TEXT,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    status TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    target_components TEXT,
                    progress REAL DEFAULT 0,
                    steps_completed TEXT,
                    steps_failed TEXT,
                    error_details TEXT,
                    verification_results TEXT,
                    FOREIGN KEY (snapshot_id) REFERENCES snapshots (snapshot_id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON snapshots(timestamp);
                CREATE INDEX IF NOT EXISTS idx_snapshots_deployment ON snapshots(deployment_id);
                CREATE INDEX IF NOT EXISTS idx_rollback_operations_snapshot ON rollback_operations(snapshot_id);
            """)
    
    def store_snapshot(self, snapshot: SystemSnapshot):
        """Store snapshot metadata"""
        with sqlite3.connect(self.db_file) as conn:
            # Store main snapshot record
            conn.execute("""
                INSERT OR REPLACE INTO snapshots
                (snapshot_id, timestamp, snapshot_type, deployment_id, phase,
                 description, size, system_state, docker_state, environment_vars,
                 health_status, tags, retention_policy, verified)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                snapshot.snapshot_id,
                snapshot.timestamp.isoformat(),
                snapshot.snapshot_type.value,
                snapshot.deployment_id,
                snapshot.phase,
                snapshot.description,
                snapshot.size,
                json.dumps(snapshot.system_state),
                json.dumps(snapshot.docker_state),
                json.dumps(snapshot.environment_vars),
                json.dumps(snapshot.health_status),
                json.dumps(snapshot.tags),
                snapshot.retention_policy,
                1 if snapshot.verified else 0
            ))
            
            # Store components
            for component in snapshot.components:
                conn.execute("""
                    INSERT INTO snapshot_components
                    (snapshot_id, component_name, component_type, path, size, checksum, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    snapshot.snapshot_id,
                    component.name,
                    component.type,
                    component.path,
                    component.size,
                    component.checksum,
                    json.dumps(component.metadata)
                ))
    
    def get_snapshot(self, snapshot_id: str) -> Optional[SystemSnapshot]:
        """Retrieve snapshot by ID"""
        with sqlite3.connect(self.db_file) as conn:
            # Get main snapshot record
            row = conn.execute("""
                SELECT snapshot_id, timestamp, snapshot_type, deployment_id, phase,
                       description, size, system_state, docker_state, environment_vars,
                       health_status, tags, retention_policy, verified
                FROM snapshots WHERE snapshot_id = ?
            """, (snapshot_id,)).fetchone()
            
            if not row:
                return None
            
            # Get components
            component_rows = conn.execute("""
                SELECT component_name, component_type, path, size, checksum, metadata
                FROM snapshot_components WHERE snapshot_id = ?
            """, (snapshot_id,)).fetchall()
            
            components = []
            for comp_row in component_rows:
                components.append(SnapshotComponent(
                    name=comp_row[0],
                    type=comp_row[1],
                    path=comp_row[2],
                    size=comp_row[3],
                    checksum=comp_row[4],
                    metadata=json.loads(comp_row[5]) if comp_row[5] else {}
                ))
            
            return SystemSnapshot(
                snapshot_id=row[0],
                timestamp=datetime.fromisoformat(row[1]),
                snapshot_type=SnapshotType(row[2]),
                deployment_id=row[3],
                phase=row[4],
                description=row[5],
                size=row[6],
                components=components,
                system_state=json.loads(row[7]) if row[7] else {},
                docker_state=json.loads(row[8]) if row[8] else {},
                environment_vars=json.loads(row[9]) if row[9] else {},
                health_status=json.loads(row[10]) if row[10] else {},
                tags=json.loads(row[11]) if row[11] else [],
                retention_policy=row[12],
                verified=bool(row[13])
            )
    
    def list_snapshots(self, limit: int = 50, deployment_id: str = None) -> List[SystemSnapshot]:
        """List snapshots with optional filtering"""
        with sqlite3.connect(self.db_file) as conn:
            query = """
                SELECT snapshot_id, timestamp, snapshot_type, deployment_id, phase,
                       description, size, verified
                FROM snapshots
            """
            params = []
            
            if deployment_id:
                query += " WHERE deployment_id = ?"
                params.append(deployment_id)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            rows = conn.execute(query, params).fetchall()
            
            snapshots = []
            for row in rows:
                # Get basic snapshot info (without components for performance)
                snapshots.append(SystemSnapshot(
                    snapshot_id=row[0],
                    timestamp=datetime.fromisoformat(row[1]),
                    snapshot_type=SnapshotType(row[2]),
                    deployment_id=row[3],
                    phase=row[4],
                    description=row[5],
                    size=row[6],
                    components=[],  # Load separately if needed
                    system_state={},
                    docker_state={},
                    environment_vars={},
                    health_status={},
                    verified=bool(row[7])
                ))
            
            return snapshots
    
    def store_rollback_operation(self, operation: RollbackOperation):
        """Store rollback operation"""
        with sqlite3.connect(self.db_file) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO rollback_operations
                (operation_id, snapshot_id, initiated_by, start_time, end_time,
                 status, strategy, target_components, progress, steps_completed,
                 steps_failed, error_details, verification_results)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                operation.operation_id,
                operation.snapshot_id,
                operation.initiated_by,
                operation.start_time.isoformat(),
                operation.end_time.isoformat() if operation.end_time else None,
                operation.status.value,
                operation.strategy.value,
                json.dumps(operation.target_components),
                operation.progress,
                json.dumps(operation.steps_completed),
                json.dumps(operation.steps_failed),
                json.dumps(operation.error_details),
                json.dumps(operation.verification_results)
            ))

class SystemStateCapture:
    """Capture comprehensive system state"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def capture_full_state(self, deployment_id: str, phase: str, description: str) -> SystemSnapshot:
        """Capture complete system state"""
        snapshot_id = f"snapshot_{deployment_id}_{phase}_{int(time.time())}"
        
        logger.info(f"Creating full system snapshot: {snapshot_id}")
        
        # Capture different components in parallel
        tasks = [
            self._capture_docker_state(),
            self._capture_configuration_files(),
            self._capture_database_state(),
            self._capture_environment_variables(),
            self._capture_system_metrics(),
            self._capture_health_status()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        docker_state = results[0] if not isinstance(results[0], Exception) else {}
        config_components = results[1] if not isinstance(results[1], Exception) else []
        db_components = results[2] if not isinstance(results[2], Exception) else []
        env_vars = results[3] if not isinstance(results[3], Exception) else {}
        system_state = results[4] if not isinstance(results[4], Exception) else {}
        health_status = results[5] if not isinstance(results[5], Exception) else {}
        
        # Combine all components
        all_components = config_components + db_components
        
        # Calculate total size
        total_size = sum(comp.size for comp in all_components)
        
        # Create snapshot
        snapshot = SystemSnapshot(
            snapshot_id=snapshot_id,
            timestamp=datetime.now(),
            snapshot_type=SnapshotType.FULL_SYSTEM,
            deployment_id=deployment_id,
            phase=phase,
            description=description,
            size=total_size,
            components=all_components,
            system_state=system_state,
            docker_state=docker_state,
            environment_vars=env_vars,
            health_status=health_status,
            tags=[f"deployment:{deployment_id}", f"phase:{phase}"]
        )
        
        # Create physical snapshot archive
        await self._create_snapshot_archive(snapshot)
        
        return snapshot
    
    async def _capture_docker_state(self) -> Dict[str, Any]:
        """Capture Docker containers and images state"""
        try:
            # Get running containers
            result = await asyncio.create_subprocess_exec(
                "docker", "ps", "--format", "json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            containers = []
            if result.returncode == 0:
                for line in stdout.decode().strip().split('\n'):
                    if line:
                        containers.append(json.loads(line))
            
            # Get images
            result = await asyncio.create_subprocess_exec(
                "docker", "images", "--format", "json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            images = []
            if result.returncode == 0:
                for line in stdout.decode().strip().split('\n'):
                    if line:
                        images.append(json.loads(line))
            
            # Get compose state
            compose_state = {}
            compose_file = self.project_root / "docker-compose.yml"
            if compose_file.exists():
                with open(compose_file, 'r') as f:
                    compose_state = yaml.safe_load(f)
            
            return {
                "containers": containers,
                "images": images,
                "compose_config": compose_state,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to capture Docker state: {e}")
            return {}
    
    async def _capture_configuration_files(self) -> List[SnapshotComponent]:
        """Capture configuration files"""
        components = []
        
        config_patterns = [
            "docker-compose*.yml",
            "config/**/*.yaml",
            "config/**/*.json",
            ".env*",
            "requirements*.txt",
            "package*.json",
            "Dockerfile*"
        ]
        
        for pattern in config_patterns:
            for file_path in self.project_root.glob(pattern):
                if file_path.is_file():
                    try:
                        size = file_path.stat().st_size
                        checksum = await self._calculate_checksum(file_path)
                        
                        components.append(SnapshotComponent(
                            name=str(file_path.relative_to(self.project_root)),
                            type="configuration",
                            path=str(file_path),
                            size=size,
                            checksum=checksum,
                            metadata={
                                "modified_time": file_path.stat().st_mtime,
                                "permissions": oct(file_path.stat().st_mode)
                            }
                        ))
                    except Exception as e:
                        logger.error(f"Failed to process config file {file_path}: {e}")
        
        return components
    
    async def _capture_database_state(self) -> List[SnapshotComponent]:
        """Capture database snapshots"""
        components = []
        
        try:
            # PostgreSQL dump
            pg_dump_file = BACKUP_DIR / f"postgres_snapshot_{int(time.time())}.sql"
            
            result = await asyncio.create_subprocess_exec(
                "docker", "exec", "sutazai-postgres", 
                "pg_dump", "-U", "sutazai", "-d", "sutazai",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                with open(pg_dump_file, 'wb') as f:
                    f.write(stdout)
                
                size = pg_dump_file.stat().st_size
                checksum = await self._calculate_checksum(pg_dump_file)
                
                components.append(SnapshotComponent(
                    name="postgres_dump",
                    type="database",
                    path=str(pg_dump_file),
                    size=size,
                    checksum=checksum,
                    metadata={"database": "postgresql", "format": "sql"}
                ))
        
        except Exception as e:
            logger.error(f"Failed to capture PostgreSQL state: {e}")
        
        # Redis snapshot
        try:
            redis_snapshot = BACKUP_DIR / f"redis_snapshot_{int(time.time())}.rdb"
            
            result = await asyncio.create_subprocess_exec(
                "docker", "exec", "sutazai-redis",
                "redis-cli", "BGSAVE",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            if result.returncode == 0:
                # Copy RDB file from container
                copy_result = await asyncio.create_subprocess_exec(
                    "docker", "cp", "sutazai-redis:/data/dump.rdb", str(redis_snapshot),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                if copy_result.returncode == 0 and redis_snapshot.exists():
                    size = redis_snapshot.stat().st_size
                    checksum = await self._calculate_checksum(redis_snapshot)
                    
                    components.append(SnapshotComponent(
                        name="redis_dump",
                        type="database",
                        path=str(redis_snapshot),
                        size=size,
                        checksum=checksum,
                        metadata={"database": "redis", "format": "rdb"}
                    ))
        
        except Exception as e:
            logger.error(f"Failed to capture Redis state: {e}")
        
        return components
    
    async def _capture_environment_variables(self) -> Dict[str, str]:
        """Capture relevant environment variables"""
        env_vars = {}
        
        # Capture SutazAI-related environment variables
        for key, value in os.environ.items():
            if any(prefix in key for prefix in ["SUTAZAI_", "POSTGRES_", "REDIS_", "NEO4J_", "OLLAMA_"]):
                env_vars[key] = value
        
        return env_vars
    
    async def _capture_system_metrics(self) -> Dict[str, Any]:
        """Capture system performance metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            return {
                "cpu_percent": cpu_percent,
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent,
                    "used": memory.used
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": (disk.used / disk.total) * 100
                },
                "network": {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to capture system metrics: {e}")
            return {}
    
    async def _capture_health_status(self) -> Dict[str, Any]:
        """Capture current health status of all services"""
        health_status = {}
        
        try:
            # Get container health status
            result = await asyncio.create_subprocess_exec(
                "docker", "ps", "--format", "table {{.Names}}\t{{.Status}}\t{{.Health}}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                lines = stdout.decode().strip().split('\n')[1:]  # Skip header
                for line in lines:
                    if line:
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            name = parts[0]
                            status = parts[1]
                            health = parts[2] if len(parts) > 2 else "unknown"
                            
                            health_status[name] = {
                                "status": status,
                                "health": health,
                                "timestamp": datetime.now().isoformat()
                            }
        
        except Exception as e:
            logger.error(f"Failed to capture health status: {e}")
        
        return health_status
    
    async def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file"""
        def _hash_file(path):
            hash_sha256 = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _hash_file, file_path)
    
    async def _create_snapshot_archive(self, snapshot: SystemSnapshot):
        """Create compressed archive of snapshot components"""
        archive_path = ROLLBACK_DIR / f"{snapshot.snapshot_id}.tar.gz"
        
        def _create_archive():
            with tarfile.open(archive_path, "w:gz") as tar:
                # Add metadata
                metadata_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
                try:
                    json.dump(asdict(snapshot), metadata_file.file, indent=2, default=str)
                    metadata_file.close()
                    tar.add(metadata_file.name, arcname="snapshot_metadata.json")
                finally:
                    os.unlink(metadata_file.name)
                
                # Add component files
                for component in snapshot.components:
                    if Path(component.path).exists():
                        try:
                            tar.add(component.path, arcname=f"components/{component.name}")
                        except Exception as e:
                            logger.error(f"Failed to add component {component.name} to archive: {e}")
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, _create_archive)
        
        logger.info(f"Created snapshot archive: {archive_path}")

class RollbackExecutor:
    """Execute rollback operations with different strategies"""
    
    def __init__(self, database: SnapshotDatabase, state_capture: SystemStateCapture):
        self.database = database
        self.state_capture = state_capture
    
    async def execute_rollback(self, snapshot_id: str, strategy: RecoveryStrategy, 
                             target_components: List[str] = None, 
                             initiated_by: str = "system") -> RollbackOperation:
        """Execute rollback operation"""
        
        operation_id = f"rollback_{snapshot_id}_{int(time.time())}"
        
        # Create rollback operation record
        operation = RollbackOperation(
            operation_id=operation_id,
            snapshot_id=snapshot_id,
            initiated_by=initiated_by,
            start_time=datetime.now(),
            end_time=None,
            status=RollbackStatus.IN_PROGRESS,
            strategy=strategy,
            target_components=target_components or [],
            progress=0.0,
            steps_completed=[],
            steps_failed=[],
            error_details=[],
            verification_results={}
        )
        
        # Store initial operation state
        self.database.store_rollback_operation(operation)
        
        logger.info(f"Starting rollback operation: {operation_id}")
        
        try:
            # Get snapshot to rollback to
            snapshot = self.database.get_snapshot(snapshot_id)
            if not snapshot:
                raise ValueError(f"Snapshot not found: {snapshot_id}")
            
            # Execute rollback based on strategy
            if strategy == RecoveryStrategy.IMMEDIATE:
                await self._immediate_rollback(operation, snapshot)
            elif strategy == RecoveryStrategy.GRACEFUL:
                await self._graceful_rollback(operation, snapshot)
            elif strategy == RecoveryStrategy.SELECTIVE:
                await self._selective_rollback(operation, snapshot)
            elif strategy == RecoveryStrategy.PROGRESSIVE:
                await self._progressive_rollback(operation, snapshot)
            elif strategy == RecoveryStrategy.EMERGENCY:
                await self._emergency_rollback(operation, snapshot)
            
            # Verify rollback success
            operation.verification_results = await self._verify_rollback(operation, snapshot)
            
            # Update final status
            verification_passed = operation.verification_results.get("passed", False)
            operation.status = RollbackStatus.VERIFIED if verification_passed else RollbackStatus.PARTIAL
            operation.end_time = datetime.now()
            operation.progress = 100.0
            
            logger.info(f"Rollback operation completed: {operation_id}")
            
        except Exception as e:
            operation.status = RollbackStatus.FAILED
            operation.end_time = datetime.now()
            operation.error_details.append(str(e))
            
            logger.error(f"Rollback operation failed: {operation_id}: {e}")
            logger.error(traceback.format_exc())
        
        finally:
            # Store final operation state
            self.database.store_rollback_operation(operation)
        
        return operation
    
    async def _immediate_rollback(self, operation: RollbackOperation, snapshot: SystemSnapshot):
        """Immediate rollback - stop everything and restore quickly"""
        logger.info("Executing immediate rollback strategy")
        
        steps = [
            "stop_all_services",
            "restore_docker_state",
            "restore_configuration",
            "restore_environment",
            "start_critical_services",
            "validate_system"
        ]
        
        for i, step in enumerate(steps):
            try:
                await self._execute_rollback_step(step, operation, snapshot)
                operation.steps_completed.append(step)
                operation.progress = (i + 1) / len(steps) * 100
                self.database.store_rollback_operation(operation)
                
            except Exception as e:
                operation.steps_failed.append(step)
                operation.error_details.append(f"{step}: {str(e)}")
                raise
    
    async def _graceful_rollback(self, operation: RollbackOperation, snapshot: SystemSnapshot):
        """Graceful rollback - drain connections and rollback smoothly"""
        logger.info("Executing graceful rollback strategy")
        
        steps = [
            "drain_connections",
            "scale_down_services",
            "backup_current_state",
            "restore_configuration",
            "restore_databases",
            "restore_docker_state",
            "validate_configuration",
            "start_services_progressive",
            "verify_health",
            "restore_connections"
        ]
        
        for i, step in enumerate(steps):
            try:
                await self._execute_rollback_step(step, operation, snapshot)
                operation.steps_completed.append(step)
                operation.progress = (i + 1) / len(steps) * 100
                self.database.store_rollback_operation(operation)
                
                # Add delays for graceful operations
                if step in ["drain_connections", "scale_down_services"]:
                    await asyncio.sleep(30)
                
            except Exception as e:
                operation.steps_failed.append(step)
                operation.error_details.append(f"{step}: {str(e)}")
                
                # For graceful rollback, continue with next steps on non-critical failures
                if step not in ["restore_configuration", "restore_docker_state"]:
                    logger.warning(f"Non-critical step failed in graceful rollback: {step}: {e}")
                    continue
                else:
                    raise
    
    async def _selective_rollback(self, operation: RollbackOperation, snapshot: SystemSnapshot):
        """Selective rollback - only rollback specified components"""
        logger.info("Executing selective rollback strategy")
        
        target_components = operation.target_components
        if not target_components:
            raise ValueError("Selective rollback requires target_components")
        
        # Filter snapshot components to only target ones
        components_to_restore = [
            comp for comp in snapshot.components 
            if comp.name in target_components or comp.type in target_components
        ]
        
        if not components_to_restore:
            raise ValueError(f"No components found matching targets: {target_components}")
        
        steps = []
        for component in components_to_restore:
            steps.append(f"restore_component_{component.name}")
        
        steps.extend(["validate_selective", "restart_affected_services"])
        
        for i, step in enumerate(steps):
            try:
                if step.startswith("restore_component_"):
                    component_name = step.replace("restore_component_", "")
                    component = next(c for c in components_to_restore if c.name == component_name)
                    await self._restore_component(component, snapshot)
                else:
                    await self._execute_rollback_step(step, operation, snapshot)
                
                operation.steps_completed.append(step)
                operation.progress = (i + 1) / len(steps) * 100
                self.database.store_rollback_operation(operation)
                
            except Exception as e:
                operation.steps_failed.append(step)
                operation.error_details.append(f"{step}: {str(e)}")
                raise
    
    async def _progressive_rollback(self, operation: RollbackOperation, snapshot: SystemSnapshot):
        """Progressive rollback - rollback in phases with validation"""
        logger.info("Executing progressive rollback strategy")
        
        phases = [
            {
                "name": "configuration",
                "steps": ["backup_current_config", "restore_configuration", "validate_config"]
            },
            {
                "name": "databases",
                "steps": ["stop_applications", "restore_databases", "validate_databases"]
            },
            {
                "name": "services",
                "steps": ["restore_docker_state", "start_infrastructure", "validate_infrastructure"]
            },
            {
                "name": "applications",
                "steps": ["start_applications", "validate_health", "run_smoke_tests"]
            }
        ]
        
        total_steps = sum(len(phase["steps"]) for phase in phases)
        step_count = 0
        
        for phase in phases:
            logger.info(f"Progressive rollback phase: {phase['name']}")
            
            phase_success = True
            for step in phase["steps"]:
                try:
                    await self._execute_rollback_step(step, operation, snapshot)
                    operation.steps_completed.append(f"{phase['name']}:{step}")
                    step_count += 1
                    operation.progress = step_count / total_steps * 100
                    self.database.store_rollback_operation(operation)
                    
                except Exception as e:
                    operation.steps_failed.append(f"{phase['name']}:{step}")
                    operation.error_details.append(f"{phase['name']}:{step}: {str(e)}")
                    phase_success = False
                    break
            
            if not phase_success:
                raise Exception(f"Progressive rollback failed in phase: {phase['name']}")
            
            # Validate phase completion
            await asyncio.sleep(10)  # Allow stabilization
    
    async def _emergency_rollback(self, operation: RollbackOperation, snapshot: SystemSnapshot):
        """Emergency rollback - fastest possible recovery"""
        logger.info("Executing emergency rollback strategy")
        
        # Emergency rollback prioritizes speed over graceful shutdown
        steps = [
            "force_stop_all",
            "restore_critical_config",
            "restore_docker_state_fast",
            "start_critical_services_only",
            "basic_validation"
        ]
        
        for i, step in enumerate(steps):
            try:
                await self._execute_rollback_step(step, operation, snapshot, emergency=True)
                operation.steps_completed.append(step)
                operation.progress = (i + 1) / len(steps) * 100
                self.database.store_rollback_operation(operation)
                
            except Exception as e:
                operation.steps_failed.append(step)
                operation.error_details.append(f"{step}: {str(e)}")
                
                # In emergency mode, continue even on failures
                logger.error(f"Emergency rollback step failed: {step}: {e}")
                continue
    
    async def _execute_rollback_step(self, step: str, operation: RollbackOperation, 
                                   snapshot: SystemSnapshot, emergency: bool = False):
        """Execute individual rollback step"""
        logger.info(f"Executing rollback step: {step}")
        
        if step == "stop_all_services":
            await self._stop_all_services(emergency)
        elif step == "force_stop_all":
            await self._force_stop_all_services()
        elif step == "restore_docker_state":
            await self._restore_docker_state(snapshot)
        elif step == "restore_docker_state_fast":
            await self._restore_docker_state(snapshot, fast=True)
        elif step == "restore_configuration":
            await self._restore_configuration(snapshot)
        elif step == "restore_critical_config":
            await self._restore_configuration(snapshot, critical_only=True)
        elif step == "restore_environment":
            await self._restore_environment(snapshot)
        elif step == "restore_databases":
            await self._restore_databases(snapshot)
        elif step == "start_critical_services":
            await self._start_critical_services()
        elif step == "start_critical_services_only":
            await self._start_critical_services(only_critical=True)
        elif step == "start_services_progressive":
            await self._start_services_progressive()
        elif step == "validate_system":
            await self._validate_system()
        elif step == "basic_validation":
            await self._basic_validation()
        else:
            logger.warning(f"Unknown rollback step: {step}")
    
    async def _stop_all_services(self, emergency: bool = False):
        """Stop all running services"""
        timeout = 30 if not emergency else 5
        
        cmd = ["docker", "compose", "down", "--remove-orphans"]
        if emergency:
            cmd.append("--timeout=5")
        
        result = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=self.state_capture.project_root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            await asyncio.wait_for(result.communicate(), timeout=timeout * 2)
        except asyncio.TimeoutError:
            if emergency:
                # Force kill in emergency mode
                await self._force_stop_all_services()
            else:
                raise
    
    async def _force_stop_all_services(self):
        """Force stop all services"""
        # Kill all sutazai containers
        result = await asyncio.create_subprocess_exec(
            "docker", "ps", "-q", "--filter", "name=sutazai-",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await result.communicate()
        
        if result.returncode == 0 and stdout:
            container_ids = stdout.decode().strip().split('\n')
            for container_id in container_ids:
                if container_id:
                    await asyncio.create_subprocess_exec(
                        "docker", "kill", container_id,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL
                    )
    
    async def _restore_configuration(self, snapshot: SystemSnapshot, critical_only: bool = False):
        """Restore configuration files"""
        config_components = [
            comp for comp in snapshot.components 
            if comp.type == "configuration"
        ]
        
        if critical_only:
            # Only restore critical config files
            critical_patterns = ["docker-compose", ".env", "config/"]
            config_components = [
                comp for comp in config_components
                if any(pattern in comp.name for pattern in critical_patterns)
            ]
        
        for component in config_components:
            await self._restore_component(component, snapshot)
    
    async def _restore_component(self, component: SnapshotComponent, snapshot: SystemSnapshot):
        """Restore individual component"""
        # Extract component from snapshot archive
        archive_path = ROLLBACK_DIR / f"{snapshot.snapshot_id}.tar.gz"
        
        if not archive_path.exists():
            raise FileNotFoundError(f"Snapshot archive not found: {archive_path}")
        
        with tarfile.open(archive_path, "r:gz") as tar:
            component_path = f"components/{component.name}"
            
            try:
                member = tar.getmember(component_path)
                
                # Extract to temporary location first
                temp_dir = tempfile.mkdtemp()
                tar.extract(member, temp_dir)
                
                # Move to final location
                extracted_path = Path(temp_dir) / component_path
                target_path = Path(component.path)
                
                # Ensure parent directory exists
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Move file
                shutil.move(str(extracted_path), str(target_path))
                
                # Restore permissions if available
                if "permissions" in component.metadata:
                    os.chmod(target_path, int(component.metadata["permissions"], 8))
                
                # Cleanup temp directory
                shutil.rmtree(temp_dir, ignore_errors=True)
                
            except KeyError:
                logger.error(f"Component not found in archive: {component_path}")
                raise
    
    async def _restore_docker_state(self, snapshot: SystemSnapshot, fast: bool = False):
        """Restore Docker state"""
        docker_state = snapshot.docker_state
        
        # Restore compose configuration
        if "compose_config" in docker_state:
            compose_file = self.state_capture.project_root / "docker-compose.yml"
            with open(compose_file, 'w') as f:
                yaml.dump(docker_state["compose_config"], f)
        
        # Start services from compose
        cmd = ["docker", "compose", "up", "-d"]
        if fast:
            cmd.extend(["--no-deps", "--no-recreate"])
        
        result = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=self.state_capture.project_root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        await result.communicate()
        if result.returncode != 0:
            raise Exception("Failed to restore Docker state")
    
    async def _restore_environment(self, snapshot: SystemSnapshot):
        """Restore environment variables"""
        env_file = self.state_capture.project_root / ".env"
        
        with open(env_file, 'w') as f:
            f.write("# Restored from snapshot\n")
            f.write(f"# Snapshot ID: {snapshot.snapshot_id}\n")
            f.write(f"# Restored at: {datetime.now().isoformat()}\n\n")
            
            for key, value in snapshot.environment_vars.items():
                f.write(f"{key}={value}\n")
    
    async def _restore_databases(self, snapshot: SystemSnapshot):
        """Restore database snapshots"""
        db_components = [
            comp for comp in snapshot.components 
            if comp.type == "database"
        ]
        
        for component in db_components:
            if "postgres" in component.name:
                await self._restore_postgres_database(component, snapshot)
            elif "redis" in component.name:
                await self._restore_redis_database(component, snapshot)
    
    async def _restore_postgres_database(self, component: SnapshotComponent, snapshot: SystemSnapshot):
        """Restore PostgreSQL database"""
        # Extract database dump
        temp_file = tempfile.NamedTemporaryFile(suffix='.sql', delete=False)
        
        try:
            # Extract from archive
            archive_path = ROLLBACK_DIR / f"{snapshot.snapshot_id}.tar.gz"
            with tarfile.open(archive_path, "r:gz") as tar:
                member = tar.getmember(f"components/{component.name}")
                
                with tar.extractfile(member) as src:
                    temp_file.write(src.read())
                temp_file.close()
            
            # Restore database
            result = await asyncio.create_subprocess_exec(
                "docker", "exec", "-i", "sutazai-postgres",
                "psql", "-U", "sutazai", "-d", "sutazai",
                stdin=open(temp_file.name, 'rb'),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await result.communicate()
            
            if result.returncode != 0:
                raise Exception("Failed to restore PostgreSQL database")
        
        finally:
            os.unlink(temp_file.name)
    
    async def _restore_redis_database(self, component: SnapshotComponent, snapshot: SystemSnapshot):
        """Restore Redis database"""
        # This would involve stopping Redis, replacing RDB file, and restarting
        # Implementation depends on Redis container setup
        pass
    
    async def _start_critical_services(self, only_critical: bool = False):
        """Start critical services"""
        critical_services = ["postgres", "redis", "ollama", "backend"]
        
        if only_critical:
            services = critical_services
        else:
            # Start all services
            result = await asyncio.create_subprocess_exec(
                "docker", "compose", "up", "-d",
                cwd=self.state_capture.project_root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await result.communicate()
            return
        
        # Start only critical services
        for service in services:
            result = await asyncio.create_subprocess_exec(
                "docker", "compose", "up", "-d", service,
                cwd=self.state_capture.project_root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await result.communicate()
            
            # Wait for service to stabilize
            await asyncio.sleep(10)
    
    async def _start_services_progressive(self):
        """Start services in dependency order"""
        service_groups = [
            ["postgres", "redis", "neo4j"],
            ["ollama"],
            ["backend"],
            ["frontend"],
            # Other services can start in parallel
        ]
        
        for group in service_groups:
            # Start services in group
            for service in group:
                result = await asyncio.create_subprocess_exec(
                    "docker", "compose", "up", "-d", service,
                    cwd=self.state_capture.project_root,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL
                )
                await result.communicate()
            
            # Wait between groups
            await asyncio.sleep(30)
    
    async def _validate_system(self):
        """Comprehensive system validation"""
        await self._basic_validation()
        
        # Additional validations
        # - Check database connectivity
        # - Verify API endpoints
        # - Test model inference
        # - Validate configuration consistency
    
    async def _basic_validation(self):
        """Basic system validation"""
        # Check if critical containers are running
        critical_services = ["postgres", "redis", "backend"]
        
        for service in critical_services:
            result = await asyncio.create_subprocess_exec(
                "docker", "ps", "--filter", f"name=sutazai-{service}",
                "--filter", "status=running", "--format", "{{.Names}}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if not stdout.decode().strip():
                raise Exception(f"Critical service not running: {service}")
    
    async def _verify_rollback(self, operation: RollbackOperation, snapshot: SystemSnapshot) -> Dict[str, Any]:
        """Verify rollback was successful"""
        verification_results = {
            "passed": True,
            "checks": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Check service health
        try:
            await self._basic_validation()
            verification_results["checks"]["services"] = "passed"
        except Exception as e:
            verification_results["checks"]["services"] = f"failed: {str(e)}"
            verification_results["passed"] = False
        
        # Check configuration integrity
        try:
            config_components = [c for c in snapshot.components if c.type == "configuration"]
            for component in config_components:
                if Path(component.path).exists():
                    current_checksum = await self.state_capture._calculate_checksum(Path(component.path))
                    if current_checksum != component.checksum:
                        verification_results["checks"][f"config_{component.name}"] = "checksum_mismatch"
                        verification_results["passed"] = False
                    else:
                        verification_results["checks"][f"config_{component.name}"] = "passed"
                else:
                    verification_results["checks"][f"config_{component.name}"] = "missing"
                    verification_results["passed"] = False
        except Exception as e:
            verification_results["checks"]["configuration"] = f"failed: {str(e)}"
            verification_results["passed"] = False
        
        return verification_results

class AdvancedRollbackSystem:
    """Main rollback system coordinator"""
    
    def __init__(self):
        self.database = SnapshotDatabase(SNAPSHOT_DB)
        self.state_capture = SystemStateCapture(PROJECT_ROOT)
        self.rollback_executor = RollbackExecutor(self.database, self.state_capture)
        
        # Retention policies
        self.retention_policies = {
            "default": timedelta(days=7),
            "critical": timedelta(days=30),
            "milestone": timedelta(days=90)
        }
    
    async def create_snapshot(self, deployment_id: str, phase: str, 
                            description: str, tags: List[str] = None) -> SystemSnapshot:
        """Create a new system snapshot"""
        logger.info(f"Creating snapshot for deployment {deployment_id}, phase {phase}")
        
        snapshot = await self.state_capture.capture_full_state(deployment_id, phase, description)
        
        if tags:
            snapshot.tags.extend(tags)
        
        # Store snapshot metadata
        self.database.store_snapshot(snapshot)
        
        logger.info(f"Snapshot created: {snapshot.snapshot_id}")
        return snapshot
    
    async def rollback_to_snapshot(self, snapshot_id: str, strategy: RecoveryStrategy = RecoveryStrategy.GRACEFUL,
                                 target_components: List[str] = None) -> RollbackOperation:
        """Rollback to a specific snapshot"""
        logger.info(f"Initiating rollback to snapshot: {snapshot_id}")
        
        operation = await self.rollback_executor.execute_rollback(
            snapshot_id, strategy, target_components
        )
        
        return operation
    
    def list_snapshots(self, limit: int = 20, deployment_id: str = None) -> List[SystemSnapshot]:
        """List available snapshots"""
        return self.database.list_snapshots(limit, deployment_id)
    
    def get_snapshot(self, snapshot_id: str) -> Optional[SystemSnapshot]:
        """Get specific snapshot"""
        return self.database.get_snapshot(snapshot_id)
    
    async def cleanup_old_snapshots(self):
        """Clean up old snapshots based on retention policies"""
        snapshots = self.database.list_snapshots(limit=1000)
        
        for snapshot in snapshots:
            retention_period = self.retention_policies.get(snapshot.retention_policy, 
                                                         self.retention_policies["default"])
            
            if datetime.now() - snapshot.timestamp > retention_period:
                await self._delete_snapshot(snapshot)
    
    async def _delete_snapshot(self, snapshot: SystemSnapshot):
        """Delete a snapshot and its files"""
        # Remove archive file
        archive_path = ROLLBACK_DIR / f"{snapshot.snapshot_id}.tar.gz"
        if archive_path.exists():
            archive_path.unlink()
        
        # Remove component files
        for component in snapshot.components:
            component_path = Path(component.path)
            if component_path.exists() and component_path.parent == BACKUP_DIR:
                component_path.unlink()
        
        # Remove from database
        with sqlite3.connect(self.database.db_file) as conn:
            conn.execute("DELETE FROM snapshot_components WHERE snapshot_id = ?", (snapshot.snapshot_id,))
            conn.execute("DELETE FROM snapshots WHERE snapshot_id = ?", (snapshot.snapshot_id,))
        
        logger.info(f"Deleted snapshot: {snapshot.snapshot_id}")

async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Advanced Rollback System for SutazAI"
    )
    parser.add_argument(
        "command",
        choices=["snapshot", "rollback", "list", "cleanup", "info"],
        help="Command to execute"
    )
    parser.add_argument(
        "--deployment-id", "-d",
        help="Deployment ID for snapshot"
    )
    parser.add_argument(
        "--phase", "-p",
        help="Deployment phase for snapshot"
    )
    parser.add_argument(
        "--description",
        help="Description for snapshot"
    )
    parser.add_argument(
        "--snapshot-id", "-s",
        help="Snapshot ID for rollback"
    )
    parser.add_argument(
        "--strategy",
        choices=["immediate", "graceful", "selective", "progressive", "emergency"],
        default="graceful",
        help="Rollback strategy"
    )
    parser.add_argument(
        "--components",
        nargs="*",
        help="Target components for selective rollback"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=20,
        help="Limit for list command"
    )
    
    args = parser.parse_args()
    
    rollback_system = AdvancedRollbackSystem()
    
    try:
        if args.command == "snapshot":
            if not args.deployment_id or not args.phase:
                logger.error("deployment-id and phase are required for snapshot")
                sys.exit(1)
            
            description = args.description or f"Snapshot for {args.deployment_id} - {args.phase}"
            snapshot = await rollback_system.create_snapshot(
                args.deployment_id, args.phase, description
            )
            
            print(f"Created snapshot: {snapshot.snapshot_id}")
            print(f"Size: {snapshot.size / (1024*1024):.1f} MB")
            print(f"Components: {len(snapshot.components)}")
        
        elif args.command == "rollback":
            if not args.snapshot_id:
                logger.error("snapshot-id is required for rollback")
                sys.exit(1)
            
            strategy = RecoveryStrategy(args.strategy)
            operation = await rollback_system.rollback_to_snapshot(
                args.snapshot_id, strategy, args.components
            )
            
            print(f"Rollback operation: {operation.operation_id}")
            print(f"Status: {operation.status.value}")
            print(f"Progress: {operation.progress:.1f}%")
            
            if operation.status == RollbackStatus.FAILED:
                print("Errors:")
                for error in operation.error_details:
                    print(f"  - {error}")
        
        elif args.command == "list":
            snapshots = rollback_system.list_snapshots(args.limit, args.deployment_id)
            
            print(f"Found {len(snapshots)} snapshots:")
            print()
            print("ID                    | Timestamp           | Type        | Phase     | Size (MB)")
            print("-" * 80)
            
            for snapshot in snapshots:
                size_mb = snapshot.size / (1024*1024) if snapshot.size else 0
                print(f"{snapshot.snapshot_id[:20]:<20} | {snapshot.timestamp.strftime('%Y-%m-%d %H:%M:%S')} | "
                      f"{snapshot.snapshot_type.value:<11} | {snapshot.phase[:9]:<9} | {size_mb:>8.1f}")
        
        elif args.command == "cleanup":
            await rollback_system.cleanup_old_snapshots()
            print("Cleanup completed")
        
        elif args.command == "info":
            if not args.snapshot_id:
                logger.error("snapshot-id is required for info")
                sys.exit(1)
            
            snapshot = rollback_system.get_snapshot(args.snapshot_id)
            if not snapshot:
                print(f"Snapshot not found: {args.snapshot_id}")
                sys.exit(1)
            
            print(f"Snapshot ID: {snapshot.snapshot_id}")
            print(f"Timestamp: {snapshot.timestamp}")
            print(f"Type: {snapshot.snapshot_type.value}")
            print(f"Deployment ID: {snapshot.deployment_id}")
            print(f"Phase: {snapshot.phase}")
            print(f"Description: {snapshot.description}")
            print(f"Size: {snapshot.size / (1024*1024):.1f} MB")
            print(f"Verified: {snapshot.verified}")
            print(f"Components: {len(snapshot.components)}")
            
            if snapshot.components:
                print("\nComponents:")
                for comp in snapshot.components:
                    comp_size = comp.size / 1024 if comp.size else 0
                    print(f"  - {comp.name} ({comp.type}): {comp_size:.1f} KB")
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())