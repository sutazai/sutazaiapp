#!/usr/bin/env python3
"""
MCP Safety Validator

Comprehensive safety validation for MCP cleanup operations. Provides multi-layered
safety checks to ensure cleanup operations never damage critical infrastructure
or break running services.

Author: Claude AI Assistant (garbage-collector.md)
Created: 2025-08-15 21:30:00 UTC
Version: 1.0.0
"""

import asyncio
import json
import psutil
import subprocess
import hashlib
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
import socket
import fnmatch

# Import parent automation components
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import get_config, MCPAutomationConfig


class SafetyLevel(Enum):
    """Safety levels for cleanup operations."""
    UNSAFE = "unsafe"      # Operation should not proceed
    LOW = "low"           # High risk, require explicit confirmation
    MEDIUM = "medium"     # Medium risk, proceed with caution
    HIGH = "high"         # Low risk, safe to proceed
    MAXIMUM = "maximum"   # No risk, completely safe


class SafetyCheckType(Enum):
    """Types of safety checks performed."""
    CRITICAL_FILE_CHECK = "critical_file_check"
    PROCESS_CHECK = "process_check"
    DEPENDENCY_CHECK = "dependency_check"
    DISK_SPACE_CHECK = "disk_space_check"
    NETWORK_CHECK = "network_check"
    CONFIGURATION_CHECK = "configuration_check"
    BACKUP_CHECK = "backup_check"
    SERVICE_HEALTH_CHECK = "service_health_check"
    ROLLBACK_CHECK = "rollback_check"


@dataclass
class SafetyViolation:
    """Details of a safety violation found during validation."""
    violation_type: SafetyCheckType
    severity: str  # 'critical', 'warning', 'info'
    resource: str
    message: str
    recommendation: str
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['violation_type'] = self.violation_type.value
        return result


@dataclass
class SafetyResult:
    """Results from safety validation."""
    safety_level: SafetyLevel
    violations: List[SafetyViolation]
    warnings: List[str]
    checks_performed: List[SafetyCheckType]
    validation_timestamp: datetime
    validation_duration_seconds: float
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['safety_level'] = self.safety_level.value
        result['violations'] = [v.to_dict() for v in self.violations]
        result['checks_performed'] = [c.value for c in self.checks_performed]
        result['validation_timestamp'] = self.validation_timestamp.isoformat()
        return result


class SafetyValidator:
    """
    Comprehensive safety validator for MCP cleanup operations.
    
    Provides multi-layered safety checks to ensure cleanup operations
    are safe and will not damage critical infrastructure.
    """
    
    def __init__(self, config: MCPAutomationConfig):
        """Initialize safety validator."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Critical file patterns that should never be deleted
        self.critical_file_patterns = [
            # MCP configuration files
            '*.mcp.json', '.mcp.json', 'mcp.json',
            
            # Application configuration
            'package.json', 'requirements.txt', 'Pipfile',
            'docker-compose*.yml', 'Dockerfile*',
            'config.json', 'config.yaml', 'config.yml',
            
            # Source code files
            '*.py', '*.js', '*.ts', '*.sh', '*.bash',
            '*.go', '*.rs', '*.java', '*.cpp', '*.c',
            
            # Documentation and licenses
            'README*', 'LICENSE*', 'CHANGELOG*',
            'CONTRIBUTING*', 'SECURITY*',
            
            # Version control
            '.git*', '.gitignore', '.gitmodules',
            
            # Build and deployment
            'Makefile', 'CMakeLists.txt', 'build.xml',
            '*.lock', 'yarn.lock', 'poetry.lock',
            
            # Database and data files
            '*.db', '*.sqlite', '*.sql',
            '*.json', '*.yaml', '*.yml', '*.xml',
            
            # Certificates and keys
            '*.pem', '*.key', '*.cert', '*.crt',
            '*.p12', '*.pfx', '*.jks'
        ]
        
        # Critical directory patterns
        self.critical_directory_patterns = [
            '.git', '.mcp', 'node_modules',
            'src', 'lib', 'bin', 'config',
            '.venv', 'venv', '__pycache__'
        ]
        
        # System paths that should never be touched
        self.forbidden_paths = [
            Path('/'),
            Path('/bin'),
            Path('/sbin'),
            Path('/usr'),
            Path('/etc'),
            Path('/var/lib'),
            Path('/proc'),
            Path('/sys'),
            Path('/dev')
        ]
        
        # Minimum disk space thresholds (in bytes)
        self.min_disk_space_threshold = 1024 * 1024 * 1024  # 1GB
        self.min_disk_space_percentage = 10  # 10%
        
        self.logger.info("SafetyValidator initialized", extra={
            'critical_patterns': len(self.critical_file_patterns),
            'forbidden_paths': len(self.forbidden_paths)
        })
    
    async def validate_cleanup_safety(
        self,
        server_name: str,
        cleanup_analysis: Dict[str, Any],
        safety_level: SafetyLevel = SafetyLevel.HIGH
    ) -> SafetyResult:
        """
        Comprehensive safety validation for cleanup operations.
        
        Args:
            server_name: Name of the MCP server
            cleanup_analysis: Analysis results from cleanup services
            safety_level: Required safety level for validation
            
        Returns:
            SafetyResult with validation results and recommendations
        """
        validation_start = datetime.now(timezone.utc)
        
        violations = []
        warnings = []
        checks_performed = []
        
        try:
            self.logger.info(f"Starting safety validation for {server_name}", extra={
                'required_safety_level': safety_level.value
            })
            
            # Critical file safety check
            file_violations = await self._check_critical_files(server_name, cleanup_analysis)
            violations.extend(file_violations)
            checks_performed.append(SafetyCheckType.CRITICAL_FILE_CHECK)
            
            # Process safety check
            process_violations = await self._check_running_processes(server_name, cleanup_analysis)
            violations.extend(process_violations)
            checks_performed.append(SafetyCheckType.PROCESS_CHECK)
            
            # Dependency safety check
            dependency_violations = await self._check_dependencies(server_name, cleanup_analysis)
            violations.extend(dependency_violations)
            checks_performed.append(SafetyCheckType.DEPENDENCY_CHECK)
            
            # Disk space safety check
            disk_violations = await self._check_disk_space(server_name, cleanup_analysis)
            violations.extend(disk_violations)
            checks_performed.append(SafetyCheckType.DISK_SPACE_CHECK)
            
            # Network safety check
            network_violations = await self._check_network_dependencies(server_name)
            violations.extend(network_violations)
            checks_performed.append(SafetyCheckType.NETWORK_CHECK)
            
            # Configuration safety check
            config_violations = await self._check_configuration_integrity(server_name)
            violations.extend(config_violations)
            checks_performed.append(SafetyCheckType.CONFIGURATION_CHECK)
            
            # Backup availability check
            backup_violations = await self._check_backup_availability(server_name)
            violations.extend(backup_violations)
            checks_performed.append(SafetyCheckType.BACKUP_CHECK)
            
            # Service health check
            health_violations = await self._check_service_health(server_name)
            violations.extend(health_violations)
            checks_performed.append(SafetyCheckType.SERVICE_HEALTH_CHECK)
            
            # Rollback capability check
            rollback_violations = await self._check_rollback_capability(server_name)
            violations.extend(rollback_violations)
            checks_performed.append(SafetyCheckType.ROLLBACK_CHECK)
            
            # Determine overall safety level
            overall_safety = await self._calculate_overall_safety(violations, warnings)
            
            # Generate warnings from non-critical violations
            for violation in violations:
                if violation.severity == 'warning':
                    warnings.append(f"{violation.violation_type.value}: {violation.message}")
            
            validation_end = datetime.now(timezone.utc)
            validation_duration = (validation_end - validation_start).total_seconds()
            
            result = SafetyResult(
                safety_level=overall_safety,
                violations=violations,
                warnings=warnings,
                checks_performed=checks_performed,
                validation_timestamp=validation_start,
                validation_duration_seconds=validation_duration,
                summary={
                    'total_violations': len(violations),
                    'critical_violations': len([v for v in violations if v.severity == 'critical']),
                    'warning_violations': len([v for v in violations if v.severity == 'warning']),
                    'checks_passed': len(checks_performed),
                    'overall_safety_level': overall_safety.value
                }
            )
            
            self.logger.info(f"Safety validation completed for {server_name}", extra={
                'safety_level': overall_safety.value,
                'violations': len(violations),
                'warnings': len(warnings),
                'duration_seconds': validation_duration
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error during safety validation for {server_name}: {e}", exc_info=True)
            
            # Return unsafe result on validation error
            return SafetyResult(
                safety_level=SafetyLevel.UNSAFE,
                violations=[SafetyViolation(
                    violation_type=SafetyCheckType.CRITICAL_FILE_CHECK,
                    severity='critical',
                    resource=server_name,
                    message=f"Safety validation failed: {e}",
                    recommendation="Do not proceed with cleanup until validation issues are resolved"
                )],
                warnings=[f"Safety validation error: {e}"],
                checks_performed=checks_performed,
                validation_timestamp=validation_start,
                validation_duration_seconds=(datetime.now(timezone.utc) - validation_start).total_seconds()
            )
    
    async def validate_server_cleanup_safety(
        self,
        server_name: str,
        cleanup_types: List[str],
        safety_level: SafetyLevel
    ) -> SafetyResult:
        """Validate safety for server-specific cleanup operations."""
        
        # Create mock cleanup analysis for validation
        cleanup_analysis = {
            'server_name': server_name,
            'cleanup_types': cleanup_types,
            'total_candidates': 0,
            'estimated_space_freed': 0
        }
        
        return await self.validate_cleanup_safety(server_name, cleanup_analysis, safety_level)
    
    async def _check_critical_files(
        self,
        server_name: str,
        cleanup_analysis: Dict[str, Any]
    ) -> List[SafetyViolation]:
        """Check if any critical files would be affected by cleanup."""
        
        violations = []
        
        try:
            # Check cleanup candidates for critical files
            candidates = cleanup_analysis.get('candidates', {})
            
            # Check version cleanup candidates
            version_candidates = candidates.get('versions', {}).get('removable_versions', [])
            for version_data in version_candidates:
                if isinstance(version_data, dict):
                    version_path = version_data.get('version_info', {}).get('path', '')
                    if await self._is_critical_path(Path(version_path)):
                        violations.append(SafetyViolation(
                            violation_type=SafetyCheckType.CRITICAL_FILE_CHECK,
                            severity='critical',
                            resource=version_path,
                            message=f"Critical version path in cleanup candidates: {version_path}",
                            recommendation="Remove this path from cleanup candidates"
                        ))
            
            # Check artifact cleanup candidates
            artifact_candidates = candidates.get('artifacts', {})
            for artifact_type, artifacts in artifact_candidates.items():
                if isinstance(artifacts, list):
                    for artifact in artifacts:
                        if isinstance(artifact, dict):
                            artifact_path = artifact.get('artifact_info', {}).get('path', '')
                            if artifact_path and await self._is_critical_path(Path(artifact_path)):
                                violations.append(SafetyViolation(
                                    violation_type=SafetyCheckType.CRITICAL_FILE_CHECK,
                                    severity='critical',
                                    resource=artifact_path,
                                    message=f"Critical artifact path in cleanup candidates: {artifact_path}",
                                    recommendation="Remove this path from cleanup candidates"
                                ))
            
            # Check server-specific critical files
            server_critical_files = await self._get_server_critical_files(server_name)
            for critical_file in server_critical_files:
                if not critical_file.exists():
                    violations.append(SafetyViolation(
                        violation_type=SafetyCheckType.CRITICAL_FILE_CHECK,
                        severity='warning',
                        resource=str(critical_file),
                        message=f"Critical server file missing: {critical_file}",
                        recommendation="Verify server configuration is intact"
                    ))
        
        except Exception as e:
            violations.append(SafetyViolation(
                violation_type=SafetyCheckType.CRITICAL_FILE_CHECK,
                severity='critical',
                resource=server_name,
                message=f"Error checking critical files: {e}",
                recommendation="Investigate file system access issues"
            ))
        
        return violations
    
    async def _check_running_processes(
        self,
        server_name: str,
        cleanup_analysis: Dict[str, Any]
    ) -> List[SafetyViolation]:
        """Check for running processes that might be affected by cleanup."""
        
        violations = []
        
        try:
            # Check for MCP server processes
            mcp_processes = await self._find_mcp_processes(server_name)
            
            if mcp_processes:
                violations.append(SafetyViolation(
                    violation_type=SafetyCheckType.PROCESS_CHECK,
                    severity='warning',
                    resource=server_name,
                    message=f"MCP server {server_name} has {len(mcp_processes)} running processes",
                    recommendation="Consider stopping server before cleanup or proceed with caution",
                    details={'processes': [p.name() for p in mcp_processes]}
                ))
            
            # Check for processes using files in cleanup candidates
            candidates = cleanup_analysis.get('candidates', {})
            
            # Check file usage for artifact candidates
            artifact_candidates = candidates.get('artifacts', {})
            for artifact_type, artifacts in artifact_candidates.items():
                if isinstance(artifacts, list):
                    for artifact in artifacts:
                        if isinstance(artifact, dict):
                            artifact_path = artifact.get('artifact_info', {}).get('path', '')
                            if artifact_path:
                                using_processes = await self._find_processes_using_file(artifact_path)
                                if using_processes:
                                    violations.append(SafetyViolation(
                                        violation_type=SafetyCheckType.PROCESS_CHECK,
                                        severity='warning',
                                        resource=artifact_path,
                                        message=f"File {artifact_path} is in use by {len(using_processes)} processes",
                                        recommendation="Wait for processes to finish or exclude file from cleanup",
                                        details={'processes': [p.name() for p in using_processes]}
                                    ))
        
        except Exception as e:
            violations.append(SafetyViolation(
                violation_type=SafetyCheckType.PROCESS_CHECK,
                severity='warning',
                resource=server_name,
                message=f"Error checking running processes: {e}",
                recommendation="Manually verify no critical processes are running"
            ))
        
        return violations
    
    async def _check_dependencies(
        self,
        server_name: str,
        cleanup_analysis: Dict[str, Any]
    ) -> List[SafetyViolation]:
        """Check for dependency issues that cleanup might cause."""
        
        violations = []
        
        try:
            # Check for dependency files in cleanup candidates
            dependency_patterns = [
                'package.json', 'requirements.txt', 'Pipfile',
                'go.mod', 'Cargo.toml', 'pom.xml'
            ]
            
            candidates = cleanup_analysis.get('candidates', {})
            
            # Check artifact candidates for dependency files
            artifact_candidates = candidates.get('artifacts', {})
            for artifact_type, artifacts in artifact_candidates.items():
                if isinstance(artifacts, list):
                    for artifact in artifacts:
                        if isinstance(artifact, dict):
                            artifact_path = artifact.get('artifact_info', {}).get('path', '')
                            if artifact_path:
                                path = Path(artifact_path)
                                for pattern in dependency_patterns:
                                    if fnmatch.fnmatch(path.name, pattern):
                                        violations.append(SafetyViolation(
                                            violation_type=SafetyCheckType.DEPENDENCY_CHECK,
                                            severity='critical',
                                            resource=artifact_path,
                                            message=f"Dependency file in cleanup candidates: {path.name}",
                                            recommendation="Exclude dependency files from cleanup"
                                        ))
            
            # Check for broken symlinks that might be created
            server_paths = await self._get_server_paths(server_name)
            for path in server_paths:
                if path.exists():
                    broken_links = await self._find_broken_symlinks(path)
                    if broken_links:
                        violations.append(SafetyViolation(
                            violation_type=SafetyCheckType.DEPENDENCY_CHECK,
                            severity='warning',
                            resource=str(path),
                            message=f"Found {len(broken_links)} broken symlinks in {path}",
                            recommendation="Review symlinks before cleanup to prevent further breakage",
                            details={'broken_links': [str(link) for link in broken_links[:10]]}
                        ))
        
        except Exception as e:
            violations.append(SafetyViolation(
                violation_type=SafetyCheckType.DEPENDENCY_CHECK,
                severity='warning',
                resource=server_name,
                message=f"Error checking dependencies: {e}",
                recommendation="Manually verify no critical dependencies would be affected"
            ))
        
        return violations
    
    async def _check_disk_space(
        self,
        server_name: str,
        cleanup_analysis: Dict[str, Any]
    ) -> List[SafetyViolation]:
        """Check disk space constraints and cleanup impact."""
        
        violations = []
        
        try:
            # Check current disk space
            disk_usage = psutil.disk_usage('/')
            free_space = disk_usage.free
            total_space = disk_usage.total
            free_percentage = (free_space / total_space) * 100
            
            # Check if we have minimum free space
            if free_space < self.min_disk_space_threshold:
                violations.append(SafetyViolation(
                    violation_type=SafetyCheckType.DISK_SPACE_CHECK,
                    severity='warning',
                    resource='filesystem',
                    message=f"Low disk space: {free_space / (1024**3):.1f}GB free",
                    recommendation="Cleanup is needed but proceed carefully"
                ))
            
            if free_percentage < self.min_disk_space_percentage:
                violations.append(SafetyViolation(
                    violation_type=SafetyCheckType.DISK_SPACE_CHECK,
                    severity='warning',
                    resource='filesystem',
                    message=f"Low disk space: {free_percentage:.1f}% free",
                    recommendation="Cleanup is urgently needed"
                ))
            
            # Check cleanup impact
            estimated_freed = cleanup_analysis.get('estimated_space_freed', 0)
            if estimated_freed > free_space * 0.5:  # Cleanup would free > 50% of available space
                violations.append(SafetyViolation(
                    violation_type=SafetyCheckType.DISK_SPACE_CHECK,
                    severity='warning',
                    resource='filesystem',
                    message=f"Large cleanup operation: {estimated_freed / (1024**3):.1f}GB to be freed",
                    recommendation="Verify cleanup candidates carefully before proceeding"
                ))
        
        except Exception as e:
            violations.append(SafetyViolation(
                violation_type=SafetyCheckType.DISK_SPACE_CHECK,
                severity='warning',
                resource=server_name,
                message=f"Error checking disk space: {e}",
                recommendation="Manually verify sufficient disk space"
            ))
        
        return violations
    
    async def _check_network_dependencies(self, server_name: str) -> List[SafetyViolation]:
        """Check for network dependencies and connectivity."""
        
        violations = []
        
        try:
            # Check if server has network listeners
            network_connections = await self._get_server_network_connections(server_name)
            
            if network_connections:
                violations.append(SafetyViolation(
                    violation_type=SafetyCheckType.NETWORK_CHECK,
                    severity='info',
                    resource=server_name,
                    message=f"Server has {len(network_connections)} active network connections",
                    recommendation="Monitor for connection issues after cleanup",
                    details={'connections': network_connections[:5]}  # First 5 connections
                ))
            
            # Check for common MCP ports
            common_ports = [8000, 8080, 3000, 5000, 9000]
            listening_ports = []
            
            for port in common_ports:
                if await self._is_port_listening(port):
                    listening_ports.append(port)
            
            if listening_ports:
                violations.append(SafetyViolation(
                    violation_type=SafetyCheckType.NETWORK_CHECK,
                    severity='info',
                    resource=server_name,
                    message=f"Ports in use: {', '.join(map(str, listening_ports))}",
                    recommendation="Ensure cleanup does not affect running services",
                    details={'listening_ports': listening_ports}
                ))
        
        except Exception as e:
            violations.append(SafetyViolation(
                violation_type=SafetyCheckType.NETWORK_CHECK,
                severity='info',
                resource=server_name,
                message=f"Could not check network dependencies: {e}",
                recommendation="Manually verify network connectivity"
            ))
        
        return violations
    
    async def _check_configuration_integrity(self, server_name: str) -> List[SafetyViolation]:
        """Check MCP configuration integrity."""
        
        violations = []
        
        try:
            # Check for MCP configuration files
            mcp_config_paths = [
                self.config.paths.mcp_root / ".mcp.json",
                self.config.paths.automation_root / ".mcp.json",
                Path.home() / ".mcp.json"
            ]
            
            mcp_config_found = False
            for config_path in mcp_config_paths:
                if config_path.exists():
                    mcp_config_found = True
                    try:
                        with open(config_path, 'r') as f:
                            config_data = json.load(f)
                        
                        # Check if server is configured
                        mcps = config_data.get('mcps', {})
                        if server_name not in mcps:
                            violations.append(SafetyViolation(
                                violation_type=SafetyCheckType.CONFIGURATION_CHECK,
                                severity='warning',
                                resource=str(config_path),
                                message=f"Server {server_name} not found in MCP configuration",
                                recommendation="Verify server name or add to configuration"
                            ))
                    
                    except json.JSONDecodeError as e:
                        violations.append(SafetyViolation(
                            violation_type=SafetyCheckType.CONFIGURATION_CHECK,
                            severity='critical',
                            resource=str(config_path),
                            message=f"Invalid JSON in MCP configuration: {e}",
                            recommendation="Fix configuration file before proceeding"
                        ))
            
            if not mcp_config_found:
                violations.append(SafetyViolation(
                    violation_type=SafetyCheckType.CONFIGURATION_CHECK,
                    severity='warning',
                    resource=server_name,
                    message="No MCP configuration file found",
                    recommendation="Verify MCP setup is correct"
                ))
        
        except Exception as e:
            violations.append(SafetyViolation(
                violation_type=SafetyCheckType.CONFIGURATION_CHECK,
                severity='warning',
                resource=server_name,
                message=f"Error checking configuration: {e}",
                recommendation="Manually verify MCP configuration"
            ))
        
        return violations
    
    async def _check_backup_availability(self, server_name: str) -> List[SafetyViolation]:
        """Check backup availability and recency."""
        
        violations = []
        
        try:
            # Check for backup directories
            backup_paths = [
                self.config.paths.backup_root / server_name,
                self.config.paths.automation_root / "backups" / server_name
            ]
            
            recent_backup_found = False
            for backup_path in backup_paths:
                if backup_path.exists():
                    # Check for recent backups (within last 7 days)
                    cutoff_time = datetime.now() - timedelta(days=7)
                    
                    for backup_file in backup_path.iterdir():
                        if backup_file.is_file():
                            backup_time = datetime.fromtimestamp(backup_file.stat().st_mtime)
                            if backup_time > cutoff_time:
                                recent_backup_found = True
                                break
            
            if not recent_backup_found:
                violations.append(SafetyViolation(
                    violation_type=SafetyCheckType.BACKUP_CHECK,
                    severity='warning',
                    resource=server_name,
                    message="No recent backups found for server",
                    recommendation="Create backup before proceeding with cleanup"
                ))
        
        except Exception as e:
            violations.append(SafetyViolation(
                violation_type=SafetyCheckType.BACKUP_CHECK,
                severity='info',
                resource=server_name,
                message=f"Could not check backup availability: {e}",
                recommendation="Manually verify backup existence"
            ))
        
        return violations
    
    async def _check_service_health(self, server_name: str) -> List[SafetyViolation]:
        """Check service health and responsiveness."""
        
        violations = []
        
        try:
            # Check if server responds to health checks
            health_check_passed = await self._perform_health_check(server_name)
            
            if not health_check_passed:
                violations.append(SafetyViolation(
                    violation_type=SafetyCheckType.SERVICE_HEALTH_CHECK,
                    severity='warning',
                    resource=server_name,
                    message="Server health check failed",
                    recommendation="Investigate server issues before cleanup"
                ))
        
        except Exception as e:
            violations.append(SafetyViolation(
                violation_type=SafetyCheckType.SERVICE_HEALTH_CHECK,
                severity='info',
                resource=server_name,
                message=f"Could not perform health check: {e}",
                recommendation="Manually verify server health"
            ))
        
        return violations
    
    async def _check_rollback_capability(self, server_name: str) -> List[SafetyViolation]:
        """Check rollback and recovery capabilities."""
        
        violations = []
        
        try:
            # Check for version management capability
            version_tracking_available = await self._check_version_tracking(server_name)
            
            if not version_tracking_available:
                violations.append(SafetyViolation(
                    violation_type=SafetyCheckType.ROLLBACK_CHECK,
                    severity='warning',
                    resource=server_name,
                    message="Version tracking not available for server",
                    recommendation="Implement version tracking for safer cleanup operations"
                ))
            
            # Check for snapshot capability
            snapshot_available = await self._check_snapshot_capability(server_name)
            
            if not snapshot_available:
                violations.append(SafetyViolation(
                    violation_type=SafetyCheckType.ROLLBACK_CHECK,
                    severity='info',
                    resource=server_name,
                    message="Snapshot capability not available",
                    recommendation="Consider implementing snapshot capability for better rollback"
                ))
        
        except Exception as e:
            violations.append(SafetyViolation(
                violation_type=SafetyCheckType.ROLLBACK_CHECK,
                severity='info',
                resource=server_name,
                message=f"Could not check rollback capability: {e}",
                recommendation="Manually verify rollback procedures"
            ))
        
        return violations
    
    async def _calculate_overall_safety(
        self,
        violations: List[SafetyViolation],
        warnings: List[str]
    ) -> SafetyLevel:
        """Calculate overall safety level based on violations."""
        
        if not violations and not warnings:
            return SafetyLevel.MAXIMUM
        
        critical_count = len([v for v in violations if v.severity == 'critical'])
        warning_count = len([v for v in violations if v.severity == 'warning'])
        info_count = len([v for v in violations if v.severity == 'info'])
        
        if critical_count > 0:
            return SafetyLevel.UNSAFE
        elif warning_count > 3:
            return SafetyLevel.LOW
        elif warning_count > 1:
            return SafetyLevel.MEDIUM
        elif warning_count > 0 or info_count > 0:
            return SafetyLevel.HIGH
        else:
            return SafetyLevel.MAXIMUM
    
    # Helper methods
    
    async def _is_critical_path(self, path: Path) -> bool:
        """Check if a path is critical and should not be deleted."""
        
        # Check against forbidden system paths
        for forbidden in self.forbidden_paths:
            try:
                if path.is_relative_to(forbidden):
                    return True
            except ValueError:
                continue
        
        # Check against critical file patterns
        for pattern in self.critical_file_patterns:
            if fnmatch.fnmatch(path.name, pattern):
                return True
        
        # Check against critical directory patterns
        for pattern in self.critical_directory_patterns:
            if pattern in path.parts:
                return True
        
        return False
    
    async def _get_server_critical_files(self, server_name: str) -> List[Path]:
        """Get list of critical files for a specific server."""
        
        critical_files = []
        
        # Server wrapper script
        wrapper_script = self.config.paths.wrappers_root / f"{server_name}.sh"
        if wrapper_script.exists():
            critical_files.append(wrapper_script)
        
        # Server configuration files
        server_config_paths = [
            self.config.paths.mcp_root / server_name / "package.json",
            self.config.paths.automation_root / server_name / "config.json"
        ]
        
        for config_path in server_config_paths:
            if config_path.exists():
                critical_files.append(config_path)
        
        return critical_files
    
    async def _find_mcp_processes(self, server_name: str) -> List[psutil.Process]:
        """Find running processes related to an MCP server."""
        
        processes = []
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    if server_name.lower() in cmdline.lower():
                        processes.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception:
            pass
        
        return processes
    
    async def _find_processes_using_file(self, file_path: str) -> List[psutil.Process]:
        """Find processes using a specific file."""
        
        processes = []
        
        try:
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    if any(file_path in f.path for f in proc.open_files()):
                        processes.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception:
            pass
        
        return processes
    
    async def _get_server_paths(self, server_name: str) -> List[Path]:
        """Get paths associated with a server."""
        
        paths = []
        
        # Server-specific directories
        server_dirs = [
            self.config.paths.staging_root / server_name,
            self.config.paths.backup_root / server_name,
            self.config.paths.logs_root / server_name
        ]
        
        for path in server_dirs:
            if path.exists():
                paths.append(path)
        
        return paths
    
    async def _find_broken_symlinks(self, directory: Path) -> List[Path]:
        """Find broken symlinks in a directory."""
        
        broken_links = []
        
        try:
            for item in directory.rglob('*'):
                if item.is_symlink() and not item.exists():
                    broken_links.append(item)
        except Exception:
            pass
        
        return broken_links
    
    async def _get_server_network_connections(self, server_name: str) -> List[Dict[str, Any]]:
        """Get network connections for server processes."""
        
        connections = []
        
        try:
            # Find server processes
            server_processes = await self._find_mcp_processes(server_name)
            
            for proc in server_processes:
                try:
                    for conn in proc.connections():
                        connections.append({
                            'local_address': f"{conn.laddr.ip}:{conn.laddr.port}" if conn.laddr else None,
                            'remote_address': f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else None,
                            'status': conn.status,
                            'type': conn.type.name if hasattr(conn.type, 'name') else str(conn.type)
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception:
            pass
        
        return connections
    
    async def _is_port_listening(self, port: int) -> bool:
        """Check if a port is currently listening."""
        
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                result = sock.connect_ex(('localhost', port))
                return result == 0
        except Exception:
            return False
    
    async def _perform_health_check(self, server_name: str) -> bool:
        """Perform a health check on the server."""
        
        try:
            # Try to run the server's self-check script
            selfcheck_script = self.config.paths.mcp_root / "selfcheck_all.sh"
            
            if selfcheck_script.exists():
                result = await asyncio.create_subprocess_exec(
                    str(selfcheck_script),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await result.communicate()
                return result.returncode == 0
            
            return True  # Assume healthy if no check available
        except Exception:
            return False
    
    async def _check_version_tracking(self, server_name: str) -> bool:
        """Check if version tracking is available for the server."""
        
        try:
            # Check if version manager can track this server
            version_file = self.config.paths.automation_root / "versions" / f"{server_name}.json"
            return version_file.exists()
        except Exception:
            return False
    
    async def _check_snapshot_capability(self, server_name: str) -> bool:
        """Check if snapshot capability is available."""
        
        try:
            # Check for snapshot tools or backup scripts
            snapshot_script = self.config.paths.automation_root / "scripts" / f"snapshot_{server_name}.sh"
            return snapshot_script.exists()
        except Exception:
            return False


async def main():
    """Main entry point for safety validator testing."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        from ..config import get_config
        
        config = get_config()
        safety_validator = SafetyValidator(config)
        
        # Example: Validate cleanup safety
        logger.info("Testing safety validation...")
        
        # Mock cleanup analysis
        cleanup_analysis = {
            'server_name': 'postgres',
            'total_candidates': 5,
            'estimated_space_freed': 1024 * 1024 * 100,  # 100MB
            'candidates': {
                'versions': {'removable_versions': []},
                'artifacts': {'temporary_files': []}
            }
        }
        
        result = await safety_validator.validate_cleanup_safety(
            'postgres', cleanup_analysis, SafetyLevel.HIGH
        )
        
        logger.info(f"\nSafety Validation Results:")
        logger.info(f"Safety Level: {result.safety_level.value}")
        logger.info(f"Violations: {len(result.violations)}")
        logger.warning(f"Warnings: {len(result.warnings)}")
        logger.info(f"Checks Performed: {len(result.checks_performed)}")
        
        if result.violations:
            logger.info("\nViolations:")
            for violation in result.violations[:3]:  # Show first 3
                logger.info(f"- {violation.violation_type.value}: {violation.message}")
        
        logger.info(json.dumps(result.to_dict(), indent=2, default=str))
        
    except Exception as e:
        logging.error(f"Error in safety validator: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())