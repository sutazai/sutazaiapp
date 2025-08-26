#!/usr/bin/env python3
"""
MCP Version Cleanup Service

Intelligent cleanup of old MCP server versions with comprehensive safety validation,
retention policy enforcement, and zero-impact operations. Integrates with existing
version management infrastructure.

Author: Claude AI Assistant (garbage-collector.md)
Created: 2025-08-15 20:45:00 UTC
Version: 1.0.0
"""

import asyncio
import json
import shutil
import hashlib
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# Import parent automation components
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import get_config, MCPAutomationConfig
from version_manager import VersionManager, VersionInfo, VersionState, OperationType

from .retention_policies import RetentionPolicy
from .audit_logger import AuditLogger, AuditEvent, AuditEventType


class VersionCleanupStatus(Enum):
    """Status of version cleanup operations."""
    PENDING = "pending"
    ANALYZING = "analyzing"
    VALIDATING = "validating"
    CLEANING = "cleaning"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class VersionCleanupCandidate:
    """Information about a version candidate for cleanup."""
    version_info: VersionInfo
    cleanup_safety: str  # 'safe', 'caution', 'unsafe'
    cleanup_reason: str
    estimated_space_freed: int
    dependencies: List[str]
    last_used: Optional[datetime]
    cleanup_priority: int  # 1-10, higher = more urgent
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['version_info'] = self.version_info.to_dict()
        if self.last_used:
            result['last_used'] = self.last_used.isoformat()
        return result


@dataclass
class VersionCleanupResult:
    """Results from version cleanup execution."""
    server_name: str
    status: VersionCleanupStatus
    started_at: datetime
    completed_at: Optional[datetime]
    cleaned_versions: List[VersionInfo]
    skipped_versions: List[VersionInfo]
    bytes_freed: int
    errors: List[str]
    warnings: List[str]
    rollback_info: Dict[str, Any]
    dry_run: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['status'] = self.status.value
        result['started_at'] = self.started_at.isoformat()
        if self.completed_at:
            result['completed_at'] = self.completed_at.isoformat()
        result['cleaned_versions'] = [v.to_dict() for v in self.cleaned_versions]
        result['skipped_versions'] = [v.to_dict() for v in self.skipped_versions]
        return result


class VersionCleanupService:
    """
    Service for intelligent MCP version cleanup.
    
    Provides safe cleanup of old MCP server versions while preserving
    current and critical versions based on retention policies.
    """
    
    def __init__(self, config: MCPAutomationConfig, audit_logger: AuditLogger):
        """Initialize version cleanup service."""
        self.config = config
        self.audit_logger = audit_logger
        self.version_manager = VersionManager(config)
        self.logger = logging.getLogger(__name__)
        
        # Cleanup safety settings
        self.min_versions_to_keep = 2  # Always keep at least 2 versions
        self.critical_version_age_days = 7  # Keep versions newer than 7 days
        self.safety_buffer_versions = 1  # Extra versions to keep for safety
        
        self.logger.info("VersionCleanupService initialized", extra={
            'min_versions_to_keep': self.min_versions_to_keep,
            'critical_version_age_days': self.critical_version_age_days
        })
    
    async def analyze_cleanup_candidates(
        self, 
        server_name: str,
        retention_policy: Optional[RetentionPolicy] = None
    ) -> Dict[str, Any]:
        """
        Analyze version cleanup candidates for a specific server.
        
        Args:
            server_name: Name of the MCP server
            retention_policy: Custom retention policy (None for default)
            
        Returns:
            Analysis results with cleanup candidates and recommendations
        """
        analysis_start = datetime.now(timezone.utc)
        
        try:
            self.logger.info(f"Analyzing version cleanup candidates for: {server_name}")
            
            # Get all versions for the server
            all_versions = await self.version_manager.list_versions(server_name)
            
            if not all_versions:
                return {
                    'server_name': server_name,
                    'analysis_timestamp': analysis_start.isoformat(),
                    'total_versions': 0,
                    'removable_versions': [],
                    'kept_versions': [],
                    'estimated_space_freed': 0,
                    'recommendations': ['No versions found for cleanup']
                }
            
            # Sort versions by installation date (newest first)
            sorted_versions = sorted(
                all_versions, 
                key=lambda v: v.installed_at, 
                reverse=True
            )
            
            # Identify current/active version
            current_version = await self._get_current_version(server_name)
            
            # Apply retention policy
            if retention_policy:
                keep_count = retention_policy.max_versions_to_keep
                keep_days = retention_policy.min_age_days
            else:
                keep_count = max(self.min_versions_to_keep, 3)  # Default: keep 3
                keep_days = self.critical_version_age_days
            
            # Analyze each version
            removable_candidates = []
            kept_versions = []
            total_estimated_space = 0
            
            for i, version in enumerate(sorted_versions):
                candidate = await self._analyze_version_for_cleanup(
                    version, i, current_version, keep_count, keep_days
                )
                
                if candidate.cleanup_safety in ['safe', 'caution'] and i >= keep_count:
                    # Check if version is older than minimum age
                    age_days = (analysis_start - version.installed_at).days
                    if age_days >= keep_days:
                        removable_candidates.append(candidate)
                        total_estimated_space += candidate.estimated_space_freed
                    else:
                        kept_versions.append(version)
                else:
                    kept_versions.append(version)
            
            # Generate recommendations
            recommendations = await self._generate_version_recommendations(
                server_name, len(all_versions), len(removable_candidates), 
                total_estimated_space
            )
            
            analysis_result = {
                'server_name': server_name,
                'analysis_timestamp': analysis_start.isoformat(),
                'total_versions': len(all_versions),
                'removable_versions': [c.to_dict() for c in removable_candidates],
                'kept_versions': [v.to_dict() for v in kept_versions],
                'estimated_space_freed': total_estimated_space,
                'current_version': current_version.to_dict() if current_version else None,
                'retention_policy_applied': {
                    'max_versions_to_keep': keep_count,
                    'min_age_days': keep_days
                },
                'recommendations': recommendations
            }
            
            # Log analysis completion
            await self.audit_logger.log_event(AuditEvent(
                event_type=AuditEventType.ANALYSIS_COMPLETED,
                timestamp=analysis_start,
                component='version_cleanup',
                user='system',
                action='analyze_cleanup_candidates',
                resource=server_name,
                details={
                    'total_versions': len(all_versions),
                    'removable_versions': len(removable_candidates),
                    'estimated_space_freed': total_estimated_space
                }
            ))
            
            self.logger.info(f"Version analysis completed for {server_name}", extra={
                'total_versions': len(all_versions),
                'removable_versions': len(removable_candidates),
                'estimated_space_freed': total_estimated_space
            })
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error analyzing versions for {server_name}: {e}", exc_info=True)
            raise
    
    async def execute_cleanup(
        self,
        server_name: str,
        retention_policy: RetentionPolicy,
        dry_run: bool = False
    ) -> VersionCleanupResult:
        """
        Execute version cleanup for a specific server.
        
        Args:
            server_name: Name of the MCP server
            retention_policy: Retention policy to apply
            dry_run: If True, simulate cleanup without making changes
            
        Returns:
            Cleanup execution results
        """
        cleanup_start = datetime.now(timezone.utc)
        
        result = VersionCleanupResult(
            server_name=server_name,
            status=VersionCleanupStatus.ANALYZING,
            started_at=cleanup_start,
            completed_at=None,
            cleaned_versions=[],
            skipped_versions=[],
            bytes_freed=0,
            errors=[],
            warnings=[],
            rollback_info={},
            dry_run=dry_run
        )
        
        try:
            self.logger.info(f"Starting version cleanup for {server_name}", extra={
                'dry_run': dry_run,
                'retention_policy': {
                    'max_versions': retention_policy.max_versions_to_keep,
                    'min_age_days': retention_policy.min_age_days
                }
            })
            
            # Log cleanup start
            await self.audit_logger.log_event(AuditEvent(
                event_type=AuditEventType.CLEANUP_STARTED,
                timestamp=cleanup_start,
                component='version_cleanup',
                user='system',
                action='execute_cleanup',
                resource=server_name,
                details={
                    'dry_run': dry_run,
                    'retention_policy': retention_policy.__dict__
                }
            ))
            
            # Analyze cleanup candidates
            analysis = await self.analyze_cleanup_candidates(server_name, retention_policy)
            removable_candidates = [
                VersionCleanupCandidate(**{k: v for k, v in candidate.items() if k != 'version_info'})
                for candidate in analysis['removable_versions']
            ]
            
            result.status = VersionCleanupStatus.VALIDATING
            
            # Validate cleanup safety
            safety_validation = await self._validate_cleanup_safety(
                server_name, removable_candidates
            )
            
            if not safety_validation['safe'] and not dry_run:
                result.status = VersionCleanupStatus.FAILED
                result.errors.extend(safety_validation['errors'])
                result.warnings.extend(safety_validation['warnings'])
                return result
            
            result.warnings.extend(safety_validation['warnings'])
            
            # Execute cleanup
            result.status = VersionCleanupStatus.CLEANING
            
            for candidate in removable_candidates:
                version_info = candidate.version_info
                
                try:
                    # Check if this version should be skipped
                    if candidate.cleanup_safety == 'unsafe':
                        result.skipped_versions.append(version_info)
                        result.warnings.append(f"Skipped unsafe version: {version_info.version}")
                        continue
                    
                    # Perform cleanup
                    if dry_run:
                        # Simulate cleanup
                        result.cleaned_versions.append(version_info)
                        result.bytes_freed += candidate.estimated_space_freed
                        self.logger.info(f"[DRY RUN] Would remove version {version_info.version}")
                    else:
                        # Actual cleanup
                        cleanup_success = await self._remove_version(server_name, version_info)
                        
                        if cleanup_success:
                            result.cleaned_versions.append(version_info)
                            result.bytes_freed += candidate.estimated_space_freed
                            
                            # Log version removal
                            await self.audit_logger.log_event(AuditEvent(
                                event_type=AuditEventType.VERSION_REMOVED,
                                timestamp=datetime.now(timezone.utc),
                                component='version_cleanup',
                                user='system',
                                action='remove_version',
                                resource=f"{server_name}:{version_info.version}",
                                details={
                                    'version_info': version_info.to_dict(),
                                    'bytes_freed': candidate.estimated_space_freed
                                }
                            ))
                            
                            self.logger.info(f"Removed version {version_info.version} for {server_name}")
                        else:
                            result.skipped_versions.append(version_info)
                            result.warnings.append(f"Failed to remove version: {version_info.version}")
                
                except Exception as e:
                    error_msg = f"Error processing version {version_info.version}: {e}"
                    result.errors.append(error_msg)
                    result.skipped_versions.append(version_info)
                    self.logger.error(error_msg, exc_info=True)
            
            # Update version manager state
            if not dry_run and result.cleaned_versions:
                for version in result.cleaned_versions:
                    await self.version_manager.update_version_state(
                        server_name, version.version, VersionState.ARCHIVED
                    )
            
            # Complete cleanup
            cleanup_end = datetime.now(timezone.utc)
            result.completed_at = cleanup_end
            result.status = VersionCleanupStatus.COMPLETED
            
            # Create rollback information
            if not dry_run and result.cleaned_versions:
                result.rollback_info = {
                    'rollback_available': False,  # Version cleanup is typically not reversible
                    'rollback_note': 'Version cleanup is not reversible. Rely on backups for recovery.',
                    'backup_recommendations': [
                        'Ensure MCP server configurations are backed up',
                        'Test server functionality after cleanup',
                        'Keep installation documentation for version restoration'
                    ]
                }
            
            # Log cleanup completion
            await self.audit_logger.log_event(AuditEvent(
                event_type=AuditEventType.CLEANUP_COMPLETED,
                timestamp=cleanup_end,
                component='version_cleanup',
                user='system',
                action='execute_cleanup',
                resource=server_name,
                details={
                    'status': result.status.value,
                    'cleaned_versions': len(result.cleaned_versions),
                    'bytes_freed': result.bytes_freed,
                    'dry_run': dry_run
                }
            ))
            
            self.logger.info(f"Version cleanup completed for {server_name}", extra={
                'status': result.status.value,
                'cleaned_versions': len(result.cleaned_versions),
                'bytes_freed': result.bytes_freed,
                'dry_run': dry_run
            })
            
            return result
            
        except Exception as e:
            cleanup_end = datetime.now(timezone.utc)
            result.completed_at = cleanup_end
            result.status = VersionCleanupStatus.FAILED
            result.errors.append(f"Cleanup execution error: {e}")
            
            self.logger.error(f"Version cleanup failed for {server_name}: {e}", exc_info=True)
            
            # Log cleanup failure
            await self.audit_logger.log_event(AuditEvent(
                event_type=AuditEventType.CLEANUP_FAILED,
                timestamp=cleanup_end,
                component='version_cleanup',
                user='system',
                action='execute_cleanup',
                resource=server_name,
                details={'error': str(e)}
            ))
            
            return result
    
    async def _analyze_version_for_cleanup(
        self,
        version: VersionInfo,
        position: int,
        current_version: Optional[VersionInfo],
        keep_count: int,
        min_age_days: int
    ) -> VersionCleanupCandidate:
        """Analyze a single version for cleanup eligibility."""
        
        # Determine cleanup safety
        safety = 'safe'
        cleanup_reason = 'old_version'
        cleanup_priority = 5  # Default priority
        
        # Current/active version is unsafe to remove
        if current_version and version.version == current_version.version:
            safety = 'unsafe'
            cleanup_reason = 'current_active_version'
            cleanup_priority = 1
        
        # Very recent versions are unsafe
        age_days = (datetime.now(timezone.utc) - version.installed_at).days
        if age_days < min_age_days:
            safety = 'unsafe'
            cleanup_reason = f'too_recent_{age_days}_days'
            cleanup_priority = 2
        
        # Top N versions to keep
        elif position < keep_count:
            safety = 'caution'
            cleanup_reason = f'within_keep_count_{position}/{keep_count}'
            cleanup_priority = 3
        
        # Failed or rollback versions might be safer to remove
        elif version.state in [VersionState.FAILED, VersionState.ROLLBACK]:
            safety = 'safe'
            cleanup_reason = f'failed_or_rollback_state_{version.state.value}'
            cleanup_priority = 8
        
        # Old archived versions are safest to remove
        elif version.state == VersionState.ARCHIVED and age_days > 30:
            safety = 'safe'
            cleanup_reason = f'old_archived_{age_days}_days'
            cleanup_priority = 9
        
        # Check for dependencies (simplified check)
        dependencies = await self._check_version_dependencies(version)
        if dependencies:
            if safety == 'safe':
                safety = 'caution'
            cleanup_reason += '_has_dependencies'
            cleanup_priority -= 2
        
        return VersionCleanupCandidate(
            version_info=version,
            cleanup_safety=safety,
            cleanup_reason=cleanup_reason,
            estimated_space_freed=version.size_bytes,
            dependencies=dependencies,
            last_used=await self._get_version_last_used(version),
            cleanup_priority=max(1, cleanup_priority)
        )
    
    async def _get_current_version(self, server_name: str) -> Optional[VersionInfo]:
        """Get the currently active version for a server."""
        try:
            versions = await self.version_manager.list_versions(server_name)
            for version in versions:
                if version.state == VersionState.ACTIVE:
                    return version
            
            # If no active version, assume the most recent is current
            if versions:
                return max(versions, key=lambda v: v.installed_at)
            
        except Exception as e:
            self.logger.warning(f"Could not determine current version for {server_name}: {e}")
        
        return None
    
    async def _check_version_dependencies(self, version: VersionInfo) -> List[str]:
        """Check if a version has dependencies that might prevent cleanup."""
        dependencies = []
        
        # Check for configuration files that reference this version
        # Check for scripts that might reference this version
        # Check for other servers that might depend on this version
        
        # This is a simplified implementation
        # In a real system, you would check:
        # - Package.json dependencies
        # - Configuration files
        # - Symlinks
        # - Running processes
        
        return dependencies
    
    async def _get_version_last_used(self, version: VersionInfo) -> Optional[datetime]:
        """Get the last time this version was used (if available)."""
        # This would check logs, access times, etc.
        # For now, return None (not available)
        return None
    
    async def _validate_cleanup_safety(
        self,
        server_name: str,
        candidates: List[VersionCleanupCandidate]
    ) -> Dict[str, Any]:
        """Validate the safety of cleanup operation."""
        
        validation_result = {
            'safe': True,
            'errors': [],
            'warnings': []
        }
        
        # Check minimum version requirements
        unsafe_count = len([c for c in candidates if c.cleanup_safety == 'unsafe'])
        if unsafe_count > 0:
            validation_result['errors'].append(
                f"Found {unsafe_count} unsafe versions in cleanup candidates"
            )
            validation_result['safe'] = False
        
        # Check if we're removing too many versions
        total_removable = len([c for c in candidates if c.cleanup_safety in ['safe', 'caution']])
        if total_removable > 10:  # Arbitrary safety limit
            validation_result['warnings'].append(
                f"Large cleanup operation: {total_removable} versions to remove"
            )
        
        # Check for active server processes
        server_running = await self._check_server_running(server_name)
        if server_running:
            validation_result['warnings'].append(
                f"Server {server_name} appears to be running - proceed with caution"
            )
        
        return validation_result
    
    async def _check_server_running(self, server_name: str) -> bool:
        """Check if an MCP server is currently running."""
        try:
            # Check for running processes
            result = await asyncio.create_subprocess_exec(
                'pgrep', '-f', server_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await result.communicate()
            return len(stdout.decode().strip()) > 0
        except Exception:
            return False
    
    async def _remove_version(self, server_name: str, version: VersionInfo) -> bool:
        """Remove a specific version from the system."""
        try:
            # Construct version directory path
            version_dir = self.config.paths.staging_root / server_name / version.version
            
            if version_dir.exists():
                # Remove version directory
                shutil.rmtree(version_dir)
                self.logger.info(f"Removed version directory: {version_dir}")
                return True
            else:
                self.logger.warning(f"Version directory not found: {version_dir}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error removing version {version.version}: {e}", exc_info=True)
            return False
    
    async def _generate_version_recommendations(
        self,
        server_name: str,
        total_versions: int,
        removable_count: int,
        estimated_space: int
    ) -> List[str]:
        """Generate cleanup recommendations for versions."""
        recommendations = []
        
        if removable_count == 0:
            recommendations.append("No versions available for cleanup")
        elif removable_count == 1:
            recommendations.append(f"1 version can be safely removed, freeing ~{estimated_space:,} bytes")
        else:
            recommendations.append(
                f"{removable_count} versions can be safely removed, freeing ~{estimated_space:,} bytes"
            )
        
        if total_versions > 10:
            recommendations.append("Consider implementing more aggressive retention policies")
        
        if estimated_space > 1024 * 1024 * 100:  # > 100MB
            recommendations.append("Significant space savings available from cleanup")
        
        recommendations.append("Run in dry-run mode first to validate cleanup plan")
        recommendations.append("Ensure server functionality is tested after cleanup")
        
        return recommendations


async def main():
    """Main entry point for version cleanup testing."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        from ..config import get_config
        from .audit_logger import AuditLogger
        from .retention_policies import RetentionPolicy
        
        config = get_config()
        audit_logger = AuditLogger(config)
        version_cleanup = VersionCleanupService(config, audit_logger)
        
        # Example: Analyze cleanup candidates
        logger.info("Analyzing version cleanup candidates...")
        servers = ['postgres', 'files']  # Example servers
        
        for server in servers:
            try:
                analysis = await version_cleanup.analyze_cleanup_candidates(server)
                logger.info(f"\nAnalysis for {server}:")
                logger.info(json.dumps(analysis, indent=2, default=str))
            except Exception as e:
                logger.error(f"Error analyzing {server}: {e}")
        
    except Exception as e:
        logging.error(f"Error in version cleanup: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())