#!/usr/bin/env python3
"""
MCP Cleanup Manager

Main orchestration service for intelligent MCP cleanup operations. Coordinates
version cleanup, artifact management, and safety validation with comprehensive
audit logging and zero-impact operations.

Author: Claude AI Assistant (garbage-collector.md)
Created: 2025-08-15 20:30:00 UTC
Version: 1.0.0
"""

import asyncio
import signal
import sys
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# Import parent automation components
sys.path.append(str(Path(__file__).parent.parent))
from config import get_config, MCPAutomationConfig
from version_manager import VersionManager, VersionInfo, VersionState

from .retention_policies import RetentionPolicyManager, RetentionPolicy
from .safety_validator import SafetyValidator, SafetyResult, SafetyLevel
from .audit_logger import AuditLogger, AuditEvent, AuditEventType
from .version_cleanup import VersionCleanupService, VersionCleanupResult
from .artifact_cleanup import ArtifactCleanupService, ArtifactType


class CleanupStatus(Enum):
    """Cleanup operation status."""
    PENDING = "pending"
    ANALYZING = "analyzing" 
    VALIDATING = "validating"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DRY_RUN = "dry_run"


class CleanupPriority(Enum):
    """Cleanup operation priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class CleanupMode(Enum):
    """Cleanup execution modes."""
    DRY_RUN = "dry_run"
    SAFE = "safe"
    AGGRESSIVE = "aggressive"
    EMERGENCY = "emergency"


@dataclass
class CleanupJobConfig:
    """Configuration for a cleanup job."""
    job_id: str
    job_name: str
    priority: CleanupPriority
    mode: CleanupMode
    target_servers: List[str]
    cleanup_types: List[str]
    retention_policy: RetentionPolicy
    safety_level: SafetyLevel
    created_at: datetime
    scheduled_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CleanupJobResult:
    """Results from a cleanup job execution."""
    job_id: str
    status: CleanupStatus
    started_at: datetime
    completed_at: Optional[datetime]
    duration_seconds: float
    items_analyzed: int
    items_cleaned: int
    bytes_freed: int
    errors: List[str]
    warnings: List[str]
    safety_violations: List[str]
    rollback_available: bool
    audit_trail: List[AuditEvent]
    detailed_results: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['status'] = self.status.value
        result['started_at'] = self.started_at.isoformat()
        if self.completed_at:
            result['completed_at'] = self.completed_at.isoformat()
        result['audit_trail'] = [event.to_dict() for event in self.audit_trail]
        return result


class CleanupManager:
    """
    Main cleanup orchestration service.
    
    Coordinates all cleanup operations with comprehensive safety validation,
    audit logging, and zero-impact execution patterns.
    """
    
    def __init__(self, config: Optional[MCPAutomationConfig] = None):
        """Initialize cleanup manager with configuration."""
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize cleanup subsystems
        self.audit_logger = AuditLogger(self.config)
        self.safety_validator = SafetyValidator(self.config)
        self.retention_manager = RetentionPolicyManager(self.config)
        self.version_cleanup = VersionCleanupService(self.config, self.audit_logger)
        self.artifact_cleanup = ArtifactCleanupService(self.config, self.audit_logger)
        self.version_manager = VersionManager(self.config)
        
        # Job tracking
        self.active_jobs: Dict[str, CleanupJobConfig] = {}
        self.job_results: Dict[str, CleanupJobResult] = {}
        self.cleanup_stats = {
            'total_jobs': 0,
            'successful_jobs': 0,
            'failed_jobs': 0,
            'total_bytes_freed': 0,
            'total_items_cleaned': 0,
            'last_cleanup': None
        }
        
        # Shutdown handling
        self._shutdown_event = asyncio.Event()
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("CleanupManager initialized", extra={
            'config_mode': self.config.update_mode.value,
            'safety_level': 'maximum',
            'audit_enabled': True
        })
    
    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown")
        self._shutdown_event.set()
    
    async def analyze_cleanup_candidates(
        self,
        servers: Optional[List[str]] = None,
        cleanup_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze potential cleanup candidates without executing cleanup.
        
        Args:
            servers: List of server names to analyze (None for all)
            cleanup_types: Types of cleanup to analyze (versions, artifacts, logs)
            
        Returns:
            Analysis results with recommendations and safety assessment
        """
        analysis_start = datetime.now(timezone.utc)
        
        try:
            # Get all available servers if not specified
            if servers is None:
                servers = await self._discover_mcp_servers()
            
            if cleanup_types is None:
                cleanup_types = ['versions', 'artifacts', 'logs']
            
            analysis_results = {
                'analysis_timestamp': analysis_start.isoformat(),
                'analyzed_servers': servers,
                'cleanup_types': cleanup_types,
                'candidates': {},
                'safety_assessment': {},
                'recommendations': {},
                'estimated_impact': {
                    'bytes_to_free': 0,
                    'items_to_clean': 0,
                    'risk_level': 'unknown'
                }
            }
            
            # Analyze each server
            for server in servers:
                self.logger.info(f"Analyzing cleanup candidates for server: {server}")
                
                server_analysis = {
                    'server_name': server,
                    'versions': {},
                    'artifacts': {},
                    'logs': {},
                    'safety_level': SafetyLevel.HIGH,
                    'recommendations': []
                }
                
                # Version analysis
                if 'versions' in cleanup_types:
                    version_candidates = await self.version_cleanup.analyze_cleanup_candidates(server)
                    server_analysis['versions'] = version_candidates
                    
                    # Update estimated impact
                    for version_info in version_candidates.get('removable_versions', []):
                        analysis_results['estimated_impact']['bytes_to_free'] += version_info.get('size_bytes', 0)
                        analysis_results['estimated_impact']['items_to_clean'] += 1
                
                # Artifact analysis
                if 'artifacts' in cleanup_types:
                    artifact_candidates = await self.artifact_cleanup.analyze_cleanup_candidates(server)
                    server_analysis['artifacts'] = artifact_candidates
                    
                    # Update estimated impact
                    for artifact_type, artifacts in artifact_candidates.items():
                        if isinstance(artifacts, list):
                            for artifact in artifacts:
                                analysis_results['estimated_impact']['bytes_to_free'] += artifact.get('size_bytes', 0)
                                analysis_results['estimated_impact']['items_to_clean'] += 1
                
                # Safety assessment for server
                safety_result = await self.safety_validator.validate_cleanup_safety(
                    server, server_analysis
                )
                server_analysis['safety_level'] = safety_result.safety_level
                server_analysis['safety_warnings'] = safety_result.warnings
                
                # Generate recommendations
                recommendations = await self._generate_cleanup_recommendations(server_analysis)
                server_analysis['recommendations'] = recommendations
                
                analysis_results['candidates'][server] = server_analysis
            
            # Overall safety assessment
            analysis_results['safety_assessment'] = await self._assess_overall_safety(
                analysis_results['candidates']
            )
            
            # Risk level determination
            total_items = analysis_results['estimated_impact']['items_to_clean']
            if total_items == 0:
                analysis_results['estimated_impact']['risk_level'] = 'none'
            elif total_items < 10:
                analysis_results['estimated_impact']['risk_level'] = 'low'
            elif total_items < 50:
                analysis_results['estimated_impact']['risk_level'] = 'medium'
            else:
                analysis_results['estimated_impact']['risk_level'] = 'high'
            
            analysis_duration = (datetime.now(timezone.utc) - analysis_start).total_seconds()
            
            self.logger.info("Cleanup analysis completed", extra={
                'duration_seconds': analysis_duration,
                'servers_analyzed': len(servers),
                'items_to_clean': analysis_results['estimated_impact']['items_to_clean'],
                'bytes_to_free': analysis_results['estimated_impact']['bytes_to_free'],
                'risk_level': analysis_results['estimated_impact']['risk_level']
            })
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Error during cleanup analysis: {e}", exc_info=True)
            raise
    
    async def create_cleanup_job(
        self,
        job_name: str,
        servers: List[str],
        cleanup_types: List[str],
        mode: CleanupMode = CleanupMode.SAFE,
        priority: CleanupPriority = CleanupPriority.NORMAL,
        retention_policy: Optional[RetentionPolicy] = None,
        safety_level: SafetyLevel = SafetyLevel.HIGH,
        scheduled_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new cleanup job.
        
        Args:
            job_name: Human-readable job name
            servers: List of MCP servers to clean
            cleanup_types: Types of cleanup to perform
            mode: Cleanup execution mode
            priority: Job priority level
            retention_policy: Custom retention policy (None for default)
            safety_level: Safety validation level
            scheduled_at: Optional scheduled execution time
            metadata: Additional job metadata
            
        Returns:
            Unique job ID
        """
        job_id = f"cleanup_{int(time.time())}_{hash(job_name) % 10000:04d}"
        
        # Use default retention policy if not provided
        if retention_policy is None:
            retention_policy = self.retention_manager.get_default_policy()
        
        job_config = CleanupJobConfig(
            job_id=job_id,
            job_name=job_name,
            priority=priority,
            mode=mode,
            target_servers=servers,
            cleanup_types=cleanup_types,
            retention_policy=retention_policy,
            safety_level=safety_level,
            created_at=datetime.now(timezone.utc),
            scheduled_at=scheduled_at,
            metadata=metadata or {}
        )
        
        self.active_jobs[job_id] = job_config
        
        # Log job creation
        await self.audit_logger.log_event(AuditEvent(
            event_type=AuditEventType.JOB_CREATED,
            timestamp=job_config.created_at,
            component='cleanup_manager',
            user='system',
            action='create_cleanup_job',
            resource=job_id,
            details={
                'job_name': job_name,
                'servers': servers,
                'cleanup_types': cleanup_types,
                'mode': mode.value,
                'priority': priority.value,
                'safety_level': safety_level.value
            }
        ))
        
        self.logger.info(f"Created cleanup job: {job_id}", extra={
            'job_name': job_name,
            'servers': servers,
            'cleanup_types': cleanup_types,
            'mode': mode.value,
            'priority': priority.value
        })
        
        return job_id
    
    async def execute_cleanup_job(
        self,
        job_id: str,
        dry_run: bool = False
    ) -> CleanupJobResult:
        """
        Execute a cleanup job with comprehensive safety validation.
        
        Args:
            job_id: Unique job identifier
            dry_run: If True, only simulate cleanup without making changes
            
        Returns:
            Cleanup job results with detailed execution information
        """
        if job_id not in self.active_jobs:
            raise ValueError(f"Job {job_id} not found")
        
        job_config = self.active_jobs[job_id]
        execution_start = datetime.now(timezone.utc)
        
        # Create initial result
        result = CleanupJobResult(
            job_id=job_id,
            status=CleanupStatus.DRY_RUN if dry_run else CleanupStatus.EXECUTING,
            started_at=execution_start,
            completed_at=None,
            duration_seconds=0.0,
            items_analyzed=0,
            items_cleaned=0,
            bytes_freed=0,
            errors=[],
            warnings=[],
            safety_violations=[],
            rollback_available=False,
            audit_trail=[],
            detailed_results={}
        )
        
        try:
            self.logger.info(f"Starting cleanup job execution: {job_id}", extra={
                'dry_run': dry_run,
                'job_name': job_config.job_name,
                'mode': job_config.mode.value
            })
            
            # Log job start
            start_event = AuditEvent(
                event_type=AuditEventType.JOB_STARTED,
                timestamp=execution_start,
                component='cleanup_manager',
                user='system',
                action='execute_cleanup_job',
                resource=job_id,
                details={
                    'dry_run': dry_run,
                    'mode': job_config.mode.value,
                    'servers': job_config.target_servers
                }
            )
            await self.audit_logger.log_event(start_event)
            result.audit_trail.append(start_event)
            
            # Phase 1: Safety validation
            result.status = CleanupStatus.VALIDATING
            safety_validation_passed = True
            
            for server in job_config.target_servers:
                safety_result = await self.safety_validator.validate_server_cleanup_safety(
                    server, job_config.cleanup_types, job_config.safety_level
                )
                
                if safety_result.safety_level == SafetyLevel.UNSAFE:
                    result.safety_violations.extend(safety_result.violations)
                    safety_validation_passed = False
                elif safety_result.warnings:
                    result.warnings.extend(safety_result.warnings)
            
            if not safety_validation_passed and not dry_run:
                result.status = CleanupStatus.FAILED
                result.errors.append("Safety validation failed - cleanup aborted")
                return result
            
            # Phase 2: Execute cleanup for each server
            result.status = CleanupStatus.EXECUTING
            
            for server in job_config.target_servers:
                server_results = {
                    'server_name': server,
                    'versions_cleaned': [],
                    'artifacts_cleaned': [],
                    'bytes_freed': 0,
                    'errors': [],
                    'warnings': []
                }
                
                try:
                    # Version cleanup
                    if 'versions' in job_config.cleanup_types:
                        version_result = await self.version_cleanup.execute_cleanup(
                            server, job_config.retention_policy, dry_run
                        )
                        server_results['versions_cleaned'] = version_result.cleaned_versions
                        server_results['bytes_freed'] += version_result.bytes_freed
                        result.items_cleaned += len(version_result.cleaned_versions)
                        
                        if version_result.errors:
                            server_results['errors'].extend(version_result.errors)
                        if version_result.warnings:
                            server_results['warnings'].extend(version_result.warnings)
                    
                    # Artifact cleanup
                    if 'artifacts' in job_config.cleanup_types:
                        artifact_result = await self.artifact_cleanup.execute_cleanup(
                            server, job_config.retention_policy, dry_run
                        )
                        server_results['artifacts_cleaned'] = artifact_result.cleaned_artifacts
                        server_results['bytes_freed'] += artifact_result.bytes_freed
                        result.items_cleaned += len(artifact_result.cleaned_artifacts)
                        
                        if artifact_result.errors:
                            server_results['errors'].extend(artifact_result.errors)
                        if artifact_result.warnings:
                            server_results['warnings'].extend(artifact_result.warnings)
                    
                    result.bytes_freed += server_results['bytes_freed']
                    result.errors.extend(server_results['errors'])
                    result.warnings.extend(server_results['warnings'])
                    
                except Exception as e:
                    error_msg = f"Error cleaning server {server}: {e}"
                    self.logger.error(error_msg, exc_info=True)
                    result.errors.append(error_msg)
                    server_results['errors'].append(error_msg)
                
                result.detailed_results[server] = server_results
            
            # Completion
            execution_end = datetime.now(timezone.utc)
            result.completed_at = execution_end
            result.duration_seconds = (execution_end - execution_start).total_seconds()
            result.status = CleanupStatus.COMPLETED if not result.errors else CleanupStatus.FAILED
            
            # Log completion
            completion_event = AuditEvent(
                event_type=AuditEventType.JOB_COMPLETED,
                timestamp=execution_end,
                component='cleanup_manager',
                user='system',
                action='execute_cleanup_job',
                resource=job_id,
                details={
                    'status': result.status.value,
                    'duration_seconds': result.duration_seconds,
                    'items_cleaned': result.items_cleaned,
                    'bytes_freed': result.bytes_freed,
                    'errors_count': len(result.errors)
                }
            )
            await self.audit_logger.log_event(completion_event)
            result.audit_trail.append(completion_event)
            
            # Update statistics
            self.cleanup_stats['total_jobs'] += 1
            if result.status == CleanupStatus.COMPLETED:
                self.cleanup_stats['successful_jobs'] += 1
                self.cleanup_stats['total_bytes_freed'] += result.bytes_freed
                self.cleanup_stats['total_items_cleaned'] += result.items_cleaned
            else:
                self.cleanup_stats['failed_jobs'] += 1
            self.cleanup_stats['last_cleanup'] = execution_end.isoformat()
            
            # Store result
            self.job_results[job_id] = result
            
            self.logger.info(f"Cleanup job completed: {job_id}", extra={
                'status': result.status.value,
                'duration_seconds': result.duration_seconds,
                'items_cleaned': result.items_cleaned,
                'bytes_freed': result.bytes_freed,
                'dry_run': dry_run
            })
            
            return result
            
        except Exception as e:
            execution_end = datetime.now(timezone.utc)
            result.completed_at = execution_end
            result.duration_seconds = (execution_end - execution_start).total_seconds()
            result.status = CleanupStatus.FAILED
            result.errors.append(f"Execution error: {e}")
            
            self.logger.error(f"Cleanup job failed: {job_id}", exc_info=True)
            
            # Log failure
            failure_event = AuditEvent(
                event_type=AuditEventType.JOB_FAILED,
                timestamp=execution_end,
                component='cleanup_manager',
                user='system',
                action='execute_cleanup_job',
                resource=job_id,
                details={
                    'error': str(e),
                    'duration_seconds': result.duration_seconds
                }
            )
            await self.audit_logger.log_event(failure_event)
            result.audit_trail.append(failure_event)
            
            self.job_results[job_id] = result
            return result
    
    async def get_job_status(self, job_id: str) -> Optional[CleanupJobResult]:
        """Get the current status of a cleanup job."""
        return self.job_results.get(job_id)
    
    async def list_active_jobs(self) -> List[CleanupJobConfig]:
        """List all active cleanup jobs."""
        return list(self.active_jobs.values())
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel an active cleanup job."""
        if job_id in self.active_jobs:
            del self.active_jobs[job_id]
            
            # Log cancellation
            await self.audit_logger.log_event(AuditEvent(
                event_type=AuditEventType.JOB_CANCELLED,
                timestamp=datetime.now(timezone.utc),
                component='cleanup_manager',
                user='system',
                action='cancel_cleanup_job',
                resource=job_id,
                details={'reason': 'user_requested'}
            ))
            
            self.logger.info(f"Cancelled cleanup job: {job_id}")
            return True
        return False
    
    async def get_cleanup_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cleanup statistics."""
        stats = self.cleanup_stats.copy()
        
        # Add current job counts
        stats.update({
            'active_jobs': len(self.active_jobs),
            'completed_jobs': len(self.job_results),
            'cleanup_manager_uptime': time.time(),
            'last_analysis': None  # Could be enhanced to track last analysis
        })
        
        return stats
    
    async def _discover_mcp_servers(self) -> List[str]:
        """Discover available MCP servers from wrappers directory."""
        wrappers_dir = self.config.paths.wrappers_root
        servers = []
        
        if wrappers_dir.exists():
            for wrapper_file in wrappers_dir.glob("*.sh"):
                if wrapper_file.name not in ['_common.sh']:
                    server_name = wrapper_file.stem
                    servers.append(server_name)
        
        return sorted(servers)
    
    async def _generate_cleanup_recommendations(
        self, 
        server_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate cleanup recommendations based on analysis."""
        recommendations = []
        
        # Version recommendations
        versions = server_analysis.get('versions', {})
        removable_count = len(versions.get('removable_versions', []))
        if removable_count > 0:
            recommendations.append(
                f"Remove {removable_count} old version(s) to free space"
            )
        
        # Artifact recommendations
        artifacts = server_analysis.get('artifacts', {})
        for artifact_type, artifact_list in artifacts.items():
            if isinstance(artifact_list, list) and len(artifact_list) > 0:
                recommendations.append(
                    f"Clean {len(artifact_list)} {artifact_type} artifact(s)"
                )
        
        # Safety recommendations
        if server_analysis.get('safety_level') == SafetyLevel.MEDIUM:
            recommendations.append("Review cleanup items manually before execution")
        elif server_analysis.get('safety_level') == SafetyLevel.LOW:
            recommendations.append("Consider running in dry-run mode first")
        
        return recommendations
    
    async def _assess_overall_safety(
        self, 
        candidates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess overall safety of cleanup operation."""
        safety_levels = []
        total_warnings = 0
        total_violations = 0
        
        for server, analysis in candidates.items():
            safety_levels.append(analysis.get('safety_level', SafetyLevel.HIGH))
            total_warnings += len(analysis.get('safety_warnings', []))
        
        # Determine overall safety level
        if all(level == SafetyLevel.HIGH for level in safety_levels):
            overall_safety = SafetyLevel.HIGH
        elif any(level == SafetyLevel.UNSAFE for level in safety_levels):
            overall_safety = SafetyLevel.UNSAFE
        elif any(level == SafetyLevel.LOW for level in safety_levels):
            overall_safety = SafetyLevel.LOW
        else:
            overall_safety = SafetyLevel.MEDIUM
        
        return {
            'overall_safety_level': overall_safety,
            'servers_analyzed': len(candidates),
            'total_warnings': total_warnings,
            'recommendations': [
                "Review all warnings before executing cleanup",
                "Consider dry-run mode for first execution",
                "Ensure backup/rollback procedures are available"
            ] if total_warnings > 0 else []
        }


async def main():
    """Main entry point for cleanup manager testing."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        cleanup_manager = CleanupManager()
        
        # Example: Analyze cleanup candidates
        analysis = await cleanup_manager.analyze_cleanup_candidates()
        logger.info("Cleanup Analysis Results:")
        logger.info(json.dumps(analysis, indent=2, default=str))
        
        # Example: Create and execute a dry-run job
        if analysis['estimated_impact']['items_to_clean'] > 0:
            job_id = await cleanup_manager.create_cleanup_job(
                job_name="Test Cleanup",
                servers=list(analysis['candidates'].keys())[:1],  # First server only
                cleanup_types=['versions', 'artifacts'],
                mode=CleanupMode.SAFE
            )
            
            result = await cleanup_manager.execute_cleanup_job(job_id, dry_run=True)
            logger.info(f"\nDry Run Results for Job {job_id}:")
            logger.info(json.dumps(result.to_dict(), indent=2, default=str))
        
    except Exception as e:
        logging.error(f"Error in cleanup manager: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())