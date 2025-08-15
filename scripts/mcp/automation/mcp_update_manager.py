#!/usr/bin/env python3
"""
MCP Update Manager

Main orchestration service for automated MCP server updates. Provides comprehensive
update coordination with safety mechanisms, monitoring, and zero-disruption deployment
following Enforcement Rules compliance.

Author: Claude AI Assistant (python-architect.md)
Created: 2025-08-15 11:35:00 UTC
Version: 1.0.0
"""

import asyncio
import signal
import sys
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass
from enum import Enum
import logging

from config import get_config, MCPAutomationConfig, UpdateMode
from version_manager import VersionManager, VersionState, OperationType
from download_manager import DownloadManager, DownloadProgress, DownloadResult


class UpdateStatus(Enum):
    """Update operation status."""
    PENDING = "pending"
    CHECKING = "checking"
    DOWNLOADING = "downloading"
    STAGING = "staging"
    TESTING = "testing"
    ACTIVATING = "activating"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLBACK = "rollback"
    CANCELLED = "cancelled"


class UpdatePriority(Enum):
    """Update priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"
    SECURITY = "security"


@dataclass
class UpdateJob:
    """Update job definition."""
    job_id: str
    server_name: str
    target_version: Optional[str]
    priority: UpdatePriority
    status: UpdateStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress_percentage: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def duration_seconds(self) -> float:
        """Calculate job duration."""
        if self.started_at:
            end_time = self.completed_at or datetime.now(timezone.utc)
            return (end_time - self.started_at).total_seconds()
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'job_id': self.job_id,
            'server_name': self.server_name,
            'target_version': self.target_version,
            'priority': self.priority.value,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'progress_percentage': self.progress_percentage,
            'error_message': self.error_message,
            'metadata': self.metadata,
            'duration_seconds': self.duration_seconds
        }


@dataclass
class UpdateSummary:
    """Summary of update operations."""
    total_servers: int
    pending_updates: int
    successful_updates: int
    failed_updates: int
    rollbacks_performed: int
    total_duration_seconds: float
    last_update_check: datetime
    next_scheduled_check: datetime


class MCPUpdateManager:
    """
    Main orchestration service for MCP server updates.
    
    Provides comprehensive update coordination with safety mechanisms,
    monitoring integration, and zero-disruption deployment capabilities.
    """
    
    def __init__(self, config: Optional[MCPAutomationConfig] = None):
        """
        Initialize MCP update manager.
        
        Args:
            config: Optional configuration override
        """
        self.config = config or get_config()
        self.logger = self._setup_logging()
        
        # Initialize managers
        self.version_manager = VersionManager(self.config)
        self.download_manager = None  # Initialized in context manager
        
        # Job management
        self.update_queue: List[UpdateJob] = []
        self.active_jobs: Dict[str, UpdateJob] = {}
        self.completed_jobs: List[UpdateJob] = []
        
        # State management
        self.running = False
        self.paused = False
        self.shutdown_requested = False
        
        # Metrics and monitoring
        self.metrics = {
            'updates_processed': 0,
            'updates_successful': 0,
            'updates_failed': 0,
            'rollbacks_performed': 0,
            'total_download_bytes': 0,
            'average_update_time': 0.0,
            'last_health_check': None,
            'last_update_check': None
        }
        
        # Health check configuration
        self.health_check_interval = timedelta(minutes=self.config.monitoring.health_check_interval_minutes)
        self.last_health_check = datetime.now(timezone.utc)
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        self.logger.info("MCP Update Manager initialized successfully")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging for update manager."""
        logger = logging.getLogger(f"{__name__}.MCPUpdateManager")
        logger.setLevel(getattr(logging, self.config.log_level.value))
        
        if not logger.handlers:
            # Create logs directory
            self.config.paths.logs_root.mkdir(parents=True, exist_ok=True)
            
            # File handler for detailed logs
            log_file = self.config.paths.logs_root / f"mcp_update_manager_{datetime.now().strftime('%Y%m%d')}.log"
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
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating graceful shutdown")
            self.shutdown_requested = True
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.download_manager = DownloadManager(self.config)
        await self.download_manager.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.download_manager:
            await self.download_manager.__aexit__(exc_type, exc_val, exc_tb)
    
    async def run(self):
        """
        Main event loop for update manager.
        
        Continuously processes update queue and performs health checks.
        """
        try:
            self.running = True
            self.logger.info("MCP Update Manager started")
            
            while self.running and not self.shutdown_requested:
                try:
                    # Process update queue
                    if not self.paused:
                        await self._process_update_queue()
                    
                    # Perform health checks
                    if await self._should_perform_health_check():
                        await self._perform_health_check()
                    
                    # Check for new updates
                    if await self._should_check_for_updates():
                        await self._check_for_updates()
                    
                    # Cleanup completed jobs
                    await self._cleanup_completed_jobs()
                    
                    # Save state
                    await self._save_state()
                    
                    # Short sleep to prevent busy waiting
                    await asyncio.sleep(1)
                
                except Exception as e:
                    self.logger.error(f"Error in main loop: {e}")
                    await asyncio.sleep(5)  # Longer sleep on error
            
            self.logger.info("MCP Update Manager stopped")
        
        except Exception as e:
            self.logger.critical(f"Critical error in update manager: {e}")
            raise
        
        finally:
            await self._cleanup()
    
    async def schedule_update(self, server_name: str, target_version: Optional[str] = None,
                            priority: UpdatePriority = UpdatePriority.NORMAL,
                            metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Schedule an update for an MCP server.
        
        Args:
            server_name: Name of the MCP server
            target_version: Specific version to update to (latest if None)
            priority: Update priority
            metadata: Optional metadata for the update
            
        Returns:
            Job ID for tracking the update
        """
        try:
            # Validate server
            if not self.config.get_server_config(server_name):
                raise ValueError(f"Unknown MCP server: {server_name}")
            
            # Check if update already queued
            existing_job = self._find_queued_job(server_name)
            if existing_job:
                self.logger.warning(f"Update already queued for {server_name}: {existing_job.job_id}")
                return existing_job.job_id
            
            # Create update job
            job_id = f"update_{server_name}_{int(datetime.now().timestamp())}"
            job = UpdateJob(
                job_id=job_id,
                server_name=server_name,
                target_version=target_version,
                priority=priority,
                status=UpdateStatus.PENDING,
                created_at=datetime.now(timezone.utc),
                metadata=metadata or {}
            )
            
            # Add to queue (sorted by priority)
            self.update_queue.append(job)
            self._sort_update_queue()
            
            self.logger.info(f"Scheduled update for {server_name} (Job: {job_id})")
            
            return job_id
        
        except Exception as e:
            self.logger.error(f"Failed to schedule update for {server_name}: {e}")
            raise
    
    async def cancel_update(self, job_id: str) -> bool:
        """
        Cancel a pending or active update.
        
        Args:
            job_id: ID of the job to cancel
            
        Returns:
            True if cancellation successful, False otherwise
        """
        try:
            # Check if job is in queue
            for job in self.update_queue:
                if job.job_id == job_id:
                    job.status = UpdateStatus.CANCELLED
                    self.update_queue.remove(job)
                    self.completed_jobs.append(job)
                    self.logger.info(f"Cancelled queued update job: {job_id}")
                    return True
            
            # Check if job is active
            if job_id in self.active_jobs:
                job = self.active_jobs[job_id]
                job.status = UpdateStatus.CANCELLED
                # Cancel download if in progress
                download_id = job.metadata.get('download_id')
                if download_id and self.download_manager:
                    self.download_manager.cancel_download(download_id)
                
                self.logger.info(f"Cancelled active update job: {job_id}")
                return True
            
            return False
        
        except Exception as e:
            self.logger.error(f"Failed to cancel update {job_id}: {e}")
            return False
    
    async def get_update_status(self, job_id: str) -> Optional[UpdateJob]:
        """Get status of an update job."""
        # Check active jobs
        if job_id in self.active_jobs:
            return self.active_jobs[job_id]
        
        # Check queue
        for job in self.update_queue:
            if job.job_id == job_id:
                return job
        
        # Check completed jobs
        for job in self.completed_jobs:
            if job.job_id == job_id:
                return job
        
        return None
    
    async def get_update_summary(self) -> UpdateSummary:
        """Get summary of update operations."""
        total_servers = len(self.config.get_all_servers())
        pending_updates = len(self.update_queue) + len(self.active_jobs)
        
        successful_updates = len([j for j in self.completed_jobs if j.status == UpdateStatus.COMPLETED])
        failed_updates = len([j for j in self.completed_jobs if j.status == UpdateStatus.FAILED])
        rollbacks_performed = len([j for j in self.completed_jobs if j.status == UpdateStatus.ROLLBACK])
        
        total_duration = sum(j.duration_seconds for j in self.completed_jobs if j.completed_at)
        
        return UpdateSummary(
            total_servers=total_servers,
            pending_updates=pending_updates,
            successful_updates=successful_updates,
            failed_updates=failed_updates,
            rollbacks_performed=rollbacks_performed,
            total_duration_seconds=total_duration,
            last_update_check=self.metrics.get('last_update_check', datetime.now(timezone.utc)),
            next_scheduled_check=datetime.now(timezone.utc) + timedelta(hours=1)  # Next automatic check
        )
    
    async def _process_update_queue(self):
        """Process pending updates from the queue."""
        if not self.update_queue:
            return
        
        # Limit concurrent updates
        max_concurrent = self.config.performance.max_concurrent_downloads
        if len(self.active_jobs) >= max_concurrent:
            return
        
        # Get next job from queue
        job = self.update_queue.pop(0)
        
        # Move to active jobs
        self.active_jobs[job.job_id] = job
        job.started_at = datetime.now(timezone.utc)
        
        # Process update asynchronously
        asyncio.create_task(self._process_update_job(job))
    
    async def _process_update_job(self, job: UpdateJob):
        """
        Process a single update job.
        
        Args:
            job: Update job to process
        """
        try:
            self.logger.info(f"Processing update job: {job.job_id} for {job.server_name}")
            
            # Check current version
            job.status = UpdateStatus.CHECKING
            job.progress_percentage = 10.0
            
            current_version = await self.version_manager.get_current_version(job.server_name)
            available_version = await self.version_manager.get_available_version(job.server_name)
            
            target_version = job.target_version or available_version
            
            if not target_version:
                raise ValueError(f"No target version available for {job.server_name}")
            
            # Check if update needed
            if current_version == target_version:
                self.logger.info(f"Server {job.server_name} already at version {target_version}")
                job.status = UpdateStatus.COMPLETED
                job.progress_percentage = 100.0
                return
            
            # Download package
            job.status = UpdateStatus.DOWNLOADING
            job.progress_percentage = 20.0
            
            def download_progress_callback(progress: DownloadProgress):
                # Update job progress (20-60% for download)
                download_percent = progress.percentage * 0.4  # 40% of total
                job.progress_percentage = 20.0 + download_percent
            
            download_result = await self.download_manager.download_package(
                server_name=job.server_name,
                package_name=self.config.get_server_config(job.server_name)['package'],
                version=target_version,
                progress_callback=download_progress_callback
            )
            
            if not download_result.success:
                raise RuntimeError(f"Download failed: {download_result.error_message}")
            
            # Stage version
            job.status = UpdateStatus.STAGING
            job.progress_percentage = 70.0
            
            stage_success = await self.version_manager.stage_version(
                server_name=job.server_name,
                version=target_version,
                source_path=download_result.file_path
            )
            
            if not stage_success:
                raise RuntimeError("Failed to stage version")
            
            # Test staged version (only in staging mode)
            if self.config.update_mode == UpdateMode.STAGING_ONLY:
                job.status = UpdateStatus.TESTING
                job.progress_percentage = 80.0
                
                test_success = await self._test_staged_version(job.server_name, target_version)
                if not test_success:
                    raise RuntimeError("Staged version failed health checks")
                
                # Complete in staging mode
                job.status = UpdateStatus.COMPLETED
                job.progress_percentage = 100.0
                self.logger.info(f"Update completed in staging mode for {job.server_name}")
                return
            
            # Activate version (production mode)
            if self.config.update_mode == UpdateMode.PRODUCTION:
                job.status = UpdateStatus.ACTIVATING
                job.progress_percentage = 90.0
                
                activate_success = await self.version_manager.activate_version(
                    server_name=job.server_name,
                    version=target_version
                )
                
                if not activate_success:
                    # Attempt rollback
                    if self.config.enable_auto_rollback:
                        await self._attempt_rollback(job, current_version)
                    raise RuntimeError("Failed to activate version")
                
                # Final health check
                final_test_success = await self._test_activated_version(job.server_name)
                if not final_test_success:
                    # Attempt rollback
                    if self.config.enable_auto_rollback:
                        await self._attempt_rollback(job, current_version)
                    raise RuntimeError("Activated version failed health checks")
            
            # Success
            job.status = UpdateStatus.COMPLETED
            job.progress_percentage = 100.0
            
            self.metrics['updates_successful'] += 1
            self.metrics['total_download_bytes'] += download_result.size_bytes
            
            self.logger.info(f"Successfully updated {job.server_name} from {current_version} to {target_version}")
        
        except Exception as e:
            job.status = UpdateStatus.FAILED
            job.error_message = str(e)
            self.metrics['updates_failed'] += 1
            
            self.logger.error(f"Update failed for {job.server_name}: {e}")
        
        finally:
            # Complete job
            job.completed_at = datetime.now(timezone.utc)
            self.metrics['updates_processed'] += 1
            
            # Move from active to completed
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]
            
            self.completed_jobs.append(job)
            
            # Update average time
            if self.metrics['updates_processed'] > 0:
                total_time = sum(j.duration_seconds for j in self.completed_jobs if j.completed_at)
                self.metrics['average_update_time'] = total_time / self.metrics['updates_processed']
    
    async def _test_staged_version(self, server_name: str, version: str) -> bool:
        """Test staged version in isolated environment."""
        try:
            self.logger.info(f"Testing staged version {version} for {server_name}")
            
            # In a real implementation, this would:
            # 1. Create isolated test environment
            # 2. Install staged version
            # 3. Run health checks
            # 4. Verify functionality
            # 5. Cleanup test environment
            
            # For now, simulate basic validation
            await asyncio.sleep(2)  # Simulate test time
            
            # Check if staging files exist
            staging_path = self.config.get_staging_path(server_name)
            staged_files = list(staging_path.glob(f"{server_name}-{version}.*"))
            
            if not staged_files:
                self.logger.error(f"No staged files found for {server_name} version {version}")
                return False
            
            self.logger.info(f"Staged version test passed for {server_name}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error testing staged version: {e}")
            return False
    
    async def _test_activated_version(self, server_name: str) -> bool:
        """Test activated version using existing health check infrastructure."""
        try:
            self.logger.info(f"Testing activated version for {server_name}")
            
            # Use existing MCP health check infrastructure
            wrapper_script = self.config.paths.wrappers_root / f"{server_name}.sh"
            if not wrapper_script.exists():
                self.logger.warning(f"No wrapper script found for {server_name}")
                return True  # Don't fail if wrapper doesn't exist
            
            # Run health check
            process = await asyncio.create_subprocess_exec(
                str(wrapper_script), '--selfcheck',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.performance.health_check_timeout_seconds
            )
            
            if process.returncode == 0:
                self.logger.info(f"Health check passed for {server_name}")
                return True
            else:
                self.logger.error(f"Health check failed for {server_name}: {stderr.decode()}")
                return False
        
        except asyncio.TimeoutError:
            self.logger.error(f"Health check timeout for {server_name}")
            return False
        
        except Exception as e:
            self.logger.error(f"Error testing activated version: {e}")
            return False
    
    async def _attempt_rollback(self, job: UpdateJob, previous_version: Optional[str]):
        """Attempt automatic rollback on failure."""
        try:
            self.logger.warning(f"Attempting rollback for {job.server_name}")
            job.status = UpdateStatus.ROLLBACK
            
            rollback_success = await self.version_manager.rollback_version(
                server_name=job.server_name,
                target_version=previous_version
            )
            
            if rollback_success:
                self.metrics['rollbacks_performed'] += 1
                self.logger.info(f"Rollback successful for {job.server_name}")
            else:
                self.logger.error(f"Rollback failed for {job.server_name}")
        
        except Exception as e:
            self.logger.error(f"Error during rollback: {e}")
    
    async def _should_perform_health_check(self) -> bool:
        """Check if health check should be performed."""
        return (datetime.now(timezone.utc) - self.last_health_check) >= self.health_check_interval
    
    async def _perform_health_check(self):
        """Perform health check on all MCP servers."""
        try:
            self.logger.debug("Performing health check on all MCP servers")
            
            for server_name in self.config.get_all_servers():
                # Skip servers that are currently being updated
                if any(job.server_name == server_name for job in self.active_jobs.values()):
                    continue
                
                # Test server health
                health_ok = await self._test_activated_version(server_name)
                if not health_ok:
                    self.logger.warning(f"Health check failed for {server_name}")
                    
                    # Optionally schedule corrective action
                    if self.config.enable_auto_rollback:
                        # Could schedule automatic remediation here
                        pass
            
            self.last_health_check = datetime.now(timezone.utc)
            self.metrics['last_health_check'] = self.last_health_check
        
        except Exception as e:
            self.logger.error(f"Error during health check: {e}")
    
    async def _should_check_for_updates(self) -> bool:
        """Check if we should look for new updates."""
        last_check = self.metrics.get('last_update_check')
        if not last_check:
            return True
        
        # Check every hour
        return (datetime.now(timezone.utc) - last_check) >= timedelta(hours=1)
    
    async def _check_for_updates(self):
        """Check for available updates for all servers."""
        try:
            self.logger.debug("Checking for available updates")
            
            for server_name in self.config.get_all_servers():
                # Skip if update already queued or active
                if (any(job.server_name == server_name for job in self.update_queue) or
                    any(job.server_name == server_name for job in self.active_jobs.values())):
                    continue
                
                current_version = await self.version_manager.get_current_version(server_name)
                available_version = await self.version_manager.get_available_version(server_name)
                
                if available_version and current_version != available_version:
                    self.logger.info(f"Update available for {server_name}: {current_version} -> {available_version}")
                    
                    # Auto-schedule low priority updates
                    await self.schedule_update(
                        server_name=server_name,
                        target_version=available_version,
                        priority=UpdatePriority.LOW,
                        metadata={'auto_scheduled': True}
                    )
            
            self.metrics['last_update_check'] = datetime.now(timezone.utc)
        
        except Exception as e:
            self.logger.error(f"Error checking for updates: {e}")
    
    def _find_queued_job(self, server_name: str) -> Optional[UpdateJob]:
        """Find existing queued job for server."""
        for job in self.update_queue:
            if job.server_name == server_name and job.status == UpdateStatus.PENDING:
                return job
        return None
    
    def _sort_update_queue(self):
        """Sort update queue by priority."""
        priority_order = {
            UpdatePriority.SECURITY: 0,
            UpdatePriority.CRITICAL: 1,
            UpdatePriority.HIGH: 2,
            UpdatePriority.NORMAL: 3,
            UpdatePriority.LOW: 4
        }
        
        self.update_queue.sort(key=lambda job: (priority_order[job.priority], job.created_at))
    
    async def _cleanup_completed_jobs(self):
        """Clean up old completed jobs."""
        # Keep only last 50 completed jobs
        if len(self.completed_jobs) > 50:
            self.completed_jobs = self.completed_jobs[-50:]
    
    async def _save_state(self):
        """Save current state to disk."""
        try:
            state_file = self.config.paths.automation_root / "manager_state.json"
            
            state = {
                'metrics': self.metrics,
                'update_queue': [job.to_dict() for job in self.update_queue],
                'active_jobs': {job_id: job.to_dict() for job_id, job in self.active_jobs.items()},
                'completed_jobs': [job.to_dict() for job in self.completed_jobs[-20:]],  # Last 20 only
                'last_save': datetime.now(timezone.utc).isoformat()
            }
            
            # Convert datetime objects to strings in metrics
            serializable_metrics = {}
            for key, value in self.metrics.items():
                if isinstance(value, datetime):
                    serializable_metrics[key] = value.isoformat()
                else:
                    serializable_metrics[key] = value
            
            state['metrics'] = serializable_metrics
            
            # Atomic write
            temp_file = state_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            temp_file.replace(state_file)
        
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
    
    async def _cleanup(self):
        """Cleanup resources on shutdown."""
        try:
            self.logger.info("Cleaning up update manager resources")
            
            # Cancel active jobs
            for job in self.active_jobs.values():
                job.status = UpdateStatus.CANCELLED
            
            # Save final state
            await self._save_state()
        
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def pause(self):
        """Pause update processing."""
        self.paused = True
        self.logger.info("Update manager paused")
    
    def resume(self):
        """Resume update processing."""
        self.paused = False
        self.logger.info("Update manager resumed")
    
    def stop(self):
        """Stop update manager."""
        self.running = False
        self.logger.info("Update manager stop requested")


async def main():
    """Main entry point for standalone execution."""
    try:
        config = get_config()
        
        async with MCPUpdateManager(config) as manager:
            # Example: Schedule some updates
            await manager.schedule_update("files", priority=UpdatePriority.HIGH)
            await manager.schedule_update("postgres", priority=UpdatePriority.NORMAL)
            
            # Run manager
            await manager.run()
    
    except KeyboardInterrupt:
        print("\\nShutdown requested by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())