#!/usr/bin/env python3
"""
MCP Cleanup Scheduler

Automated cleanup scheduling and orchestration service. Provides flexible
scheduling capabilities with cron-like expressions, event-driven triggers,
and intelligent cleanup coordination across multiple servers.

Author: Claude AI Assistant (garbage-collector.md)
Created: 2025-08-15 22:00:00 UTC
Version: 1.0.0
"""

import asyncio
import json
import signal
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
import croniter
import threading
import uuid

# Import parent automation components
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import get_config, MCPAutomationConfig

from .cleanup_manager import CleanupManager, CleanupMode, CleanupPriority
from .retention_policies import RetentionPolicyManager, PolicyType
from .safety_validator import SafetyLevel
from .audit_logger import AuditLogger, AuditEvent, AuditEventType


class ScheduleType(Enum):
    """Types of cleanup schedules."""
    CRON = "cron"                    # Standard cron expression
    INTERVAL = "interval"            # Fixed interval (seconds)
    EVENT_DRIVEN = "event_driven"    # Triggered by events
    THRESHOLD_BASED = "threshold_based"  # Triggered by thresholds
    ONE_TIME = "one_time"           # Single execution


class TriggerCondition(Enum):
    """Trigger conditions for event-driven schedules."""
    DISK_SPACE_LOW = "disk_space_low"
    SERVER_UPDATE = "server_update"
    MANUAL_TRIGGER = "manual_trigger"
    SYSTEM_STARTUP = "system_startup"
    ERROR_THRESHOLD = "error_threshold"
    SIZE_THRESHOLD = "size_threshold"


class ScheduleStatus(Enum):
    """Status of scheduled cleanup jobs."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    EXECUTING = "executing"
    FAILED = "failed"
    COMPLETED = "completed"


@dataclass
class CleanupSchedule:
    """Comprehensive cleanup schedule definition."""
    # Schedule identification
    schedule_id: str
    schedule_name: str
    description: str = ""
    
    # Schedule timing
    schedule_type: ScheduleType = ScheduleType.CRON
    cron_expression: Optional[str] = None
    interval_seconds: Optional[int] = None
    next_run_time: Optional[datetime] = None
    
    # Cleanup configuration
    target_servers: List[str] = field(default_factory=list)
    cleanup_types: List[str] = field(default_factory=lambda: ['versions', 'artifacts'])
    cleanup_mode: CleanupMode = CleanupMode.SAFE
    priority: CleanupPriority = CleanupPriority.NORMAL
    safety_level: SafetyLevel = SafetyLevel.HIGH
    dry_run: bool = False
    
    # Policy and constraints
    retention_policy_id: Optional[str] = None
    max_concurrent_jobs: int = 1
    timeout_minutes: int = 60
    retry_attempts: int = 3
    retry_delay_seconds: int = 300
    
    # Trigger conditions
    trigger_conditions: List[TriggerCondition] = field(default_factory=list)
    threshold_config: Dict[str, Any] = field(default_factory=dict)
    
    # Schedule metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = "system"
    enabled: bool = True
    
    # Execution tracking
    status: ScheduleStatus = ScheduleStatus.ACTIVE
    last_execution: Optional[datetime] = None
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    execution_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    
    # Execution constraints
    max_executions: Optional[int] = None
    expire_at: Optional[datetime] = None
    only_if_idle: bool = False
    
    def __post_init__(self):
        """Validate and initialize schedule."""
        self._validate_schedule()
        if self.next_run_time is None:
            self.next_run_time = self._calculate_next_run_time()
    
    def _validate_schedule(self):
        """Validate schedule configuration."""
        if self.schedule_type == ScheduleType.CRON and not self.cron_expression:
            raise ValueError("CRON schedule requires cron_expression")
        
        if self.schedule_type == ScheduleType.INTERVAL and not self.interval_seconds:
            raise ValueError("INTERVAL schedule requires interval_seconds")
        
        if self.schedule_type == ScheduleType.CRON and self.cron_expression:
            try:
                croniter.croniter(self.cron_expression)
            except ValueError as e:
                raise ValueError(f"Invalid cron expression: {e}")
        
        if self.max_executions is not None and self.max_executions <= 0:
            raise ValueError("max_executions must be positive")
        
        if self.timeout_minutes <= 0:
            raise ValueError("timeout_minutes must be positive")
    
    def _calculate_next_run_time(self) -> Optional[datetime]:
        """Calculate the next execution time."""
        now = datetime.now(timezone.utc)
        
        if self.schedule_type == ScheduleType.CRON and self.cron_expression:
            cron = croniter.croniter(self.cron_expression, now)
            return cron.get_next(datetime)
        
        elif self.schedule_type == ScheduleType.INTERVAL and self.interval_seconds:
            return now + timedelta(seconds=self.interval_seconds)
        
        elif self.schedule_type == ScheduleType.ONE_TIME:
            return now  # Execute immediately
        
        return None  # Event-driven or threshold-based schedules
    
    def should_execute_now(self) -> bool:
        """Check if schedule should execute now."""
        if not self.enabled or self.status != ScheduleStatus.ACTIVE:
            return False
        
        # Check expiration
        if self.expire_at and datetime.now(timezone.utc) > self.expire_at:
            return False
        
        # Check execution limits
        if self.max_executions and self.execution_count >= self.max_executions:
            return False
        
        # Check next run time
        if self.next_run_time and datetime.now(timezone.utc) >= self.next_run_time:
            return True
        
        return False
    
    def update_next_run_time(self):
        """Update next run time after execution."""
        if self.schedule_type == ScheduleType.CRON and self.cron_expression:
            cron = croniter.croniter(self.cron_expression, datetime.now(timezone.utc))
            self.next_run_time = cron.get_next(datetime)
        
        elif self.schedule_type == ScheduleType.INTERVAL and self.interval_seconds:
            self.next_run_time = datetime.now(timezone.utc) + timedelta(seconds=self.interval_seconds)
        
        elif self.schedule_type == ScheduleType.ONE_TIME:
            self.enabled = False  # Disable after single execution
            self.status = ScheduleStatus.COMPLETED
        
        self.updated_at = datetime.now(timezone.utc)
    
    def record_execution(self, success: bool):
        """Record execution results."""
        now = datetime.now(timezone.utc)
        self.last_execution = now
        self.execution_count += 1
        
        if success:
            self.last_success = now
            self.success_count += 1
        else:
            self.last_failure = now
            self.failure_count += 1
        
        self.updated_at = now
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['schedule_type'] = self.schedule_type.value
        result['cleanup_mode'] = self.cleanup_mode.value
        result['priority'] = self.priority.value
        result['safety_level'] = self.safety_level.value
        result['status'] = self.status.value
        result['trigger_conditions'] = [tc.value for tc in self.trigger_conditions]
        
        # Convert datetime fields
        datetime_fields = [
            'created_at', 'updated_at', 'next_run_time', 'last_execution',
            'last_success', 'last_failure', 'expire_at'
        ]
        for field in datetime_fields:
            if result[field]:
                result[field] = result[field].isoformat()
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CleanupSchedule':
        """Create from dictionary."""
        data = data.copy()
        
        # Convert enums
        data['schedule_type'] = ScheduleType(data['schedule_type'])
        data['cleanup_mode'] = CleanupMode(data['cleanup_mode'])
        data['priority'] = CleanupPriority(data['priority'])
        data['safety_level'] = SafetyLevel(data['safety_level'])
        data['status'] = ScheduleStatus(data['status'])
        data['trigger_conditions'] = [TriggerCondition(tc) for tc in data['trigger_conditions']]
        
        # Convert datetime fields
        datetime_fields = [
            'created_at', 'updated_at', 'next_run_time', 'last_execution',
            'last_success', 'last_failure', 'expire_at'
        ]
        for field in datetime_fields:
            if data.get(field):
                data[field] = datetime.fromisoformat(data[field])
        
        return cls(**data)


@dataclass
class ScheduleExecutionResult:
    """Results from a scheduled cleanup execution."""
    schedule_id: str
    execution_id: str
    started_at: datetime
    completed_at: Optional[datetime]
    success: bool
    cleanup_job_id: Optional[str]
    cleanup_results: Optional[Dict[str, Any]]
    error_message: Optional[str]
    retry_attempt: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['started_at'] = self.started_at.isoformat()
        if result['completed_at']:
            result['completed_at'] = result['completed_at'].isoformat()
        return result


class CleanupScheduler:
    """
    Automated cleanup scheduler and orchestrator.
    
    Provides flexible scheduling capabilities with cron-like expressions,
    event-driven triggers, and intelligent cleanup coordination.
    """
    
    def __init__(self, config: Optional[MCPAutomationConfig] = None):
        """Initialize cleanup scheduler."""
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.cleanup_manager = CleanupManager(self.config)
        self.retention_policy_manager = RetentionPolicyManager(self.config)
        self.audit_logger = AuditLogger(self.config)
        
        # Schedule storage
        self.schedules: Dict[str, CleanupSchedule] = {}
        self.execution_results: Dict[str, ScheduleExecutionResult] = {}
        self.active_executions: Dict[str, asyncio.Task] = {}
        
        # Scheduler configuration
        self.check_interval_seconds = 60  # Check schedules every minute
        self.max_concurrent_executions = 5
        self.execution_timeout_multiplier = 1.2  # Add 20% buffer to timeouts
        
        # Scheduler control
        self.running = False
        self.scheduler_task: Optional[asyncio.Task] = None
        self.shutdown_event = asyncio.Event()
        
        # Statistics
        self.stats = {
            'scheduler_started_at': None,
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'active_schedules': 0,
            'last_check_time': None,
            'execution_queue_size': 0
        }
        
        # Load existing schedules
        self.load_schedules()
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("CleanupScheduler initialized", extra={
            'schedules_loaded': len(self.schedules),
            'check_interval': self.check_interval_seconds
        })
    
    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, initiating scheduler shutdown")
        asyncio.create_task(self.stop())
    
    async def start(self):
        """Start the cleanup scheduler."""
        if self.running:
            self.logger.warning("Scheduler is already running")
            return
        
        self.running = True
        self.shutdown_event.clear()
        self.stats['scheduler_started_at'] = datetime.now(timezone.utc).isoformat()
        
        # Start the main scheduler loop
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        
        # Log scheduler start
        await self.audit_logger.log_event(AuditEvent(
            event_type=AuditEventType.SYSTEM_ERROR,  # Using available enum
            component='cleanup_scheduler',
            user='system',
            action='start_scheduler',
            resource='scheduler',
            details={
                'active_schedules': len([s for s in self.schedules.values() if s.enabled]),
                'check_interval_seconds': self.check_interval_seconds
            }
        ))
        
        self.logger.info("Cleanup scheduler started", extra={
            'active_schedules': len([s for s in self.schedules.values() if s.enabled])
        })
    
    async def stop(self):
        """Stop the cleanup scheduler gracefully."""
        if not self.running:
            return
        
        self.logger.info("Stopping cleanup scheduler...")
        
        # Signal shutdown
        self.running = False
        self.shutdown_event.set()
        
        # Cancel scheduler task
        if self.scheduler_task and not self.scheduler_task.done():
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        
        # Wait for active executions to complete (with timeout)
        if self.active_executions:
            self.logger.info(f"Waiting for {len(self.active_executions)} active executions to complete...")
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self.active_executions.values(), return_exceptions=True),
                    timeout=300  # 5 minutes max wait
                )
            except asyncio.TimeoutError:
                self.logger.warning("Timeout waiting for executions to complete")
                for execution_id, task in self.active_executions.items():
                    if not task.done():
                        task.cancel()
                        self.logger.warning(f"Cancelled execution: {execution_id}")
        
        # Save schedules
        self.save_schedules()
        
        # Log scheduler stop
        await self.audit_logger.log_event(AuditEvent(
            event_type=AuditEventType.SYSTEM_ERROR,  # Using available enum
            component='cleanup_scheduler',
            user='system',
            action='stop_scheduler',
            resource='scheduler',
            details={
                'total_executions': self.stats['total_executions'],
                'successful_executions': self.stats['successful_executions'],
                'failed_executions': self.stats['failed_executions']
            }
        ))
        
        self.logger.info("Cleanup scheduler stopped")
    
    async def _scheduler_loop(self):
        """Main scheduler loop."""
        self.logger.info("Starting scheduler loop")
        
        try:
            while self.running and not self.shutdown_event.is_set():
                try:
                    # Check and execute due schedules
                    await self._check_and_execute_schedules()
                    
                    # Clean up completed executions
                    await self._cleanup_completed_executions()
                    
                    # Update statistics
                    self._update_statistics()
                    
                    # Wait for next check interval
                    try:
                        await asyncio.wait_for(
                            self.shutdown_event.wait(),
                            timeout=self.check_interval_seconds
                        )
                        break  # Shutdown event was set
                    except asyncio.TimeoutError:
                        continue  # Normal timeout, continue loop
                
                except Exception as e:
                    self.logger.error(f"Error in scheduler loop: {e}", exc_info=True)
                    # Wait a bit before retrying
                    await asyncio.sleep(min(self.check_interval_seconds, 60))
        
        except asyncio.CancelledError:
            self.logger.info("Scheduler loop cancelled")
        except Exception as e:
            self.logger.error(f"Fatal error in scheduler loop: {e}", exc_info=True)
        finally:
            self.logger.info("Scheduler loop exited")
    
    async def _check_and_execute_schedules(self):
        """Check schedules and execute those that are due."""
        current_time = datetime.now(timezone.utc)
        self.stats['last_check_time'] = current_time.isoformat()
        
        # Find schedules that should execute
        due_schedules = []
        for schedule in self.schedules.values():
            if schedule.should_execute_now():
                # Check if already executing
                if schedule.schedule_id not in self.active_executions:
                    due_schedules.append(schedule)
        
        if not due_schedules:
            return
        
        self.logger.info(f"Found {len(due_schedules)} due schedules")
        
        # Sort by priority (higher priority first)
        priority_order = {
            CleanupPriority.CRITICAL: 4,
            CleanupPriority.HIGH: 3,
            CleanupPriority.NORMAL: 2,
            CleanupPriority.LOW: 1
        }
        due_schedules.sort(key=lambda s: priority_order.get(s.priority, 0), reverse=True)
        
        # Execute schedules (respecting concurrency limits)
        executions_started = 0
        for schedule in due_schedules:
            if len(self.active_executions) >= self.max_concurrent_executions:
                self.logger.info("Maximum concurrent executions reached, queuing remaining schedules")
                break
            
            if executions_started >= 3:  # Limit per cycle to avoid overload
                break
            
            # Start execution
            await self._start_schedule_execution(schedule)
            executions_started += 1
    
    async def _start_schedule_execution(self, schedule: CleanupSchedule):
        """Start execution of a scheduled cleanup."""
        execution_id = f"{schedule.schedule_id}_{int(time.time())}"
        
        try:
            # Update schedule status
            schedule.status = ScheduleStatus.EXECUTING
            
            # Create execution result
            execution_result = ScheduleExecutionResult(
                schedule_id=schedule.schedule_id,
                execution_id=execution_id,
                started_at=datetime.now(timezone.utc),
                completed_at=None,
                success=False,
                cleanup_job_id=None,
                cleanup_results=None,
                error_message=None,
                retry_attempt=0
            )
            
            self.execution_results[execution_id] = execution_result
            
            # Start execution task
            execution_task = asyncio.create_task(
                self._execute_schedule(schedule, execution_result)
            )
            self.active_executions[execution_id] = execution_task
            
            # Log execution start
            await self.audit_logger.log_event(AuditEvent(
                event_type=AuditEventType.JOB_STARTED,
                component='cleanup_scheduler',
                user='scheduler',
                action='start_scheduled_cleanup',
                resource=schedule.schedule_id,
                details={
                    'schedule_name': schedule.schedule_name,
                    'execution_id': execution_id,
                    'cleanup_types': schedule.cleanup_types,
                    'target_servers': schedule.target_servers
                }
            ))
            
            self.logger.info(f"Started execution of schedule: {schedule.schedule_name}", extra={
                'schedule_id': schedule.schedule_id,
                'execution_id': execution_id
            })
        
        except Exception as e:
            self.logger.error(f"Failed to start schedule execution: {e}", exc_info=True)
            schedule.status = ScheduleStatus.FAILED
    
    async def _execute_schedule(
        self,
        schedule: CleanupSchedule,
        execution_result: ScheduleExecutionResult
    ):
        """Execute a cleanup schedule."""
        
        try:
            # Get retention policy
            retention_policy = None
            if schedule.retention_policy_id:
                retention_policy = self.retention_policy_manager.get_policy(schedule.retention_policy_id)
            
            if not retention_policy:
                retention_policy = self.retention_policy_manager.get_default_policy()
            
            # Create cleanup job
            job_id = await self.cleanup_manager.create_cleanup_job(
                job_name=f"Scheduled: {schedule.schedule_name}",
                servers=schedule.target_servers,
                cleanup_types=schedule.cleanup_types,
                mode=schedule.cleanup_mode,
                priority=schedule.priority,
                retention_policy=retention_policy,
                safety_level=schedule.safety_level,
                metadata={
                    'schedule_id': schedule.schedule_id,
                    'execution_id': execution_result.execution_id,
                    'scheduled_execution': True
                }
            )
            
            execution_result.cleanup_job_id = job_id
            
            # Execute cleanup job
            cleanup_result = await self.cleanup_manager.execute_cleanup_job(
                job_id, dry_run=schedule.dry_run
            )
            
            # Record results
            execution_result.cleanup_results = cleanup_result.to_dict()
            execution_result.success = cleanup_result.status.value == 'completed'
            execution_result.completed_at = datetime.now(timezone.utc)
            
            if not execution_result.success:
                execution_result.error_message = '; '.join(cleanup_result.errors)
            
            # Update schedule
            schedule.record_execution(execution_result.success)
            schedule.update_next_run_time()
            schedule.status = ScheduleStatus.ACTIVE if schedule.enabled else ScheduleStatus.INACTIVE
            
            # Update statistics
            self.stats['total_executions'] += 1
            if execution_result.success:
                self.stats['successful_executions'] += 1
            else:
                self.stats['failed_executions'] += 1
            
            # Log completion
            await self.audit_logger.log_event(AuditEvent(
                event_type=AuditEventType.JOB_COMPLETED if execution_result.success else AuditEventType.JOB_FAILED,
                component='cleanup_scheduler',
                user='scheduler',
                action='complete_scheduled_cleanup',
                resource=schedule.schedule_id,
                details={
                    'execution_id': execution_result.execution_id,
                    'success': execution_result.success,
                    'cleanup_job_id': job_id,
                    'items_cleaned': cleanup_result.items_cleaned if execution_result.success else 0,
                    'bytes_freed': cleanup_result.bytes_freed if execution_result.success else 0,
                    'error_message': execution_result.error_message
                }
            ))
            
            self.logger.info(f"Completed scheduled cleanup: {schedule.schedule_name}", extra={
                'success': execution_result.success,
                'items_cleaned': cleanup_result.items_cleaned if execution_result.success else 0,
                'bytes_freed': cleanup_result.bytes_freed if execution_result.success else 0
            })
        
        except Exception as e:
            execution_result.success = False
            execution_result.error_message = str(e)
            execution_result.completed_at = datetime.now(timezone.utc)
            
            schedule.record_execution(False)
            schedule.status = ScheduleStatus.FAILED
            
            self.stats['total_executions'] += 1
            self.stats['failed_executions'] += 1
            
            self.logger.error(f"Schedule execution failed: {schedule.schedule_name}: {e}", exc_info=True)
        
        finally:
            # Remove from active executions
            if execution_result.execution_id in self.active_executions:
                del self.active_executions[execution_result.execution_id]
    
    async def _cleanup_completed_executions(self):
        """Clean up completed execution records."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
        
        completed_executions = [
            exec_id for exec_id, result in self.execution_results.items()
            if result.completed_at and result.completed_at < cutoff_time
        ]
        
        for exec_id in completed_executions:
            del self.execution_results[exec_id]
        
        if completed_executions:
            self.logger.debug(f"Cleaned up {len(completed_executions)} old execution records")
    
    def _update_statistics(self):
        """Update scheduler statistics."""
        self.stats.update({
            'active_schedules': len([s for s in self.schedules.values() if s.enabled]),
            'execution_queue_size': len(self.active_executions),
            'total_schedules': len(self.schedules)
        })
    
    def create_schedule(
        self,
        schedule_name: str,
        schedule_type: ScheduleType,
        target_servers: List[str],
        cleanup_types: List[str] = None,
        cron_expression: Optional[str] = None,
        interval_seconds: Optional[int] = None,
        cleanup_mode: CleanupMode = CleanupMode.SAFE,
        **kwargs
    ) -> str:
        """
        Create a new cleanup schedule.
        
        Args:
            schedule_name: Human-readable schedule name
            schedule_type: Type of schedule (cron, interval, etc.)
            target_servers: List of servers to clean
            cleanup_types: Types of cleanup to perform
            cron_expression: Cron expression for CRON schedules
            interval_seconds: Interval for INTERVAL schedules
            cleanup_mode: Cleanup execution mode
            **kwargs: Additional schedule parameters
            
        Returns:
            Schedule ID
        """
        
        schedule_id = f"schedule_{int(time.time())}_{hash(schedule_name) % 10000:04d}"
        
        if cleanup_types is None:
            cleanup_types = ['versions', 'artifacts']
        
        schedule = CleanupSchedule(
            schedule_id=schedule_id,
            schedule_name=schedule_name,
            schedule_type=schedule_type,
            target_servers=target_servers,
            cleanup_types=cleanup_types,
            cron_expression=cron_expression,
            interval_seconds=interval_seconds,
            cleanup_mode=cleanup_mode,
            **kwargs
        )
        
        self.schedules[schedule_id] = schedule
        
        # Save schedules
        self.save_schedules()
        
        self.logger.info(f"Created cleanup schedule: {schedule_name}", extra={
            'schedule_id': schedule_id,
            'schedule_type': schedule_type.value,
            'target_servers': target_servers
        })
        
        return schedule_id
    
    def get_schedule(self, schedule_id: str) -> Optional[CleanupSchedule]:
        """Get a schedule by ID."""
        return self.schedules.get(schedule_id)
    
    def list_schedules(self, enabled_only: bool = False) -> List[CleanupSchedule]:
        """List all schedules."""
        schedules = list(self.schedules.values())
        
        if enabled_only:
            schedules = [s for s in schedules if s.enabled]
        
        return sorted(schedules, key=lambda s: s.schedule_name)
    
    def update_schedule(self, schedule_id: str, **updates) -> bool:
        """Update a schedule."""
        schedule = self.schedules.get(schedule_id)
        if not schedule:
            return False
        
        # Update fields
        for key, value in updates.items():
            if hasattr(schedule, key):
                setattr(schedule, key, value)
        
        schedule.updated_at = datetime.now(timezone.utc)
        
        # Recalculate next run time if timing changed
        if any(key in updates for key in ['cron_expression', 'interval_seconds', 'schedule_type']):
            schedule.next_run_time = schedule._calculate_next_run_time()
        
        self.save_schedules()
        
        self.logger.info(f"Updated schedule: {schedule.schedule_name}")
        return True
    
    def delete_schedule(self, schedule_id: str) -> bool:
        """Delete a schedule."""
        if schedule_id in self.schedules:
            schedule = self.schedules[schedule_id]
            del self.schedules[schedule_id]
            
            # Cancel any active execution
            for exec_id, task in list(self.active_executions.items()):
                if exec_id.startswith(schedule_id):
                    task.cancel()
                    del self.active_executions[exec_id]
            
            self.save_schedules()
            
            self.logger.info(f"Deleted schedule: {schedule.schedule_name}")
            return True
        return False
    
    def enable_schedule(self, schedule_id: str) -> bool:
        """Enable a schedule."""
        return self.update_schedule(schedule_id, enabled=True, status=ScheduleStatus.ACTIVE)
    
    def disable_schedule(self, schedule_id: str) -> bool:
        """Disable a schedule."""
        return self.update_schedule(schedule_id, enabled=False, status=ScheduleStatus.INACTIVE)
    
    async def trigger_schedule(self, schedule_id: str, force: bool = False) -> bool:
        """Manually trigger a schedule execution."""
        schedule = self.schedules.get(schedule_id)
        if not schedule:
            return False
        
        if not force and schedule_id in self.active_executions:
            self.logger.warning(f"Schedule {schedule.schedule_name} is already executing")
            return False
        
        # Temporarily override next_run_time to trigger execution
        original_next_run = schedule.next_run_time
        schedule.next_run_time = datetime.now(timezone.utc) - timedelta(seconds=1)
        
        try:
            await self._start_schedule_execution(schedule)
            return True
        finally:
            # Restore original next_run_time if execution failed to start
            if schedule_id not in self.active_executions:
                schedule.next_run_time = original_next_run
    
    def get_execution_results(self, schedule_id: Optional[str] = None) -> List[ScheduleExecutionResult]:
        """Get execution results, optionally filtered by schedule."""
        results = list(self.execution_results.values())
        
        if schedule_id:
            results = [r for r in results if r.schedule_id == schedule_id]
        
        return sorted(results, key=lambda r: r.started_at, reverse=True)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        stats = self.stats.copy()
        
        # Add schedule breakdown
        stats['schedule_breakdown'] = {}
        for schedule_type in ScheduleType:
            count = len([s for s in self.schedules.values() if s.schedule_type == schedule_type])
            stats['schedule_breakdown'][schedule_type.value] = count
        
        # Add status breakdown
        stats['status_breakdown'] = {}
        for status in ScheduleStatus:
            count = len([s for s in self.schedules.values() if s.status == status])
            stats['status_breakdown'][status.value] = count
        
        return stats
    
    def save_schedules(self, file_path: Optional[Path] = None):
        """Save schedules to file."""
        if file_path is None:
            file_path = self.config.paths.automation_root / "cleanup" / "schedules.json"
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        schedules_data = {
            'saved_at': datetime.now(timezone.utc).isoformat(),
            'schedules': {sid: schedule.to_dict() for sid, schedule in self.schedules.items()}
        }
        
        with open(file_path, 'w') as f:
            json.dump(schedules_data, f, indent=2)
        
        self.logger.debug(f"Saved {len(self.schedules)} schedules to {file_path}")
    
    def load_schedules(self, file_path: Optional[Path] = None):
        """Load schedules from file."""
        if file_path is None:
            file_path = self.config.paths.automation_root / "cleanup" / "schedules.json"
        
        if not file_path.exists():
            self.logger.info(f"No existing schedules file found at {file_path}")
            return
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            schedules_data = data.get('schedules', {})
            loaded_count = 0
            
            for schedule_id, schedule_dict in schedules_data.items():
                try:
                    schedule = CleanupSchedule.from_dict(schedule_dict)
                    self.schedules[schedule_id] = schedule
                    loaded_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to load schedule {schedule_id}: {e}")
            
            self.logger.info(f"Loaded {loaded_count} schedules from {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load schedules from {file_path}: {e}")


async def main():
    """Main entry point for cleanup scheduler testing."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        from ..config import get_config
        
        config = get_config()
        scheduler = CleanupScheduler(config)
        
        # Example: Create some test schedules
        print("Creating test schedules...")
        
        # Daily cleanup schedule
        daily_schedule_id = scheduler.create_schedule(
            schedule_name="Daily Cleanup",
            schedule_type=ScheduleType.CRON,
            cron_expression="0 2 * * *",  # 2 AM daily
            target_servers=['postgres', 'files'],
            cleanup_types=['versions', 'artifacts'],
            cleanup_mode=CleanupMode.SAFE,
            description="Daily automated cleanup of old versions and artifacts"
        )
        
        # Weekly aggressive cleanup
        weekly_schedule_id = scheduler.create_schedule(
            schedule_name="Weekly Deep Clean",
            schedule_type=ScheduleType.CRON,
            cron_expression="0 3 * * 0",  # 3 AM on Sundays
            target_servers=['postgres', 'files'],
            cleanup_types=['versions', 'artifacts', 'logs'],
            cleanup_mode=CleanupMode.AGGRESSIVE,
            safety_level=SafetyLevel.MEDIUM,
            description="Weekly comprehensive cleanup"
        )
        
        # Interval-based cleanup
        interval_schedule_id = scheduler.create_schedule(
            schedule_name="Hourly Temp Cleanup",
            schedule_type=ScheduleType.INTERVAL,
            interval_seconds=3600,  # Every hour
            target_servers=['postgres', 'files'],
            cleanup_types=['artifacts'],
            dry_run=True,
            description="Hourly temporary file cleanup (dry run)"
        )
        
        # List schedules
        print(f"\nCreated schedules:")
        for schedule in scheduler.list_schedules():
            print(f"- {schedule.schedule_name} ({schedule.schedule_type.value})")
            print(f"  Next run: {schedule.next_run_time}")
            print(f"  Targets: {', '.join(schedule.target_servers)}")
        
        # Test manual trigger
        print(f"\nManually triggering schedule: {daily_schedule_id}")
        success = await scheduler.trigger_schedule(daily_schedule_id)
        print(f"Trigger success: {success}")
        
        # Wait a moment
        await asyncio.sleep(2)
        
        # Get statistics
        print(f"\nScheduler statistics:")
        stats = scheduler.get_statistics()
        print(json.dumps(stats, indent=2, default=str))
        
        # Start scheduler briefly
        print("\nStarting scheduler for 10 seconds...")
        await scheduler.start()
        await asyncio.sleep(10)
        await scheduler.stop()
        
        print("Scheduler test completed")
        
    except Exception as e:
        logging.error(f"Error in cleanup scheduler: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())