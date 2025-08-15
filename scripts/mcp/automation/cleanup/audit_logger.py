#!/usr/bin/env python3
"""
MCP Audit Logger

Comprehensive audit logging for all MCP cleanup operations. Provides detailed
tracking, compliance reporting, and forensic capabilities for cleanup activities
with multiple output formats and retention management.

Author: Claude AI Assistant (garbage-collector.md)
Created: 2025-08-15 21:45:00 UTC
Version: 1.0.0
"""

import asyncio
import json
import csv
import hashlib
import gzip
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, TextIO
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
import threading
import queue
import uuid

# Import parent automation components
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import get_config, MCPAutomationConfig


class AuditEventType(Enum):
    """Types of audit events."""
    # Job lifecycle events
    JOB_CREATED = "job_created"
    JOB_STARTED = "job_started"
    JOB_COMPLETED = "job_completed"
    JOB_FAILED = "job_failed"
    JOB_CANCELLED = "job_cancelled"
    
    # Analysis events
    ANALYSIS_STARTED = "analysis_started"
    ANALYSIS_COMPLETED = "analysis_completed"
    ANALYSIS_FAILED = "analysis_failed"
    
    # Cleanup events
    CLEANUP_STARTED = "cleanup_started"
    CLEANUP_COMPLETED = "cleanup_completed"
    CLEANUP_FAILED = "cleanup_failed"
    
    # Item-specific events
    VERSION_REMOVED = "version_removed"
    ARTIFACT_REMOVED = "artifact_removed"
    FILE_DELETED = "file_deleted"
    DIRECTORY_REMOVED = "directory_removed"
    
    # Safety and validation events
    SAFETY_CHECK_PERFORMED = "safety_check_performed"
    SAFETY_VIOLATION_DETECTED = "safety_violation_detected"
    VALIDATION_PASSED = "validation_passed"
    VALIDATION_FAILED = "validation_failed"
    
    # Policy events
    POLICY_APPLIED = "policy_applied"
    POLICY_OVERRIDE = "policy_override"
    RETENTION_POLICY_ENFORCED = "retention_policy_enforced"
    
    # System events
    SYSTEM_ERROR = "system_error"
    CONFIGURATION_CHANGE = "configuration_change"
    BACKUP_CREATED = "backup_created"
    ROLLBACK_PERFORMED = "rollback_performed"


class AuditSeverity(Enum):
    """Severity levels for audit events."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Comprehensive audit event record."""
    # Core event information
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: AuditEventType = AuditEventType.SYSTEM_ERROR
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    severity: AuditSeverity = AuditSeverity.INFO
    
    # Context information
    component: str = "unknown"
    user: str = "system"
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    
    # Action details
    action: str = ""
    resource: str = ""
    resource_type: str = ""
    
    # Event details and metadata
    details: Dict[str, Any] = field(default_factory=dict)
    before_state: Optional[Dict[str, Any]] = None
    after_state: Optional[Dict[str, Any]] = None
    
    # Security and compliance
    checksum: Optional[str] = None
    digital_signature: Optional[str] = None
    
    def __post_init__(self):
        """Calculate checksum for integrity verification."""
        if self.checksum is None:
            self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate SHA-256 checksum of event data."""
        # Create deterministic string representation
        data_to_hash = {
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'component': self.component,
            'user': self.user,
            'action': self.action,
            'resource': self.resource,
            'details': json.dumps(self.details, sort_keys=True, default=str)
        }
        
        hash_string = json.dumps(data_to_hash, sort_keys=True)
        return hashlib.sha256(hash_string.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['event_type'] = self.event_type.value
        result['severity'] = self.severity.value
        result['timestamp'] = self.timestamp.isoformat()
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)
    
    def to_csv_row(self) -> List[str]:
        """Convert to CSV row."""
        return [
            self.event_id,
            self.event_type.value,
            self.timestamp.isoformat(),
            self.severity.value,
            self.component,
            self.user,
            self.action,
            self.resource,
            json.dumps(self.details, default=str),
            self.checksum or ""
        ]
    
    @classmethod
    def csv_headers(cls) -> List[str]:
        """Get CSV headers."""
        return [
            'event_id', 'event_type', 'timestamp', 'severity',
            'component', 'user', 'action', 'resource', 'details', 'checksum'
        ]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditEvent':
        """Create from dictionary."""
        data = data.copy()
        data['event_type'] = AuditEventType(data['event_type'])
        data['severity'] = AuditSeverity(data['severity'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)
    
    def verify_integrity(self) -> bool:
        """Verify event integrity using checksum."""
        expected_checksum = self._calculate_checksum()
        return self.checksum == expected_checksum


class AuditLogFormat(Enum):
    """Supported audit log formats."""
    JSON = "json"
    JSON_LINES = "jsonl"
    CSV = "csv"
    STRUCTURED_TEXT = "txt"


class AuditLogRotation:
    """Audit log rotation configuration."""
    
    def __init__(
        self,
        max_size_mb: int = 100,
        max_files: int = 10,
        rotation_interval_hours: int = 24,
        compress_rotated: bool = True
    ):
        self.max_size_mb = max_size_mb
        self.max_files = max_files
        self.rotation_interval_hours = rotation_interval_hours
        self.compress_rotated = compress_rotated


class AuditLogger:
    """
    Comprehensive audit logger for MCP cleanup operations.
    
    Provides detailed tracking, compliance reporting, and forensic
    capabilities with multiple output formats and retention management.
    """
    
    def __init__(self, config: MCPAutomationConfig):
        """Initialize audit logger."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Audit configuration
        self.audit_enabled = True
        self.log_formats = [AuditLogFormat.JSON_LINES, AuditLogFormat.CSV]
        self.audit_retention_days = 90
        self.buffer_size = 1000
        self.sync_interval_seconds = 30
        
        # Log rotation configuration
        self.rotation_config = AuditLogRotation()
        
        # Audit storage paths
        self.audit_root = self.config.paths.logs_root / "audit"
        self.audit_root.mkdir(parents=True, exist_ok=True)
        
        # Log files for different formats
        self.log_files = {}
        self._initialize_log_files()
        
        # Event buffering and async processing
        self.event_buffer = queue.Queue(maxsize=self.buffer_size)
        self.processing_thread = None
        self.shutdown_event = threading.Event()
        
        # Session and correlation tracking
        self.current_session_id = str(uuid.uuid4())
        self.correlation_stack = []
        
        # Statistics and monitoring
        self.stats = {
            'events_logged': 0,
            'events_failed': 0,
            'bytes_written': 0,
            'rotations_performed': 0,
            'last_rotation': None,
            'buffer_overflows': 0
        }
        
        # Start background processing
        self._start_background_processing()
        
        self.logger.info("AuditLogger initialized", extra={
            'audit_root': str(self.audit_root),
            'formats': [f.value for f in self.log_formats],
            'retention_days': self.audit_retention_days
        })
    
    def _initialize_log_files(self):
        """Initialize log files for different formats."""
        
        for log_format in self.log_formats:
            if log_format == AuditLogFormat.JSON_LINES:
                file_path = self.audit_root / "audit.jsonl"
            elif log_format == AuditLogFormat.CSV:
                file_path = self.audit_root / "audit.csv"
            elif log_format == AuditLogFormat.JSON:
                file_path = self.audit_root / "audit.json"
            else:
                file_path = self.audit_root / "audit.txt"
            
            self.log_files[log_format] = {
                'path': file_path,
                'handle': None,
                'size_bytes': file_path.stat().st_size if file_path.exists() else 0,
                'created_at': datetime.now(timezone.utc)
            }
            
            # Initialize CSV with headers if needed
            if log_format == AuditLogFormat.CSV and not file_path.exists():
                self._write_csv_headers(file_path)
    
    def _write_csv_headers(self, file_path: Path):
        """Write CSV headers to a new file."""
        try:
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(AuditEvent.csv_headers())
        except Exception as e:
            self.logger.error(f"Failed to write CSV headers to {file_path}: {e}")
    
    def _start_background_processing(self):
        """Start background thread for processing audit events."""
        
        def process_events():
            """Background event processing loop."""
            while not self.shutdown_event.is_set():
                try:
                    # Process events from buffer
                    events_to_process = []
                    
                    # Collect events with timeout
                    try:
                        # Get first event with timeout
                        event = self.event_buffer.get(timeout=self.sync_interval_seconds)
                        events_to_process.append(event)
                        
                        # Get additional events without blocking
                        while len(events_to_process) < 100:  # Process in batches
                            try:
                                event = self.event_buffer.get_nowait()
                                events_to_process.append(event)
                            except queue.Empty:
                                break
                    
                    except queue.Empty:
                        continue  # Timeout, check for shutdown
                    
                    # Write events to log files
                    if events_to_process:
                        self._write_events_to_files(events_to_process)
                    
                    # Check for log rotation
                    self._check_log_rotation()
                    
                    # Clean up old log files
                    self._cleanup_old_logs()
                
                except Exception as e:
                    self.logger.error(f"Error in audit event processing: {e}", exc_info=True)
        
        self.processing_thread = threading.Thread(target=process_events, daemon=True)
        self.processing_thread.start()
    
    async def log_event(self, event: AuditEvent) -> bool:
        """
        Log an audit event asynchronously.
        
        Args:
            event: AuditEvent to log
            
        Returns:
            True if event was queued successfully, False otherwise
        """
        
        if not self.audit_enabled:
            return True
        
        try:
            # Set session and correlation context
            if not event.session_id:
                event.session_id = self.current_session_id
            
            if not event.correlation_id and self.correlation_stack:
                event.correlation_id = self.correlation_stack[-1]
            
            # Ensure checksum is calculated
            if not event.checksum:
                event.checksum = event._calculate_checksum()
            
            # Add to processing queue
            try:
                self.event_buffer.put_nowait(event)
                self.stats['events_logged'] += 1
                return True
            except queue.Full:
                self.stats['buffer_overflows'] += 1
                self.logger.warning("Audit event buffer overflow, dropping event")
                return False
        
        except Exception as e:
            self.stats['events_failed'] += 1
            self.logger.error(f"Failed to queue audit event: {e}", exc_info=True)
            return False
    
    def _write_events_to_files(self, events: List[AuditEvent]):
        """Write events to all configured log files."""
        
        for log_format, file_info in self.log_files.items():
            try:
                self._write_events_to_format(events, log_format, file_info)
            except Exception as e:
                self.logger.error(f"Failed to write events to {log_format.value} format: {e}")
    
    def _write_events_to_format(
        self,
        events: List[AuditEvent],
        log_format: AuditLogFormat,
        file_info: Dict[str, Any]
    ):
        """Write events to a specific format file."""
        
        file_path = file_info['path']
        
        try:
            with open(file_path, 'a', encoding='utf-8') as f:
                if log_format == AuditLogFormat.JSON_LINES:
                    for event in events:
                        line = event.to_json() + '\n'
                        f.write(line)
                        file_info['size_bytes'] += len(line.encode('utf-8'))
                
                elif log_format == AuditLogFormat.CSV:
                    writer = csv.writer(f)
                    for event in events:
                        row = event.to_csv_row()
                        writer.writerow(row)
                        # Estimate size increase
                        file_info['size_bytes'] += len(','.join(row).encode('utf-8')) + 1
                
                elif log_format == AuditLogFormat.JSON:
                    # For JSON format, we maintain an array (more complex to append)
                    self._append_to_json_file(events, file_path, file_info)
                
                elif log_format == AuditLogFormat.STRUCTURED_TEXT:
                    for event in events:
                        line = self._format_structured_text(event) + '\n'
                        f.write(line)
                        file_info['size_bytes'] += len(line.encode('utf-8'))
                
                f.flush()
                self.stats['bytes_written'] += sum(len(e.to_json().encode('utf-8')) for e in events)
        
        except Exception as e:
            self.logger.error(f"Failed to write to {file_path}: {e}")
            raise
    
    def _append_to_json_file(
        self,
        events: List[AuditEvent],
        file_path: Path,
        file_info: Dict[str, Any]
    ):
        """Append events to JSON array file."""
        
        # For simplicity, we'll append as separate JSON objects with newlines
        # This makes it easier to append without rewriting the entire file
        with open(file_path, 'a', encoding='utf-8') as f:
            for event in events:
                line = event.to_json() + '\n'
                f.write(line)
                file_info['size_bytes'] += len(line.encode('utf-8'))
    
    def _format_structured_text(self, event: AuditEvent) -> str:
        """Format event as structured text."""
        
        return (
            f"[{event.timestamp.isoformat()}] "
            f"{event.severity.value.upper()} "
            f"{event.component}:{event.action} "
            f"user={event.user} "
            f"resource={event.resource} "
            f"type={event.event_type.value} "
            f"id={event.event_id} "
            f"details={json.dumps(event.details, default=str)}"
        )
    
    def _check_log_rotation(self):
        """Check if log rotation is needed."""
        
        current_time = datetime.now(timezone.utc)
        
        for log_format, file_info in self.log_files.items():
            should_rotate = False
            
            # Check size-based rotation
            if file_info['size_bytes'] > self.rotation_config.max_size_mb * 1024 * 1024:
                should_rotate = True
            
            # Check time-based rotation
            time_since_creation = current_time - file_info['created_at']
            if time_since_creation.total_seconds() > self.rotation_config.rotation_interval_hours * 3600:
                should_rotate = True
            
            if should_rotate:
                self._rotate_log_file(log_format, file_info)
    
    def _rotate_log_file(self, log_format: AuditLogFormat, file_info: Dict[str, Any]):
        """Rotate a log file."""
        
        try:
            file_path = file_info['path']
            
            # Generate rotated filename with timestamp
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            rotated_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
            rotated_path = file_path.parent / rotated_name
            
            # Move current log to rotated name
            if file_path.exists():
                file_path.rename(rotated_path)
                
                # Compress if configured
                if self.rotation_config.compress_rotated:
                    compressed_path = rotated_path.with_suffix(rotated_path.suffix + '.gz')
                    with open(rotated_path, 'rb') as f_in:
                        with gzip.open(compressed_path, 'wb') as f_out:
                            f_out.writelines(f_in)
                    rotated_path.unlink()  # Remove uncompressed file
            
            # Reset file info
            file_info['size_bytes'] = 0
            file_info['created_at'] = datetime.now(timezone.utc)
            
            # Initialize new file
            if log_format == AuditLogFormat.CSV:
                self._write_csv_headers(file_path)
            
            self.stats['rotations_performed'] += 1
            self.stats['last_rotation'] = datetime.now(timezone.utc).isoformat()
            
            self.logger.info(f"Rotated log file: {file_path}")
            
            # Clean up old rotated files
            self._cleanup_rotated_files(file_path)
        
        except Exception as e:
            self.logger.error(f"Failed to rotate log file {file_path}: {e}")
    
    def _cleanup_rotated_files(self, base_path: Path):
        """Clean up old rotated files."""
        
        try:
            pattern = f"{base_path.stem}_*{base_path.suffix}"
            rotated_files = list(base_path.parent.glob(pattern))
            
            # Also check for compressed files
            compressed_pattern = f"{base_path.stem}_*{base_path.suffix}.gz"
            rotated_files.extend(base_path.parent.glob(compressed_pattern))
            
            # Sort by modification time (oldest first)
            rotated_files.sort(key=lambda p: p.stat().st_mtime)
            
            # Remove files beyond max_files limit
            files_to_remove = rotated_files[:-self.rotation_config.max_files]
            
            for file_to_remove in files_to_remove:
                file_to_remove.unlink()
                self.logger.debug(f"Removed old rotated log: {file_to_remove}")
        
        except Exception as e:
            self.logger.error(f"Failed to cleanup rotated files: {e}")
    
    def _cleanup_old_logs(self):
        """Clean up logs older than retention period."""
        
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(days=self.audit_retention_days)
            cutoff_timestamp = cutoff_time.timestamp()
            
            for log_file in self.audit_root.rglob('*'):
                if log_file.is_file():
                    file_mtime = log_file.stat().st_mtime
                    if file_mtime < cutoff_timestamp:
                        log_file.unlink()
                        self.logger.debug(f"Removed expired log: {log_file}")
        
        except Exception as e:
            self.logger.error(f"Failed to cleanup old logs: {e}")
    
    def start_correlation(self, correlation_id: Optional[str] = None) -> str:
        """Start a new correlation context."""
        
        if correlation_id is None:
            correlation_id = str(uuid.uuid4())
        
        self.correlation_stack.append(correlation_id)
        return correlation_id
    
    def end_correlation(self) -> Optional[str]:
        """End the current correlation context."""
        
        if self.correlation_stack:
            return self.correlation_stack.pop()
        return None
    
    def new_session(self) -> str:
        """Start a new session."""
        
        self.current_session_id = str(uuid.uuid4())
        return self.current_session_id
    
    async def query_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[List[AuditEventType]] = None,
        users: Optional[List[str]] = None,
        components: Optional[List[str]] = None,
        resources: Optional[List[str]] = None,
        limit: int = 1000
    ) -> List[AuditEvent]:
        """
        Query audit events with filtering.
        
        Args:
            start_time: Filter events after this time
            end_time: Filter events before this time
            event_types: Filter by event types
            users: Filter by users
            components: Filter by components
            resources: Filter by resources
            limit: Maximum number of events to return
            
        Returns:
            List of matching audit events
        """
        
        events = []
        
        try:
            # Read from JSONL file (most efficient for querying)
            jsonl_file = self.log_files[AuditLogFormat.JSON_LINES]['path']
            
            if not jsonl_file.exists():
                return events
            
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if len(events) >= limit:
                        break
                    
                    try:
                        event_data = json.loads(line.strip())
                        event = AuditEvent.from_dict(event_data)
                        
                        # Apply filters
                        if start_time and event.timestamp < start_time:
                            continue
                        if end_time and event.timestamp > end_time:
                            continue
                        if event_types and event.event_type not in event_types:
                            continue
                        if users and event.user not in users:
                            continue
                        if components and event.component not in components:
                            continue
                        if resources and event.resource not in resources:
                            continue
                        
                        events.append(event)
                    
                    except json.JSONDecodeError:
                        continue  # Skip malformed lines
        
        except Exception as e:
            self.logger.error(f"Failed to query audit events: {e}")
        
        return events
    
    async def generate_compliance_report(
        self,
        start_time: datetime,
        end_time: datetime,
        output_format: str = 'json'
    ) -> Dict[str, Any]:
        """
        Generate compliance report for a time period.
        
        Args:
            start_time: Report start time
            end_time: Report end time
            output_format: Output format ('json', 'csv', 'html')
            
        Returns:
            Compliance report data
        """
        
        try:
            # Query all events in the time period
            events = await self.query_events(start_time=start_time, end_time=end_time)
            
            # Analyze events for compliance metrics
            report = {
                'report_generated_at': datetime.now(timezone.utc).isoformat(),
                'period': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat()
                },
                'summary': {
                    'total_events': len(events),
                    'unique_users': len(set(e.user for e in events)),
                    'unique_resources': len(set(e.resource for e in events)),
                    'unique_components': len(set(e.component for e in events))
                },
                'event_breakdown': {},
                'user_activity': {},
                'component_activity': {},
                'integrity_verification': {
                    'total_events_checked': 0,
                    'integrity_violations': 0,
                    'verification_success_rate': 100.0
                },
                'security_events': [],
                'errors_and_warnings': []
            }
            
            # Event type breakdown
            event_type_counts = {}
            for event in events:
                event_type = event.event_type.value
                event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
            report['event_breakdown'] = event_type_counts
            
            # User activity breakdown
            user_activity = {}
            for event in events:
                user = event.user
                if user not in user_activity:
                    user_activity[user] = {'total_events': 0, 'event_types': {}}
                user_activity[user]['total_events'] += 1
                event_type = event.event_type.value
                user_activity[user]['event_types'][event_type] = user_activity[user]['event_types'].get(event_type, 0) + 1
            report['user_activity'] = user_activity
            
            # Component activity breakdown
            component_activity = {}
            for event in events:
                component = event.component
                if component not in component_activity:
                    component_activity[component] = {'total_events': 0, 'event_types': {}}
                component_activity[component]['total_events'] += 1
                event_type = event.event_type.value
                component_activity[component]['event_types'][event_type] = component_activity[component]['event_types'].get(event_type, 0) + 1
            report['component_activity'] = component_activity
            
            # Integrity verification
            integrity_violations = 0
            for event in events:
                if not event.verify_integrity():
                    integrity_violations += 1
            
            report['integrity_verification'] = {
                'total_events_checked': len(events),
                'integrity_violations': integrity_violations,
                'verification_success_rate': ((len(events) - integrity_violations) / len(events) * 100) if events else 100.0
            }
            
            # Security events (errors, warnings, violations)
            security_events = [
                event for event in events
                if event.severity in [AuditSeverity.WARNING, AuditSeverity.ERROR, AuditSeverity.CRITICAL]
                or event.event_type in [AuditEventType.SAFETY_VIOLATION_DETECTED, AuditEventType.VALIDATION_FAILED]
            ]
            report['security_events'] = [event.to_dict() for event in security_events[:100]]  # Limit to 100
            
            # Errors and warnings summary
            errors_and_warnings = []
            for event in events:
                if event.severity in [AuditSeverity.WARNING, AuditSeverity.ERROR, AuditSeverity.CRITICAL]:
                    errors_and_warnings.append({
                        'timestamp': event.timestamp.isoformat(),
                        'severity': event.severity.value,
                        'component': event.component,
                        'action': event.action,
                        'resource': event.resource,
                        'details': event.details
                    })
            report['errors_and_warnings'] = errors_and_warnings[:50]  # Limit to 50
            
            return report
        
        except Exception as e:
            self.logger.error(f"Failed to generate compliance report: {e}")
            return {
                'error': f"Failed to generate report: {e}",
                'report_generated_at': datetime.now(timezone.utc).isoformat()
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get audit logging statistics."""
        
        stats = self.stats.copy()
        
        # Add current buffer status
        stats.update({
            'buffer_size': self.event_buffer.qsize(),
            'buffer_capacity': self.buffer_size,
            'audit_enabled': self.audit_enabled,
            'current_session_id': self.current_session_id,
            'correlation_stack_depth': len(self.correlation_stack),
            'log_formats': [f.value for f in self.log_formats],
            'audit_retention_days': self.audit_retention_days
        })
        
        # Add file information
        stats['log_files'] = {}
        for log_format, file_info in self.log_files.items():
            stats['log_files'][log_format.value] = {
                'path': str(file_info['path']),
                'size_bytes': file_info['size_bytes'],
                'size_mb': round(file_info['size_bytes'] / (1024 * 1024), 2),
                'created_at': file_info['created_at'].isoformat()
            }
        
        return stats
    
    def shutdown(self):
        """Shutdown audit logger gracefully."""
        
        self.logger.info("Shutting down audit logger...")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Process remaining events in buffer
        remaining_events = []
        while not self.event_buffer.empty():
            try:
                event = self.event_buffer.get_nowait()
                remaining_events.append(event)
            except queue.Empty:
                break
        
        if remaining_events:
            self.logger.info(f"Processing {len(remaining_events)} remaining audit events")
            self._write_events_to_files(remaining_events)
        
        # Wait for processing thread to finish
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=10)
        
        # Close any open file handles
        for file_info in self.log_files.values():
            if file_info.get('handle'):
                file_info['handle'].close()
        
        self.logger.info("Audit logger shutdown complete")


async def main():
    """Main entry point for audit logger testing."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        from ..config import get_config
        
        config = get_config()
        audit_logger = AuditLogger(config)
        
        # Example: Log some audit events
        print("Testing audit logging...")
        
        # Start a correlation context
        correlation_id = audit_logger.start_correlation("test_cleanup_session")
        
        # Log various types of events
        events = [
            AuditEvent(
                event_type=AuditEventType.JOB_CREATED,
                component='cleanup_manager',
                user='test_user',
                action='create_cleanup_job',
                resource='test_server',
                details={'job_name': 'Test Cleanup', 'cleanup_types': ['versions']}
            ),
            AuditEvent(
                event_type=AuditEventType.ANALYSIS_STARTED,
                component='version_cleanup',
                user='test_user',
                action='analyze_versions',
                resource='test_server',
                details={'version_count': 5}
            ),
            AuditEvent(
                event_type=AuditEventType.VERSION_REMOVED,
                component='version_cleanup',
                user='test_user',
                action='remove_version',
                resource='test_server:v1.0.0',
                details={'version': 'v1.0.0', 'size_bytes': 1024000}
            )
        ]
        
        for event in events:
            await audit_logger.log_event(event)
        
        # Wait a moment for background processing
        await asyncio.sleep(2)
        
        # End correlation context
        audit_logger.end_correlation()
        
        # Query events
        print("\nQuerying audit events...")
        queried_events = await audit_logger.query_events(
            event_types=[AuditEventType.JOB_CREATED, AuditEventType.VERSION_REMOVED]
        )
        
        print(f"Found {len(queried_events)} events")
        for event in queried_events:
            print(f"- {event.event_type.value}: {event.action} on {event.resource}")
        
        # Generate compliance report
        print("\nGenerating compliance report...")
        start_time = datetime.now(timezone.utc) - timedelta(hours=1)
        end_time = datetime.now(timezone.utc)
        
        report = await audit_logger.generate_compliance_report(start_time, end_time)
        print(f"Report summary: {report['summary']}")
        
        # Get statistics
        print("\nAudit statistics:")
        stats = audit_logger.get_statistics()
        print(json.dumps(stats, indent=2, default=str))
        
        # Shutdown
        audit_logger.shutdown()
        
    except Exception as e:
        logging.error(f"Error in audit logger: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())