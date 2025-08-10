"""
Data Audit Logger
================

Comprehensive audit logging system for data governance activities.
Tracks all data access, modifications, and governance actions for compliance.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import uuid


class AuditEventType(Enum):
    """Types of audit events"""
    DATA_ACCESS = "data_access"
    DATA_CREATION = "data_creation"
    DATA_MODIFICATION = "data_modification"
    DATA_DELETION = "data_deletion"
    DATA_EXPORT = "data_export"
    DATA_CLASSIFICATION = "data_classification"
    POLICY_CHANGE = "policy_change"
    LIFECYCLE_ACTION = "lifecycle_action"
    COMPLIANCE_CHECK = "compliance_check"
    QUALITY_ASSESSMENT = "quality_assessment"
    SECURITY_EVENT = "security_event"
    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"


class AuditSeverity(Enum):
    """Severity levels for audit events"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Represents a single audit event"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    event_type: AuditEventType = AuditEventType.SYSTEM_EVENT
    severity: AuditSeverity = AuditSeverity.INFO
    
    # Core event data
    source: str = ""  # System, user, or service that triggered the event
    target: str = ""  # What was acted upon (data asset, policy, etc.)
    action: str = ""  # What action was performed
    description: str = ""
    
    # Context information
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    # Data classification context
    data_classification: Optional[str] = None
    data_types: List[str] = field(default_factory=list)
    
    # Results and outcomes
    success: bool = True
    error_message: Optional[str] = None
    before_state: Optional[Dict[str, Any]] = None
    after_state: Optional[Dict[str, Any]] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    # Integrity protection
    checksum: Optional[str] = None
    
    def __post_init__(self):
        """Calculate checksum after initialization"""
        self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate integrity checksum for the audit event"""
        # Create deterministic string representation
        content = {
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type.value,
            'source': self.source,
            'target': self.target,
            'action': self.action,
            'description': self.description,
            'success': self.success
        }
        
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify the integrity of the audit event"""
        current_checksum = self.checksum
        self.checksum = None  # Temporarily remove for recalculation
        expected_checksum = self._calculate_checksum()
        self.checksum = current_checksum
        
        return current_checksum == expected_checksum


@dataclass
class AuditFilter:
    """Filter criteria for audit log queries"""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    event_types: Optional[List[AuditEventType]] = None
    severities: Optional[List[AuditSeverity]] = None
    sources: Optional[List[str]] = None
    targets: Optional[List[str]] = None
    user_ids: Optional[List[str]] = None
    data_classifications: Optional[List[str]] = None
    success_only: Optional[bool] = None
    tags: Optional[List[str]] = None
    limit: int = 1000


class DataAuditLogger:
    """
    Comprehensive audit logging system for data governance.
    Provides secure, tamper-evident logging of all data-related activities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger("data_audit")
        
        # Storage configuration
        self.audit_log_path = Path(self.config.get('audit_log_path', '/data/audit'))
        self.audit_log_path.mkdir(parents=True, exist_ok=True)
        
        # Retention settings
        self.retention_days = self.config.get('retention_days', 2555)  # 7 years default
        self.archive_after_days = self.config.get('archive_after_days', 365)
        
        # Performance settings
        self.batch_size = self.config.get('batch_size', 1000)
        self.flush_interval_seconds = self.config.get('flush_interval_seconds', 60)
        
        # Security settings
        self.enable_encryption = self.config.get('enable_encryption', True)
        self.enable_compression = self.config.get('enable_compression', True)
        
        # In-memory storage for recent events (for performance)
        self.event_buffer: List[AuditEvent] = []
        self.buffer_lock = asyncio.Lock()
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
    
    async def initialize(self) -> bool:
        """Initialize the audit logger"""
        try:
            self.logger.info("Initializing data audit logger")
            
            # Start background tasks
            await self._start_background_tasks()
            
            # Log initialization
            await self.log_event(AuditEvent(
                event_type=AuditEventType.SYSTEM_EVENT,
                source="audit_logger",
                action="initialize",
                description="Data audit logger initialized",
                metadata={"config": self.config}
            ))
            
            self.logger.info("Data audit logger initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize audit logger: {e}")
            return False
    
    async def log_event(self, event: AuditEvent):
        """Log a single audit event"""
        try:
            async with self.buffer_lock:
                self.event_buffer.append(event)
                
                # Flush buffer if it's getting large
                if len(self.event_buffer) >= self.batch_size:
                    await self._flush_buffer()
            
        except Exception as e:
            self.logger.error(f"Failed to log audit event: {e}")
    
    async def log_data_access(self, user_id: str, data_id: str, access_type: str,
                            classification: Optional[str] = None,
                            ip_address: Optional[str] = None,
                            success: bool = True,
                            additional_info: Optional[Dict[str, Any]] = None):
        """Log data access event"""
        
        event = AuditEvent(
            event_type=AuditEventType.DATA_ACCESS,
            severity=AuditSeverity.INFO if success else AuditSeverity.WARNING,
            source=f"user:{user_id}",
            target=f"data:{data_id}",
            action=access_type,
            description=f"User {user_id} {access_type} data {data_id}",
            user_id=user_id,
            ip_address=ip_address,
            data_classification=classification,
            success=success,
            metadata=additional_info or {}
        )
        
        # Add tags based on classification
        if classification:
            event.tags.append(f"classification:{classification}")
            if classification in ['confidential', 'restricted']:
                event.severity = AuditSeverity.WARNING  # Elevated importance
        
        await self.log_event(event)
    
    async def log_data_modification(self, user_id: str, data_id: str, 
                                  modification_type: str,
                                  before_state: Optional[Dict[str, Any]] = None,
                                  after_state: Optional[Dict[str, Any]] = None,
                                  classification: Optional[str] = None,
                                  success: bool = True):
        """Log data modification event"""
        
        event = AuditEvent(
            event_type=AuditEventType.DATA_MODIFICATION,
            severity=AuditSeverity.INFO if success else AuditSeverity.ERROR,
            source=f"user:{user_id}",
            target=f"data:{data_id}",
            action=modification_type,
            description=f"User {user_id} {modification_type} data {data_id}",
            user_id=user_id,
            data_classification=classification,
            success=success,
            before_state=before_state,
            after_state=after_state
        )
        
        if classification:
            event.tags.append(f"classification:{classification}")
        
        await self.log_event(event)
    
    async def log_data_processing(self, data_id: str, processing_results: Dict[str, Any]):
        """Log data processing through governance pipeline"""
        
        event = AuditEvent(
            event_type=AuditEventType.DATA_CLASSIFICATION,
            source="governance_framework",
            target=f"data:{data_id}",
            action="process_governance_pipeline",
            description=f"Processed data {data_id} through governance pipeline",
            success=True,
            metadata=processing_results
        )
        
        # Add classification info if available
        if 'classification' in processing_results:
            classification_info = processing_results['classification']
            event.data_classification = classification_info.get('level')
            event.data_types = classification_info.get('data_types', [])
            
            if event.data_classification:
                event.tags.append(f"classification:{event.data_classification}")
        
        await self.log_event(event)
    
    async def log_policy_change(self, user_id: str, policy_name: str, 
                              change_type: str, changes: Dict[str, Any]):
        """Log governance policy changes"""
        
        event = AuditEvent(
            event_type=AuditEventType.POLICY_CHANGE,
            severity=AuditSeverity.WARNING,  # Policy changes are important
            source=f"user:{user_id}",
            target=f"policy:{policy_name}",
            action=change_type,
            description=f"User {user_id} {change_type} policy {policy_name}",
            user_id=user_id,
            success=True,
            metadata={"changes": changes},
            tags=["policy_change", "governance"]
        )
        
        await self.log_event(event)
    
    async def log_lifecycle_action(self, data_id: str, action: str, 
                                 policy_name: Optional[str] = None,
                                 success: bool = True,
                                 details: Optional[Dict[str, Any]] = None):
        """Log data lifecycle management actions"""
        
        event = AuditEvent(
            event_type=AuditEventType.LIFECYCLE_ACTION,
            severity=AuditSeverity.INFO if success else AuditSeverity.ERROR,
            source="lifecycle_manager",
            target=f"data:{data_id}",
            action=action,
            description=f"Lifecycle action {action} on data {data_id}",
            success=success,
            metadata={
                "policy": policy_name,
                "details": details or {}
            },
            tags=["lifecycle", "automation"]
        )
        
        await self.log_event(event)
    
    async def log_compliance_check(self, data_id: str, regulation: str,
                                 compliant: bool, details: Dict[str, Any]):
        """Log compliance assessment results"""
        
        event = AuditEvent(
            event_type=AuditEventType.COMPLIANCE_CHECK,
            severity=AuditSeverity.INFO if compliant else AuditSeverity.CRITICAL,
            source="compliance_manager",
            target=f"data:{data_id}",
            action=f"check_{regulation}_compliance",
            description=f"Compliance check for {regulation} on data {data_id}",
            success=compliant,
            metadata={
                "regulation": regulation,
                "compliance_details": details
            },
            tags=["compliance", regulation.lower()]
        )
        
        await self.log_event(event)
    
    async def log_security_event(self, event_type: str, description: str,
                               severity: AuditSeverity = AuditSeverity.WARNING,
                               user_id: Optional[str] = None,
                               ip_address: Optional[str] = None,
                               additional_info: Optional[Dict[str, Any]] = None):
        """Log security-related events"""
        
        event = AuditEvent(
            event_type=AuditEventType.SECURITY_EVENT,
            severity=severity,
            source="security_system" if not user_id else f"user:{user_id}",
            action=event_type,
            description=description,
            user_id=user_id,
            ip_address=ip_address,
            metadata=additional_info or {},
            tags=["security"]
        )
        
        await self.log_event(event)
    
    async def query_events(self, filter_criteria: AuditFilter) -> List[AuditEvent]:
        """Query audit events based on filter criteria"""
        try:
            # First flush any pending events
            async with self.buffer_lock:
                if self.event_buffer:
                    await self._flush_buffer()
            
            # Load events from storage (simplified - in production would query database)
            events = await self._load_events_from_storage(filter_criteria)
            
            return events
            
        except Exception as e:
            self.logger.error(f"Failed to query audit events: {e}")
            return []
    
    async def get_audit_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get audit statistics for the specified period"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        
        filter_criteria = AuditFilter(
            start_time=start_time,
            end_time=end_time,
            limit=10000  # Large limit for statistics
        )
        
        events = await self.query_events(filter_criteria)
        
        # Calculate statistics
        stats = {
            "total_events": len(events),
            "events_by_type": {},
            "events_by_severity": {},
            "events_by_day": {},
            "top_users": {},
            "top_targets": {},
            "security_events": 0,
            "compliance_violations": 0,
            "data_access_events": 0
        }
        
        for event in events:
            # Count by type
            event_type = event.event_type.value
            stats["events_by_type"][event_type] = stats["events_by_type"].get(event_type, 0) + 1
            
            # Count by severity
            severity = event.severity.value
            stats["events_by_severity"][severity] = stats["events_by_severity"].get(severity, 0) + 1
            
            # Count by day
            day_key = event.timestamp.strftime("%Y-%m-%d")
            stats["events_by_day"][day_key] = stats["events_by_day"].get(day_key, 0) + 1
            
            # Count by user
            if event.user_id:
                stats["top_users"][event.user_id] = stats["top_users"].get(event.user_id, 0) + 1
            
            # Count by target
            if event.target:
                stats["top_targets"][event.target] = stats["top_targets"].get(event.target, 0) + 1
            
            # Special counts
            if event.event_type == AuditEventType.SECURITY_EVENT:
                stats["security_events"] += 1
            
            if event.event_type == AuditEventType.COMPLIANCE_CHECK and not event.success:
                stats["compliance_violations"] += 1
            
            if event.event_type == AuditEventType.DATA_ACCESS:
                stats["data_access_events"] += 1
        
        # Sort top lists
        stats["top_users"] = dict(sorted(stats["top_users"].items(), key=lambda x: x[1], reverse=True)[:10])
        stats["top_targets"] = dict(sorted(stats["top_targets"].items(), key=lambda x: x[1], reverse=True)[:10])
        
        return stats
    
    async def _flush_buffer(self):
        """Flush event buffer to persistent storage"""
        if not self.event_buffer:
            return
        
        try:
            # In production, this would write to a database or file system
            # For now, we'll simulate by logging the count
            event_count = len(self.event_buffer)
            
            # Write events to storage (simplified implementation)
            await self._write_events_to_storage(self.event_buffer)
            
            self.event_buffer.clear()
            self.logger.debug(f"Flushed {event_count} audit events to storage")
            
        except Exception as e:
            self.logger.error(f"Failed to flush audit events: {e}")
    
    async def _write_events_to_storage(self, events: List[AuditEvent]):
        """Write events to persistent storage"""
        # Create daily log file
        today = datetime.utcnow().strftime("%Y-%m-%d")
        log_file = self.audit_log_path / f"audit-{today}.jsonl"
        
        # Write events as JSON lines
        with open(log_file, 'a', encoding='utf-8') as f:
            for event in events:
                event_dict = {
                    'id': event.id,
                    'timestamp': event.timestamp.isoformat(),
                    'event_type': event.event_type.value,
                    'severity': event.severity.value,
                    'source': event.source,
                    'target': event.target,
                    'action': event.action,
                    'description': event.description,
                    'user_id': event.user_id,
                    'session_id': event.session_id,
                    'ip_address': event.ip_address,
                    'data_classification': event.data_classification,
                    'data_types': event.data_types,
                    'success': event.success,
                    'error_message': event.error_message,
                    'before_state': event.before_state,
                    'after_state': event.after_state,
                    'metadata': event.metadata,
                    'tags': event.tags,
                    'checksum': event.checksum
                }
                f.write(json.dumps(event_dict) + '\n')
    
    async def _load_events_from_storage(self, filter_criteria: AuditFilter) -> List[AuditEvent]:
        """Load events from persistent storage based on filter criteria"""
        events = []
        
        # Determine which log files to check based on time range
        if filter_criteria.start_time:
            start_date = filter_criteria.start_time.date()
        else:
            start_date = (datetime.utcnow() - timedelta(days=30)).date()
        
        if filter_criteria.end_time:
            end_date = filter_criteria.end_time.date()
        else:
            end_date = datetime.utcnow().date()
        
        # Read events from daily log files
        current_date = start_date
        while current_date <= end_date:
            log_file = self.audit_log_path / f"audit-{current_date.strftime('%Y-%m-%d')}.jsonl"
            
            if log_file.exists():
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if not line.strip():
                                continue
                            
                            event_dict = json.loads(line)
                            
                            # Create AuditEvent object
                            event = AuditEvent(
                                id=event_dict['id'],
                                timestamp=datetime.fromisoformat(event_dict['timestamp']),
                                event_type=AuditEventType(event_dict['event_type']),
                                severity=AuditSeverity(event_dict['severity']),
                                source=event_dict['source'],
                                target=event_dict['target'],
                                action=event_dict['action'],
                                description=event_dict['description'],
                                user_id=event_dict.get('user_id'),
                                session_id=event_dict.get('session_id'),
                                ip_address=event_dict.get('ip_address'),
                                data_classification=event_dict.get('data_classification'),
                                data_types=event_dict.get('data_types', []),
                                success=event_dict['success'],
                                error_message=event_dict.get('error_message'),
                                before_state=event_dict.get('before_state'),
                                after_state=event_dict.get('after_state'),
                                metadata=event_dict.get('metadata', {}),
                                tags=event_dict.get('tags', [])
                            )
                            event.checksum = event_dict.get('checksum')
                            
                            # Apply filters
                            if self._event_matches_filter(event, filter_criteria):
                                events.append(event)
                                
                                if len(events) >= filter_criteria.limit:
                                    return events
                
                except Exception as e:
                    self.logger.error(f"Error reading audit log file {log_file}: {e}")
            
            current_date += timedelta(days=1)
        
        return events
    
    def _event_matches_filter(self, event: AuditEvent, filter_criteria: AuditFilter) -> bool:
        """Check if an event matches the filter criteria"""
        
        # Time range filter
        if filter_criteria.start_time and event.timestamp < filter_criteria.start_time:
            return False
        if filter_criteria.end_time and event.timestamp > filter_criteria.end_time:
            return False
        
        # Event type filter
        if filter_criteria.event_types and event.event_type not in filter_criteria.event_types:
            return False
        
        # Severity filter
        if filter_criteria.severities and event.severity not in filter_criteria.severities:
            return False
        
        # Source filter
        if filter_criteria.sources and event.source not in filter_criteria.sources:
            return False
        
        # Target filter
        if filter_criteria.targets and event.target not in filter_criteria.targets:
            return False
        
        # User ID filter
        if filter_criteria.user_ids and event.user_id not in filter_criteria.user_ids:
            return False
        
        # Data classification filter
        if (filter_criteria.data_classifications and 
            event.data_classification not in filter_criteria.data_classifications):
            return False
        
        # Success filter
        if filter_criteria.success_only is not None and event.success != filter_criteria.success_only:
            return False
        
        # Tags filter
        if filter_criteria.tags:
            if not any(tag in event.tags for tag in filter_criteria.tags):
                return False
        
        return True
    
    async def _start_background_tasks(self):
        """Start background tasks for audit log maintenance"""
        
        async def periodic_flush():
            """Periodically flush the event buffer"""
            while not self._shutdown_event.is_set():
                try:
                    async with self.buffer_lock:
                        if self.event_buffer:
                            await self._flush_buffer()
                    
                    await asyncio.sleep(self.flush_interval_seconds)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Error in periodic flush task: {e}")
                    await asyncio.sleep(10)  # Wait before retry
        
        async def cleanup_old_logs():
            """Clean up old audit logs based on retention policy"""
            while not self._shutdown_event.is_set():
                try:
                    cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
                    
                    # Find and remove old log files
                    for log_file in self.audit_log_path.glob("audit-*.jsonl"):
                        try:
                            date_str = log_file.stem.replace("audit-", "")
                            file_date = datetime.strptime(date_str, "%Y-%m-%d")
                            
                            if file_date.date() < cutoff_date.date():
                                log_file.unlink()
                                self.logger.info(f"Removed old audit log: {log_file}")
                        
                        except (ValueError, OSError) as e:
                            self.logger.warning(f"Error processing log file {log_file}: {e}")
                    
                    # Run daily
                    await asyncio.sleep(24 * 60 * 60)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Error in cleanup task: {e}")
                    await asyncio.sleep(60 * 60)  # Wait 1 hour before retry
        
        # Start background tasks
        self._background_tasks = [
            asyncio.create_task(periodic_flush()),
            asyncio.create_task(cleanup_old_logs())
        ]
        
        self.logger.info("Started audit logger background tasks")
    
    async def shutdown(self):
        """Shutdown the audit logger"""
        try:
            self.logger.info("Shutting down data audit logger")
            
            # Signal shutdown
            self._shutdown_event.set()
            
            # Cancel background tasks
            for task in self._background_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
            # Final flush of remaining events
            async with self.buffer_lock:
                if self.event_buffer:
                    await self._flush_buffer()
            
            # Log shutdown
            await self.log_event(AuditEvent(
                event_type=AuditEventType.SYSTEM_EVENT,
                source="audit_logger",
                action="shutdown",
                description="Data audit logger shutdown complete"
            ))
            
            # Final flush
            async with self.buffer_lock:
                if self.event_buffer:
                    await self._flush_buffer()
            
            self.logger.info("Data audit logger shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during audit logger shutdown: {e}")