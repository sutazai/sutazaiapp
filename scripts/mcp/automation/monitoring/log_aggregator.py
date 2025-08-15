#!/usr/bin/env python3
"""
MCP Automation Log Aggregator
Structured logging and aggregation for MCP infrastructure
"""

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Pattern, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import httpx
from collections import defaultdict, deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Log severity levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class LogSource(Enum):
    """Log sources"""
    MCP_SERVER = "mcp_server"
    AUTOMATION = "automation"
    SYSTEM = "system"
    APPLICATION = "application"
    SECURITY = "security"
    AUDIT = "audit"


@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: datetime
    level: LogLevel
    source: LogSource
    component: str
    message: str
    structured_data: Dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    user: Optional[str] = None
    session_id: Optional[str] = None
    error_code: Optional[str] = None
    stack_trace: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['level'] = self.level.value
        data['source'] = self.source.value
        return json.dumps(data)
        
    def to_loki_format(self) -> Dict[str, Any]:
        """Convert to Loki log format"""
        return {
            "streams": [{
                "stream": {
                    "source": self.source.value,
                    "component": self.component,
                    "level": self.level.value
                },
                "values": [[
                    str(int(self.timestamp.timestamp() * 1e9)),  # Nanoseconds
                    self.message
                ]]
            }]
        }


@dataclass
class LogPattern:
    """Pattern for log parsing"""
    name: str
    pattern: Pattern
    level: LogLevel
    extractor: callable
    tags: List[str] = field(default_factory=list)


@dataclass
class LogAggregation:
    """Aggregated log statistics"""
    component: str
    time_window: timedelta
    total_count: int = 0
    level_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    error_patterns: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    top_messages: List[Tuple[str, int]] = field(default_factory=list)
    average_rate: float = 0.0
    peak_rate: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


class LogAggregator:
    """Log aggregation and processing system"""
    
    def __init__(self,
                 loki_url: str = "http://localhost:10202",
                 buffer_size: int = 10000,
                 batch_size: int = 100,
                 flush_interval: int = 5):
        """
        Initialize log aggregator
        
        Args:
            loki_url: Loki server URL
            buffer_size: Maximum number of logs to buffer
            batch_size: Number of logs to send in each batch
            flush_interval: Interval to flush logs in seconds
        """
        self.loki_url = loki_url
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        
        # Log buffer
        self.log_buffer: deque = deque(maxlen=buffer_size)
        self.batch_queue: List[LogEntry] = []
        
        # Patterns for log parsing
        self.patterns = self._init_patterns()
        
        # Aggregation storage
        self.aggregations: Dict[str, LogAggregation] = {}
        self.recent_logs: deque = deque(maxlen=1000)
        
        # HTTP client for Loki
        self.http_client = httpx.AsyncClient(timeout=10.0)
        
        # Statistics
        self.stats = {
            'logs_processed': 0,
            'logs_sent': 0,
            'logs_failed': 0,
            'batches_sent': 0,
            'parse_errors': 0
        }
        
    def _init_patterns(self) -> List[LogPattern]:
        """Initialize log parsing patterns"""
        patterns = [
            LogPattern(
                name="error_pattern",
                pattern=re.compile(r"(?i)(error|exception|failed|failure):\s*(.+)"),
                level=LogLevel.ERROR,
                extractor=lambda m: {"error_message": m.group(2)},
                tags=["error"]
            ),
            LogPattern(
                name="warning_pattern",
                pattern=re.compile(r"(?i)(warning|warn):\s*(.+)"),
                level=LogLevel.WARNING,
                extractor=lambda m: {"warning_message": m.group(2)},
                tags=["warning"]
            ),
            LogPattern(
                name="mcp_server_pattern",
                pattern=re.compile(r"MCP Server \[([^\]]+)\]:\s*(.+)"),
                level=LogLevel.INFO,
                extractor=lambda m: {"server_name": m.group(1), "message": m.group(2)},
                tags=["mcp_server"]
            ),
            LogPattern(
                name="performance_pattern",
                pattern=re.compile(r"Performance:\s*(\w+)\s*took\s*([\d.]+)ms"),
                level=LogLevel.INFO,
                extractor=lambda m: {"operation": m.group(1), "duration_ms": float(m.group(2))},
                tags=["performance"]
            ),
            LogPattern(
                name="security_pattern",
                pattern=re.compile(r"(?i)security\s*(alert|violation|event):\s*(.+)"),
                level=LogLevel.CRITICAL,
                extractor=lambda m: {"security_type": m.group(1), "details": m.group(2)},
                tags=["security"]
            ),
            LogPattern(
                name="stack_trace_pattern",
                pattern=re.compile(r"Traceback \(most recent call last\):(.+?)(?=\n\n|\Z)", re.DOTALL),
                level=LogLevel.ERROR,
                extractor=lambda m: {"stack_trace": m.group(0)},
                tags=["stack_trace", "exception"]
            ),
            LogPattern(
                name="http_request_pattern",
                pattern=re.compile(r'(\w+)\s+"([^"]+)"\s+(\d+)\s+([\d.]+)ms'),
                level=LogLevel.INFO,
                extractor=lambda m: {
                    "method": m.group(1),
                    "path": m.group(2),
                    "status_code": int(m.group(3)),
                    "duration_ms": float(m.group(4))
                },
                tags=["http", "request"]
            ),
            LogPattern(
                name="database_query_pattern",
                pattern=re.compile(r"Query:\s*(.+?)\s*Duration:\s*([\d.]+)ms"),
                level=LogLevel.DEBUG,
                extractor=lambda m: {"query": m.group(1), "duration_ms": float(m.group(2))},
                tags=["database", "query"]
            )
        ]
        
        return patterns
        
    def parse_log_line(self, line: str, source: LogSource = LogSource.SYSTEM) -> Optional[LogEntry]:
        """
        Parse a log line into structured format
        
        Args:
            line: Raw log line
            source: Log source
            
        Returns:
            Parsed LogEntry or None if parsing failed
        """
        try:
            # Try to parse as JSON first
            if line.strip().startswith('{'):
                try:
                    data = json.loads(line)
                    return LogEntry(
                        timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat())),
                        level=LogLevel(data.get('level', 'info')),
                        source=source,
                        component=data.get('component', 'unknown'),
                        message=data.get('message', ''),
                        structured_data=data.get('data', {}),
                        trace_id=data.get('trace_id'),
                        span_id=data.get('span_id'),
                        user=data.get('user'),
                        session_id=data.get('session_id'),
                        error_code=data.get('error_code'),
                        stack_trace=data.get('stack_trace'),
                        tags=data.get('tags', [])
                    )
                except (json.JSONDecodeError, KeyError):
                    pass
                    
            # Try pattern matching
            level = LogLevel.INFO
            structured_data = {}
            tags = []
            
            for pattern in self.patterns:
                match = pattern.pattern.search(line)
                if match:
                    level = pattern.level
                    structured_data.update(pattern.extractor(match))
                    tags.extend(pattern.tags)
                    break
                    
            # Extract timestamp if present
            timestamp = datetime.now()
            timestamp_match = re.match(r'^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+(.+)$', line)
            if timestamp_match:
                try:
                    timestamp = datetime.fromisoformat(timestamp_match.group(1))
                    line = timestamp_match.group(2)
                except ValueError:
                    pass
                    
            # Extract component if present
            component = "unknown"
            component_match = re.match(r'^\[([^\]]+)\]\s+(.+)$', line)
            if component_match:
                component = component_match.group(1)
                line = component_match.group(2)
                
            return LogEntry(
                timestamp=timestamp,
                level=level,
                source=source,
                component=component,
                message=line.strip(),
                structured_data=structured_data,
                tags=tags
            )
            
        except Exception as e:
            logger.error(f"Failed to parse log line: {e}")
            self.stats['parse_errors'] += 1
            return None
            
    async def ingest_log(self, log_entry: LogEntry):
        """
        Ingest a log entry
        
        Args:
            log_entry: Log entry to ingest
        """
        # Add to buffer
        self.log_buffer.append(log_entry)
        self.recent_logs.append(log_entry)
        self.batch_queue.append(log_entry)
        
        # Update statistics
        self.stats['logs_processed'] += 1
        
        # Update aggregations
        self._update_aggregations(log_entry)
        
        # Check if batch is ready
        if len(self.batch_queue) >= self.batch_size:
            await self.flush_logs()
            
    def _update_aggregations(self, log_entry: LogEntry):
        """Update log aggregations with new entry"""
        key = f"{log_entry.component}_{log_entry.source.value}"
        
        if key not in self.aggregations:
            self.aggregations[key] = LogAggregation(
                component=log_entry.component,
                time_window=timedelta(minutes=5)
            )
            
        agg = self.aggregations[key]
        agg.total_count += 1
        agg.level_counts[log_entry.level.value] += 1
        agg.last_updated = datetime.now()
        
        # Track error patterns
        if log_entry.level in [LogLevel.ERROR, LogLevel.CRITICAL]:
            error_key = log_entry.error_code or log_entry.message[:50]
            agg.error_patterns[error_key] += 1
            
    async def flush_logs(self):
        """Flush buffered logs to Loki"""
        if not self.batch_queue:
            return
            
        batch = self.batch_queue[:self.batch_size]
        self.batch_queue = self.batch_queue[self.batch_size:]
        
        try:
            # Group logs by stream
            streams = defaultdict(list)
            
            for log in batch:
                stream_key = f"{log.source.value}_{log.component}_{log.level.value}"
                streams[stream_key].append(log)
                
            # Send to Loki
            for stream_key, logs in streams.items():
                loki_data = {
                    "streams": [{
                        "stream": self._parse_stream_key(stream_key),
                        "values": [
                            [
                                str(int(log.timestamp.timestamp() * 1e9)),
                                log.to_json()
                            ]
                            for log in logs
                        ]
                    }]
                }
                
                response = await self.http_client.post(
                    f"{self.loki_url}/loki/api/v1/push",
                    json=loki_data
                )
                
                if response.status_code == 204:
                    self.stats['logs_sent'] += len(logs)
                else:
                    logger.error(f"Failed to send logs to Loki: {response.status_code}")
                    self.stats['logs_failed'] += len(logs)
                    
            self.stats['batches_sent'] += 1
            
        except Exception as e:
            logger.error(f"Error flushing logs: {e}")
            self.stats['logs_failed'] += len(batch)
            
    def _parse_stream_key(self, stream_key: str) -> Dict[str, str]:
        """Parse stream key into labels"""
        parts = stream_key.split('_')
        return {
            "source": parts[0] if len(parts) > 0 else "unknown",
            "component": parts[1] if len(parts) > 1 else "unknown",
            "level": parts[2] if len(parts) > 2 else "info"
        }
        
    async def tail_log_file(self, file_path: str, source: LogSource = LogSource.SYSTEM):
        """
        Tail a log file and ingest new entries
        
        Args:
            file_path: Path to log file
            source: Log source
        """
        path = Path(file_path)
        
        if not path.exists():
            logger.error(f"Log file not found: {file_path}")
            return
            
        # Track file position
        last_position = 0
        
        while True:
            try:
                with open(path, 'r') as f:
                    f.seek(last_position)
                    
                    for line in f:
                        if line.strip():
                            log_entry = self.parse_log_line(line, source)
                            if log_entry:
                                await self.ingest_log(log_entry)
                                
                    last_position = f.tell()
                    
            except Exception as e:
                logger.error(f"Error tailing log file: {e}")
                
            await asyncio.sleep(1)
            
    async def start_aggregation_loop(self):
        """Start continuous log aggregation"""
        logger.info(f"Starting log aggregation with {self.flush_interval}s flush interval")
        
        while True:
            try:
                # Flush logs periodically
                await self.flush_logs()
                
                # Clean old aggregations
                cutoff_time = datetime.now() - timedelta(hours=1)
                old_keys = [
                    key for key, agg in self.aggregations.items()
                    if agg.last_updated < cutoff_time
                ]
                for key in old_keys:
                    del self.aggregations[key]
                    
            except Exception as e:
                logger.error(f"Error in aggregation loop: {e}")
                
            await asyncio.sleep(self.flush_interval)
            
    def search_logs(self,
                   query: str,
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None,
                   level: Optional[LogLevel] = None,
                   component: Optional[str] = None,
                   limit: int = 100) -> List[LogEntry]:
        """
        Search recent logs
        
        Args:
            query: Search query
            start_time: Start time filter
            end_time: End time filter
            level: Log level filter
            component: Component filter
            limit: Maximum results
            
        Returns:
            List of matching log entries
        """
        results = []
        
        for log in reversed(self.recent_logs):
            # Apply filters
            if start_time and log.timestamp < start_time:
                continue
            if end_time and log.timestamp > end_time:
                continue
            if level and log.level != level:
                continue
            if component and log.component != component:
                continue
                
            # Search in message and structured data
            if query.lower() in log.message.lower():
                results.append(log)
            elif any(query.lower() in str(v).lower() for v in log.structured_data.values()):
                results.append(log)
                
            if len(results) >= limit:
                break
                
        return results
        
    def get_aggregation_summary(self) -> Dict[str, Any]:
        """Get aggregation summary"""
        summary = {
            'total_logs': self.stats['logs_processed'],
            'logs_sent': self.stats['logs_sent'],
            'logs_failed': self.stats['logs_failed'],
            'parse_errors': self.stats['parse_errors'],
            'components': {}
        }
        
        for key, agg in self.aggregations.items():
            summary['components'][key] = {
                'total_count': agg.total_count,
                'level_distribution': dict(agg.level_counts),
                'error_patterns': dict(list(agg.error_patterns.items())[:5]),  # Top 5
                'last_updated': agg.last_updated.isoformat()
            }
            
        return summary
        
    def get_error_analysis(self) -> Dict[str, Any]:
        """Analyze error patterns"""
        error_analysis = {
            'total_errors': 0,
            'error_rate': 0.0,
            'top_errors': [],
            'error_trends': {},
            'affected_components': []
        }
        
        # Aggregate error data
        all_errors = defaultdict(int)
        component_errors = defaultdict(int)
        
        for agg in self.aggregations.values():
            error_count = agg.level_counts.get('error', 0) + agg.level_counts.get('critical', 0)
            error_analysis['total_errors'] += error_count
            
            if error_count > 0:
                component_errors[agg.component] += error_count
                
            for pattern, count in agg.error_patterns.items():
                all_errors[pattern] += count
                
        # Calculate error rate
        if self.stats['logs_processed'] > 0:
            error_analysis['error_rate'] = error_analysis['total_errors'] / self.stats['logs_processed']
            
        # Top errors
        error_analysis['top_errors'] = [
            {'pattern': pattern, 'count': count}
            for pattern, count in sorted(all_errors.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
        
        # Affected components
        error_analysis['affected_components'] = [
            {'component': comp, 'error_count': count}
            for comp, count in sorted(component_errors.items(), key=lambda x: x[1], reverse=True)
        ]
        
        return error_analysis
        
    async def cleanup(self):
        """Cleanup resources"""
        # Flush remaining logs
        await self.flush_logs()
        await self.http_client.aclose()


async def main():
    """Main function for testing"""
    aggregator = LogAggregator()
    
    # Test log ingestion
    test_logs = [
        "2024-01-15 10:30:00 [mcp-server] ERROR: Connection failed to database",
        '{"timestamp": "2024-01-15T10:31:00", "level": "warning", "component": "automation", "message": "Slow query detected", "data": {"duration_ms": 1500}}',
        "2024-01-15 10:32:00 [security] CRITICAL: Security violation detected: Unauthorized access attempt",
        "Performance: API request took 250.5ms",
        "MCP Server [filesystem]: Health check completed successfully"
    ]
    
    for log_line in test_logs:
        log_entry = aggregator.parse_log_line(log_line)
        if log_entry:
            await aggregator.ingest_log(log_entry)
            
    # Get summary
    summary = aggregator.get_aggregation_summary()
    print("Aggregation Summary:")
    print(json.dumps(summary, indent=2, default=str))
    
    # Get error analysis
    errors = aggregator.get_error_analysis()
    print("\nError Analysis:")
    print(json.dumps(errors, indent=2))
    
    # Cleanup
    await aggregator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())