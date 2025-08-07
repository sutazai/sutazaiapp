#!/usr/bin/env python3
"""
Comprehensive Logging Infrastructure for Hygiene Monitoring
Purpose: Structured logging, rotation, aggregation, and real-time log streaming
Author: AI Observability and Monitoring Engineer
Version: 1.0.0 - Production Logging System
"""

import asyncio
import json
import logging
import logging.handlers
import os
import re
import sqlite3
import time
import threading
from collections import defaultdict, deque, Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Generator
from dataclasses import dataclass, asdict
import gzip
import shutil
import queue
import uuid

import aiofiles
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

@dataclass
class LogEntry:
    """Structured log entry"""
    id: str
    timestamp: str
    level: str
    logger_name: str
    message: str
    module: str
    function: str
    line_number: int
    process_id: int
    thread_id: int
    agent_id: Optional[str] = None
    rule_id: Optional[str] = None
    file_path: Optional[str] = None
    duration_ms: Optional[float] = None
    error_details: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class LogAggregator:
    """Real-time log aggregation and analysis"""
    
    def __init__(self, project_root: str = "/opt/sutazaiapp"):
        self.project_root = Path(project_root)
        self.logs_dir = self.project_root / "logs"
        self.db_path = self.project_root / "monitoring" / "logs.db"
        
        # Ensure directories exist
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # In-memory storage for real-time access
        self.recent_logs = deque(maxlen=1000)  # Keep last 1000 log entries
        self.log_stats = defaultdict(int)
        self.error_patterns = Counter()
        self.agent_logs = defaultdict(lambda: deque(maxlen=100))
        
        # WebSocket clients for real-time log streaming
        self.websocket_clients = set()
        
        # Initialize database
        self._init_database()
        
        # Log processing queue
        self.log_queue = queue.Queue()
        self.processing_thread = threading.Thread(target=self._process_log_queue, daemon=True)
        self.processing_thread.start()
        
        print(f"Log Aggregator initialized - Logs dir: {self.logs_dir}")

    def _init_database(self):
        """Initialize SQLite database for log storage"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS log_entries (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                level TEXT NOT NULL,
                logger_name TEXT NOT NULL,
                message TEXT NOT NULL,
                module TEXT,
                function TEXT,
                line_number INTEGER,
                process_id INTEGER,
                thread_id INTEGER,
                agent_id TEXT,
                rule_id TEXT,
                file_path TEXT,
                duration_ms REAL,
                error_details TEXT,
                metadata TEXT
            )
        ''')
        
        # Create indexes for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON log_entries(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_level ON log_entries(level)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_agent_id ON log_entries(agent_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_rule_id ON log_entries(rule_id)')
        
        conn.commit()
        conn.close()

    def _process_log_queue(self):
        """Background thread to process log entries"""
        while True:
            try:
                log_entry = self.log_queue.get(timeout=1)
                self._store_log_entry(log_entry)
                self._update_stats(log_entry)
                self._broadcast_log_entry(log_entry)
                self.log_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing log entry: {e}")

    def _store_log_entry(self, log_entry: LogEntry):
        """Store log entry in database"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO log_entries (
                    id, timestamp, level, logger_name, message, module, function,
                    line_number, process_id, thread_id, agent_id, rule_id, file_path,
                    duration_ms, error_details, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                log_entry.id, log_entry.timestamp, log_entry.level,
                log_entry.logger_name, log_entry.message, log_entry.module,
                log_entry.function, log_entry.line_number, log_entry.process_id,
                log_entry.thread_id, log_entry.agent_id, log_entry.rule_id,
                log_entry.file_path, log_entry.duration_ms,
                json.dumps(log_entry.error_details) if log_entry.error_details else None,
                json.dumps(log_entry.metadata) if log_entry.metadata else None
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error storing log entry: {e}")

    def _update_stats(self, log_entry: LogEntry):
        """Update log statistics"""
        self.log_stats[f"level_{log_entry.level.lower()}"] += 1
        self.log_stats["total_logs"] += 1
        
        if log_entry.agent_id:
            self.log_stats[f"agent_{log_entry.agent_id}"] += 1
            self.agent_logs[log_entry.agent_id].append(log_entry)
        
        if log_entry.level == "ERROR":
            # Extract error patterns
            pattern = self._extract_error_pattern(log_entry.message)
            self.error_patterns[pattern] += 1
        
        # Keep recent logs in memory
        self.recent_logs.append(log_entry)

    def _extract_error_pattern(self, message: str) -> str:
        """Extract error pattern from error message"""
        # Remove specific details like file paths, line numbers, timestamps
        pattern = re.sub(r'/[/\w\.-]+', '<path>', message)
        pattern = re.sub(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', '<timestamp>', pattern)
        pattern = re.sub(r'line \d+', 'line <number>', pattern)
        pattern = re.sub(r'\d+', '<number>', pattern)
        return pattern[:200]  # Limit pattern length

    def _broadcast_log_entry(self, log_entry: LogEntry):
        """Broadcast log entry to WebSocket clients"""
        if not self.websocket_clients:
            return
        
        message = {
            'type': 'log_entry',
            'data': asdict(log_entry)
        }
        
        # Remove closed connections
        closed_clients = set()
        for ws in self.websocket_clients:
            try:
                asyncio.create_task(ws.send_str(json.dumps(message)))
            except Exception:
                closed_clients.add(ws)
        
        self.websocket_clients -= closed_clients

    def add_log_entry(self, 
                     level: str,
                     logger_name: str,
                     message: str,
                     module: str = None,
                     function: str = None,
                     line_number: int = None,
                     agent_id: str = None,
                     rule_id: str = None,
                     file_path: str = None,
                     duration_ms: float = None,
                     error_details: Dict[str, Any] = None,
                     metadata: Dict[str, Any] = None):
        """Add a new log entry"""
        
        log_entry = LogEntry(
            id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            level=level.upper(),
            logger_name=logger_name,
            message=message,
            module=module or "unknown",
            function=function or "unknown",
            line_number=line_number or 0,
            process_id=os.getpid(),
            thread_id=threading.get_ident(),
            agent_id=agent_id,
            rule_id=rule_id,
            file_path=file_path,
            duration_ms=duration_ms,
            error_details=error_details,
            metadata=metadata
        )
        
        # Add to processing queue
        self.log_queue.put(log_entry)

    def get_recent_logs(self, limit: int = 100, level: str = None, agent_id: str = None) -> List[LogEntry]:
        """Get recent log entries with optional filtering"""
        logs = list(self.recent_logs)
        
        if level:
            logs = [log for log in logs if log.level == level.upper()]
        
        if agent_id:
            logs = [log for log in logs if log.agent_id == agent_id]
        
        return logs[-limit:]

    def get_log_stats(self) -> Dict[str, Any]:
        """Get current log statistics"""
        return {
            'total_logs': self.log_stats.get('total_logs', 0),
            'error_count': self.log_stats.get('level_error', 0),
            'warning_count': self.log_stats.get('level_warning', 0),
            'info_count': self.log_stats.get('level_info', 0),
            'debug_count': self.log_stats.get('level_debug', 0),
            'top_error_patterns': dict(self.error_patterns.most_common(10)),
            'active_agents': len([k for k in self.log_stats.keys() if k.startswith('agent_')]),
            'recent_log_count': len(self.recent_logs)
        }

    def search_logs(self, 
                   query: str,
                   start_time: datetime = None,
                   end_time: datetime = None,
                   level: str = None,
                   limit: int = 100) -> List[LogEntry]:
        """Search logs by query with optional filters"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            sql = "SELECT * FROM log_entries WHERE message LIKE ?"
            params = [f"%{query}%"]
            
            if start_time:
                sql += " AND timestamp >= ?"
                params.append(start_time.isoformat())
            
            if end_time:
                sql += " AND timestamp <= ?"
                params.append(end_time.isoformat())
            
            if level:
                sql += " AND level = ?"
                params.append(level.upper())
            
            sql += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            
            logs = []
            for row in rows:
                log_entry = LogEntry(
                    id=row[0], timestamp=row[1], level=row[2], logger_name=row[3],
                    message=row[4], module=row[5], function=row[6], line_number=row[7],
                    process_id=row[8], thread_id=row[9], agent_id=row[10],
                    rule_id=row[11], file_path=row[12], duration_ms=row[13],
                    error_details=json.loads(row[14]) if row[14] else None,
                    metadata=json.loads(row[15]) if row[15] else None
                )
                logs.append(log_entry)
            
            conn.close()
            return logs
            
        except Exception as e:
            print(f"Error searching logs: {e}")
            return []

class StructuredLogger:
    """Enhanced logger with structured output"""
    
    def __init__(self, name: str, log_aggregator: LogAggregator):
        self.name = name
        self.aggregator = log_aggregator
        self.logger = logging.getLogger(name)
        
        # Set up file handler with rotation
        log_file = log_aggregator.logs_dir / f"{name}.log"
        handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        
        # JSON formatter for structured logs
        formatter = JsonFormatter()
        handler.setFormatter(formatter)
        
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def info(self, message: str, **kwargs):
        """Log info message with optional metadata"""
        self.logger.info(message, extra=kwargs)
        self.aggregator.add_log_entry(
            level="INFO",
            logger_name=self.name,
            message=message,
            **kwargs
        )

    def warning(self, message: str, **kwargs):
        """Log warning message with optional metadata"""
        self.logger.warning(message, extra=kwargs)
        self.aggregator.add_log_entry(
            level="WARNING",
            logger_name=self.name,
            message=message,
            **kwargs
        )

    def error(self, message: str, error_details: Dict[str, Any] = None, **kwargs):
        """Log error message with optional error details"""
        self.logger.error(message, extra=kwargs)
        self.aggregator.add_log_entry(
            level="ERROR",
            logger_name=self.name,
            message=message,
            error_details=error_details,
            **kwargs
        )

    def debug(self, message: str, **kwargs):
        """Log debug message with optional metadata"""
        self.logger.debug(message, extra=kwargs)
        self.aggregator.add_log_entry(
            level="DEBUG",
            logger_name=self.name,
            message=message,
            **kwargs
        )

class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'process': record.process,
            'thread': record.thread
        }
        
        # Add any extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 
                          'pathname', 'filename', 'module', 'exc_info',
                          'exc_text', 'stack_info', 'lineno', 'funcName',
                          'created', 'msecs', 'relativeCreated', 'thread',
                          'threadName', 'processName', 'process', 'getMessage']:
                log_data[key] = value
        
        return json.dumps(log_data)

class LogFileWatcher(FileSystemEventHandler):
    """Watch log files for real-time updates"""
    
    def __init__(self, log_aggregator: LogAggregator):
        self.aggregator = log_aggregator
        self.file_positions = {}

    def on_modified(self, event):
        """Handle file modification events"""
        if event.is_directory or not event.src_path.endswith('.log'):
            return
        
        try:
            self._process_log_file(event.src_path)
        except Exception as e:
            print(f"Error processing log file {event.src_path}: {e}")

    def _process_log_file(self, file_path: str):
        """Process new lines in a log file"""
        try:
            with open(file_path, 'r') as f:
                # Get current position or start from beginning
                current_pos = self.file_positions.get(file_path, 0)
                f.seek(current_pos)
                
                # Read new lines
                new_lines = f.readlines()
                
                # Update position
                self.file_positions[file_path] = f.tell()
                
                # Process each new line
                for line in new_lines:
                    self._parse_log_line(line.strip(), file_path)
                    
        except Exception as e:
            print(f"Error reading log file {file_path}: {e}")

    def _parse_log_line(self, line: str, file_path: str):
        """Parse a log line and extract structured data"""
        try:
            # Try to parse as JSON first
            try:
                log_data = json.loads(line)
                self.aggregator.add_log_entry(
                    level=log_data.get('level', 'INFO'),
                    logger_name=log_data.get('logger', 'file_watcher'),
                    message=log_data.get('message', line),
                    module=log_data.get('module'),
                    function=log_data.get('function'),
                    line_number=log_data.get('line'),
                    file_path=file_path,
                    metadata=log_data
                )
                return
            except json.JSONDecodeError:
                pass
            
            # Parse standard log format
            # Example: 2024-08-03 10:30:45 - hygiene-agent - INFO - Starting hygiene scan
            pattern = r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s*-\s*([^-]+)\s*-\s*(\w+)\s*-\s*(.+)'
            match = re.match(pattern, line)
            
            if match:
                timestamp_str, logger_name, level, message = match.groups()
                
                self.aggregator.add_log_entry(
                    level=level.strip(),
                    logger_name=logger_name.strip(),
                    message=message.strip(),
                    file_path=file_path
                )
            else:
                # Fallback: treat entire line as message
                self.aggregator.add_log_entry(
                    level="INFO",
                    logger_name="file_watcher",
                    message=line,
                    file_path=file_path
                )
                
        except Exception as e:
            print(f"Error parsing log line: {e}")

def setup_logging_infrastructure(project_root: str = "/opt/sutazaiapp") -> LogAggregator:
    """Set up the complete logging infrastructure"""
    # Create log aggregator
    aggregator = LogAggregator(project_root)
    
    # Set up file watcher
    observer = Observer()
    event_handler = LogFileWatcher(aggregator)
    observer.schedule(event_handler, str(aggregator.logs_dir), recursive=True)
    observer.start()
    
    print(f"Logging infrastructure initialized")
    print(f"Logs directory: {aggregator.logs_dir}")
    print(f"Database: {aggregator.db_path}")
    print(f"File watcher started for: {aggregator.logs_dir}")
    
    return aggregator

def create_agent_logger(agent_id: str, aggregator: LogAggregator) -> StructuredLogger:
    """Create a structured logger for an agent"""
    return StructuredLogger(f"agent-{agent_id}", aggregator)

# Example usage and testing
if __name__ == '__main__':
    # Set up logging infrastructure
    aggregator = setup_logging_infrastructure()
    
    # Create test loggers
    hygiene_logger = create_agent_logger("hygiene-scanner", aggregator)
    cleanup_logger = create_agent_logger("cleanup-agent", aggregator)
    
    # Test logging
    hygiene_logger.info("Hygiene scan started", 
                       agent_id="hygiene-scanner",
                       rule_id="rule_1",
                       file_path="test.py")
    
    hygiene_logger.warning("Potential violation detected",
                          agent_id="hygiene-scanner", 
                          rule_id="rule_1",
                          file_path="test.py",
                          metadata={"line_number": 42, "pattern": "magic"})
    
    cleanup_logger.error("Failed to clean up file",
                        agent_id="cleanup-agent",
                        error_details={"error_type": "PermissionError", "file": "locked.tmp"})
    
    # Print stats
    print("\nLog Statistics:")
    stats = aggregator.get_log_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\nRecent logs: {len(aggregator.get_recent_logs())}")
    
    # Keep running to monitor file changes
    try:
        print("\nMonitoring logs (Ctrl+C to stop)...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")