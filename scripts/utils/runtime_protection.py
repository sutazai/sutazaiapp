#!/usr/bin/env python3
"""
Runtime Application Self-Protection (RASP) System
Implements real-time application security monitoring and protection
"""

import asyncio
import logging
import json
import sys
import os
import inspect
import traceback
import time
import threading
import hashlib
import re
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from functools import wraps
import ast
import types
import redis
import psutil
import sqlite3
from contextlib import contextmanager
import tempfile
import shutil
import subprocess

class AttackType(Enum):
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    COMMAND_INJECTION = "command_injection"
    PATH_TRAVERSAL = "path_traversal"
    LDAP_INJECTION = "ldap_injection"
    XXE = "xxe"
    SSRF = "ssrf"
    DESERIALIZATION = "deserialization"
    CODE_INJECTION = "code_injection"
    BUFFER_OVERFLOW = "buffer_overflow"

class Severity(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class ActionType(Enum):
    MONITOR = "monitor"
    BLOCK = "block"
    SANITIZE = "sanitize"
    TERMINATE = "terminate"

@dataclass
class SecurityEvent:
    """Security event detected by RASP"""
    event_id: str
    attack_type: AttackType
    severity: Severity
    description: str
    source_ip: str
    user_agent: str
    request_path: str
    payload: str
    timestamp: datetime
    stack_trace: Optional[str] = None
    blocked: bool = False

@dataclass
class SecurityRule:
    """RASP security rule"""
    rule_id: str
    name: str
    attack_type: AttackType
    pattern: str
    action: ActionType
    severity: Severity
    enabled: bool = True
    description: str = ""

class RASPEngine:
    """Runtime Application Self-Protection Engine"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.redis_client = None
        self.rules: Dict[str, SecurityRule] = {}
        self.event_history: List[SecurityEvent] = []
        self.protection_enabled = True
        self.monitored_functions: Set[str] = set()
        self.original_functions: Dict[str, Callable] = {}
        self.attack_patterns = {}
        self._initialize_components()

    def _initialize_components(self):
        """Initialize RASP components"""
        try:
            # Initialize Redis for event storage
            self.redis_client = redis.Redis(
                host=self.config.get('redis_host', 'redis'),
                port=self.config.get('redis_port', 6379),
                password=self.config.get('redis_password'),
                ssl=True,
                decode_responses=True
            )
            
            # Load security rules
            self._load_security_rules()
            
            # Load attack patterns
            self._load_attack_patterns()
            
            # Initialize function hooks
            self._initialize_function_hooks()
            
            self.logger.info("RASP Engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RASP Engine: {e}")
            raise

    def _load_security_rules(self):
        """Load security rules for various attack types"""
        default_rules = [
            # SQL Injection Rules
            SecurityRule(
                "sql_001", "SQL Injection - UNION", AttackType.SQL_INJECTION,
                r"(?i)(union\s+(all\s+)?select|select\s+.*\s+from\s+.*\s+union)",
                ActionType.BLOCK, Severity.HIGH,
                description="Detects SQL UNION-based injection attempts"
            ),
            SecurityRule(
                "sql_002", "SQL Injection - Boolean", AttackType.SQL_INJECTION,
                r"(?i)(\s+(or|and)\s+[\w\s]*=[\w\s]*--|'.*'.*=.*'.*')",
                ActionType.BLOCK, Severity.HIGH,
                description="Detects boolean-based SQL injection"
            ),
            SecurityRule(
                "sql_003", "SQL Injection - Time-based", AttackType.SQL_INJECTION,
                r"(?i)(sleep\s*\(|waitfor\s+delay|benchmark\s*\(|pg_sleep\s*\()",
                ActionType.BLOCK, Severity.HIGH,
                description="Detects time-based SQL injection"
            ),
            
            # XSS Rules
            SecurityRule(
                "xss_001", "XSS - Script Tags", AttackType.XSS,
                r"(?i)<script[^>]*>.*?</script>|<script[^>]*/>|javascript:",
                ActionType.SANITIZE, Severity.MEDIUM,
                description="Detects script-based XSS attempts"
            ),
            SecurityRule(
                "xss_002", "XSS - Event Handlers", AttackType.XSS,
                r"(?i)on(load|error|click|mouseover|focus|blur)\s*=",
                ActionType.SANITIZE, Severity.MEDIUM,
                description="Detects event handler XSS"
            ),
            
            # Command Injection Rules
            SecurityRule(
                "cmd_001", "Command Injection - Basic", AttackType.COMMAND_INJECTION,
                r"[;&|`\$\(\)]|(\|\||&&)",
                ActionType.BLOCK, Severity.CRITICAL,
                description="Detects command injection metacharacters"
            ),
            SecurityRule(
                "cmd_002", "Command Injection - System Commands", AttackType.COMMAND_INJECTION,
                r"(?i)(wget|curl|nc|netcat|ping|nslookup|dig|cat|ls|ps|whoami|id|uname)",
                ActionType.BLOCK, Severity.HIGH,
                description="Detects system command injection"
            ),
            
            # Path Traversal Rules
            SecurityRule(
                "path_001", "Path Traversal - Directory", AttackType.PATH_TRAVERSAL,
                r"(\.\./|\.\.\\\|%2e%2e%2f|%2e%2e\\|%252e%252e%252f)",
                ActionType.BLOCK, Severity.HIGH,
                description="Detects directory traversal attempts"
            ),
            
            # XXE Rules
            SecurityRule(
                "xxe_001", "XXE - External Entity", AttackType.XXE,
                r"(?i)<!entity[^>]*system|<!entity[^>]*public",
                ActionType.BLOCK, Severity.HIGH,
                description="Detects XXE external entity references"
            ),
            
            # SSRF Rules
            SecurityRule(
                "ssrf_001", "SSRF - Internal IPs", AttackType.SSRF,
                r"(127\.0\.0\.1|localhost|10\.|172\.(1[6-9]|2[0-9]|3[01])\.|192\.168\.)",
                ActionType.BLOCK, Severity.HIGH,
                description="Detects SSRF to internal networks"
            ),
        ]
        
        for rule in default_rules:
            self.rules[rule.rule_id] = rule
        
        self.logger.info(f"Loaded {len(default_rules)} security rules")

    def _load_attack_patterns(self):
        """Load attack patterns for detection"""
        self.attack_patterns = {
            AttackType.SQL_INJECTION: [
                re.compile(pattern.pattern, re.IGNORECASE) 
                for rule in self.rules.values() 
                if rule.attack_type == AttackType.SQL_INJECTION
            ],
            AttackType.XSS: [
                re.compile(pattern.pattern, re.IGNORECASE)
                for rule in self.rules.values()
                if rule.attack_type == AttackType.XSS
            ],
            AttackType.COMMAND_INJECTION: [
                re.compile(pattern.pattern, re.IGNORECASE)
                for rule in self.rules.values()
                if rule.attack_type == AttackType.COMMAND_INJECTION
            ],
            AttackType.PATH_TRAVERSAL: [
                re.compile(pattern.pattern, re.IGNORECASE)
                for rule in self.rules.values()
                if rule.attack_type == AttackType.PATH_TRAVERSAL
            ],
            AttackType.XXE: [
                re.compile(pattern.pattern, re.IGNORECASE)
                for rule in self.rules.values()
                if rule.attack_type == AttackType.XXE
            ],
            AttackType.SSRF: [
                re.compile(pattern.pattern, re.IGNORECASE)
                for rule in self.rules.values()
                if rule.attack_type == AttackType.SSRF
            ],
        }

    def _initialize_function_hooks(self):
        """Initialize hooks for critical functions"""
        # Hook database operations
        self._hook_database_functions()
        
        # Hook file operations
        self._hook_file_operations()
        
        # Hook network operations
        self._hook_network_operations()
        
        # Hook system operations
        self._hook_system_operations()

    def _hook_database_functions(self):
        """Hook database-related functions"""
        try:
            # Hook sqlite3.execute
            if hasattr(sqlite3.Cursor, 'execute'):
                self._hook_function(sqlite3.Cursor, 'execute', self._protected_db_execute)
            
            # Hook psycopg2 if available
            try:
                import psycopg2
                if hasattr(psycopg2.extras.RealDictCursor, 'execute'):
                    self._hook_function(psycopg2.extras.RealDictCursor, 'execute', self._protected_db_execute)
            except ImportError:
                pass
            
        except Exception as e:
            self.logger.error(f"Failed to hook database functions: {e}")

    def _hook_file_operations(self):
        """Hook file operation functions"""
        try:
            # Hook built-in open function
            self._hook_builtin_function('open', self._protected_file_open)
            
            # Hook os.system
            if hasattr(os, 'system'):
                self._hook_function(os, 'system', self._protected_system_command)
            
            # Hook subprocess functions
            if hasattr(subprocess, 'run'):
                self._hook_function(subprocess, 'run', self._protected_subprocess_run)
                
        except Exception as e:
            self.logger.error(f"Failed to hook file operations: {e}")

    def _hook_network_operations(self):
        """Hook network operation functions"""
        try:
            # Hook urllib
            import urllib.request
            if hasattr(urllib.request, 'urlopen'):
                self._hook_function(urllib.request, 'urlopen', self._protected_urlopen)
                
        except Exception as e:
            self.logger.error(f"Failed to hook network operations: {e}")

    def _hook_system_operations(self):
        """Hook system operation functions"""
        try:
            # Hook eval and exec
            self._hook_builtin_function('eval', self._protected_eval)
            self._hook_builtin_function('exec', self._protected_exec)
            
        except Exception as e:
            self.logger.error(f"Failed to hook system operations: {e}")

    def _hook_function(self, module_or_class, function_name: str, wrapper_function: Callable):
        """Hook a function with protection wrapper"""
        try:
            original_function = getattr(module_or_class, function_name)
            self.original_functions[f"{module_or_class.__name__}.{function_name}"] = original_function
            
            @wraps(original_function)
            def protected_wrapper(*args, **kwargs):
                return wrapper_function(original_function, *args, **kwargs)
            
            setattr(module_or_class, function_name, protected_wrapper)
            self.monitored_functions.add(f"{module_or_class.__name__}.{function_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to hook {module_or_class.__name__}.{function_name}: {e}")

    def _hook_builtin_function(self, function_name: str, wrapper_function: Callable):
        """Hook a built-in function"""
        try:
            import builtins
            original_function = getattr(builtins, function_name)
            self.original_functions[f"builtins.{function_name}"] = original_function
            
            @wraps(original_function)
            def protected_wrapper(*args, **kwargs):
                return wrapper_function(original_function, *args, **kwargs)
            
            setattr(builtins, function_name, protected_wrapper)
            self.monitored_functions.add(f"builtins.{function_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to hook builtin {function_name}: {e}")

    def _protected_db_execute(self, original_func, *args, **kwargs):
        """Protected database execute function"""
        if not self.protection_enabled:
            return original_func(*args, **kwargs)
        
        # Extract SQL query (first argument)
        if args and isinstance(args[0], str):
            query = args[0]
            
            # Check for SQL injection
            event = self._check_sql_injection(query)
            if event:
                self._handle_security_event(event)
                if event.blocked:
                    raise SecurityException(f"SQL injection blocked: {event.description}")
        
        return original_func(*args, **kwargs)

    def _protected_file_open(self, original_func, *args, **kwargs):
        """Protected file open function"""
        if not self.protection_enabled:
            return original_func(*args, **kwargs)
        
        # Extract file path
        if args:
            file_path = str(args[0])
            
            # Check for path traversal
            event = self._check_path_traversal(file_path)
            if event:
                self._handle_security_event(event)
                if event.blocked:
                    raise SecurityException(f"Path traversal blocked: {event.description}")
        
        return original_func(*args, **kwargs)

    def _protected_system_command(self, original_func, *args, **kwargs):
        """Protected system command function"""
        if not self.protection_enabled:
            return original_func(*args, **kwargs)
        
        # Extract command
        if args:
            command = str(args[0])
            
            # Check for command injection
            event = self._check_command_injection(command)
            if event:
                self._handle_security_event(event)
                if event.blocked:
                    raise SecurityException(f"Command injection blocked: {event.description}")
        
        return original_func(*args, **kwargs)

    def _protected_subprocess_run(self, original_func, *args, **kwargs):
        """Protected subprocess run function"""
        if not self.protection_enabled:
            return original_func(*args, **kwargs)
        
        # Extract command
        if args:
            command = args[0]
            if isinstance(command, list):
                command = ' '.join(command)
            command = str(command)
            
            # Check for command injection
            event = self._check_command_injection(command)
            if event:
                self._handle_security_event(event)
                if event.blocked:
                    raise SecurityException(f"Command injection blocked: {event.description}")
        
        return original_func(*args, **kwargs)

    def _protected_urlopen(self, original_func, *args, **kwargs):
        """Protected URL open function"""
        if not self.protection_enabled:
            return original_func(*args, **kwargs)
        
        # Extract URL
        if args:
            url = str(args[0])
            
            # Check for SSRF
            event = self._check_ssrf(url)
            if event:
                self._handle_security_event(event)
                if event.blocked:
                    raise SecurityException(f"SSRF blocked: {event.description}")
        
        return original_func(*args, **kwargs)

    def _protected_eval(self, original_func, *args, **kwargs):
        """Protected eval function"""
        if not self.protection_enabled:
            return original_func(*args, **kwargs)
        
        # Extract code
        if args:
            code = str(args[0])
            
            # Check for code injection
            event = self._check_code_injection(code)
            if event:
                self._handle_security_event(event)
                if event.blocked:
                    raise SecurityException(f"Code injection blocked: {event.description}")
        
        return original_func(*args, **kwargs)

    def _protected_exec(self, original_func, *args, **kwargs):
        """Protected exec function"""
        if not self.protection_enabled:
            return original_func(*args, **kwargs)
        
        # Extract code
        if args:
            code = str(args[0])
            
            # Check for code injection
            event = self._check_code_injection(code)
            if event:
                self._handle_security_event(event)
                if event.blocked:
                    raise SecurityException(f"Code injection blocked: {event.description}")
        
        return original_func(*args, **kwargs)

    def _check_sql_injection(self, query: str) -> Optional[SecurityEvent]:
        """Check for SQL injection patterns"""
        for rule in self.rules.values():
            if rule.attack_type == AttackType.SQL_INJECTION and rule.enabled:
                pattern = re.compile(rule.pattern, re.IGNORECASE)
                if pattern.search(query):
                    return SecurityEvent(
                        event_id=self._generate_event_id(),
                        attack_type=AttackType.SQL_INJECTION,
                        severity=rule.severity,
                        description=f"SQL injection detected: {rule.name}",
                        source_ip=self._get_client_ip(),
                        user_agent=self._get_user_agent(),
                        request_path=self._get_request_path(),
                        payload=query,
                        timestamp=datetime.utcnow(),
                        stack_trace=self._get_stack_trace(),
                        blocked=(rule.action == ActionType.BLOCK)
                    )
        return None

    def _check_path_traversal(self, path: str) -> Optional[SecurityEvent]:
        """Check for path traversal patterns"""
        for rule in self.rules.values():
            if rule.attack_type == AttackType.PATH_TRAVERSAL and rule.enabled:
                pattern = re.compile(rule.pattern, re.IGNORECASE)
                if pattern.search(path):
                    return SecurityEvent(
                        event_id=self._generate_event_id(),
                        attack_type=AttackType.PATH_TRAVERSAL,
                        severity=rule.severity,
                        description=f"Path traversal detected: {rule.name}",
                        source_ip=self._get_client_ip(),
                        user_agent=self._get_user_agent(),
                        request_path=self._get_request_path(),
                        payload=path,
                        timestamp=datetime.utcnow(),
                        stack_trace=self._get_stack_trace(),
                        blocked=(rule.action == ActionType.BLOCK)
                    )
        return None

    def _check_command_injection(self, command: str) -> Optional[SecurityEvent]:
        """Check for command injection patterns"""
        for rule in self.rules.values():
            if rule.attack_type == AttackType.COMMAND_INJECTION and rule.enabled:
                pattern = re.compile(rule.pattern, re.IGNORECASE)
                if pattern.search(command):
                    return SecurityEvent(
                        event_id=self._generate_event_id(),
                        attack_type=AttackType.COMMAND_INJECTION,
                        severity=rule.severity,
                        description=f"Command injection detected: {rule.name}",
                        source_ip=self._get_client_ip(),
                        user_agent=self._get_user_agent(),
                        request_path=self._get_request_path(),
                        payload=command,
                        timestamp=datetime.utcnow(),
                        stack_trace=self._get_stack_trace(),
                        blocked=(rule.action == ActionType.BLOCK)
                    )
        return None

    def _check_ssrf(self, url: str) -> Optional[SecurityEvent]:
        """Check for SSRF patterns"""
        for rule in self.rules.values():
            if rule.attack_type == AttackType.SSRF and rule.enabled:
                pattern = re.compile(rule.pattern, re.IGNORECASE)
                if pattern.search(url):
                    return SecurityEvent(
                        event_id=self._generate_event_id(),
                        attack_type=AttackType.SSRF,
                        severity=rule.severity,
                        description=f"SSRF detected: {rule.name}",
                        source_ip=self._get_client_ip(),
                        user_agent=self._get_user_agent(),
                        request_path=self._get_request_path(),
                        payload=url,
                        timestamp=datetime.utcnow(),
                        stack_trace=self._get_stack_trace(),
                        blocked=(rule.action == ActionType.BLOCK)
                    )
        return None

    def _check_code_injection(self, code: str) -> Optional[SecurityEvent]:
        """Check for code injection patterns"""
        # Basic code injection detection
        dangerous_patterns = [
            r'__import__',
            r'exec\s*\(',
            r'eval\s*\(',
            r'compile\s*\(',
            r'open\s*\(',
            r'file\s*\(',
            r'input\s*\(',
            r'raw_input\s*\('
        ]
        
        for pattern_str in dangerous_patterns:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            if pattern.search(code):
                return SecurityEvent(
                    event_id=self._generate_event_id(),
                    attack_type=AttackType.CODE_INJECTION,
                    severity=Severity.CRITICAL,
                    description=f"Code injection detected: {pattern_str}",
                    source_ip=self._get_client_ip(),
                    user_agent=self._get_user_agent(),
                    request_path=self._get_request_path(),
                    payload=code,
                    timestamp=datetime.utcnow(),
                    stack_trace=self._get_stack_trace(),
                    blocked=True
                )
        return None

    def _handle_security_event(self, event: SecurityEvent):
        """Handle detected security event"""
        # Store event
        self.event_history.append(event)
        
        # Log event
        self.logger.warning(
            f"RASP Event: {event.attack_type.value} - {event.description} "
            f"from {event.source_ip} (blocked: {event.blocked})"
        )
        
        # Store in Redis for real-time monitoring
        try:
            event_data = asdict(event)
            event_data['timestamp'] = event_data['timestamp'].isoformat()
            event_data['attack_type'] = event_data['attack_type'].value
            event_data['severity'] = event_data['severity'].value
            
            self.redis_client.lpush("rasp_events", json.dumps(event_data))
            self.redis_client.ltrim("rasp_events", 0, 9999)  # Keep last 10k events
            
        except Exception as e:
            self.logger.error(f"Failed to store RASP event: {e}")
        
        # Real-time alerting for critical events
        if event.severity == Severity.CRITICAL:
            self._send_critical_alert(event)

    def _send_critical_alert(self, event: SecurityEvent):
        """Send critical security alert"""
        try:
            alert_data = {
                "type": "rasp_critical_event",
                "event_id": event.event_id,
                "attack_type": event.attack_type.value,
                "description": event.description,
                "source_ip": event.source_ip,
                "timestamp": event.timestamp.isoformat(),
                "blocked": event.blocked
            }
            
            # Store critical alert
            self.redis_client.lpush("critical_alerts", json.dumps(alert_data))
            
            # You could also integrate with external alerting systems here
            # e.g., Slack, PagerDuty, email, etc.
            
        except Exception as e:
            self.logger.error(f"Failed to send critical alert: {e}")

    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        timestamp = str(int(time.time() * 1000))
        random_part = hashlib.md5(os.urandom(16)).hexdigest()[:8]
        return f"rasp_{timestamp}_{random_part}"

    def _get_client_ip(self) -> str:
        """Get client IP address from request context"""
        # This would be implemented based on your framework
        # For now, return placeholder
        return "127.0.0.1"

    def _get_user_agent(self) -> str:
        """Get user agent from request context"""
        # This would be implemented based on your framework
        return "Unknown"

    def _get_request_path(self) -> str:
        """Get request path from request context"""
        # This would be implemented based on your framework
        return "/"

    def _get_stack_trace(self) -> str:
        """Get current stack trace"""
        return ''.join(traceback.format_stack())

    def enable_protection(self):
        """Enable RASP protection"""
        self.protection_enabled = True
        self.logger.info("RASP protection enabled")

    def disable_protection(self):
        """Disable RASP protection"""
        self.protection_enabled = False
        self.logger.warning("RASP protection disabled")

    def add_rule(self, rule: SecurityRule):
        """Add new security rule"""
        self.rules[rule.rule_id] = rule
        self._load_attack_patterns()  # Reload patterns
        self.logger.info(f"Added security rule: {rule.name}")

    def remove_rule(self, rule_id: str):
        """Remove security rule"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            self._load_attack_patterns()  # Reload patterns
            self.logger.info(f"Removed security rule: {rule_id}")

    def get_events(self, limit: int = 100) -> List[SecurityEvent]:
        """Get recent security events"""
        return self.event_history[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        """Get RASP statistics"""
        total_events = len(self.event_history)
        blocked_events = sum(1 for event in self.event_history if event.blocked)
        
        attack_counts = {}
        for event in self.event_history:
            attack_type = event.attack_type.value
            attack_counts[attack_type] = attack_counts.get(attack_type, 0) + 1
        
        return {
            "total_events": total_events,
            "blocked_events": blocked_events,
            "attack_counts": attack_counts,
            "protection_enabled": self.protection_enabled,
            "monitored_functions": len(self.monitored_functions),
            "active_rules": len([r for r in self.rules.values() if r.enabled])
        }

class SecurityException(Exception):
    """Exception raised when security threat is blocked"""
    pass

# Context manager for temporary RASP disable
@contextmanager
def rasp_disabled(rasp_engine: RASPEngine):
    """Temporarily disable RASP protection"""
    was_enabled = rasp_engine.protection_enabled
    rasp_engine.disable_protection()
    try:
        yield
    finally:
        if was_enabled:
            rasp_engine.enable_protection()

# Decorator for protecting specific functions
def rasp_protected(attack_types: List[AttackType] = None):
    """Decorator to add RASP protection to specific functions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Add custom protection logic here
            return func(*args, **kwargs)
        return wrapper
    return decorator

if __name__ == "__main__":
    # Example usage
    config = {
        'redis_host': 'redis',
        'redis_port': 6379,
        'enable_logging': True,
        'alert_webhook_url': None
    }
    
    rasp = RASPEngine(config)
    
    # Example of using RASP
    try:
        # This would trigger SQL injection detection
        # rasp._check_sql_injection("' OR '1'='1' --")
        
        logger.info("RASP Engine initialized and monitoring...")
        logger.info(f"Statistics: {rasp.get_statistics()}")
        
    except SecurityException as e:
        logger.info(f"Security threat blocked: {e}")