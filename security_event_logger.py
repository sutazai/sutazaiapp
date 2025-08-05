#!/usr/bin/env python3
"""
Security Event Logging System for SutazAI
Centralized security event collection, processing, and analysis
"""

import json
import time
import sqlite3
import threading
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
import hashlib
import re
from collections import defaultdict
import syslog

class SecurityEventLogger:
    def __init__(self, config_file='/opt/sutazaiapp/config/security_logging_config.json'):
        self.config = self.load_config(config_file)
        self.running = False
        
        # Initialize logging
        self.setup_logging()
        
        # Initialize database
        self.init_database()
        
        # Event processors
        self.event_processors = {
            'authentication': self.process_auth_event,
            'authorization': self.process_authz_event,
            'network': self.process_network_event,
            'container': self.process_container_event,
            'application': self.process_app_event,
            'system': self.process_system_event
        }
        
        # Threat detection patterns
        self.threat_patterns = self.load_threat_patterns()
        
        # Event correlation engine
        self.correlation_engine = EventCorrelationEngine(self.db_path)
    
    def load_config(self, config_file):
        """Load security logging configuration"""
        default_config = {
            'logging': {
                'log_level': 'INFO',
                'log_file': '/opt/sutazaiapp/logs/security_events.log',
                'max_file_size': 100 * 1024 * 1024,  # 100MB
                'backup_count': 10,
                'syslog_enabled': True,
                'syslog_facility': 'LOG_LOCAL0'
            },
            'sources': {
                'docker_events': True,
                'system_logs': True,
                'application_logs': True,
                'network_logs': True,
                'auth_logs': True
            },
            'filters': {
                'severity_threshold': 'INFO',
                'excluded_events': ['container_stats_update'],
                'rate_limiting': {
                    'enabled': True,
                    'max_events_per_minute': 1000
                }
            },
            'alerts': {
                'correlation_window': 300,  # 5 minutes
                'threat_score_threshold': 7.0,
                'auto_response': True
            }
        }
        
        try:
            with open(config_file, 'r') as f:
                user_config = json.load(f)
            config = {**default_config, **user_config}
            return config
        except FileNotFoundError:
            # Save default config
            Path(config_file).parent.mkdir(parents=True, exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config
    
    def setup_logging(self):
        """Setup logging configuration"""
        # Create logs directory
        log_dir = Path(self.config['logging']['log_file']).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure main logger
        self.logger = logging.getLogger('security_events')
        self.logger.setLevel(getattr(logging, self.config['logging']['log_level']))
        
        # File handler with rotation
        file_handler = RotatingFileHandler(
            self.config['logging']['log_file'],
            maxBytes=self.config['logging']['max_file_size'],
            backupCount=self.config['logging']['backup_count']
        )
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler for debugging
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Syslog handler if enabled
        if self.config['logging']['syslog_enabled']:
            try:
                syslog_handler = logging.handlers.SysLogHandler(
                    address='/dev/log',
                    facility=getattr(syslog, self.config['logging']['syslog_facility'])
                )
                syslog_formatter = logging.Formatter('SutazAI-Security: %(message)s')
                syslog_handler.setFormatter(syslog_formatter)
                self.logger.addHandler(syslog_handler)
            except Exception as e:
                self.logger.warning(f"Failed to setup syslog handler: {e}")
    
    def init_database(self):
        """Initialize database for event storage"""
        self.db_path = '/opt/sutazaiapp/data/security_events.db'
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Security events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS security_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                source TEXT NOT NULL,
                source_ip TEXT,
                user_id TEXT,
                session_id TEXT,
                event_data TEXT NOT NULL,
                threat_score REAL DEFAULT 0.0,
                processed BOOLEAN DEFAULT FALSE,
                tags TEXT,
                correlation_id TEXT
            )
        ''')
        
        # Event correlations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS event_correlations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                correlation_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                event_count INTEGER NOT NULL,
                threat_score REAL NOT NULL,
                description TEXT NOT NULL,
                events TEXT NOT NULL,
                status TEXT DEFAULT 'open'
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON security_events(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_event_type ON security_events(event_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_source_ip ON security_events(source_ip)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_threat_score ON security_events(threat_score)')
        
        conn.commit()
        conn.close()
    
    def load_threat_patterns(self):
        """Load threat detection patterns"""
        return {
            'authentication': [
                {
                    'name': 'brute_force_login',
                    'pattern': r'Failed.*password.*from.*(\d+\.\d+\.\d+\.\d+)',
                    'severity': 'HIGH',
                    'threat_score': 8.0
                },
                {
                    'name': 'privilege_escalation',
                    'pattern': r'sudo.*COMMAND=.*',
                    'severity': 'MEDIUM',
                    'threat_score': 6.0
                }
            ],
            'network': [
                {
                    'name': 'port_scan',
                    'pattern': r'SYN.*scan.*from.*(\d+\.\d+\.\d+\.\d+)',
                    'severity': 'HIGH',
                    'threat_score': 7.5
                },
                {
                    'name': 'ddos_attempt',
                    'pattern': r'Possible.*DDoS.*from.*(\d+\.\d+\.\d+\.\d+)',
                    'severity': 'CRITICAL',
                    'threat_score': 9.0
                }
            ],
            'application': [
                {
                    'name': 'sql_injection',
                    'pattern': r'(?i)(union.*select|or.*1=1|drop.*table)',
                    'severity': 'HIGH',
                    'threat_score': 8.5
                },
                {
                    'name': 'xss_attempt',
                    'pattern': r'(?i)(<script|javascript:|onerror=)',
                    'severity': 'MEDIUM',
                    'threat_score': 6.5
                }
            ]
        }
    
    def log_security_event(self, event_type, severity, source, event_data, source_ip=None, 
                          user_id=None, session_id=None, tags=None):
        """Log a security event"""
        
        event_id = hashlib.md5(f"{time.time()}{event_type}{source}".encode()).hexdigest()
        
        # Calculate threat score
        threat_score = self.calculate_threat_score(event_type, event_data, source_ip)
        
        # Create event record
        event_record = {
            'id': event_id,
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'severity': severity,
            'source': source,
            'source_ip': source_ip,
            'user_id': user_id,
            'session_id': session_id,
            'event_data': json.dumps(event_data) if isinstance(event_data, dict) else str(event_data),
            'threat_score': threat_score,
            'tags': json.dumps(tags) if tags else None
        }
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO security_events 
            (timestamp, event_type, severity, source, source_ip, user_id, session_id, 
             event_data, threat_score, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            event_record['timestamp'], event_type, severity, source, source_ip,
            user_id, session_id, event_record['event_data'], threat_score,
            event_record['tags']
        ))
        conn.commit()
        conn.close()
        
        # Log to file
        log_message = f"EventType={event_type} | Severity={severity} | Source={source}"
        if source_ip:
            log_message += f" | SourceIP={source_ip}"
        if user_id:
            log_message += f" | UserID={user_id}"
        log_message += f" | ThreatScore={threat_score:.1f} | Data={event_record['event_data'][:200]}..."
        
        getattr(self.logger, severity.lower())(log_message)
        
        # Trigger correlation analysis
        if threat_score >= 5.0:
            self.correlation_engine.analyze_event(event_record)
        
        return event_id
    
    def calculate_threat_score(self, event_type, event_data, source_ip=None):
        """Calculate threat score for an event"""
        base_score = {
            'authentication': 3.0,
            'authorization': 4.0,
            'network': 5.0,
            'container': 2.0,
            'application': 6.0,
            'system': 4.0
        }.get(event_type, 1.0)
        
        score_modifiers = 0.0
        event_str = str(event_data).lower()
        
        # Pattern-based scoring
        for category, patterns in self.threat_patterns.items():
            for pattern_info in patterns:
                if re.search(pattern_info['pattern'], event_str, re.IGNORECASE):
                    score_modifiers += pattern_info['threat_score'] - base_score
                    break
        
        # IP-based scoring
        if source_ip:
            # Check if IP is from suspicious ranges
            if source_ip.startswith('192.168.') or source_ip.startswith('10.'):
                score_modifiers -= 1.0  # Lower score for internal IPs
            elif self.is_suspicious_ip(source_ip):
                score_modifiers += 3.0
        
        # Time-based scoring (higher score for off-hours)
        current_hour = datetime.now().hour
        if current_hour < 6 or current_hour > 22:
            score_modifiers += 1.0
        
        final_score = max(0.0, min(10.0, base_score + score_modifiers))
        return round(final_score, 1)
    
    def is_suspicious_ip(self, ip):
        """Check if IP is suspicious based on threat intelligence"""
        # Implement threat intelligence lookups here
        # For now, simple heuristics
        suspicious_patterns = [
            r'^192\.168\.1\.(100|200|250)$',  # Example suspicious internal IPs
            r'^10\.0\.0\.(50|100|200)$'
        ]
        
        for pattern in suspicious_patterns:
            if re.match(pattern, ip):
                return True
        return False
    
    def process_auth_event(self, event_data):
        """Process authentication events"""
        # Extract relevant information
        if 'failed' in str(event_data).lower():
            severity = 'WARNING'
            threat_score_boost = 2.0
        elif 'success' in str(event_data).lower():
            severity = 'INFO'
            threat_score_boost = 0.0
        else:
            severity = 'INFO'
            threat_score_boost = 0.0
        
        return severity, threat_score_boost
    
    def process_authz_event(self, event_data):
        """Process authorization events"""
        if 'denied' in str(event_data).lower() or 'forbidden' in str(event_data).lower():
            return 'WARNING', 3.0
        return 'INFO', 0.0
    
    def process_network_event(self, event_data):
        """Process network events"""
        event_str = str(event_data).lower()
        if any(keyword in event_str for keyword in ['scan', 'probe', 'attack']):
            return 'HIGH', 4.0
        elif 'connection' in event_str and 'refused' in event_str:
            return 'WARNING', 2.0
        return 'INFO', 0.0
    
    def process_container_event(self, event_data):
        """Process container events"""
        event_str = str(event_data).lower()
        if any(keyword in event_str for keyword in ['privilege', 'root', 'escape']):
            return 'HIGH', 5.0
        elif 'exec' in event_str:
            return 'WARNING', 2.0
        return 'INFO', 0.0
    
    def process_app_event(self, event_data):
        """Process application events"""
        event_str = str(event_data).lower()
        if any(keyword in event_str for keyword in ['error', 'exception', 'failure']):
            return 'WARNING', 1.0
        return 'INFO', 0.0
    
    def process_system_event(self, event_data):
        """Process system events"""
        event_str = str(event_data).lower()
        if any(keyword in event_str for keyword in ['critical', 'emergency', 'panic']):
            return 'CRITICAL', 6.0
        elif 'warning' in event_str:
            return 'WARNING', 2.0
        return 'INFO', 0.0
    
    def monitor_docker_events(self):
        """Monitor Docker events"""
        self.logger.info("Starting Docker event monitoring")
        
        while self.running:
            try:
                process = subprocess.Popen([
                    'docker', 'events', '--format', 'json'
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
                
                for line in process.stdout:
                    if not self.running:
                        break
                    
                    try:
                        event = json.loads(line.strip())
                        
                        # Filter out noise
                        if event.get('Action') in self.config['filters']['excluded_events']:
                            continue
                        
                        # Log security-relevant Docker events
                        if event.get('Type') == 'container':
                            action = event.get('Action', '')
                            if action in ['start', 'stop', 'kill', 'exec_create', 'exec_start']:
                                self.log_security_event(
                                    'container',
                                    'INFO',
                                    'docker',
                                    event,
                                    tags=['docker', 'container', action]
                                )
                    
                    except json.JSONDecodeError:
                        continue
                
            except Exception as e:
                self.logger.error(f"Docker monitoring error: {e}")
                time.sleep(60)
    
    def monitor_system_logs(self):
        """Monitor system logs"""
        self.logger.info("Starting system log monitoring")
        
        log_files = [
            '/var/log/auth.log',
            '/var/log/syslog',
            '/var/log/kern.log'
        ]
        
        while self.running:
            for log_file in log_files:
                try:
                    # Use tail to get recent entries
                    result = subprocess.run([
                        'tail', '-n', '50', log_file
                    ], capture_output=True, text=True, timeout=10)
                    
                    if result.returncode == 0:
                        for line in result.stdout.split('\n'):
                            if line.strip():
                                # Parse and categorize log entries
                                event_type = self.categorize_log_entry(line)
                                if event_type:
                                    self.log_security_event(
                                        event_type,
                                        'INFO',
                                        f'system:{Path(log_file).name}',
                                        line.strip()
                                    )
                
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    continue
            
            time.sleep(30)
    
    def categorize_log_entry(self, log_line):
        """Categorize log entry by type"""
        line_lower = log_line.lower()
        
        if any(keyword in line_lower for keyword in ['ssh', 'login', 'password', 'auth']):
            return 'authentication'
        elif any(keyword in line_lower for keyword in ['denied', 'permission', 'access']):
            return 'authorization'
        elif any(keyword in line_lower for keyword in ['network', 'tcp', 'udp', 'connection']):
            return 'network'
        elif any(keyword in line_lower for keyword in ['kernel', 'hardware', 'memory']):
            return 'system'
        
        return None
    
    def generate_security_report(self):
        """Generate comprehensive security report"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get statistics
        cursor.execute('''
            SELECT 
                event_type,
                severity,
                COUNT(*) as count,
                AVG(threat_score) as avg_threat_score,
                MAX(threat_score) as max_threat_score
            FROM security_events 
            WHERE timestamp >= datetime('now', '-24 hours')
            GROUP BY event_type, severity
            ORDER BY count DESC
        ''')
        
        event_stats = cursor.fetchall()
        
        # Get top threats
        cursor.execute('''
            SELECT source_ip, COUNT(*) as event_count, AVG(threat_score) as avg_score
            FROM security_events 
            WHERE source_ip IS NOT NULL 
            AND timestamp >= datetime('now', '-24 hours')
            GROUP BY source_ip
            HAVING avg_score > 5.0
            ORDER BY avg_score DESC, event_count DESC
            LIMIT 10
        ''')
        
        top_threats = cursor.fetchall()
        
        # Get recent high-threat events
        cursor.execute('''
            SELECT timestamp, event_type, severity, source_ip, threat_score, event_data
            FROM security_events 
            WHERE threat_score >= 7.0
            AND timestamp >= datetime('now', '-24 hours')
            ORDER BY threat_score DESC, timestamp DESC
            LIMIT 20
        ''')
        
        high_threat_events = cursor.fetchall()
        
        conn.close()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'period': '24 hours',
            'summary': {
                'total_events': sum(stat[2] for stat in event_stats),
                'high_threat_events': len(high_threat_events),
                'unique_source_ips': len(top_threats)
            },
            'event_statistics': [
                {
                    'event_type': stat[0],
                    'severity': stat[1],
                    'count': stat[2],
                    'avg_threat_score': round(stat[3], 2),
                    'max_threat_score': round(stat[4], 2)
                } for stat in event_stats
            ],
            'top_threats': [
                {
                    'source_ip': threat[0],
                    'event_count': threat[1],
                    'avg_threat_score': round(threat[2], 2)
                } for threat in top_threats
            ],
            'high_threat_events': [
                {
                    'timestamp': event[0],
                    'event_type': event[1],
                    'severity': event[2],
                    'source_ip': event[3],
                    'threat_score': event[4],
                    'description': event[5][:100] + '...' if len(event[5]) > 100 else event[5]
                } for event in high_threat_events
            ]
        }
        
        return report
    
    def start(self):
        """Start the security event logger"""
        self.logger.info("Starting SutazAI Security Event Logger")
        self.running = True
        
        # Start monitoring threads
        threads = []
        
        if self.config['sources']['docker_events']:
            threads.append(threading.Thread(target=self.monitor_docker_events, daemon=True))
        
        if self.config['sources']['system_logs']:
            threads.append(threading.Thread(target=self.monitor_system_logs, daemon=True))
        
        for thread in threads:
            thread.start()
        
        self.logger.info(f"Started {len(threads)} monitoring threads")
        
        # Main loop for maintenance tasks
        try:
            while self.running:
                # Generate periodic reports
                report = self.generate_security_report()
                with open('/opt/sutazaiapp/data/security_report.json', 'w') as f:
                    json.dump(report, f, indent=2)
                
                time.sleep(300)  # Run every 5 minutes
                
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """Stop the security event logger"""
        self.running = False
        self.logger.info("Security Event Logger stopped")

class EventCorrelationEngine:
    """Engine for correlating related security events"""
    
    def __init__(self, db_path):
        self.db_path = db_path
        self.correlation_rules = self.load_correlation_rules()
    
    def load_correlation_rules(self):
        """Load event correlation rules"""
        return [
            {
                'name': 'brute_force_attack',
                'conditions': {
                    'event_types': ['authentication'],
                    'time_window': 300,  # 5 minutes
                    'min_events': 5,
                    'same_source_ip': True
                },
                'threat_score': 8.5,
                'description': 'Multiple failed authentication attempts from same IP'
            },
            {
                'name': 'privilege_escalation_sequence',
                'conditions': {
                    'event_types': ['authentication', 'authorization', 'system'],
                    'time_window': 600,  # 10 minutes
                    'min_events': 3,
                    'same_user': True
                },
                'threat_score': 9.0,
                'description': 'Potential privilege escalation sequence detected'
            }
        ]
    
    def analyze_event(self, event):
        """Analyze event for correlations"""
        for rule in self.correlation_rules:
            if self.check_correlation_rule(event, rule):
                self.create_correlation(event, rule)
    
    def check_correlation_rule(self, event, rule):
        """Check if event matches correlation rule"""
        # Implementation would check database for related events
        # This is a simplified version
        return event['threat_score'] >= 5.0
    
    def create_correlation(self, event, rule):
        """Create correlation record"""
        correlation_id = hashlib.md5(f"{time.time()}{rule['name']}".encode()).hexdigest()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO event_correlations 
            (correlation_id, timestamp, event_count, threat_score, description, events)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            correlation_id,
            datetime.now().isoformat(),
            1,  # Would count related events
            rule['threat_score'],
            rule['description'],
            json.dumps([event['id']])
        ))
        conn.commit()
        conn.close()

def main():
    logger = SecurityEventLogger()
    
    # Test event logging
    logger.log_security_event(
        'authentication',
        'WARNING',
        'test_source',
        {'message': 'Failed login attempt', 'user': 'admin'},
        source_ip='192.168.1.100'
    )
    
    try:
        logger.start()
    except KeyboardInterrupt:
        logger.stop()

if __name__ == "__main__":
    main()