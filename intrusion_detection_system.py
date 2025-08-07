#!/usr/bin/env python3
"""
Intrusion Detection System for SutazAI
Real-time monitoring and detection of security threats
"""

import asyncio
import json
import time
import re
import subprocess
import threading
from datetime import datetime, timedelta
from collections import defaultdict, deque
import sqlite3
import hashlib
import ipaddress
from pathlib import Path

class IntrusionDetectionSystem:
    def __init__(self, config_file='/opt/sutazaiapp/config/ids_config.json'):
        self.config = self.load_config(config_file)
        self.alerts = []
        self.threat_intelligence = {}
        self.blocked_ips = set()
        self.connection_tracking = defaultdict(list)
        self.anomaly_baselines = {}
        self.running = False
        
        # Initialize database
        self.init_database()
        
        # Load threat intelligence
        self.load_threat_intelligence()
        
    def load_config(self, config_file):
        """Load IDS configuration"""
        default_config = {
            'monitoring': {
                'log_files': [
                    '/var/log/auth.log',
                    '/var/log/syslog',
                    '/opt/sutazaiapp/logs/*.log'
                ],
                'network_interfaces': ['eth0', 'docker0'],
                'ports_to_monitor': list(range(10000, 10600))
            },
            'detection_rules': {
                'failed_login_threshold': 5,
                'failed_login_window': 300,  # 5 minutes
                'port_scan_threshold': 10,
                'port_scan_window': 60,      # 1 minute
                'suspicious_user_agents': [
                    'sqlmap', 'nmap', 'nikto', 'burp', 'dirbuster'
                ],
                'max_connections_per_ip': 100,
                'rate_limit_threshold': 1000  # requests per minute
            },
            'response': {
                'auto_block': True,
                'block_duration': 3600,      # 1 hour
                'alert_email': None,
                'alert_webhook': None
            }
        }
        
        try:
            with open(config_file, 'r') as f:
                user_config = json.load(f)
            # Merge configs
            config = {**default_config, **user_config}
            return config
        except FileNotFoundError:
            # Save default config
            Path(config_file).parent.mkdir(parents=True, exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config
    
    def init_database(self):
        """Initialize SQLite database for storing alerts and logs"""
        self.db_path = '/opt/sutazaiapp/data/ids_database.db'
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                severity TEXT NOT NULL,
                type TEXT NOT NULL,
                source_ip TEXT,
                target_port INTEGER,
                description TEXT NOT NULL,
                raw_data TEXT,
                status TEXT DEFAULT 'open'
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS blocked_ips (
                ip TEXT PRIMARY KEY,
                blocked_at TEXT NOT NULL,
                expires_at TEXT,
                reason TEXT NOT NULL,
                block_count INTEGER DEFAULT 1
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS network_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                source_ip TEXT,
                destination_ip TEXT,
                source_port INTEGER,
                destination_port INTEGER,
                protocol TEXT,
                bytes_transferred INTEGER,
                flags TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def load_threat_intelligence(self):
        """Load threat intelligence data"""
        # Known malicious IPs (example list)
        self.threat_intelligence = {
            'malicious_ips': set([
                '192.168.1.100',  # Example local malicious IP
                '10.0.0.50'       # Another example
            ]),
            'malicious_domains': set([
                'malware.example.com',
                'phishing.example.com'
            ]),
            'attack_patterns': [
                r'(?i)(union.*select|or.*1=1|drop.*table)',  # SQL Injection
                r'(?i)(<script|javascript:|onerror=)',        # XSS
                r'(?i)(\.\.\/|\.\.\\)',                       # Directory traversal
                r'(?i)(cmd\.exe|/bin/sh|powershell)',         # Command injection
            ]
        }
    
    def create_alert(self, severity, alert_type, description, source_ip=None, target_port=None, raw_data=None):
        """Create and store security alert"""
        alert = {
            'id': len(self.alerts) + 1,
            'timestamp': datetime.now().isoformat(),
            'severity': severity,
            'type': alert_type,
            'source_ip': source_ip,
            'target_port': target_port,
            'description': description,
            'raw_data': raw_data,
            'status': 'open'
        }
        
        self.alerts.append(alert)
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO alerts (timestamp, severity, type, source_ip, target_port, description, raw_data)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (alert['timestamp'], severity, alert_type, source_ip, target_port, description, raw_data))
        conn.commit()
        conn.close()
        
        print(f"[ALERT] {severity}: {description} (Source: {source_ip or 'N/A'})")
        
        # Auto-response if configured
        if self.config['response']['auto_block'] and source_ip and severity in ['HIGH', 'CRITICAL']:
            self.block_ip(source_ip, f"Auto-blocked due to {alert_type}")
        
        return alert
    
    def block_ip(self, ip, reason):
        """Block IP address using iptables"""
        if ip in self.blocked_ips:
            return
        
        try:
            # Validate IP address
            ipaddress.ip_address(ip)
            
            # Block using iptables
            subprocess.run([
                'iptables', '-I', 'INPUT', '-s', ip, '-j', 'DROP'
            ], check=True)
            
            self.blocked_ips.add(ip)
            
            # Store in database
            expires_at = (datetime.now() + timedelta(seconds=self.config['response']['block_duration'])).isoformat()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO blocked_ips (ip, blocked_at, expires_at, reason, block_count)
                VALUES (?, ?, ?, ?, 
                    COALESCE((SELECT block_count + 1 FROM blocked_ips WHERE ip = ?), 1))
            ''', (ip, datetime.now().isoformat(), expires_at, reason, ip))
            conn.commit()
            conn.close()
            
            print(f"[BLOCK] IP {ip} blocked: {reason}")
            
        except Exception as e:
            print(f"[ERROR] Failed to block IP {ip}: {e}")
    
    def unblock_expired_ips(self):
        """Unblock IPs whose block duration has expired"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get expired blocks
        cursor.execute('''
            SELECT ip FROM blocked_ips 
            WHERE expires_at < ? AND expires_at IS NOT NULL
        ''', (datetime.now().isoformat(),))
        
        expired_ips = cursor.fetchall()
        
        for (ip,) in expired_ips:
            try:
                # Remove iptables rule
                subprocess.run([
                    'iptables', '-D', 'INPUT', '-s', ip, '-j', 'DROP'
                ], check=True)
                
                self.blocked_ips.discard(ip)
                
                # Remove from database
                cursor.execute('DELETE FROM blocked_ips WHERE ip = ?', (ip,))
                
                print(f"[UNBLOCK] IP {ip} unblocked (expired)")
                
            except subprocess.CalledProcessError:
                # Rule might not exist, continue
                pass
        
        conn.commit()
        conn.close()
    
    def monitor_log_files(self):
        """Monitor log files for suspicious activity"""
        print("[*] Starting log file monitoring...")
        
        # Track failed login attempts
        failed_logins = defaultdict(list)
        
        # Patterns to detect
        patterns = {
            'failed_ssh_login': r'Failed password for .* from (\d+\.\d+\.\d+\.\d+)',
            'failed_web_auth': r'HTTP.*401.*from (\d+\.\d+\.\d+\.\d+)',
            'suspicious_user_agent': r'User-Agent.*(' + '|'.join(self.config['detection_rules']['suspicious_user_agents']) + ')',
            'sql_injection': r'(?i)(union.*select|or.*1=1|drop.*table)',
            'xss_attempt': r'(?i)(<script|javascript:|onerror=)',
            'directory_traversal': r'(?i)(\.\.\/|\.\.\\)',
            'command_injection': r'(?i)(cmd\.exe|/bin/sh|powershell)'
        }
        
        while self.running:
            try:
                # Monitor auth.log for failed logins
                result = subprocess.run([
                    'tail', '-n', '100', '/var/log/auth.log'
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    lines = result.stdout.split('\n')
                    for line in lines:
                        # Check for failed SSH logins
                        match = re.search(patterns['failed_ssh_login'], line)
                        if match:
                            ip = match.group(1)
                            now = time.time()
                            failed_logins[ip].append(now)
                            
                            # Clean old attempts
                            failed_logins[ip] = [t for t in failed_logins[ip] 
                                               if now - t < self.config['detection_rules']['failed_login_window']]
                            
                            # Check threshold
                            if len(failed_logins[ip]) >= self.config['detection_rules']['failed_login_threshold']:
                                self.create_alert(
                                    'HIGH',
                                    'BRUTE_FORCE_ATTACK',
                                    f'Multiple failed login attempts from {ip}',
                                    source_ip=ip,
                                    raw_data=line
                                )
                
                # Monitor application logs
                for log_pattern in self.config['monitoring']['log_files']:
                    log_files = Path('/opt/sutazaiapp/logs').glob('*.log')
                    for log_file in log_files:
                        try:
                            with open(log_file, 'r') as f:
                                # Read last 100 lines
                                lines = deque(f, maxlen=100)
                                for line in lines:
                                    # Check for attack patterns
                                    for pattern_name, pattern in patterns.items():
                                        if re.search(pattern, line):
                                            self.create_alert(
                                                'MEDIUM',
                                                pattern_name.upper(),
                                                f'Suspicious pattern detected: {pattern_name}',
                                                raw_data=line
                                            )
                        except (FileNotFoundError, PermissionError):
                            continue
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                print(f"[ERROR] Log monitoring error: {e}")
                time.sleep(60)
    
    def monitor_network_traffic(self):
        """Monitor network traffic for anomalies"""
        print("[*] Starting network traffic monitoring...")
        
        # Connection tracking
        connection_counts = defaultdict(int)
        port_scan_tracking = defaultdict(set)
        
        while self.running:
            try:
                # Get network connections
                result = subprocess.run([
                    'netstat', '-tn'
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    lines = result.stdout.split('\n')
                    current_connections = defaultdict(int)
                    
                    for line in lines:
                        # Parse connection line
                        if 'ESTABLISHED' in line:
                            parts = line.split()
                            if len(parts) >= 5:
                                foreign_addr = parts[4]
                                if ':' in foreign_addr:
                                    ip = foreign_addr.split(':')[0]
                                    port = foreign_addr.split(':')[1]
                                    
                                    current_connections[ip] += 1
                                    
                                    # Check for port scanning
                                    port_scan_tracking[ip].add(port)
                                    
                                    # Clean old tracking data
                                    if len(port_scan_tracking[ip]) >= self.config['detection_rules']['port_scan_threshold']:
                                        self.create_alert(
                                            'HIGH',
                                            'PORT_SCAN_DETECTED',
                                            f'Port scan detected from {ip}',
                                            source_ip=ip
                                        )
                                        port_scan_tracking[ip].clear()
                    
                    # Check connection limits
                    for ip, count in current_connections.items():
                        if count > self.config['detection_rules']['max_connections_per_ip']:
                            self.create_alert(
                                'MEDIUM',
                                'EXCESSIVE_CONNECTIONS',
                                f'Excessive connections from {ip}: {count}',
                                source_ip=ip
                            )
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                print(f"[ERROR] Network monitoring error: {e}")
                time.sleep(120)
    
    def monitor_docker_events(self):
        """Monitor Docker events for security issues"""
        print("[*] Starting Docker event monitoring...")
        
        while self.running:
            try:
                # Monitor Docker events
                process = subprocess.Popen([
                    'docker', 'events', '--format', 'json'
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                
                for line in process.stdout:
                    if not self.running:
                        break
                    
                    try:
                        event = json.loads(line.strip())
                        event_type = event.get('Type')
                        action = event.get('Action')
                        
                        # Monitor for suspicious activities
                        if event_type == 'container':
                            if action == 'start':
                                container_name = event.get('Actor', {}).get('Attributes', {}).get('name', 'unknown')
                                if 'privilege' in str(event).lower():
                                    self.create_alert(
                                        'HIGH',
                                        'PRIVILEGED_CONTAINER',
                                        f'Privileged container started: {container_name}',
                                        raw_data=json.dumps(event)
                                    )
                            
                            elif action == 'exec_create':
                                exec_cmd = event.get('Actor', {}).get('Attributes', {}).get('execID', '')
                                if any(cmd in exec_cmd.lower() for cmd in ['sh', 'bash', 'cmd']):
                                    self.create_alert(
                                        'MEDIUM',
                                        'CONTAINER_SHELL_ACCESS',
                                        'Shell access to container detected',
                                        raw_data=json.dumps(event)
                                    )
                    
                    except json.JSONDecodeError:
                        continue
                
            except Exception as e:
                print(f"[ERROR] Docker monitoring error: {e}")
                time.sleep(60)
    
    def check_threat_intelligence(self):
        """Check current connections against threat intelligence"""
        print("[*] Starting threat intelligence checks...")
        
        while self.running:
            try:
                # Get current connections
                result = subprocess.run([
                    'netstat', '-tn'
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'ESTABLISHED' in line:
                            parts = line.split()
                            if len(parts) >= 5:
                                foreign_addr = parts[4]
                                if ':' in foreign_addr:
                                    ip = foreign_addr.split(':')[0]
                                    
                                    # Check against threat intelligence
                                    if ip in self.threat_intelligence['malicious_ips']:
                                        self.create_alert(
                                            'CRITICAL',
                                            'MALICIOUS_IP_CONNECTION',
                                            f'Connection to known malicious IP: {ip}',
                                            source_ip=ip
                                        )
                
                time.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                print(f"[ERROR] Threat intelligence check error: {e}")
                time.sleep(300)
    
    def generate_security_report(self):
        """Generate security report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_alerts': len(self.alerts),
                'blocked_ips': len(self.blocked_ips),
                'alert_severity_breakdown': defaultdict(int)
            },
            'recent_alerts': self.alerts[-10:] if self.alerts else [],
            'top_threats': {},
            'recommendations': []
        }
        
        # Count alerts by severity
        for alert in self.alerts:
            report['summary']['alert_severity_breakdown'][alert['severity']] += 1
        
        # Top threat sources
        threat_sources = defaultdict(int)
        for alert in self.alerts:
            if alert.get('source_ip'):
                threat_sources[alert['source_ip']] += 1
        
        report['top_threats'] = dict(sorted(threat_sources.items(), key=lambda x: x[1], reverse=True)[:5])
        
        # Generate recommendations
        if report['summary']['total_alerts'] > 50:
            report['recommendations'].append(
                "High number of alerts detected. Consider reviewing and tuning detection rules."
            )
        
        if len(self.blocked_ips) > 10:
            report['recommendations'].append(
                "Multiple IPs blocked. Review blocked IPs and consider implementing additional preventive measures."
            )
        
        return report
    
    def start(self):
        """Start the intrusion detection system"""
        print("=" * 60)
        print("Starting SutazAI Intrusion Detection System")
        print("=" * 60)
        
        self.running = True
        
        # Start monitoring threads
        threads = [
            threading.Thread(target=self.monitor_log_files, daemon=True),
            threading.Thread(target=self.monitor_network_traffic, daemon=True),
            threading.Thread(target=self.monitor_docker_events, daemon=True),
            threading.Thread(target=self.check_threat_intelligence, daemon=True)
        ]
        
        for thread in threads:
            thread.start()
        
        print("[*] All monitoring threads started")
        print("[*] IDS is now active and monitoring for threats")
        
        # Maintenance loop
        try:
            while self.running:
                # Unblock expired IPs
                self.unblock_expired_ips()
                
                # Generate and save report
                report = self.generate_security_report()
                with open('/opt/sutazaiapp/data/ids_report.json', 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                
                time.sleep(300)  # Run maintenance every 5 minutes
                
        except KeyboardInterrupt:
            print("\n[*] Shutting down IDS...")
            self.stop()
    
    def stop(self):
        """Stop the intrusion detection system"""
        self.running = False
        print("[*] IDS stopped")
    
    def get_status(self):
        """Get current IDS status"""
        return {
            'running': self.running,
            'total_alerts': len(self.alerts),
            'blocked_ips': list(self.blocked_ips),
            'recent_alerts': self.alerts[-5:] if self.alerts else []
        }

def main():
    ids = IntrusionDetectionSystem()
    
    try:
        ids.start()
    except KeyboardInterrupt:
        ids.stop()

if __name__ == "__main__":
    main()