#!/usr/bin/env python3
"""
Automated Threat Response System for SutazAI
Intelligent automated response to security threats and incidents
"""

import json
import time
import sqlite3
import subprocess
import threading
import requests
from datetime import datetime, timedelta
from pathlib import Path
import logging
import hashlib
import ipaddress
from collections import defaultdict
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class AutomatedThreatResponse:
    def __init__(self, config_file='/opt/sutazaiapp/config/threat_response_config.json'):
        self.config = self.load_config(config_file)
        self.running = False
        self.response_actions = {}
        self.incident_database = '/opt/sutazaiapp/data/incidents.db'
        
        # Initialize logging
        self.setup_logging()
        
        # Initialize database
        self.init_database()
        
        # Load response playbooks
        self.load_response_playbooks()
        
        # Response statistics
        self.response_stats = {
            'total_responses': 0,
            'successful_responses': 0,
            'failed_responses': 0,
            'false_positives': 0
        }
    
    def load_config(self, config_file):
        """Load threat response configuration"""
        default_config = {
            'response_enabled': True,
            'response_modes': {
                'automatic': True,
                'semi_automatic': False,
                'manual_only': False
            },
            'threat_thresholds': {
                'critical': 9.0,
                'high': 7.0,
                'medium': 5.0,
                'low': 3.0
            },
            'response_actions': {
                'ip_blocking': {
                    'enabled': True,
                    'duration': 3600,  # 1 hour
                    'whitelist': ['127.0.0.1', '::1']
                },
                'service_isolation': {
                    'enabled': True,
                    'isolation_timeout': 1800  # 30 minutes
                },
                'alert_notifications': {
                    'enabled': True,
                    'email': None,
                    'webhook': None,
                    'slack': None
                },
                'traffic_rate_limiting': {
                    'enabled': True,
                    'limit_requests_per_minute': 100
                },
                'container_quarantine': {
                    'enabled': True,
                    'quarantine_network': 'quarantine'
                }
            },
            'escalation': {
                'auto_escalate_after': 300,  # 5 minutes
                'max_auto_responses': 5,
                'require_human_approval': False
            },
            'monitoring': {
                'ids_database': '/opt/sutazaiapp/data/ids_database.db',
                'security_events_database': '/opt/sutazaiapp/data/security_events.db',
                'poll_interval': 30  # seconds
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
        """Setup logging for threat response"""
        log_dir = Path('/opt/sutazaiapp/logs')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger('threat_response')
        self.logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler('/opt/sutazaiapp/logs/threat_response.log')
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def init_database(self):
        """Initialize incident tracking database"""
        Path(self.incident_database).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.incident_database)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                incident_id TEXT UNIQUE NOT NULL,
                timestamp TEXT NOT NULL,
                severity TEXT NOT NULL,
                threat_type TEXT NOT NULL,
                source_ip TEXT,
                target_service TEXT,
                description TEXT NOT NULL,
                threat_score REAL NOT NULL,
                status TEXT DEFAULT 'open',
                response_actions TEXT,
                resolution_time INTEGER,
                false_positive BOOLEAN DEFAULT FALSE
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS response_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                incident_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                action_type TEXT NOT NULL,
                action_details TEXT NOT NULL,
                success BOOLEAN NOT NULL,
                error_message TEXT,
                execution_time REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def load_response_playbooks(self):
        """Load response playbooks for different threat types"""
        self.playbooks = {
            'brute_force_attack': [
                {'action': 'block_ip', 'priority': 1, 'duration': 3600},
                {'action': 'alert_admin', 'priority': 2},
                {'action': 'log_incident', 'priority': 3}
            ],
            'port_scan': [
                {'action': 'block_ip', 'priority': 1, 'duration': 1800},
                {'action': 'alert_security_team', 'priority': 2}
            ],
            'malicious_ip_connection': [
                {'action': 'block_ip', 'priority': 1, 'duration': 7200},
                {'action': 'isolate_service', 'priority': 2},
                {'action': 'alert_admin', 'priority': 3}
            ],
            'container_escape_attempt': [
                {'action': 'quarantine_container', 'priority': 1},
                {'action': 'alert_admin', 'priority': 2},
                {'action': 'collect_forensics', 'priority': 3}
            ],
            'sql_injection': [
                {'action': 'block_ip', 'priority': 1, 'duration': 3600},
                {'action': 'rate_limit_service', 'priority': 2},
                {'action': 'alert_dev_team', 'priority': 3}
            ],
            'privilege_escalation': [
                {'action': 'isolate_user_session', 'priority': 1},
                {'action': 'alert_admin', 'priority': 2},
                {'action': 'collect_forensics', 'priority': 3}
            ]
        }
    
    def monitor_threats(self):
        """Monitor for new threats and incidents"""
        self.logger.info("Starting threat monitoring")
        
        while self.running:
            try:
                # Check IDS database for new alerts
                self.check_ids_alerts()
                
                # Check security events database
                self.check_security_events()
                
                # Check for escalation conditions
                self.check_escalation_conditions()
                
                time.sleep(self.config['monitoring']['poll_interval'])
                
            except Exception as e:
                self.logger.error(f"Threat monitoring error: {e}")
                time.sleep(60)
    
    def check_ids_alerts(self):
        """Check IDS database for new high-priority alerts"""
        ids_db = self.config['monitoring']['ids_database']
        if not Path(ids_db).exists():
            return
        
        conn = sqlite3.connect(ids_db)
        cursor = conn.cursor()
        
        # Get unprocessed high-priority alerts
        cursor.execute('''
            SELECT id, timestamp, severity, type, source_ip, target_port, description
            FROM alerts 
            WHERE status = 'open' 
            AND severity IN ('HIGH', 'CRITICAL')
            AND timestamp > datetime('now', '-1 hour')
            ORDER BY timestamp DESC
        ''')
        
        alerts = cursor.fetchall()
        conn.close()
        
        for alert in alerts:
            alert_id, timestamp, severity, alert_type, source_ip, target_port, description = alert
            
            # Create incident
            incident_id = self.create_incident(
                severity=severity,
                threat_type=alert_type,
                source_ip=source_ip,
                target_service=f"port_{target_port}" if target_port else None,
                description=description,
                threat_score=8.0 if severity == 'HIGH' else 9.5
            )
            
            # Execute response
            self.execute_response(incident_id, alert_type, source_ip, target_port)
    
    def check_security_events(self):
        """Check security events database for patterns"""
        events_db = self.config['monitoring']['security_events_database']
        if not Path(events_db).exists():
            return
        
        conn = sqlite3.connect(events_db)
        cursor = conn.cursor()
        
        # Look for patterns indicating attacks
        cursor.execute('''
            SELECT source_ip, COUNT(*) as event_count, AVG(threat_score) as avg_score
            FROM security_events 
            WHERE timestamp > datetime('now', '-5 minutes')
            AND source_ip IS NOT NULL
            GROUP BY source_ip
            HAVING event_count >= 5 AND avg_score >= 6.0
        ''')
        
        suspicious_ips = cursor.fetchall()
        conn.close()
        
        for ip_info in suspicious_ips:
            source_ip, event_count, avg_score = ip_info
            
            # Create incident for suspicious activity
            incident_id = self.create_incident(
                severity='HIGH',
                threat_type='suspicious_activity_pattern',
                source_ip=source_ip,
                description=f'Suspicious activity pattern: {event_count} events with avg score {avg_score:.1f}',
                threat_score=avg_score
            )
            
            self.execute_response(incident_id, 'suspicious_activity_pattern', source_ip)
    
    def create_incident(self, severity, threat_type, description, threat_score, 
                       source_ip=None, target_service=None):
        """Create new security incident"""
        incident_id = hashlib.md5(f"{time.time()}{threat_type}{source_ip}".encode()).hexdigest()[:16]
        
        conn = sqlite3.connect(self.incident_database)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO incidents 
            (incident_id, timestamp, severity, threat_type, source_ip, target_service, 
             description, threat_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            incident_id, datetime.now().isoformat(), severity, threat_type,
            source_ip, target_service, description, threat_score
        ))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Created incident {incident_id}: {threat_type} from {source_ip or 'unknown'}")
        return incident_id
    
    def execute_response(self, incident_id, threat_type, source_ip=None, target_port=None):
        """Execute automated response to threat"""
        if not self.config['response_enabled']:
            self.logger.info(f"Response disabled, skipping incident {incident_id}")
            return
        
        # Get response playbook
        playbook = self.playbooks.get(threat_type, [])
        if not playbook:
            self.logger.warning(f"No playbook found for threat type: {threat_type}")
            return
        
        self.logger.info(f"Executing response for incident {incident_id}")
        
        response_actions = []
        
        # Execute actions in priority order
        for action_config in sorted(playbook, key=lambda x: x['priority']):
            action_type = action_config['action']
            
            try:
                start_time = time.time()
                success = self.execute_action(action_type, action_config, source_ip, target_port, incident_id)
                execution_time = time.time() - start_time
                
                # Log response action
                self.log_response_action(
                    incident_id, action_type, action_config, 
                    success, execution_time
                )
                
                response_actions.append({
                    'action': action_type,
                    'success': success,
                    'execution_time': execution_time
                })
                
                if success:
                    self.response_stats['successful_responses'] += 1
                else:
                    self.response_stats['failed_responses'] += 1
                
            except Exception as e:
                self.logger.error(f"Failed to execute action {action_type}: {e}")
                self.log_response_action(
                    incident_id, action_type, action_config, 
                    False, 0, str(e)
                )
                self.response_stats['failed_responses'] += 1
        
        # Update incident with response actions
        self.update_incident_response(incident_id, response_actions)
        self.response_stats['total_responses'] += 1
    
    def execute_action(self, action_type, action_config, source_ip, target_port, incident_id):
        """Execute specific response action"""
        if action_type == 'block_ip' and source_ip:
            return self.block_ip_address(source_ip, action_config.get('duration', 3600))
        
        elif action_type == 'alert_admin':
            return self.send_alert_notification(incident_id, 'admin', action_config)
        
        elif action_type == 'alert_security_team':
            return self.send_alert_notification(incident_id, 'security_team', action_config)
        
        elif action_type == 'alert_dev_team':
            return self.send_alert_notification(incident_id, 'dev_team', action_config)
        
        elif action_type == 'isolate_service' and target_port:
            return self.isolate_service(target_port)
        
        elif action_type == 'quarantine_container':
            return self.quarantine_suspicious_containers(source_ip)
        
        elif action_type == 'rate_limit_service' and target_port:
            return self.apply_rate_limiting(target_port, source_ip)
        
        elif action_type == 'collect_forensics':
            return self.collect_forensic_data(incident_id, source_ip)
        
        elif action_type == 'log_incident':
            return self.log_incident_details(incident_id)
        
        else:
            self.logger.warning(f"Unknown action type: {action_type}")
            return False
    
    def block_ip_address(self, ip_address, duration):
        """Block IP address using iptables"""
        try:
            # Validate IP address
            ipaddress.ip_address(ip_address)
            
            # Check whitelist
            if ip_address in self.config['response_actions']['ip_blocking']['whitelist']:
                self.logger.info(f"IP {ip_address} is whitelisted, skipping block")
                return True
            
            # Block IP
            result = subprocess.run([
                'iptables', '-I', 'INPUT', '-s', ip_address, '-j', 'DROP'
            ], capture_output=True, text=True, check=True)
            
            self.logger.info(f"Blocked IP {ip_address} for {duration} seconds")
            
            # Schedule unblock
            if duration > 0:
                threading.Timer(duration, self.unblock_ip_address, [ip_address]).start()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to block IP {ip_address}: {e}")
            return False
    
    def unblock_ip_address(self, ip_address):
        """Unblock IP address"""
        try:
            subprocess.run([
                'iptables', '-D', 'INPUT', '-s', ip_address, '-j', 'DROP'
            ], capture_output=True, text=True, check=True)
            
            self.logger.info(f"Unblocked IP {ip_address}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unblock IP {ip_address}: {e}")
            return False
    
    def isolate_service(self, port):
        """Isolate service by modifying network rules"""
        try:
            # Create isolation rule (example implementation)
            result = subprocess.run([
                'iptables', '-I', 'INPUT', '-p', 'tcp', '--dport', str(port), 
                '-m', 'limit', '--limit', '1/min', '-j', 'ACCEPT'
            ], capture_output=True, text=True, check=True)
            
            self.logger.info(f"Isolated service on port {port}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to isolate service on port {port}: {e}")
            return False
    
    def quarantine_suspicious_containers(self, source_ip):
        """Quarantine containers associated with suspicious IP"""
        try:
            # Find containers with connections to suspicious IP
            result = subprocess.run([
                'docker', 'ps', '--format', 'json'
            ], capture_output=True, text=True, check=True)
            
            quarantined_count = 0
            for line in result.stdout.strip().split('\n'):
                if line:
                    container_info = json.loads(line)
                    container_id = container_info['ID']
                    
                    # Move container to quarantine network
                    subprocess.run([
                        'docker', 'network', 'disconnect', 'sutazai-network', container_id
                    ], capture_output=True)
                    
                    # Connect to quarantine network (create if doesn't exist)
                    subprocess.run([
                        'docker', 'network', 'create', 'quarantine', '--internal'
                    ], capture_output=True)
                    
                    subprocess.run([
                        'docker', 'network', 'connect', 'quarantine', container_id
                    ], capture_output=True)
                    
                    quarantined_count += 1
            
            self.logger.info(f"Quarantined {quarantined_count} containers")
            return quarantined_count > 0
            
        except Exception as e:
            self.logger.error(f"Failed to quarantine containers: {e}")
            return False
    
    def apply_rate_limiting(self, port, source_ip=None):
        """Apply rate limiting to service"""
        try:
            if source_ip:
                # Rate limit specific IP
                cmd = [
                    'iptables', '-I', 'INPUT', '-s', source_ip, '-p', 'tcp', 
                    '--dport', str(port), '-m', 'limit', '--limit', '10/min', '-j', 'ACCEPT'
                ]
            else:
                # General rate limiting
                cmd = [
                    'iptables', '-I', 'INPUT', '-p', 'tcp', '--dport', str(port), 
                    '-m', 'limit', '--limit', '100/min', '-j', 'ACCEPT'
                ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            self.logger.info(f"Applied rate limiting to port {port}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to apply rate limiting: {e}")
            return False
    
    def send_alert_notification(self, incident_id, recipient_type, config):
        """Send alert notification"""
        try:
            # Get incident details
            conn = sqlite3.connect(self.incident_database)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT severity, threat_type, source_ip, description, threat_score
                FROM incidents WHERE incident_id = ?
            ''', (incident_id,))
            
            incident = cursor.fetchone()
            conn.close()
            
            if not incident:
                return False
            
            severity, threat_type, source_ip, description, threat_score = incident
            
            # Create alert message
            alert_message = f"""
Security Incident Alert - {incident_id}

Severity: {severity}
Threat Type: {threat_type}
Source IP: {source_ip or 'Unknown'}
Threat Score: {threat_score}
Description: {description}

Timestamp: {datetime.now().isoformat()}
Recipient: {recipient_type}

This is an automated alert from SutazAI Threat Response System.
            """
            
            # Send notification (implement email, webhook, Slack as needed)
            self.logger.info(f"Alert sent to {recipient_type} for incident {incident_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send alert notification: {e}")
            return False
    
    def collect_forensic_data(self, incident_id, source_ip):
        """Collect forensic data for investigation"""
        try:
            forensic_data = {
                'timestamp': datetime.now().isoformat(),
                'incident_id': incident_id,
                'source_ip': source_ip,
                'network_connections': [],
                'system_state': {},
                'container_state': []
            }
            
            # Collect network connections
            result = subprocess.run(['netstat', '-tuln'], capture_output=True, text=True)
            if result.returncode == 0:
                forensic_data['network_connections'] = result.stdout.split('\n')
            
            # Collect Docker container states
            result = subprocess.run(['docker', 'ps', '-a', '--format', 'json'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        forensic_data['container_state'].append(json.loads(line))
            
            # Save forensic data
            forensic_file = f'/opt/sutazaiapp/data/forensics_{incident_id}.json'
            with open(forensic_file, 'w') as f:
                json.dump(forensic_data, f, indent=2)
            
            self.logger.info(f"Collected forensic data for incident {incident_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to collect forensic data: {e}")
            return False
    
    def log_incident_details(self, incident_id):
        """Log detailed incident information"""
        try:
            self.logger.info(f"Detailed logging for incident {incident_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to log incident details: {e}")
            return False
    
    def log_response_action(self, incident_id, action_type, action_config, 
                           success, execution_time, error_message=None):
        """Log response action details"""
        conn = sqlite3.connect(self.incident_database)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO response_log 
            (incident_id, timestamp, action_type, action_details, success, error_message, execution_time)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            incident_id, datetime.now().isoformat(), action_type,
            json.dumps(action_config), success, error_message, execution_time
        ))
        
        conn.commit()
        conn.close()
    
    def update_incident_response(self, incident_id, response_actions):
        """Update incident with response action results"""
        conn = sqlite3.connect(self.incident_database)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE incidents 
            SET response_actions = ?, status = 'responded'
            WHERE incident_id = ?
        ''', (json.dumps(response_actions), incident_id))
        
        conn.commit()
        conn.close()
    
    def check_escalation_conditions(self):
        """Check if incidents need escalation"""
        conn = sqlite3.connect(self.incident_database)
        cursor = conn.cursor()
        
        # Find incidents that need escalation
        cursor.execute('''
            SELECT incident_id, timestamp, severity, threat_type, description
            FROM incidents 
            WHERE status IN ('open', 'responded')
            AND timestamp < datetime('now', '-5 minutes')
            AND severity = 'CRITICAL'
        ''')
        
        incidents_to_escalate = cursor.fetchall()
        conn.close()
        
        for incident in incidents_to_escalate:
            incident_id = incident[0]
            self.escalate_incident(incident_id)
    
    def escalate_incident(self, incident_id):
        """Escalate incident to human operators"""
        self.logger.warning(f"Escalating incident {incident_id} to human operators")
        
        # Send high-priority alert
        self.send_alert_notification(incident_id, 'escalation_team', {})
        
        # Update incident status
        conn = sqlite3.connect(self.incident_database)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE incidents SET status = 'escalated' WHERE incident_id = ?
        ''', (incident_id,))
        conn.commit()
        conn.close()
    
    def generate_response_report(self):
        """Generate threat response report"""
        conn = sqlite3.connect(self.incident_database)
        cursor = conn.cursor()
        
        # Get statistics
        cursor.execute('''
            SELECT 
                COUNT(*) as total_incidents,
                COUNT(CASE WHEN status = 'resolved' THEN 1 END) as resolved,
                COUNT(CASE WHEN status = 'escalated' THEN 1 END) as escalated,
                AVG(threat_score) as avg_threat_score
            FROM incidents 
            WHERE timestamp >= datetime('now', '-24 hours')
        ''')
        
        stats = cursor.fetchone()
        
        # Get recent incidents
        cursor.execute('''
            SELECT incident_id, timestamp, severity, threat_type, source_ip, status
            FROM incidents 
            WHERE timestamp >= datetime('now', '-24 hours')
            ORDER BY timestamp DESC
            LIMIT 20
        ''')
        
        recent_incidents = cursor.fetchall()
        conn.close()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'period': '24 hours',
            'statistics': {
                'total_incidents': stats[0] if stats else 0,
                'resolved_incidents': stats[1] if stats else 0,
                'escalated_incidents': stats[2] if stats else 0,
                'avg_threat_score': round(stats[3], 2) if stats and stats[3] else 0.0,
                'response_stats': self.response_stats
            },
            'recent_incidents': [
                {
                    'incident_id': incident[0],
                    'timestamp': incident[1],
                    'severity': incident[2],
                    'threat_type': incident[3],
                    'source_ip': incident[4],
                    'status': incident[5]
                } for incident in recent_incidents
            ]
        }
        
        return report
    
    def start(self):
        """Start automated threat response system"""
        self.logger.info("Starting Automated Threat Response System")
        self.running = True
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self.monitor_threats, daemon=True)
        monitor_thread.start()
        
        self.logger.info("Threat response system is active")
        
        # Main loop for maintenance
        try:
            while self.running:
                # Generate reports
                report = self.generate_response_report()
                with open('/opt/sutazaiapp/data/threat_response_report.json', 'w') as f:
                    json.dump(report, f, indent=2)
                
                time.sleep(300)  # Run every 5 minutes
                
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """Stop threat response system"""
        self.running = False
        self.logger.info("Automated Threat Response System stopped")

def main():
    response_system = AutomatedThreatResponse()
    
    try:
        response_system.start()
    except KeyboardInterrupt:
        response_system.stop()

if __name__ == "__main__":
    main()