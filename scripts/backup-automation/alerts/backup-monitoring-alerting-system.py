#!/usr/bin/env python3
"""
SutazAI Backup Monitoring and Alerting System
Monitors backup processes and sends alerts for failures or issues
"""

import os
import sys
import json
import logging
import datetime
import time
import smtplib
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/logs/backup-monitoring.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BackupMonitoringAlertingSystem:
    """Backup monitoring and alerting system"""
    
    def __init__(self, backup_root: str = "/opt/sutazaiapp/data/backups"):
        self.backup_root = Path(backup_root)
        self.monitoring_dir = self.backup_root / 'monitoring'
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure monitoring directory exists
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)
        
        # Load monitoring configuration
        self.config = self.load_monitoring_config()
        
        # Alert thresholds and conditions
        self.alert_conditions = {
            'backup_failure': 'Any backup process fails',
            'backup_missing': 'No backup found within expected timeframe',
            'backup_size_anomaly': 'Backup size deviates significantly from norm',
            'verification_failure': 'Backup verification fails',
            'offsite_failure': 'Offsite replication fails',
            'storage_space_low': 'Backup storage space below threshold',
            'restore_test_failure': 'Restore test fails'
        }
        
        # Notification methods
        self.notification_methods = {
            'email': self.send_email_alert,
            'slack': self.send_slack_alert,
            'webhook': self.send_webhook_alert,
            'sms': self.send_sms_alert,
            'system_log': self.send_system_log_alert
        }
    
    def load_monitoring_config(self) -> Dict:
        """Load monitoring and alerting configuration"""
        config_file = Path('/opt/sutazaiapp/config/backup-monitoring-config.json')
        
        default_config = {
            'monitoring': {
                'enabled': True,
                'check_interval_minutes': 60,
                'backup_max_age_hours': 26,  # Alert if no backup in 26 hours
                'backup_size_deviation_percent': 50,  # Alert if size changes by 50%
                'storage_threshold_percent': 90,  # Alert if storage 90% full
                'retention_days': 30
            },
            'alerts': {
                'enabled': True,
                'methods': ['system_log', 'email'],
                'severity_levels': ['critical', 'warning', 'info'],
                'rate_limiting': {
                    'max_alerts_per_hour': 10,
                    'cooldown_minutes': 30
                }
            },
            'email': {
                'enabled': False,
                'smtp_host': 'localhost',
                'smtp_port': 587,
                'smtp_user': '',
                'smtp_password_env': 'BACKUP_SMTP_PASSWORD',
                'from_address': 'backups@sutazai.local',
                'to_addresses': ['admin@sutazai.local'],
                'use_tls': True
            },
            'slack': {
                'enabled': False,
                'webhook_url_env': 'BACKUP_SLACK_WEBHOOK',
                'channel': '#backups',
                'username': 'SutazAI Backup Monitor'
            },
            'webhook': {
                'enabled': False,
                'url': '',
                'method': 'POST',
                'headers': {},
                'timeout': 30
            },
            'sms': {
                'enabled': False,
                'service': 'twilio',
                'account_sid_env': 'TWILIO_ACCOUNT_SID',
                'auth_token_env': 'TWILIO_AUTH_TOKEN',
                'from_number': '',
                'to_numbers': []
            }
        }
        
        try:
            if config_file.exists():
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                # Merge configurations
                return {**default_config, **loaded_config}
            else:
                # Create default config
                config_file.parent.mkdir(parents=True, exist_ok=True)
                with open(config_file, 'w') as f:
                    json.dump(default_config, f, indent=2)
                return default_config
        except Exception as e:
            logger.error(f"Error loading monitoring config: {e}")
            return default_config
    
    def discover_backup_status(self) -> Dict:
        """Discover current backup status across all categories"""
        backup_status = {
            'categories': {},
            'overall_health': 'unknown',
            'last_backup_time': None,
            'total_backup_size': 0,
            'issues': []
        }
        
        backup_categories = ['daily', 'weekly', 'monthly', 'postgres', 'sqlite', 
                           'config', 'agents', 'models', 'monitoring', 'logs']
        
        latest_backup_time = None
        
        for category in backup_categories:
            category_path = self.backup_root / category
            category_status = {
                'exists': category_path.exists(),
                'file_count': 0,
                'total_size': 0,
                'latest_backup': None,
                'latest_backup_age_hours': None,
                'manifests': []
            }
            
            if category_path.exists():
                # Scan for backup files
                backup_files = []
                manifest_files = []
                
                for item in category_path.rglob('*'):
                    if item.is_file():
                        try:
                            file_info = {
                                'path': str(item),
                                'size': item.stat().st_size,
                                'modified': datetime.datetime.fromtimestamp(item.stat().st_mtime)
                            }
                            
                            if 'manifest' in item.name.lower():
                                manifest_files.append(file_info)
                            else:
                                backup_files.append(file_info)
                            
                            category_status['total_size'] += file_info['size']
                            
                        except Exception as e:
                            logger.warning(f"Error processing {item}: {e}")
                
                category_status['file_count'] = len(backup_files)
                category_status['manifests'] = manifest_files
                
                # Find latest backup
                if backup_files:
                    latest_backup = max(backup_files, key=lambda x: x['modified'])
                    category_status['latest_backup'] = latest_backup
                    
                    age_hours = (datetime.datetime.now() - latest_backup['modified']).total_seconds() / 3600
                    category_status['latest_backup_age_hours'] = age_hours
                    
                    if latest_backup_time is None or latest_backup['modified'] > latest_backup_time:
                        latest_backup_time = latest_backup['modified']
            
            backup_status['categories'][category] = category_status
            backup_status['total_backup_size'] += category_status['total_size']
        
        # Set overall backup time
        if latest_backup_time:
            backup_status['last_backup_time'] = latest_backup_time.isoformat()
        
        # Determine overall health
        backup_status['overall_health'] = self.assess_overall_health(backup_status)
        
        return backup_status
    
    def assess_overall_health(self, backup_status: Dict) -> str:
        """Assess overall backup system health"""
        issues = []
        
        max_age_hours = self.config['monitoring']['backup_max_age_hours']
        
        # Check if any backups are too old
        for category, status in backup_status['categories'].items():
            if status['exists'] and status['latest_backup_age_hours']:
                if status['latest_backup_age_hours'] > max_age_hours:
                    issues.append(f"{category} backup is {status['latest_backup_age_hours']:.1f} hours old")
        
        # Check storage space
        storage_usage = self.check_storage_usage()
        if storage_usage['usage_percent'] > self.config['monitoring']['storage_threshold_percent']:
            issues.append(f"Backup storage {storage_usage['usage_percent']:.1f}% full")
        
        # Check for recent failures
        recent_failures = self.check_recent_failures()
        if recent_failures:
            issues.extend(recent_failures)
        
        backup_status['issues'] = issues
        
        if not issues:
            return 'healthy'
        elif len(issues) <= 2:
            return 'warning'
        else:
            return 'critical'
    
    def check_storage_usage(self) -> Dict:
        """Check backup storage space usage"""
        try:
            statvfs = os.statvfs(str(self.backup_root))
            
            total_bytes = statvfs.f_frsize * statvfs.f_blocks
            available_bytes = statvfs.f_frsize * statvfs.f_bavail
            used_bytes = total_bytes - available_bytes
            
            usage_percent = (used_bytes / total_bytes) * 100 if total_bytes > 0 else 0
            
            return {
                'total_bytes': total_bytes,
                'used_bytes': used_bytes,
                'available_bytes': available_bytes,
                'usage_percent': usage_percent
            }
            
        except Exception as e:
            logger.error(f"Error checking storage usage: {e}")
            return {
                'total_bytes': 0,
                'used_bytes': 0,
                'available_bytes': 0,
                'usage_percent': 0,
                'error': str(e)
            }
    
    def check_recent_failures(self) -> List[str]:
        """Check for recent backup failures in logs"""
        failures = []
        
        try:
            # Check recent log files for failure indicators
            log_patterns = [
                '/opt/sutazaiapp/logs/database-backup.log',
                '/opt/sutazaiapp/logs/config-backup.log',
                '/opt/sutazaiapp/logs/agent-state-backup.log',
                '/opt/sutazaiapp/logs/ollama-model-backup.log',
                '/opt/sutazaiapp/logs/offsite-backup.log'
            ]
            
            failure_keywords = ['ERROR', 'FAILED', 'failed', 'error:', 'Exception:', 'CRITICAL']
            
            for log_path in log_patterns:
                if os.path.exists(log_path):
                    try:
                        # Read last 100 lines of log file
                        result = subprocess.run(
                            ['tail', '-100', log_path],
                            capture_output=True,
                            text=True,
                            timeout=10
                        )
                        
                        if result.returncode == 0:
                            log_content = result.stdout
                            
                            # Look for recent failures (last 24 hours)
                            lines = log_content.split('\n')
                            for line in lines:
                                if any(keyword in line for keyword in failure_keywords):
                                    # Try to extract timestamp
                                    if datetime.datetime.now().strftime('%Y-%m-%d') in line:
                                        failures.append(f"Recent failure in {os.path.basename(log_path)}")
                                        break
                                        
                    except Exception as e:
                        logger.warning(f"Error checking log {log_path}: {e}")
            
        except Exception as e:
            logger.error(f"Error checking recent failures: {e}")
        
        return failures
    
    def analyze_backup_trends(self) -> Dict:
        """Analyze backup size and frequency trends"""
        trends = {
            'size_trend': 'stable',
            'frequency_trend': 'stable',
            'anomalies': []
        }
        
        try:
            # Read historical monitoring data
            historical_data = self.load_historical_monitoring_data()
            
            if len(historical_data) >= 7:  # Need at least a week of data
                # Analyze size trends
                recent_sizes = [entry['total_backup_size'] for entry in historical_data[-7:]]
                older_sizes = [entry['total_backup_size'] for entry in historical_data[-14:-7]] if len(historical_data) >= 14 else recent_sizes
                
                if recent_sizes and older_sizes:
                    recent_avg = sum(recent_sizes) / len(recent_sizes)
                    older_avg = sum(older_sizes) / len(older_sizes)
                    
                    if older_avg > 0:
                        size_change_percent = ((recent_avg - older_avg) / older_avg) * 100
                        
                        if abs(size_change_percent) > self.config['monitoring']['backup_size_deviation_percent']:
                            trends['size_trend'] = 'increasing' if size_change_percent > 0 else 'decreasing'
                            trends['anomalies'].append(f"Backup size changed by {size_change_percent:.1f}%")
                
        except Exception as e:
            logger.warning(f"Error analyzing backup trends: {e}")
        
        return trends
    
    def load_historical_monitoring_data(self) -> List[Dict]:
        """Load historical monitoring data"""
        try:
            # Look for recent monitoring reports
            monitoring_files = list(self.monitoring_dir.glob('backup_monitoring_report_*.json'))
            monitoring_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            historical_data = []
            
            for monitoring_file in monitoring_files[:30]:  # Last 30 reports
                try:
                    with open(monitoring_file, 'r') as f:
                        data = json.load(f)
                        historical_data.append(data)
                except Exception as e:
                    logger.warning(f"Error loading monitoring file {monitoring_file}: {e}")
            
            return historical_data
            
        except Exception as e:
            logger.error(f"Error loading historical monitoring data: {e}")
            return []
    
    def generate_alert(self, alert_type: str, severity: str, message: str, details: Dict = None) -> Dict:
        """Generate an alert"""
        alert = {
            'timestamp': datetime.datetime.now().isoformat(),
            'alert_id': f"{alert_type}_{self.timestamp}_{int(time.time())}",
            'alert_type': alert_type,
            'severity': severity,
            'message': message,
            'details': details or {},
            'hostname': os.uname().nodename,
            'system': 'SutazAI Backup System'
        }
        
        # Check rate limiting
        if self.should_send_alert(alert):
            # Send alert via configured methods
            self.send_alert(alert)
            
            # Log alert
            self.log_alert(alert)
            
            return alert
        else:
            logger.info(f"Alert rate limited: {alert_type}")
            return None
    
    def should_send_alert(self, alert: Dict) -> bool:
        """Check if alert should be sent based on rate limiting"""
        try:
            rate_limit_config = self.config['alerts']['rate_limiting']
            max_alerts_per_hour = rate_limit_config.get('max_alerts_per_hour', 10)
            cooldown_minutes = rate_limit_config.get('cooldown_minutes', 30)
            
            # Read recent alerts
            alert_log_file = self.monitoring_dir / 'alert_log.json'
            recent_alerts = []
            
            if alert_log_file.exists():
                with open(alert_log_file, 'r') as f:
                    recent_alerts = json.load(f)
            
            # Filter alerts from last hour
            one_hour_ago = datetime.datetime.now() - datetime.timedelta(hours=1)
            recent_hour_alerts = [
                a for a in recent_alerts 
                if datetime.datetime.fromisoformat(a['timestamp']) > one_hour_ago
            ]
            
            # Check rate limit
            if len(recent_hour_alerts) >= max_alerts_per_hour:
                return False
            
            # Check cooldown for same alert type
            cooldown_time = datetime.datetime.now() - datetime.timedelta(minutes=cooldown_minutes)
            same_type_recent = [
                a for a in recent_alerts
                if (a['alert_type'] == alert['alert_type'] and 
                    datetime.datetime.fromisoformat(a['timestamp']) > cooldown_time)
            ]
            
            if same_type_recent:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking alert rate limiting: {e}")
            return True  # Default to allowing alert
    
    def send_alert(self, alert: Dict):
        """Send alert via configured notification methods"""
        enabled_methods = self.config['alerts'].get('methods', ['system_log'])
        
        for method in enabled_methods:
            if method in self.notification_methods:
                try:
                    self.notification_methods[method](alert)
                    logger.info(f"Alert sent via {method}: {alert['alert_type']}")
                except Exception as e:
                    logger.error(f"Error sending alert via {method}: {e}")
    
    def send_email_alert(self, alert: Dict):
        """Send alert via email"""
        email_config = self.config.get('email', {})
        
        if not email_config.get('enabled', False):
            return
        
        try:
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = email_config['from_address']
            msg['To'] = ', '.join(email_config['to_addresses'])
            msg['Subject'] = f"[{alert['severity'].upper()}] SutazAI Backup Alert: {alert['alert_type']}"
            
            # Create email body
            body = f"""
SutazAI Backup System Alert

Alert Type: {alert['alert_type']}
Severity: {alert['severity'].upper()}
Timestamp: {alert['timestamp']}
Hostname: {alert['hostname']}

Message: {alert['message']}

Details:
{json.dumps(alert['details'], indent=2)}

Alert ID: {alert['alert_id']}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            smtp_password = os.environ.get(email_config.get('smtp_password_env', ''))
            
            server = smtplib.SMTP(email_config['smtp_host'], email_config['smtp_port'])
            
            if email_config.get('use_tls', True):
                server.starttls()
            
            if email_config.get('smtp_user') and smtp_password:
                server.login(email_config['smtp_user'], smtp_password)
            
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            logger.error(f"Error sending email alert: {e}")
    
    def send_slack_alert(self, alert: Dict):
        """Send alert via Slack webhook"""
        slack_config = self.config.get('slack', {})
        
        if not slack_config.get('enabled', False):
            return
        
        try:
            webhook_url = os.environ.get(slack_config.get('webhook_url_env', ''))
            if not webhook_url:
                return
            
            # Create Slack message
            color = {
                'critical': 'danger',
                'warning': 'warning',
                'info': 'good'
            }.get(alert['severity'], 'warning')
            
            slack_message = {
                'channel': slack_config.get('channel', '#backups'),
                'username': slack_config.get('username', 'SutazAI Backup Monitor'),
                'attachments': [{
                    'color': color,
                    'title': f"Backup Alert: {alert['alert_type']}",
                    'text': alert['message'],
                    'fields': [
                        {'title': 'Severity', 'value': alert['severity'].upper(), 'short': True},
                        {'title': 'Hostname', 'value': alert['hostname'], 'short': True},
                        {'title': 'Timestamp', 'value': alert['timestamp'], 'short': False}
                    ],
                    'footer': f"Alert ID: {alert['alert_id']}"
                }]
            }
            
            response = requests.post(webhook_url, json=slack_message, timeout=30)
            response.raise_for_status()
            
        except Exception as e:
            logger.error(f"Error sending Slack alert: {e}")
    
    def send_webhook_alert(self, alert: Dict):
        """Send alert via generic webhook"""
        webhook_config = self.config.get('webhook', {})
        
        if not webhook_config.get('enabled', False):
            return
        
        try:
            url = webhook_config.get('url')
            if not url:
                return
            
            method = webhook_config.get('method', 'POST').upper()
            headers = webhook_config.get('headers', {})
            timeout = webhook_config.get('timeout', 30)
            
            if method == 'POST':
                response = requests.post(url, json=alert, headers=headers, timeout=timeout)
            elif method == 'PUT':
                response = requests.put(url, json=alert, headers=headers, timeout=timeout)
            else:
                response = requests.get(url, params={'alert': json.dumps(alert)}, headers=headers, timeout=timeout)
            
            response.raise_for_status()
            
        except Exception as e:
            logger.error(f"Error sending webhook alert: {e}")
    
    def send_sms_alert(self, alert: Dict):
        """Send alert via SMS (Twilio)"""
        sms_config = self.config.get('sms', {})
        
        if not sms_config.get('enabled', False):
            return
        
        try:
            if sms_config.get('service') == 'twilio':
                from twilio.rest import Client
                
                account_sid = os.environ.get(sms_config.get('account_sid_env', ''))
                auth_token = os.environ.get(sms_config.get('auth_token_env', ''))
                
                if not (account_sid and auth_token):
                    return
                
                client = Client(account_sid, auth_token)
                
                message_body = f"SutazAI Backup Alert [{alert['severity'].upper()}]: {alert['message']}"
                
                for to_number in sms_config.get('to_numbers', []):
                    client.messages.create(
                        body=message_body,
                        from_=sms_config.get('from_number'),
                        to=to_number
                    )
            
        except Exception as e:
            logger.error(f"Error sending SMS alert: {e}")
    
    def send_system_log_alert(self, alert: Dict):
        """Send alert to system log"""
        try:
            log_message = f"BACKUP ALERT [{alert['severity'].upper()}] {alert['alert_type']}: {alert['message']}"
            
            if alert['severity'] == 'critical':
                logger.critical(log_message)
            elif alert['severity'] == 'warning':
                logger.warning(log_message)
            else:
                logger.info(log_message)
                
        except Exception as e:
            logger.error(f"Error sending system log alert: {e}")
    
    def log_alert(self, alert: Dict):
        """Log alert to alert history"""
        try:
            alert_log_file = self.monitoring_dir / 'alert_log.json'
            
            # Load existing alerts
            alerts = []
            if alert_log_file.exists():
                with open(alert_log_file, 'r') as f:
                    alerts = json.load(f)
            
            # Add new alert
            alerts.append(alert)
            
            # Keep only recent alerts (last 1000)
            alerts = alerts[-1000:]
            
            # Save alerts
            with open(alert_log_file, 'w') as f:
                json.dump(alerts, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error logging alert: {e}")
    
    def run_monitoring_cycle(self) -> Dict:
        """Run a complete monitoring cycle"""
        start_time = time.time()
        logger.info(f"Starting backup monitoring cycle - {self.timestamp}")
        
        # Check if monitoring is enabled
        if not self.config['monitoring'].get('enabled', True):
            return {
                'timestamp': self.timestamp,
                'status': 'disabled',
                'message': 'Backup monitoring is disabled'
            }
        
        # Discover current backup status
        backup_status = self.discover_backup_status()
        
        # Check storage usage
        storage_usage = self.check_storage_usage()
        
        # Analyze trends
        trends = self.analyze_backup_trends()
        
        # Generate alerts based on findings
        alerts_generated = []
        
        # Check for backup age issues
        max_age_hours = self.config['monitoring']['backup_max_age_hours']
        for category, status in backup_status['categories'].items():
            if (status['exists'] and status['latest_backup_age_hours'] and 
                status['latest_backup_age_hours'] > max_age_hours):
                
                alert = self.generate_alert(
                    'backup_missing',
                    'warning',
                    f"{category} backup is {status['latest_backup_age_hours']:.1f} hours old (threshold: {max_age_hours}h)",
                    {
                        'category': category,
                        'age_hours': status['latest_backup_age_hours'],
                        'threshold_hours': max_age_hours
                    }
                )
                if alert:
                    alerts_generated.append(alert)
        
        # Check storage space
        storage_threshold = self.config['monitoring']['storage_threshold_percent']
        if storage_usage['usage_percent'] > storage_threshold:
            alert = self.generate_alert(
                'storage_space_low',
                'critical' if storage_usage['usage_percent'] > 95 else 'warning',
                f"Backup storage {storage_usage['usage_percent']:.1f}% full (threshold: {storage_threshold}%)",
                {
                    'usage_percent': storage_usage['usage_percent'],
                    'threshold_percent': storage_threshold,
                    'available_gb': storage_usage['available_bytes'] / (1024**3)
                }
            )
            if alert:
                alerts_generated.append(alert)
        
        # Check for size anomalies
        for anomaly in trends['anomalies']:
            alert = self.generate_alert(
                'backup_size_anomaly',
                'warning',
                f"Backup size anomaly detected: {anomaly}",
                {'anomaly': anomaly, 'trend': trends['size_trend']}
            )
            if alert:
                alerts_generated.append(alert)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Create monitoring report
        monitoring_report = {
            'timestamp': self.timestamp,
            'monitoring_date': datetime.datetime.now().isoformat(),
            'duration_seconds': duration,
            'backup_status': backup_status,
            'storage_usage': storage_usage,
            'trends': trends,
            'alerts_generated': len(alerts_generated),
            'alerts': alerts_generated,
            'monitoring_config': {
                'check_interval_minutes': self.config['monitoring']['check_interval_minutes'],
                'backup_max_age_hours': self.config['monitoring']['backup_max_age_hours'],
                'storage_threshold_percent': self.config['monitoring']['storage_threshold_percent']
            }
        }
        
        # Save monitoring report
        report_file = self.monitoring_dir / f"backup_monitoring_report_{self.timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(monitoring_report, f, indent=2)
        
        # Clean up old monitoring reports
        self.cleanup_old_monitoring_reports()
        
        logger.info(f"Backup monitoring cycle completed in {duration:.2f} seconds")
        logger.info(f"Overall health: {backup_status['overall_health']}, Alerts generated: {len(alerts_generated)}")
        
        return monitoring_report
    
    def cleanup_old_monitoring_reports(self):
        """Clean up old monitoring reports"""
        try:
            retention_days = self.config['monitoring'].get('retention_days', 30)
            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=retention_days)
            
            for report_file in self.monitoring_dir.glob('backup_monitoring_report_*.json'):
                try:
                    file_time = datetime.datetime.fromtimestamp(report_file.stat().st_mtime)
                    if file_time < cutoff_date:
                        report_file.unlink()
                        logger.info(f"Cleaned up old monitoring report: {report_file}")
                except Exception as e:
                    logger.warning(f"Error cleaning up report {report_file}: {e}")
                    
        except Exception as e:
            logger.error(f"Error cleaning up old monitoring reports: {e}")

def main():
    """Main entry point"""
    try:
        monitoring_system = BackupMonitoringAlertingSystem()
        result = monitoring_system.run_monitoring_cycle()
        
        # Write summary to log
        summary_file = f"/opt/sutazaiapp/logs/backup_monitoring_summary_{monitoring_system.timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Exit with appropriate code based on health status
        overall_health = result.get('backup_status', {}).get('overall_health', 'unknown')
        if overall_health == 'critical':
            sys.exit(2)
        elif overall_health == 'warning':
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Backup monitoring failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()