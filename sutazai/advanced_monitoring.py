#!/usr/bin/env python3

import os
import sys
import subprocess
import logging
import json
from datetime import datetime
import psutil
import socket
import threading
import time
import smtplib
import traceback
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class SutazAIMonitor:
    def __init__(self, project_root="/opt/sutazai_project/SutazAI"):
        self.project_root = project_root
        self.log_dir = os.path.join(project_root, 'logs', 'monitoring')
        self.health_log_file = os.path.join(self.log_dir, 'system_health.log')
        self.sync_log_file = os.path.join(self.log_dir, 'sync_status.log')
        self.error_log_file = os.path.join(self.log_dir, 'critical_errors.log')
        
        # Notification Configuration
        self.notification_config = {
            'email': {
                'enabled': True,
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'sender_email': 'sutazai.monitor@gmail.com',
                'sender_password': self._load_email_credentials(),
                'recipient_emails': ['admin@sutazai.com']
            },
            'threshold': {
                'cpu_usage': 90,
                'memory_usage': 90,
                'disk_usage': 90
            }
        }
        
        # Ensure log directories exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler(self.health_log_file),
                logging.FileHandler(self.error_log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)

    def _load_email_credentials(self):
        """Load email credentials securely."""
        try:
            # In a real-world scenario, use a secure credential management system
            # This is a placeholder - replace with actual secure method
            return os.environ.get('SUTAZAI_EMAIL_PASSWORD', '')
        except Exception as e:
            self.logger.error(f"Error loading email credentials: {e}")
            return ''

    def send_notification(self, subject, message, is_critical=False):
        """Send email notification."""
        if not self.notification_config['email']['enabled']:
            return

        try:
            msg = MIMEMultipart()
            msg['From'] = self.notification_config['email']['sender_email']
            msg['To'] = ', '.join(self.notification_config['email']['recipient_emails'])
            msg['Subject'] = f"{'🚨 CRITICAL: ' if is_critical else ''}SutazAI Monitor - {subject}"
            
            msg.attach(MIMEText(message, 'plain'))
            
            with smtplib.SMTP(
                self.notification_config['email']['smtp_server'], 
                self.notification_config['email']['smtp_port']
            ) as server:
                server.starttls()
                server.login(
                    self.notification_config['email']['sender_email'], 
                    self.notification_config['email']['sender_password']
                )
                server.send_message(msg)
            
            self.logger.info(f"Notification sent: {subject}")
        except Exception as e:
            self.logger.error(f"Failed to send notification: {e}")
            self._log_critical_error(f"Notification Error: {e}")

    def _log_critical_error(self, error_message):
        """Log critical errors to a separate file."""
        try:
            with open(self.error_log_file, 'a') as f:
                error_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'error': error_message,
                    'traceback': traceback.format_exc()
                }
                json.dump(error_entry, f)
                f.write('\n')
        except Exception as e:
            print(f"Error logging critical error: {e}")

    def get_system_metrics(self):
        """Collect comprehensive system metrics."""
        try:
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'cpu': {
                    'usage': psutil.cpu_percent(interval=1),
                    'cores': psutil.cpu_count(logical=False),
                    'logical_cores': psutil.cpu_count(logical=True)
                },
                'memory': {
                    'total': psutil.virtual_memory().total / (1024 ** 3),  # GB
                    'available': psutil.virtual_memory().available / (1024 ** 3),  # GB
                    'percent_used': psutil.virtual_memory().percent
                },
                'disk': {
                    'total': psutil.disk_usage('/').total / (1024 ** 3),  # GB
                    'free': psutil.disk_usage('/').free / (1024 ** 3),  # GB
                    'percent_used': psutil.disk_usage('/').percent
                },
                'network': {
                    'hostname': socket.gethostname(),
                    'ip_address': socket.gethostbyname(socket.gethostname())
                }
            }
            return metrics
        except Exception as e:
            self._log_critical_error(f"System Metrics Collection Error: {e}")
            return None

    def log_system_health(self):
        """Log system health metrics and send alerts."""
        try:
            metrics = self.get_system_metrics()
            if not metrics:
                return

            # Log metrics to file
            with open(os.path.join(self.log_dir, 'system_health_metrics.json'), 'a') as f:
                json.dump(metrics, f)
                f.write('\n')
            
            # Log system health
            self.logger.info(
                f"System Health: "
                f"CPU {metrics['cpu']['usage']}%, "
                f"Memory {metrics['memory']['percent_used']}%, "
                f"Disk {metrics['disk']['percent_used']}%"
            )
            
            # Check and send alerts for critical conditions
            critical_alerts = []
            if metrics['cpu']['usage'] > self.notification_config['threshold']['cpu_usage']:
                critical_alerts.append(f"High CPU Usage: {metrics['cpu']['usage']}%")
            
            if metrics['memory']['percent_used'] > self.notification_config['threshold']['memory_usage']:
                critical_alerts.append(f"High Memory Usage: {metrics['memory']['percent_used']}%")
            
            if metrics['disk']['percent_used'] > self.notification_config['threshold']['disk_usage']:
                critical_alerts.append(f"High Disk Usage: {metrics['disk']['percent_used']}%")
            
            # Send critical alerts if any
            if critical_alerts:
                alert_message = "Critical System Resource Alerts:\n" + "\n".join(critical_alerts)
                self.logger.warning(alert_message)
                self.send_notification("System Resource Alert", alert_message, is_critical=True)
        
        except Exception as e:
            self._log_critical_error(f"System Health Logging Error: {e}")

    def monitor_project_sync(self):
        """Monitor project synchronization status with enhanced error handling."""
        try:
            sync_script_path = os.path.join(self.project_root, 'server_sync.sh')
            
            # Run sync script and capture output
            result = subprocess.run(
                [sync_script_path], 
                capture_output=True, 
                text=True,
                timeout=1800  # 30-minute timeout
            )
            
            sync_status = {
                'timestamp': datetime.now().isoformat(),
                'exit_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
            # Log sync status
            with open(self.sync_log_file, 'a') as f:
                json.dump(sync_status, f)
                f.write('\n')
            
            # Handle sync results
            if result.returncode == 0:
                self.logger.info("Project Synchronization Successful")
            else:
                error_msg = f"Project Synchronization Failed\nExit Code: {result.returncode}\nError: {result.stderr}"
                self.logger.error(error_msg)
                self.send_notification("Sync Failure", error_msg, is_critical=True)
                self._log_critical_error(error_msg)
        
        except subprocess.TimeoutExpired:
            timeout_msg = "Project Synchronization Timed Out"
            self.logger.error(timeout_msg)
            self.send_notification("Sync Timeout", timeout_msg, is_critical=True)
        
        except Exception as e:
            error_msg = f"Error during project sync monitoring: {e}"
            self.logger.error(error_msg)
            self.send_notification("Sync Monitoring Error", error_msg, is_critical=True)
            self._log_critical_error(error_msg)

    def continuous_monitoring(self, interval=300):
        """Run continuous monitoring in background with error recovery."""
        def monitor_thread():
            while True:
                try:
                    self.log_system_health()
                    self.monitor_project_sync()
                    time.sleep(interval)
                except Exception as e:
                    error_msg = f"Monitoring Thread Error: {e}"
                    self.logger.error(error_msg)
                    self.send_notification("Monitoring Thread Failure", error_msg, is_critical=True)
                    time.sleep(interval)  # Prevent rapid error loops
        
        thread = threading.Thread(target=monitor_thread, daemon=True)
        thread.start()
        return thread

def main():
    monitor = SutazAIMonitor()
    
    # Initial startup notification
    monitor.send_notification(
        "System Monitoring Started", 
        "SutazAI monitoring system has been initialized and is now running."
    )
    
    # Start continuous monitoring
    monitor_thread = monitor.continuous_monitoring()
    
    try:
        # Keep main thread running
        monitor_thread.join()
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
        monitor.send_notification(
            "Monitoring Stopped", 
            "SutazAI monitoring system has been manually stopped."
        )

if __name__ == "__main__":
    main() 