#!/usr/bin/env python3.11
"""
SutazAI System Monitoring Script

This script provides comprehensive system monitoring capabilities including:
- Resource monitoring (CPU, memory, disk)
- Service health checks
- Performance metrics collection
- Alert management
- Log analysis
"""

import json
import logging
import os
import psutil
import requests
import smtplib
import subprocess
import sys
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configure logging
import os

# Use project relative paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "monitoring.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("SutazAI.Monitoring")

class SystemMonitor:
    def __init__(self, config_path=None):
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.alert_history_file = os.path.join(self.config['log_dir'], 'alert_history.json')
        self._init_alert_history()
        
    def _load_config(self, config_path):
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return {
            'log_dir': LOG_DIR,
            'alert_cooldown': 3600,  # 1 hour in seconds
            'thresholds': {
                'cpu': 90,
                'memory': 90,
                'disk': 90
            },
            'email': {
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'username': 'alerts@sutazai.com',
                'password': os.getenv('SMTP_PASSWORD'),
                'from_addr': 'alerts@sutazai.com',
                'to_addr': 'admin@sutazai.com'
            }
        }
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config['log_dir'], 'monitoring.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('SystemMonitor')
        
    def _init_alert_history(self):
        if not os.path.exists(self.alert_history_file):
            with open(self.alert_history_file, 'w') as f:
                json.dump({}, f)

    def should_alert(self, alert_type):
        try:
            with open(self.alert_history_file, 'r') as f:
                history = json.load(f)
            
            current_time = datetime.now()
            
            if alert_type not in history:
                history[alert_type] = current_time.isoformat()
                with open(self.alert_history_file, 'w') as f:
                    json.dump(history, f)
                return True
                
            last_alert = datetime.fromisoformat(history[alert_type])
            cooldown = timedelta(seconds=self.config['alert_cooldown'])
            
            if current_time - last_alert > cooldown:
                history[alert_type] = current_time.isoformat()
                with open(self.alert_history_file, 'w') as f:
                    json.dump(history, f)
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking alert history: {str(e)}")
            return True  # If there's an error, better to alert than not

    def check_resources(self):
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            metrics = {
                'cpu_usage': cpu_percent,
                'memory_usage': memory.percent,
                'disk_usage': disk.percent,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save metrics
            metrics_file = os.path.join(self.config['log_dir'], 'system_metrics.json')
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f)
            
            # Check thresholds
            alerts = []
            if cpu_percent > self.config['thresholds']['cpu'] and self.should_alert('cpu'):
                alerts.append(f"High CPU usage: {cpu_percent}%")
            
            if memory.percent > self.config['thresholds']['memory'] and self.should_alert('memory'):
                alerts.append(f"High memory usage: {memory.percent}%")
            
            if disk.percent > self.config['thresholds']['disk'] and self.should_alert('disk'):
                alerts.append(f"High disk usage: {disk.percent}%")
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Error checking resources: {str(e)}")
            return []

    def check_services(self):
        try:
            services = ['sutazai-backend', 'sutazai-web']
            alerts = []
            
            for service in services:
                result = os.system(f'systemctl is-active --quiet {service}')
                if result != 0 and self.should_alert(f'service_{service}'):
                    alerts.append(f"Service {service} is not running")
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Error checking services: {str(e)}")
            return []

    def send_alert(self, message):
        try:
            email_config = self.config['email']
            
            msg = MIMEText(message)
            msg['Subject'] = 'SutazAI System Alert'
            msg['From'] = email_config['from_addr']
            msg['To'] = email_config['to_addr']
            
            with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as server:
                server.starttls()
                server.login(email_config['username'], email_config['password'])
                server.send_message(msg)
            
            self.logger.info(f"Alert sent: {message}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending alert: {str(e)}")
            return False

    def run_monitoring(self):
        try:
            # Check system resources
            resource_alerts = self.check_resources()
            
            # Check service status
            service_alerts = self.check_services()
            
            # Send alerts if any
            all_alerts = resource_alerts + service_alerts
            if all_alerts:
                alert_message = "System Alerts:\n" + "\n".join(all_alerts)
                self.send_alert(alert_message)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in monitoring run: {str(e)}")
            return False

def main():
    """Main entry point."""
    monitor = SystemMonitor()
    monitor.run_monitoring()

if __name__ == "__main__":
    main()