#!/usr/bin/env python3
"""
Advanced Threat Detection and Security Monitoring Framework

Comprehensive security monitoring with:
- Intrusion detection
- Anomaly tracking
- Behavioral analysis
- Real-time threat assessment
- Multi-layered security checks
"""

import hashlib
import json
import logging
import os
import re
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import psutil
import requests
from cryptography.fernet import Fernet


class ThreatDetector:
    def __init__(
        self, 
        log_dir: str = 'security_logs', 
        config_path: str = 'security_config.json'
    ):
        """
        Initialize the advanced threat detection system.
        
        Args:
            log_dir (str): Directory to store security logs
            config_path (str): Path to security configuration
        """
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure logging
        self.log_dir = log_dir
        self.config_path = config_path
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'threat_detection.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('ThreatDetector')
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize encryption key
        self.encryption_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        # Track system baseline
        self.baseline_metrics = self._capture_baseline_metrics()
        
        # Threat tracking
        self.active_threats: Set[str] = set()
        
    def _load_config(self) -> Dict[str, Any]:
        """
        Load security configuration.
        
        Returns:
            Dict[str, Any]: Security configuration
        """
        default_config = {
            'max_failed_logins': 5,
            'suspicious_ports': [22, 23, 3389],  # SSH, Telnet, RDP
            'blacklisted_ips': [],
            'whitelisted_ips': [],
            'threat_sensitivity': 'medium'
        }
        
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    # Merge default with loaded config
                    default_config.update(config)
            else:
                # Save default config if not exists
                with open(self.config_path, 'w') as f:
                    json.dump(default_config, f, indent=2)
            
            return default_config
        except Exception as e:
            self.logger.error(f"Config loading error: {e}")
            return default_config
    
    def _capture_baseline_metrics(self) -> Dict[str, Any]:
        """
        Capture baseline system metrics for anomaly detection.
        
        Returns:
            Dict[str, Any]: Baseline system metrics
        """
        try:
            return {
                'network_connections': len(psutil.net_connections()),
                'running_processes': len(psutil.pids()),
                'cpu_baseline': psutil.cpu_percent(),
                'memory_baseline': psutil.virtual_memory().percent
            }
        except Exception as e:
            self.logger.warning(f"Baseline metrics capture failed: {e}")
            return {}
    
    def detect_network_anomalies(self) -> List[Dict[str, Any]]:
        """
        Detect network-related security anomalies.
        
        Returns:
            List[Dict[str, Any]]: Detected network anomalies
        """
        anomalies = []
        
        try:
            # Check current network connections
            current_connections = psutil.net_connections()
            
            for conn in current_connections:
                # Check for suspicious ports
                if conn.laddr.port in self.config.get('suspicious_ports', []):
                    anomaly = {
                        'type': 'suspicious_port',
                        'port': conn.laddr.port,
                        'pid': conn.pid,
                        'status': conn.status
                    }
                    anomalies.append(anomaly)
                
                # Check IP reputation
                remote_ip = conn.raddr.ip if conn.raddr else None
                if remote_ip and self._check_ip_reputation(remote_ip):
                    anomaly = {
                        'type': 'suspicious_ip',
                        'ip': remote_ip,
                        'pid': conn.pid
                    }
                    anomalies.append(anomaly)
        
        except Exception as e:
            self.logger.error(f"Network anomaly detection error: {e}")
        
        return anomalies
    
    def _check_ip_reputation(self, ip: str) -> bool:
        """
        Check IP reputation using external API.
        
        Args:
            ip (str): IP address to check
        
        Returns:
            bool: Whether IP is suspicious
        """
        try:
            # Use AbuseIPDB for IP reputation check
            response = requests.get(
                f"https://api.abuseipdb.com/api/v2/check",
                params={'ipAddress': ip},
                headers={
                    'Key': os.environ.get('ABUSEIPDB_API_KEY', ''),
                    'Accept': 'application/json'
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                # Consider IP suspicious if abuse confidence is high
                return data.get('data', {}).get('abuseConfidenceScore', 0) > 50
            
            return False
        except Exception as e:
            self.logger.warning(f"IP reputation check failed for {ip}: {e}")
            return False
    
    def monitor_process_behavior(self) -> List[Dict[str, Any]]:
        """
        Monitor and analyze process behaviors for potential threats.
        
        Returns:
            List[Dict[str, Any]]: Suspicious process behaviors
        """
        suspicious_processes = []
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent']):
                try:
                    # Check for high resource consumption
                    if proc.info['cpu_percent'] > 80:
                        suspicious_processes.append({
                            'type': 'high_resource_usage',
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'cpu_percent': proc.info['cpu_percent']
                        })
                    
                    # Check for suspicious command lines
                    cmdline = proc.info.get('cmdline', [])
                    if self._is_suspicious_cmdline(cmdline):
                        suspicious_processes.append({
                            'type': 'suspicious_cmdline',
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'cmdline': cmdline
                        })
                
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        
        except Exception as e:
            self.logger.error(f"Process behavior monitoring error: {e}")
        
        return suspicious_processes
    
    def _is_suspicious_cmdline(self, cmdline: List[str]) -> bool:
        """
        Check if a command line is suspicious.
        
        Args:
            cmdline (List[str]): Command line arguments
        
        Returns:
            bool: Whether command line is suspicious
        """
        suspicious_patterns = [
            r'wget\s+http',  # Potential remote download
            r'curl\s+-O',    # File download
            r'nc\s+-e',      # Netcat with executable
            r'python\s+-m\s+http\.server',  # Simple HTTP server
            r'bash\s+-i',    # Interactive bash
            r'rm\s+-rf',     # Recursive force remove
        ]
        
        cmdline_str = ' '.join(cmdline)
        return any(re.search(pattern, cmdline_str) for pattern in suspicious_patterns)
    
    def log_security_event(
        self, 
        event_type: str, 
        details: Dict[str, Any]
    ) -> None:
        """
        Log a security event with encryption.
        
        Args:
            event_type (str): Type of security event
            details (Dict[str, Any]): Event details
        """
        try:
            event = {
                'timestamp': datetime.now().isoformat(),
                'type': event_type,
                'details': details
            }
            
            # Encrypt event log
            encrypted_event = self.cipher_suite.encrypt(
                json.dumps(event).encode()
            )
            
            log_file = os.path.join(
                self.log_dir, 
                f'security_events_{datetime.now().date()}.log'
            )
            
            with open(log_file, 'ab') as f:
                f.write(encrypted_event + b'\n')
        
        except Exception as e:
            self.logger.error(f"Security event logging error: {e}")
    
    def run_comprehensive_threat_scan(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Perform a comprehensive threat detection scan.
        
        Returns:
            Dict[str, List[Dict[str, Any]]]: Detected threats
        """
        threats = {
            'network_anomalies': self.detect_network_anomalies(),
            'process_anomalies': self.monitor_process_behavior()
        }
        
        # Log detected threats
        for threat_type, threat_list in threats.items():
            for threat in threat_list:
                self.log_security_event(threat_type, threat)
                self.active_threats.add(str(threat))
        
        return threats
    
    def start_continuous_monitoring(self, interval: int = 60) -> None:
        """
        Start continuous threat monitoring in a separate thread.
        
        Args:
            interval (int): Scan interval in seconds
        """
        def monitor_thread():
            while True:
                try:
                    self.run_comprehensive_threat_scan()
                    time.sleep(interval)
                except Exception as e:
                    self.logger.error(f"Continuous monitoring error: {e}")
                    break
        
        thread = threading.Thread(target=monitor_thread, daemon=True)
        thread.start()
        self.logger.info("Continuous threat monitoring started.")


def main():
    threat_detector = ThreatDetector()
    
    # Run initial comprehensive scan
    threats = threat_detector.run_comprehensive_threat_scan()
    
    # Print detected threats
    for threat_type, threat_list in threats.items():
        print(f"{threat_type.upper()}:")
        for threat in threat_list:
            print(json.dumps(threat, indent=2))
    
    # Optional: Start continuous monitoring
    threat_detector.start_continuous_monitoring()


if __name__ == '__main__':
    main() 