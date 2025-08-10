#!/usr/bin/env python3
"""
Basic Intrusion Detection System for SutazAI
Monitors logs for suspicious activities
"""

import re
import logging
from pathlib import Path
from collections import defaultdict

class IntrusionDetector:
    def __init__(self):
        self.suspicious_patterns = [
            r'SQL injection',
            r'<script.*?>',
            r'../../../../',
            r'eval\(',
            r'base64_decode',
            r'union.*select',
            r'drop\s+table',
        ]
        self.failed_attempts = defaultdict(int)
        self.blocked_ips = set()
        
    def analyze_log_line(self, line):
        """Analyze a single log line for suspicious activity"""
        for pattern in self.suspicious_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                self.handle_suspicious_activity(line, pattern)
                
    def handle_suspicious_activity(self, line, pattern):
        """Handle detected suspicious activity"""
        timestamp = datetime.now().isoformat()
        logging.warning(f"[{timestamp}] Suspicious activity detected: {pattern}")
        logging.warning(f"Log line: {line.strip()}")
        
        # Extract IP if possible
        ip_match = re.search(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', line)
        if ip_match:
            ip = ip_match.group()
            self.failed_attempts[ip] += 1
            if self.failed_attempts[ip] > 5:
                self.block_ip(ip)
                
    def block_ip(self, ip):
        """Block an IP address (placeholder - implement with iptables)"""
        if ip not in self.blocked_ips:
            self.blocked_ips.add(ip)
            logging.error(f"IP {ip} has been flagged for blocking")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    detector = IntrusionDetector()
    
    # Monitor nginx logs (adjust path as needed)
    log_files = [
        "/var/log/nginx/access.log",
        "/var/log/nginx/error.log",
    ]
    
    for log_file in log_files:
        if Path(log_file).exists():
            with open(log_file, 'r') as f:
                for line in f:
                    detector.analyze_log_line(line)
