#!/usr/bin/env python3
"""
Defense-in-Depth Network Security Layer
Implements comprehensive network security controls for SutazAI
"""

import asyncio
import logging
import json
import ipaddress
import socket
import time
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import iptables
import scapy.all as scapy
from scapy.layers.inet import IP, TCP, UDP, ICMP
import psutil
import subprocess
import redis
import threading
from collections import defaultdict, deque
import hashlib
import ssl
import os

class ThreatLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class ActionType(Enum):
    ALLOW = "allow"
    BLOCK = "block"
    RATE_LIMIT = "rate_limit"
    LOG = "log"
    ALERT = "alert"

@dataclass
class NetworkRule:
    """Network security rule definition"""
    name: str
    source_ip: str
    destination_ip: str
    port: int
    protocol: str
    action: ActionType
    priority: int = 100
    enabled: bool = True
    expires_at: Optional[datetime] = None

@dataclass
class ThreatIndicator:
    """Network threat indicator"""
    indicator_type: str
    value: str
    threat_level: ThreatLevel
    description: str
    source: str
    created_at: datetime
    expires_at: Optional[datetime] = None

class NetworkSecurityEngine:
    """Core network security engine implementing defense-in-depth"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.redis_client = None
        self.rules: List[NetworkRule] = []
        self.threat_indicators: Dict[str, ThreatIndicator] = {}
        self.connection_tracker: Dict[str, List[float]] = defaultdict(list)
        self.blocked_ips: Set[str] = set()
        self.rate_limits: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.running = False
        self._initialize_components()

    def _initialize_components(self):
        """Initialize network security components"""
        try:
            # Initialize Redis for distributed tracking
            self.redis_client = redis.Redis(
                host=self.config.get('redis_host', 'redis'),
                port=self.config.get('redis_port', 6379),
                password=self.config.get('redis_password'),
                ssl=True,
                decode_responses=True
            )
            
            # Load default security rules
            self._load_default_rules()
            
            # Load threat intelligence
            self._load_threat_indicators()
            
            # Initialize firewall rules
            self._setup_firewall_rules()
            
            self.logger.info("Network Security Engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Network Security Engine: {e}")
            raise

    def _load_default_rules(self):
        """Load default network security rules"""
        default_rules = [
            # Block known malicious ports
            NetworkRule("block_telnet", "0.0.0.0/0", "0.0.0.0/0", 23, "tcp", ActionType.BLOCK, 10),
            NetworkRule("block_ftp", "0.0.0.0/0", "0.0.0.0/0", 21, "tcp", ActionType.BLOCK, 10),
            NetworkRule("block_smtp_open", "0.0.0.0/0", "0.0.0.0/0", 25, "tcp", ActionType.BLOCK, 10),
            
            # Rate limit common services
            NetworkRule("limit_ssh", "0.0.0.0/0", "0.0.0.0/0", 22, "tcp", ActionType.RATE_LIMIT, 50),
            NetworkRule("limit_http", "0.0.0.0/0", "0.0.0.0/0", 80, "tcp", ActionType.RATE_LIMIT, 50),
            NetworkRule("limit_https", "0.0.0.0/0", "0.0.0.0/0", 443, "tcp", ActionType.RATE_LIMIT, 50),
            
            # Allow internal communication
            NetworkRule("allow_internal", "172.20.0.0/16", "172.20.0.0/16", 0, "any", ActionType.ALLOW, 90),
            
            # Block private IP ranges from external sources
            NetworkRule("block_external_private", "0.0.0.0/0", "10.0.0.0/8", 0, "any", ActionType.BLOCK, 20),
            NetworkRule("block_external_private2", "0.0.0.0/0", "192.168.0.0/16", 0, "any", ActionType.BLOCK, 20),
            
            # Default logging for all traffic
            NetworkRule("log_all", "0.0.0.0/0", "0.0.0.0/0", 0, "any", ActionType.LOG, 1000),
        ]
        
        self.rules.extend(default_rules)
        self.logger.info(f"Loaded {len(default_rules)} default security rules")

    def _load_threat_indicators(self):
        """Load threat indicators from various sources"""
        # Load from local threat intelligence database
        threat_indicators = [
            ThreatIndicator("ip", "192.0.2.1", ThreatLevel.HIGH, "Known botnet C&C", "local_intel", datetime.utcnow()),
            ThreatIndicator("domain", "malicious.example.com", ThreatLevel.CRITICAL, "Malware distribution", "local_intel", datetime.utcnow()),
            ThreatIndicator("hash", "d41d8cd98f00b204e9800998ecf8427e", ThreatLevel.MEDIUM, "Suspicious file hash", "local_intel", datetime.utcnow()),
        ]
        
        for indicator in threat_indicators:
            self.threat_indicators[indicator.value] = indicator
        
        self.logger.info(f"Loaded {len(threat_indicators)} threat indicators")

    def _setup_firewall_rules(self):
        """Setup iptables firewall rules"""
        try:
            # Clear existing rules
            subprocess.run(['iptables', '-F'], check=False)
            subprocess.run(['iptables', '-X'], check=False)
            
            # Default policies
            subprocess.run(['iptables', '-P', 'INPUT', 'DROP'], check=True)
            subprocess.run(['iptables', '-P', 'FORWARD', 'DROP'], check=True)
            subprocess.run(['iptables', '-P', 'OUTPUT', 'ACCEPT'], check=True)
            
            # Allow loopback
            subprocess.run(['iptables', '-A', 'INPUT', '-i', 'lo', '-j', 'ACCEPT'], check=True)
            subprocess.run(['iptables', '-A', 'OUTPUT', '-o', 'lo', '-j', 'ACCEPT'], check=True)
            
            # Allow established connections
            subprocess.run(['iptables', '-A', 'INPUT', '-m', 'conntrack', '--ctstate', 'ESTABLISHED,RELATED', '-j', 'ACCEPT'], check=True)
            
            # Create custom chains for security rules
            subprocess.run(['iptables', '-N', 'SUTAZAI_SECURITY'], check=True)
            subprocess.run(['iptables', '-A', 'INPUT', '-j', 'SUTAZAI_SECURITY'], check=True)
            
            # Apply security rules
            for rule in sorted(self.rules, key=lambda x: x.priority):
                if rule.enabled:
                    self._apply_firewall_rule(rule)
            
            self.logger.info("Firewall rules configured successfully")
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to setup firewall rules: {e}")
            raise

    def _apply_firewall_rule(self, rule: NetworkRule):
        """Apply individual firewall rule"""
        try:
            cmd = ['iptables', '-A', 'SUTAZAI_SECURITY']
            
            # Source IP
            if rule.source_ip != "0.0.0.0/0":
                cmd.extend(['-s', rule.source_ip])
            
            # Destination IP
            if rule.destination_ip != "0.0.0.0/0":
                cmd.extend(['-d', rule.destination_ip])
            
            # Protocol
            if rule.protocol != "any":
                cmd.extend(['-p', rule.protocol])
            
            # Port
            if rule.port > 0:
                if rule.protocol in ['tcp', 'udp']:
                    cmd.extend(['--dport', str(rule.port)])
            
            # Action
            if rule.action == ActionType.BLOCK:
                cmd.extend(['-j', 'DROP'])
            elif rule.action == ActionType.ALLOW:
                cmd.extend(['-j', 'ACCEPT'])
            elif rule.action == ActionType.RATE_LIMIT:
                # Implement rate limiting with recent module
                cmd.extend(['-m', 'recent', '--name', f'rl_{rule.name}', '--set'])
                cmd.extend(['-m', 'recent', '--name', f'rl_{rule.name}', '--rcheck', '--seconds', '60', '--hitcount', '10', '-j', 'DROP'])
            elif rule.action == ActionType.LOG:
                cmd.extend(['-j', 'LOG', '--log-prefix', f'[{rule.name}] '])
            
            subprocess.run(cmd, check=True)
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to apply firewall rule {rule.name}: {e}")

    async def start_monitoring(self):
        """Start network monitoring and threat detection"""
        self.running = True
        
        # Start packet capture thread
        capture_thread = threading.Thread(target=self._packet_capture_loop)
        capture_thread.daemon = True
        capture_thread.start()
        
        # Start connection monitoring
        monitor_thread = threading.Thread(target=self._connection_monitor_loop)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Start threat intelligence updates
        intel_thread = threading.Thread(target=self._threat_intel_update_loop)
        intel_thread.daemon = True
        intel_thread.start()
        
        self.logger.info("Network monitoring started")

    def _packet_capture_loop(self):
        """Main packet capture and analysis loop"""
        try:
            # Create packet filter for monitoring
            def packet_handler(packet):
                if not self.running:
                    return
                
                try:
                    self._analyze_packet(packet)
                except Exception as e:
                    self.logger.error(f"Packet analysis error: {e}")
            
            # Start packet capture
            scapy.sniff(prn=packet_handler, store=0, timeout=1)
            
        except Exception as e:
            self.logger.error(f"Packet capture error: {e}")

    def _analyze_packet(self, packet):
        """Analyze individual packet for threats"""
        if not packet.haslayer(IP):
            return
        
        ip_layer = packet[IP]
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst
        
        # Check against threat indicators
        if src_ip in self.threat_indicators:
            indicator = self.threat_indicators[src_ip]
            self._handle_threat_detection(src_ip, indicator, packet)
        
        # Analyze traffic patterns
        self._analyze_traffic_patterns(packet)
        
        # Check for suspicious behavior
        self._detect_suspicious_behavior(packet)

    def _analyze_traffic_patterns(self, packet):
        """Analyze traffic patterns for anomalies"""
        if not packet.haslayer(IP):
            return
        
        ip_layer = packet[IP]
        src_ip = ip_layer.src
        
        # Track connection attempts
        current_time = time.time()
        self.connection_tracker[src_ip].append(current_time)
        
        # Remove old entries (older than 1 minute)
        cutoff_time = current_time - 60
        self.connection_tracker[src_ip] = [
            t for t in self.connection_tracker[src_ip] if t > cutoff_time
        ]
        
        # Check for rate limit violations
        if len(self.connection_tracker[src_ip]) > self.config.get('max_connections_per_minute', 100):
            self._handle_rate_limit_violation(src_ip, len(self.connection_tracker[src_ip]))

    def _detect_suspicious_behavior(self, packet):
        """Detect suspicious network behavior"""
        if not packet.haslayer(IP):
            return
        
        ip_layer = packet[IP]
        
        # Port scanning detection
        if packet.haslayer(TCP):
            tcp_layer = packet[TCP]
            if tcp_layer.flags == 2:  # SYN flag
                self._detect_port_scan(ip_layer.src, ip_layer.dst, tcp_layer.dport)
        
        # DNS tunneling detection
        if packet.haslayer(scapy.DNS):
            self._detect_dns_tunneling(packet)
        
        # DDoS detection
        self._detect_ddos_patterns(packet)

    def _detect_port_scan(self, src_ip: str, dst_ip: str, port: int):
        """Detect port scanning attempts"""
        scan_key = f"portscan:{src_ip}:{dst_ip}"
        
        try:
            # Track ports accessed by this source to this destination
            scanned_ports = self.redis_client.sadd(scan_key, port)
            self.redis_client.expire(scan_key, 300)  # 5 minute window
            
            port_count = self.redis_client.scard(scan_key)
            
            # If more than threshold ports scanned, it's likely a port scan
            if port_count > self.config.get('port_scan_threshold', 10):
                self._handle_port_scan_detection(src_ip, dst_ip, port_count)
                
        except Exception as e:
            self.logger.error(f"Port scan detection error: {e}")

    def _detect_dns_tunneling(self, packet):
        """Detect DNS tunneling attempts"""
        dns_layer = packet[scapy.DNS]
        
        if dns_layer.qr == 0:  # Query
            query_name = dns_layer.qd.qname.decode()
            
            # Check for suspicious DNS patterns
            if len(query_name) > 100:  # Unusually long DNS query
                self._handle_dns_tunneling_detection(packet[IP].src, query_name)
            
            # Check for high entropy in subdomain (possible data exfiltration)
            subdomain = query_name.split('.')[0] if '.' in query_name else query_name
            if self._calculate_entropy(subdomain) > 4.5:
                self._handle_dns_tunneling_detection(packet[IP].src, query_name)

    def _detect_ddos_patterns(self, packet):
        """Detect DDoS attack patterns"""
        if not packet.haslayer(IP):
            return
        
        src_ip = packet[IP].src
        dst_ip = packet[IP].dst
        
        # Track packet rates per source
        rate_key = f"ddos:{src_ip}"
        current_time = int(time.time())
        
        try:
            # Increment counter for current second
            self.redis_client.hincrby(rate_key, current_time, 1)
            self.redis_client.expire(rate_key, 60)
            
            # Check last 10 seconds
            total_packets = 0
            for i in range(10):
                count = self.redis_client.hget(rate_key, current_time - i)
                if count:
                    total_packets += int(count)
            
            # If packet rate exceeds threshold, potential DDoS
            if total_packets > self.config.get('ddos_threshold', 1000):
                self._handle_ddos_detection(src_ip, dst_ip, total_packets)
                
        except Exception as e:
            self.logger.error(f"DDoS detection error: {e}")

    def _calculate_entropy(self, data: str) -> float:
        """Calculate Shannon entropy of data"""
        if not data:
            return 0
        
        entropy = 0
        data_len = len(data)
        char_counts = {}
        
        for char in data:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        for count in char_counts.values():
            prob = count / data_len
            entropy -= prob * (prob.bit_length() - 1)
        
        return entropy

    def _connection_monitor_loop(self):
        """Monitor network connections"""
        while self.running:
            try:
                # Get current network connections
                connections = psutil.net_connections(kind='inet')
                
                for conn in connections:
                    if conn.raddr:  # Has remote address
                        self._analyze_connection(conn)
                
                time.sleep(self.config.get('connection_monitor_interval', 30))
                
            except Exception as e:
                self.logger.error(f"Connection monitoring error: {e}")
                time.sleep(30)

    def _analyze_connection(self, connection):
        """Analyze individual network connection"""
        remote_ip = connection.raddr.ip if connection.raddr else None
        remote_port = connection.raddr.port if connection.raddr else None
        
        if not remote_ip:
            return
        
        # Check against threat indicators
        if remote_ip in self.threat_indicators:
            indicator = self.threat_indicators[remote_ip]
            self._handle_connection_threat(remote_ip, remote_port, indicator)
        
        # Check for suspicious ports
        if remote_port in self.config.get('suspicious_ports', [135, 139, 445, 1433, 3389]):
            self._handle_suspicious_connection(remote_ip, remote_port)

    def _threat_intel_update_loop(self):
        """Update threat intelligence periodically"""
        while self.running:
            try:
                self._update_threat_intelligence()
                time.sleep(self.config.get('threat_intel_update_interval', 3600))  # 1 hour
                
            except Exception as e:
                self.logger.error(f"Threat intelligence update error: {e}")
                time.sleep(3600)

    def _update_threat_intelligence(self):
        """Update threat intelligence from external sources"""
        # This would integrate with threat intelligence feeds
        # For now, we'll implement a placeholder
        self.logger.info("Updating threat intelligence...")
        
        # Example: Load from file or API
        # new_indicators = self._fetch_threat_indicators()
        # self.threat_indicators.update(new_indicators)

    def _handle_threat_detection(self, src_ip: str, indicator: ThreatIndicator, packet):
        """Handle threat detection"""
        self.logger.warning(f"Threat detected from {src_ip}: {indicator.description}")
        
        # Block IP immediately for critical/high threats
        if indicator.threat_level in [ThreatLevel.CRITICAL, ThreatLevel.HIGH]:
            self._block_ip(src_ip, f"Threat: {indicator.description}")
        
        # Log the event
        self._log_security_event("threat_detected", {
            "src_ip": src_ip,
            "threat_level": indicator.threat_level.name,
            "description": indicator.description,
            "source": indicator.source
        })

    def _handle_rate_limit_violation(self, src_ip: str, connection_count: int):
        """Handle rate limit violation"""
        self.logger.warning(f"Rate limit violation from {src_ip}: {connection_count} connections")
        
        # Temporary block for rate limit violations
        self._block_ip(src_ip, f"Rate limit violation: {connection_count} connections", duration=300)
        
        self._log_security_event("rate_limit_violation", {
            "src_ip": src_ip,
            "connection_count": connection_count
        })

    def _handle_port_scan_detection(self, src_ip: str, dst_ip: str, port_count: int):
        """Handle port scan detection"""
        self.logger.warning(f"Port scan detected from {src_ip} to {dst_ip}: {port_count} ports")
        
        # Block source IP
        self._block_ip(src_ip, f"Port scan: {port_count} ports scanned")
        
        self._log_security_event("port_scan_detected", {
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "port_count": port_count
        })

    def _handle_dns_tunneling_detection(self, src_ip: str, query_name: str):
        """Handle DNS tunneling detection"""
        self.logger.warning(f"DNS tunneling detected from {src_ip}: {query_name}")
        
        self._log_security_event("dns_tunneling_detected", {
            "src_ip": src_ip,
            "query_name": query_name
        })

    def _handle_ddos_detection(self, src_ip: str, dst_ip: str, packet_count: int):
        """Handle DDoS detection"""
        self.logger.critical(f"DDoS attack detected from {src_ip}: {packet_count} packets")
        
        # Block source IP immediately
        self._block_ip(src_ip, f"DDoS attack: {packet_count} packets")
        
        self._log_security_event("ddos_detected", {
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "packet_count": packet_count
        })

    def _handle_connection_threat(self, remote_ip: str, remote_port: int, indicator: ThreatIndicator):
        """Handle connection to known threat"""
        self.logger.warning(f"Connection to threat indicator {remote_ip}:{remote_port}: {indicator.description}")
        
        self._log_security_event("connection_threat", {
            "remote_ip": remote_ip,
            "remote_port": remote_port,
            "threat_level": indicator.threat_level.name,
            "description": indicator.description
        })

    def _handle_suspicious_connection(self, remote_ip: str, remote_port: int):
        """Handle connection to suspicious port"""
        self.logger.warning(f"Suspicious connection to {remote_ip}:{remote_port}")
        
        self._log_security_event("suspicious_connection", {
            "remote_ip": remote_ip,
            "remote_port": remote_port
        })

    def _block_ip(self, ip_address: str, reason: str, duration: Optional[int] = None):
        """Block IP address using iptables"""
        try:
            # Add to blocked IPs set
            self.blocked_ips.add(ip_address)
            
            # Add iptables rule
            subprocess.run([
                'iptables', '-I', 'SUTAZAI_SECURITY', '1',
                '-s', ip_address, '-j', 'DROP'
            ], check=True)
            
            # Store in Redis for distributed blocking
            block_key = f"blocked_ip:{ip_address}"
            block_data = {
                "reason": reason,
                "blocked_at": datetime.utcnow().isoformat(),
                "expires_at": (datetime.utcnow() + timedelta(seconds=duration)).isoformat() if duration else None
            }
            
            if duration:
                self.redis_client.setex(block_key, duration, json.dumps(block_data))
            else:
                self.redis_client.set(block_key, json.dumps(block_data))
            
            self.logger.info(f"Blocked IP {ip_address}: {reason}")
            
            # Schedule unblock if duration specified
            if duration:
                threading.Timer(duration, self._unblock_ip, args=[ip_address]).start()
                
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to block IP {ip_address}: {e}")

    def _unblock_ip(self, ip_address: str):
        """Unblock IP address"""
        try:
            # Remove from blocked IPs set
            self.blocked_ips.discard(ip_address)
            
            # Remove iptables rule
            subprocess.run([
                'iptables', '-D', 'SUTAZAI_SECURITY',
                '-s', ip_address, '-j', 'DROP'
            ], check=False)
            
            # Remove from Redis
            self.redis_client.delete(f"blocked_ip:{ip_address}")
            
            self.logger.info(f"Unblocked IP {ip_address}")
            
        except Exception as e:
            self.logger.error(f"Failed to unblock IP {ip_address}: {e}")

    def _log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security event"""
        try:
            event_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": event_type,
                "details": details,
                "source": "network_security"
            }
            
            # Store in Redis for real-time monitoring
            self.redis_client.lpush("security_events", json.dumps(event_data))
            self.redis_client.ltrim("security_events", 0, 9999)  # Keep last 10k events
            
            # Log to file
            self.logger.info(f"Security event: {event_type} - {json.dumps(details)}")
            
        except Exception as e:
            self.logger.error(f"Failed to log security event: {e}")

    def stop_monitoring(self):
        """Stop network monitoring"""
        self.running = False
        self.logger.info("Network monitoring stopped")

if __name__ == "__main__":
    # Example usage
    config = {
        'redis_host': 'redis',
        'redis_port': 6379,
        'max_connections_per_minute': 100,
        'port_scan_threshold': 10,
        'ddos_threshold': 1000,
        'connection_monitor_interval': 30,
        'threat_intel_update_interval': 3600,
        'suspicious_ports': [135, 139, 445, 1433, 3389]
    }
    
    network_security = NetworkSecurityEngine(config)
    
    # Start monitoring
    asyncio.run(network_security.start_monitoring())