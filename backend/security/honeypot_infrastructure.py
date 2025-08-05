"""
Comprehensive Honeypot Infrastructure for SutazAI System
Enterprise-grade deception technology for early threat detection and intelligence gathering
"""

import asyncio
import json
import logging
import time
import random
import hashlib
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import ipaddress
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import os

# Import existing security infrastructure
try:
    from app.core.security import security_manager, SecurityEvent
    SECURITY_INTEGRATION = True
except ImportError:
    SECURITY_INTEGRATION = False
    security_manager = None

logger = logging.getLogger(__name__)

class HoneypotType(Enum):
    """Types of honeypots available"""
    SSH = "ssh"
    HTTP = "http"
    HTTPS = "https" 
    DATABASE = "database"
    AI_AGENT = "ai_agent"
    FTP = "ftp"
    TELNET = "telnet"
    SMTP = "smtp"
    DNS = "dns"

class InteractionLevel(Enum):
    """Honeypot interaction levels"""
    LOW = "low"          # Basic service emulation
    MEDIUM = "medium"    # Partial service functionality
    HIGH = "high"        # Full service emulation

class ThreatSeverity(Enum):
    """Threat severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class HoneypotEvent:
    """Honeypot interaction event"""
    id: str
    timestamp: datetime
    honeypot_id: str
    honeypot_type: str
    source_ip: str
    source_port: int
    destination_port: int
    event_type: str
    payload: str
    severity: str
    attack_vector: Optional[str] = None
    user_agent: Optional[str] = None
    credentials: Optional[Dict[str, str]] = None
    session_id: Optional[str] = None
    geolocation: Optional[Dict[str, Any]] = None
    threat_indicators: List[str] = None
    
    def __post_init__(self):
        if self.threat_indicators is None:
            self.threat_indicators = []

@dataclass
class AttackerProfile:
    """Attacker behavior profile"""
    source_ip: str
    first_seen: datetime
    last_seen: datetime
    total_attempts: int
    honeypots_hit: Set[str]
    attack_patterns: List[str]
    credentials_tried: List[Dict[str, str]]
    user_agents: Set[str]
    threat_score: float
    country: Optional[str] = None
    isp: Optional[str] = None
    tor_exit_node: bool = False
    
    def __post_init__(self):
        if isinstance(self.honeypots_hit, list):
            self.honeypots_hit = set(self.honeypots_hit)
        if isinstance(self.user_agents, list):
            self.user_agents = set(self.user_agents)

class HoneypotDatabase:
    """Database for storing honeypot events and attacker profiles"""
    
    def __init__(self, db_path: str = "/opt/sutazaiapp/backend/data/honeypot.db"):
        self.db_path = db_path
        self._init_database()
        
    def _init_database(self):
        """Initialize honeypot database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS honeypot_events (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    honeypot_id TEXT NOT NULL,
                    honeypot_type TEXT NOT NULL,
                    source_ip TEXT NOT NULL,
                    source_port INTEGER,
                    destination_port INTEGER,
                    event_type TEXT NOT NULL,
                    payload TEXT,
                    severity TEXT NOT NULL,
                    attack_vector TEXT,
                    user_agent TEXT,
                    credentials TEXT,
                    session_id TEXT,
                    geolocation TEXT,
                    threat_indicators TEXT
                );
                
                CREATE TABLE IF NOT EXISTS attacker_profiles (
                    source_ip TEXT PRIMARY KEY,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    total_attempts INTEGER DEFAULT 0,
                    honeypots_hit TEXT,
                    attack_patterns TEXT,
                    credentials_tried TEXT,
                    user_agents TEXT,
                    threat_score REAL DEFAULT 0.0,
                    country TEXT,
                    isp TEXT,
                    tor_exit_node BOOLEAN DEFAULT FALSE
                );
                
                CREATE TABLE IF NOT EXISTS honeypot_deployments (
                    id TEXT PRIMARY KEY,
                    honeypot_type TEXT NOT NULL,
                    port INTEGER NOT NULL,
                    interface TEXT DEFAULT '0.0.0.0',
                    status TEXT DEFAULT 'active',
                    config TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                
                CREATE INDEX IF NOT EXISTS idx_events_timestamp ON honeypot_events(timestamp);
                CREATE INDEX IF NOT EXISTS idx_events_source_ip ON honeypot_events(source_ip);
                CREATE INDEX IF NOT EXISTS idx_events_honeypot_type ON honeypot_events(honeypot_type);
                CREATE INDEX IF NOT EXISTS idx_profiles_threat_score ON attacker_profiles(threat_score DESC);
            ''')
    
    def store_event(self, event: HoneypotEvent):
        """Store honeypot event"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO honeypot_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event.id,
                event.timestamp.isoformat(),
                event.honeypot_id,
                event.honeypot_type,
                event.source_ip,
                event.source_port,
                event.destination_port,
                event.event_type,
                event.payload,
                event.severity,
                event.attack_vector,
                event.user_agent,
                json.dumps(event.credentials) if event.credentials else None,
                event.session_id,
                json.dumps(event.geolocation) if event.geolocation else None,
                json.dumps(event.threat_indicators)
            ))
    
    def update_attacker_profile(self, profile: AttackerProfile):
        """Update or create attacker profile"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO attacker_profiles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                profile.source_ip,
                profile.first_seen.isoformat(),
                profile.last_seen.isoformat(),
                profile.total_attempts,
                json.dumps(list(profile.honeypots_hit)),
                json.dumps(profile.attack_patterns),
                json.dumps(profile.credentials_tried),
                json.dumps(list(profile.user_agents)),
                profile.threat_score,
                profile.country,
                profile.isp,
                profile.tor_exit_node
            ))
    
    def get_attacker_profile(self, source_ip: str) -> Optional[AttackerProfile]:
        """Get attacker profile by IP"""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                'SELECT * FROM attacker_profiles WHERE source_ip = ?',
                (source_ip,)
            ).fetchone()
            
            if row:
                return AttackerProfile(
                    source_ip=row[0],
                    first_seen=datetime.fromisoformat(row[1]),
                    last_seen=datetime.fromisoformat(row[2]),
                    total_attempts=row[3],
                    honeypots_hit=set(json.loads(row[4]) if row[4] else []),
                    attack_patterns=json.loads(row[5]) if row[5] else [],
                    credentials_tried=json.loads(row[6]) if row[6] else [],
                    user_agents=set(json.loads(row[7]) if row[7] else []),
                    threat_score=row[8],
                    country=row[9],
                    isp=row[10],
                    tor_exit_node=bool(row[11])
                )
        return None
    
    def get_events(self, limit: int = 100, hours: int = 24) -> List[HoneypotEvent]:
        """Get recent honeypot events"""
        since = datetime.utcnow() - timedelta(hours=hours)
        
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute('''
                SELECT * FROM honeypot_events 
                WHERE timestamp > ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (since.isoformat(), limit)).fetchall()
            
            events = []
            for row in rows:
                events.append(HoneypotEvent(
                    id=row[0],
                    timestamp=datetime.fromisoformat(row[1]),
                    honeypot_id=row[2],
                    honeypot_type=row[3],
                    source_ip=row[4],
                    source_port=row[5],
                    destination_port=row[6],
                    event_type=row[7],
                    payload=row[8],
                    severity=row[9],
                    attack_vector=row[10],
                    user_agent=row[11],
                    credentials=json.loads(row[12]) if row[12] else None,
                    session_id=row[13],
                    geolocation=json.loads(row[14]) if row[14] else None,
                    threat_indicators=json.loads(row[15]) if row[15] else []
                ))
            
            return events

class ThreatIntelligenceEngine:
    """Threat intelligence gathering and analysis"""
    
    def __init__(self, database: HoneypotDatabase):
        self.database = database
        self.threat_feeds = {}
        self.indicators = {
            'malicious_ips': set(),
            'tor_exit_nodes': set(),
            'known_attack_patterns': [],
            'suspicious_user_agents': set()
        }
        
    async def analyze_event(self, event: HoneypotEvent) -> Dict[str, Any]:
        """Analyze honeypot event for threat intelligence"""
        analysis = {
            'threat_score': 0.0,
            'indicators': [],
            'attack_classification': 'unknown',
            'recommended_action': 'monitor'
        }
        
        # Check against known threat indicators
        if event.source_ip in self.indicators['malicious_ips']:
            analysis['threat_score'] += 0.8
            analysis['indicators'].append('known_malicious_ip')
        
        if event.source_ip in self.indicators['tor_exit_nodes']:
            analysis['threat_score'] += 0.6
            analysis['indicators'].append('tor_exit_node')
        
        # Analyze payload for attack patterns
        payload_analysis = await self._analyze_payload(event.payload, event.honeypot_type)
        analysis['threat_score'] += payload_analysis['score']
        analysis['indicators'].extend(payload_analysis['indicators'])
        analysis['attack_classification'] = payload_analysis['classification']
        
        # Check user agent for suspicious patterns
        if event.user_agent and event.user_agent in self.indicators['suspicious_user_agents']:
            analysis['threat_score'] += 0.4
            analysis['indicators'].append('suspicious_user_agent')
        
        # Determine recommended action
        if analysis['threat_score'] >= 0.8:
            analysis['recommended_action'] = 'block'
        elif analysis['threat_score'] >= 0.6:
            analysis['recommended_action'] = 'alert'
        elif analysis['threat_score'] >= 0.4:
            analysis['recommended_action'] = 'monitor_closely'
        
        return analysis
    
    async def _analyze_payload(self, payload: str, honeypot_type: str) -> Dict[str, Any]:
        """Analyze payload for attack patterns"""
        analysis = {
            'score': 0.0,
            'indicators': [],
            'classification': 'reconnaissance'
        }
        
        if not payload:
            return analysis
        
        payload_lower = payload.lower()
        
        # SQL injection patterns
        sql_patterns = [
            'union select', 'or 1=1', 'drop table', 'exec xp_',
            'sp_executesql', 'information_schema', 'sysobjects'
        ]
        for pattern in sql_patterns:
            if pattern in payload_lower:
                analysis['score'] += 0.7
                analysis['indicators'].append('sql_injection')
                analysis['classification'] = 'sql_injection'
                break
        
        # XSS patterns
        xss_patterns = [
            '<script', 'javascript:', 'onerror=', 'onload=',
            'eval(', 'document.cookie', 'alert('
        ]
        for pattern in xss_patterns:
            if pattern in payload_lower:
                analysis['score'] += 0.6
                analysis['indicators'].append('xss_attempt')
                analysis['classification'] = 'xss'
                break
        
        # Command injection patterns
        cmd_patterns = [
            ';cat ', '|wget', '&&curl', 'nc -e', '/bin/sh',
            'bash -i', 'python -c', 'perl -e'
        ]
        for pattern in cmd_patterns:
            if pattern in payload_lower:
                analysis['score'] += 0.8
                analysis['indicators'].append('command_injection')
                analysis['classification'] = 'command_injection'
                break
        
        # Path traversal patterns
        path_patterns = ['../../../', '..\\..\\', '%2e%2e%2f', 'etc/passwd']
        for pattern in path_patterns:
            if pattern in payload_lower:
                analysis['score'] += 0.5
                analysis['indicators'].append('path_traversal')
                analysis['classification'] = 'path_traversal'
                break
        
        # Check for encoded payloads
        if any(char in payload for char in ['%', '\\x', '\\u']):
            analysis['score'] += 0.3
            analysis['indicators'].append('encoded_payload')
        
        return analysis
    
    def calculate_attacker_threat_score(self, profile: AttackerProfile) -> float:
        """Calculate overall threat score for attacker"""
        score = 0.0
        
        # Base score from number of attempts
        if profile.total_attempts > 100:
            score += 0.8
        elif profile.total_attempts > 50:
            score += 0.6
        elif profile.total_attempts > 10:
            score += 0.4
        else:
            score += 0.2
        
        # Diversity of targets
        honeypot_diversity = len(profile.honeypots_hit) / 10.0  # Normalize
        score += min(honeypot_diversity, 0.5)
        
        # Attack pattern sophistication
        sophisticated_patterns = [
            'sql_injection', 'command_injection', 'buffer_overflow',
            'privilege_escalation', 'lateral_movement'
        ]
        sophistication_score = sum(
            0.1 for pattern in profile.attack_patterns 
            if pattern in sophisticated_patterns
        )
        score += min(sophistication_score, 0.3)
        
        # Time-based patterns (persistent attacks)
        time_diff = profile.last_seen - profile.first_seen
        if time_diff.days > 7:
            score += 0.3
        elif time_diff.days > 1:
            score += 0.2
        
        # Known threat indicators
        if profile.source_ip in self.indicators['malicious_ips']:
            score += 0.4
        
        if profile.tor_exit_node:
            score += 0.2
        
        return min(score, 1.0)  # Cap at 1.0

class BaseHoneypot:
    """Base class for all honeypots"""
    
    def __init__(self, honeypot_id: str, port: int, database: HoneypotDatabase, 
                 intelligence_engine: ThreatIntelligenceEngine):
        self.honeypot_id = honeypot_id
        self.port = port
        self.database = database
        self.intelligence_engine = intelligence_engine
        self.is_running = False
        self.server = None
        self.logger = logging.getLogger(f"honeypot.{honeypot_id}")
        
    async def start(self):
        """Start the honeypot"""
        self.is_running = True
        self.logger.info(f"Starting honeypot {self.honeypot_id} on port {self.port}")
        
    async def stop(self):
        """Stop the honeypot"""
        self.is_running = False
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        self.logger.info(f"Stopped honeypot {self.honeypot_id}")
        
    async def log_interaction(self, source_ip: str, source_port: int, event_type: str,
                            payload: str = "", severity: str = "medium", **kwargs):
        """Log honeypot interaction"""
        event_id = f"{self.honeypot_id}_{int(time.time())}_{random.randint(1000, 9999)}"
        
        event = HoneypotEvent(
            id=event_id,
            timestamp=datetime.utcnow(),
            honeypot_id=self.honeypot_id,
            honeypot_type=self.honeypot_type.value,
            source_ip=source_ip,
            source_port=source_port,
            destination_port=self.port,
            event_type=event_type,
            payload=payload,
            severity=severity,
            **kwargs
        )
        
        # Analyze event for threat intelligence
        analysis = await self.intelligence_engine.analyze_event(event)
        event.threat_indicators = analysis['indicators']
        
        # Store event
        self.database.store_event(event)
        
        # Update attacker profile
        await self._update_attacker_profile(source_ip, event, analysis)
        
        # Alert if high severity
        if severity in ['critical', 'high'] or analysis['threat_score'] >= 0.7:
            await self._send_alert(event, analysis)
        
        self.logger.info(f"Logged interaction: {event_type} from {source_ip}")
        
    async def _update_attacker_profile(self, source_ip: str, event: HoneypotEvent, analysis: Dict[str, Any]):
        """Update attacker profile"""
        profile = self.database.get_attacker_profile(source_ip)
        
        if not profile:
            profile = AttackerProfile(
                source_ip=source_ip,
                first_seen=datetime.utcnow(),
                last_seen=datetime.utcnow(),
                total_attempts=1,
                honeypots_hit={self.honeypot_id},
                attack_patterns=[analysis['attack_classification']],
                credentials_tried=[event.credentials] if event.credentials else [],
                user_agents={event.user_agent} if event.user_agent else set(),
                threat_score=0.0
            )
        else:
            profile.last_seen = datetime.utcnow()
            profile.total_attempts += 1
            profile.honeypots_hit.add(self.honeypot_id)
            if analysis['attack_classification'] not in profile.attack_patterns:
                profile.attack_patterns.append(analysis['attack_classification'])
            if event.credentials and event.credentials not in profile.credentials_tried:
                profile.credentials_tried.append(event.credentials)
            if event.user_agent:
                profile.user_agents.add(event.user_agent)
        
        # Recalculate threat score
        profile.threat_score = self.intelligence_engine.calculate_attacker_threat_score(profile)
        
        # Store updated profile
        self.database.update_attacker_profile(profile)
        
    async def _send_alert(self, event: HoneypotEvent, analysis: Dict[str, Any]):
        """Send security alert"""
        if SECURITY_INTEGRATION and security_manager:
            await security_manager.audit.log_event(
                "honeypot_high_threat_detected",
                "high" if analysis['threat_score'] >= 0.8 else "medium",
                f"honeypot_{self.honeypot_type.value}",
                {
                    "honeypot_id": self.honeypot_id,
                    "source_ip": event.source_ip,
                    "attack_type": analysis['attack_classification'],
                    "threat_score": analysis['threat_score'],
                    "indicators": analysis['indicators'],
                    "payload_preview": event.payload[:200] if event.payload else ""
                },
                ip_address=event.source_ip
            )
        
        # Log critical alerts
        if analysis['threat_score'] >= 0.8:
            self.logger.critical(
                f"HIGH THREAT DETECTED: {event.source_ip} -> {self.honeypot_id} "
                f"({analysis['attack_classification']}, score: {analysis['threat_score']:.2f})"
            )

class SSHHoneypot(BaseHoneypot):
    """SSH honeypot for detecting brute force attacks and capturing credentials"""
    
    def __init__(self, honeypot_id: str, port: int, database: HoneypotDatabase, 
                 intelligence_engine: ThreatIntelligenceEngine):
        super().__init__(honeypot_id, port, database, intelligence_engine)
        self.honeypot_type = HoneypotType.SSH
        self.fake_version = "OpenSSH_8.2p1 Ubuntu-4ubuntu0.5"
        
    async def start(self):
        """Start SSH honeypot"""
        await super().start()
        
        # Start SSH server emulation
        self.server = await asyncio.start_server(
            self.handle_connection,
            '0.0.0.0',
            self.port
        )
        
        self.logger.info(f"SSH honeypot listening on port {self.port}")
        
    async def handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle SSH connection"""
        client_addr = writer.get_extra_info('peername')
        if not client_addr:
            return
            
        source_ip, source_port = client_addr[0], client_addr[1]
        
        try:
            # Send SSH banner
            banner = f"SSH-2.0-{self.fake_version}\r\n"
            writer.write(banner.encode())
            await writer.drain()
            
            # Read client banner
            client_banner = await reader.readline()
            client_version = client_banner.decode().strip()
            
            await self.log_interaction(
                source_ip, source_port, "ssh_connection_attempt",
                payload=f"Client version: {client_version}",
                severity="low",
                user_agent=client_version
            )
            
            # Simulate authentication process
            await self._simulate_ssh_auth(reader, writer, source_ip, source_port)
            
        except Exception as e:
            self.logger.debug(f"SSH connection error from {source_ip}: {e}")
        finally:
            writer.close()
            await writer.wait_closed()
            
    async def _simulate_ssh_auth(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
                                source_ip: str, source_port: int):
        """Simulate SSH authentication to capture credentials"""
        # This is a simplified simulation
        # In a real implementation, you'd implement the full SSH protocol
        
        # Wait for authentication attempts
        auth_attempts = 0
        max_attempts = 3
        
        while auth_attempts < max_attempts:
            try:
                # Read authentication data (simplified)
                data = await reader.read(1024)
                if not data:
                    break
                    
                # Try to extract username/password (this is very simplified)
                # Real SSH protocol parsing would be much more complex
                auth_data = data.decode('utf-8', errors='ignore')
                
                # Look for common authentication patterns
                username = self._extract_username(auth_data)
                password = self._extract_password(auth_data)
                
                if username or password:
                    await self.log_interaction(
                        source_ip, source_port, "ssh_auth_attempt",
                        payload=f"Auth attempt: {auth_data[:100]}",
                        severity="medium",
                        credentials={"username": username, "password": password}
                    )
                
                # Send authentication failure
                failure_msg = b"Authentication failed\n"
                writer.write(failure_msg)
                await writer.drain()
                
                auth_attempts += 1
                
            except Exception as e:
                self.logger.debug(f"SSH auth simulation error: {e}")
                break
                
    def _extract_username(self, data: str) -> Optional[str]:
        """Extract username from SSH data (simplified)"""
        # This is a very basic extraction - real implementation would parse SSH protocol
        common_usernames = ['admin', 'root', 'user', 'test', 'oracle', 'postgres']
        for username in common_usernames:
            if username in data.lower():
                return username
        return None
        
    def _extract_password(self, data: str) -> Optional[str]:
        """Extract password from SSH data (simplified)"""
        # This is a very basic extraction - real implementation would parse SSH protocol
        common_passwords = ['password', '123456', 'admin', 'root', '']
        for password in common_passwords:
            if password in data.lower():
                return password
        return None

class HTTPHoneypot(BaseHoneypot):
    """HTTP/HTTPS honeypot for detecting web attacks"""
    
    def __init__(self, honeypot_id: str, port: int, database: HoneypotDatabase, 
                 intelligence_engine: ThreatIntelligenceEngine, ssl_context=None):
        super().__init__(honeypot_id, port, database, intelligence_engine)
        self.honeypot_type = HoneypotType.HTTPS if ssl_context else HoneypotType.HTTP
        self.ssl_context = ssl_context
        
    async def start(self):
        """Start HTTP honeypot"""
        await super().start()
        
        # Start HTTP server
        self.server = await asyncio.start_server(
            self.handle_connection,
            '0.0.0.0',
            self.port,
            ssl=self.ssl_context
        )
        
        protocol = "HTTPS" if self.ssl_context else "HTTP"
        self.logger.info(f"{protocol} honeypot listening on port {self.port}")
        
    async def handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle HTTP connection"""
        client_addr = writer.get_extra_info('peername')
        if not client_addr:
            return
            
        source_ip, source_port = client_addr[0], client_addr[1]
        
        try:
            # Read HTTP request
            request_data = await reader.read(8192)
            request_str = request_data.decode('utf-8', errors='ignore')
            
            # Parse HTTP request
            request_info = self._parse_http_request(request_str)
            
            await self.log_interaction(
                source_ip, source_port, "http_request",
                payload=request_str,
                severity=self._assess_request_severity(request_info),
                user_agent=request_info.get('user_agent'),
                attack_vector=self._identify_attack_vector(request_info)
            )
            
            # Send HTTP response
            response = self._generate_response(request_info)
            writer.write(response.encode())
            await writer.drain()
            
        except Exception as e:
            self.logger.debug(f"HTTP connection error from {source_ip}: {e}")
        finally:
            writer.close()
            await writer.wait_closed()
            
    def _parse_http_request(self, request: str) -> Dict[str, Any]:
        """Parse HTTP request"""
        lines = request.split('\n')
        if not lines:
            return {}
            
        # Parse request line
        request_line = lines[0].strip()
        parts = request_line.split(' ')
        
        info = {
            'method': parts[0] if len(parts) > 0 else '',
            'path': parts[1] if len(parts) > 1 else '',
            'version': parts[2] if len(parts) > 2 else '',
            'headers': {},
            'body': ''
        }
        
        # Parse headers
        header_end = False
        body_start = 0
        for i, line in enumerate(lines[1:], 1):
            if not header_end:
                if line.strip() == '':
                    header_end = True
                    body_start = i + 1
                else:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        info['headers'][key.strip().lower()] = value.strip()
            
        # Extract body
        if body_start < len(lines):
            info['body'] = '\n'.join(lines[body_start:])
            
        # Extract user agent
        info['user_agent'] = info['headers'].get('user-agent', '')
        
        return info
        
    def _assess_request_severity(self, request_info: Dict[str, Any]) -> str:
        """Assess severity of HTTP request"""
        path = request_info.get('path', '').lower()
        body = request_info.get('body', '').lower()
        headers = request_info.get('headers', {})
        
        # High severity indicators
        high_indicators = [
            '/etc/passwd', '/bin/sh', 'cmd.exe', 'system(',
            'exec(', 'eval(', '<script', 'union select',
            'drop table', '../../../'
        ]
        
        for indicator in high_indicators:
            if indicator in path or indicator in body:
                return "high"
        
        # Medium severity indicators
        medium_indicators = [
            '/admin', '/wp-admin', '/config', '/.env',
            '/backup', '/test', '/debug', 'php?'
        ]
        
        for indicator in medium_indicators:
            if indicator in path:
                return "medium"
        
        # Check for suspicious user agents
        user_agent = headers.get('user-agent', '').lower()
        suspicious_agents = ['sqlmap', 'nmap', 'nikto', 'gobuster', 'dirb']
        for agent in suspicious_agents:
            if agent in user_agent:
                return "high"
        
        return "low"
        
    def _identify_attack_vector(self, request_info: Dict[str, Any]) -> Optional[str]:
        """Identify attack vector from request"""
        path = request_info.get('path', '').lower()
        body = request_info.get('body', '').lower()
        
        # SQL injection
        sql_indicators = ['union select', 'or 1=1', 'drop table', 'exec xp_']
        for indicator in sql_indicators:
            if indicator in path or indicator in body:
                return "sql_injection"
        
        # XSS
        xss_indicators = ['<script', 'javascript:', 'onerror=', 'alert(']
        for indicator in xss_indicators:
            if indicator in path or indicator in body:
                return "xss"
        
        # Path traversal
        if '../' in path or '..\\' in path:
            return "path_traversal"
        
        # Command injection
        cmd_indicators = [';cat', '|wget', '&&curl', '/bin/sh']
        for indicator in cmd_indicators:
            if indicator in path or indicator in body:
                return "command_injection"
        
        return None
        
    def _generate_response(self, request_info: Dict[str, Any]) -> str:
        """Generate HTTP response"""
        path = request_info.get('path', '/')
        
        # Serve different content based on path to appear realistic
        if path == '/' or path == '/index.html':
            content = self._generate_fake_homepage()
        elif '/admin' in path:
            content = self._generate_fake_admin_page()
        elif '/login' in path:
            content = self._generate_fake_login_page()
        elif '/api/' in path:
            content = self._generate_fake_api_response()
        else:
            content = self._generate_fake_404()
            
        response = f"""HTTP/1.1 200 OK\r
Content-Type: text/html\r
Content-Length: {len(content)}\r
Server: Apache/2.4.41 (Ubuntu)\r
\r
{content}"""
        
        return response
        
    def _generate_fake_homepage(self) -> str:
        """Generate fake homepage content"""
        return """<!DOCTYPE html>
<html>
<head>
    <title>SutazAI Management Portal</title>
</head>
<body>
    <h1>Welcome to SutazAI Management Portal</h1>
    <p>Please <a href="/login">login</a> to access the system.</p>
    <p>For administrative access, visit <a href="/admin">admin panel</a>.</p>
</body>
</html>"""
        
    def _generate_fake_admin_page(self) -> str:
        """Generate fake admin page"""
        return """<!DOCTYPE html>
<html>
<head>
    <title>Admin Panel - SutazAI</title>
</head>
<body>
    <h1>Administrative Access Required</h1>
    <form action="/admin/login" method="post">
        <input type="text" name="username" placeholder="Username">
        <input type="password" name="password" placeholder="Password">
        <input type="submit" value="Login">
    </form>
</body>
</html>"""
        
    def _generate_fake_login_page(self) -> str:
        """Generate fake login page"""
        return """<!DOCTYPE html>
<html>
<head>
    <title>Login - SutazAI</title>
</head>
<body>
    <h1>SutazAI System Login</h1>
    <form action="/auth" method="post">
        <input type="text" name="username" placeholder="Username">
        <input type="password" name="password" placeholder="Password">
        <input type="submit" value="Login">
    </form>
</body>
</html>"""
        
    def _generate_fake_api_response(self) -> str:
        """Generate fake API response"""
        return """{"status": "error", "message": "Authentication required", "code": 401}"""
        
    def _generate_fake_404(self) -> str:
        """Generate 404 page"""
        return """<!DOCTYPE html>
<html>
<head>
    <title>404 Not Found</title>
</head>
<body>
    <h1>404 - Page Not Found</h1>
    <p>The requested page could not be found.</p>
</body>
</html>"""

class DatabaseHoneypot(BaseHoneypot):
    """Database honeypot for detecting SQL injection and unauthorized access"""
    
    def __init__(self, honeypot_id: str, port: int, database: HoneypotDatabase, 
                 intelligence_engine: ThreatIntelligenceEngine, db_type: str = "mysql"):
        super().__init__(honeypot_id, port, database, intelligence_engine)
        self.honeypot_type = HoneypotType.DATABASE
        self.db_type = db_type
        
    async def start(self):
        """Start database honeypot"""
        await super().start()
        
        # Start database server emulation
        self.server = await asyncio.start_server(
            self.handle_connection,
            '0.0.0.0',
            self.port
        )
        
        self.logger.info(f"{self.db_type.upper()} honeypot listening on port {self.port}")
        
    async def handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle database connection"""
        client_addr = writer.get_extra_info('peername')
        if not client_addr:
            return
            
        source_ip, source_port = client_addr[0], client_addr[1]
        
        try:
            # Send database greeting
            if self.db_type == "mysql":
                await self._handle_mysql_connection(reader, writer, source_ip, source_port)
            elif self.db_type == "postgresql":
                await self._handle_postgresql_connection(reader, writer, source_ip, source_port)
            else:
                await self._handle_generic_db_connection(reader, writer, source_ip, source_port)
                
        except Exception as e:
            self.logger.debug(f"Database connection error from {source_ip}: {e}")
        finally:
            writer.close()
            await writer.wait_closed()
            
    async def _handle_mysql_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
                                     source_ip: str, source_port: int):
        """Handle MySQL protocol simulation"""
        # Send MySQL handshake
        handshake = self._create_mysql_handshake()
        writer.write(handshake)
        await writer.drain()
        
        await self.log_interaction(
            source_ip, source_port, "mysql_connection_attempt",
            payload="MySQL handshake sent",
            severity="medium"
        )
        
        # Read authentication packet
        try:
            auth_data = await reader.read(1024)
            if auth_data:
                # Extract authentication info (simplified)
                username, password = self._parse_mysql_auth(auth_data)
                
                await self.log_interaction(
                    source_ip, source_port, "mysql_auth_attempt",
                    payload=f"Auth data: {auth_data[:100].hex()}",
                    severity="high",
                    credentials={"username": username, "password": password}
                )
                
                # Send authentication error
                error_packet = self._create_mysql_error("Access denied")
                writer.write(error_packet)
                await writer.drain()
                
        except Exception as e:
            self.logger.debug(f"MySQL auth error: {e}")
            
    def _create_mysql_handshake(self) -> bytes:
        """Create MySQL handshake packet (simplified)"""
        # This is a very simplified MySQL handshake
        # Real implementation would follow MySQL protocol exactly
        handshake = (
            b'\x0a'  # Protocol version
            b'5.7.33-0ubuntu0.18.04.1\x00'  # Server version
            b'\x01\x00\x00\x00'  # Connection ID
            b'salt1234\x00'  # Salt part 1
            b'\x00\xff'  # Capabilities
            b'\x08'  # Character set
            b'\x02\x00'  # Status flags
            b'\x00\x0c'  # Extended capabilities
            b'\x00'  # Plugin length
            b'salt5678\x00'  # Salt part 2
        )
        
        # Add packet header
        packet = bytes([len(handshake)]) + b'\x00\x00\x00' + handshake
        return packet
        
    def _parse_mysql_auth(self, data: bytes) -> tuple[Optional[str], Optional[str]]:
        """Parse MySQL authentication packet (simplified)"""
        # This is very simplified - real MySQL protocol parsing is complex
        try:
            # Skip packet header and capability flags
            pos = 36  # Approximate position after standard fields
            
            # Extract username (null-terminated)
            username_end = data.find(b'\x00', pos)
            if username_end != -1:
                username = data[pos:username_end].decode('utf-8', errors='ignore')
                return username, "hidden"  # Password is hashed, so we can't see it
            
        except Exception:
            pass
            
        return None, None
        
    def _create_mysql_error(self, message: str) -> bytes:
        """Create MySQL error packet"""
        error_data = (
            b'\xff'  # Error packet marker
            b'\x15\x04'  # Error code (Access denied)
            b'#28000'  # SQL state
            + message.encode('utf-8')
        )
        
        # Add packet header
        packet = bytes([len(error_data)]) + b'\x01\x00\x00' + error_data
        return packet
        
    async def _handle_postgresql_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
                                          source_ip: str, source_port: int):
        """Handle PostgreSQL protocol simulation"""
        # This would implement PostgreSQL wire protocol
        # For now, just log the connection
        await self.log_interaction(
            source_ip, source_port, "postgresql_connection_attempt",
            payload="PostgreSQL connection attempted",
            severity="medium"
        )
        
    async def _handle_generic_db_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
                                          source_ip: str, source_port: int):
        """Handle generic database connection"""
        await self.log_interaction(
            source_ip, source_port, "database_connection_attempt",
            payload=f"Generic {self.db_type} connection attempted",
            severity="medium"
        )

class AIAgentHoneypot(BaseHoneypot):
    """AI Agent honeypot mimicking SutazAI services"""
    
    def __init__(self, honeypot_id: str, port: int, database: HoneypotDatabase, 
                 intelligence_engine: ThreatIntelligenceEngine):
        super().__init__(honeypot_id, port, database, intelligence_engine)
        self.honeypot_type = HoneypotType.AI_AGENT
        
    async def start(self):
        """Start AI Agent honeypot"""
        await super().start()
        
        # Start HTTP server for AI API endpoints
        self.server = await asyncio.start_server(
            self.handle_connection,
            '0.0.0.0',
            self.port
        )
        
        self.logger.info(f"AI Agent honeypot listening on port {self.port}")
        
    async def handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle AI Agent API connection"""
        client_addr = writer.get_extra_info('peername')
        if not client_addr:
            return
            
        source_ip, source_port = client_addr[0], client_addr[1]
        
        try:
            # Read HTTP request
            request_data = await reader.read(8192)
            request_str = request_data.decode('utf-8', errors='ignore')
            
            # Parse request
            request_info = self._parse_http_request(request_str)
            
            # Determine if this is targeting AI endpoints
            is_ai_attack = self._is_ai_targeted_attack(request_info)
            severity = "high" if is_ai_attack else "medium"
            
            await self.log_interaction(
                source_ip, source_port, "ai_agent_access_attempt",
                payload=request_str,
                severity=severity,
                user_agent=request_info.get('user_agent'),
                attack_vector="ai_service_exploitation" if is_ai_attack else None
            )
            
            # Send realistic AI service response
            response = self._generate_ai_response(request_info)
            writer.write(response.encode())
            await writer.drain()
            
        except Exception as e:
            self.logger.debug(f"AI Agent connection error from {source_ip}: {e}")
        finally:
            writer.close()
            await writer.wait_closed()
            
    def _parse_http_request(self, request: str) -> Dict[str, Any]:
        """Parse HTTP request (reuse from HTTPHoneypot)"""
        lines = request.split('\n')
        if not lines:
            return {}
            
        request_line = lines[0].strip()
        parts = request_line.split(' ')
        
        info = {
            'method': parts[0] if len(parts) > 0 else '',
            'path': parts[1] if len(parts) > 1 else '',
            'version': parts[2] if len(parts) > 2 else '',
            'headers': {},
            'body': ''
        }
        
        # Parse headers and body
        header_end = False
        body_start = 0
        for i, line in enumerate(lines[1:], 1):
            if not header_end:
                if line.strip() == '':
                    header_end = True
                    body_start = i + 1
                else:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        info['headers'][key.strip().lower()] = value.strip()
        
        if body_start < len(lines):
            info['body'] = '\n'.join(lines[body_start:])
            
        info['user_agent'] = info['headers'].get('user-agent', '')
        
        return info
        
    def _is_ai_targeted_attack(self, request_info: Dict[str, Any]) -> bool:
        """Check if request is targeting AI services"""
        path = request_info.get('path', '').lower()
        body = request_info.get('body', '').lower()
        
        # AI-specific attack indicators
        ai_indicators = [
            '/api/v1/coordinator', '/api/v1/agents', '/ollama',
            'model', 'prompt', 'generate', 'completion',
            'agent', 'ai', 'ml', 'llm', 'gpt'
        ]
        
        for indicator in ai_indicators:
            if indicator in path or indicator in body:
                return True
                
        return False
        
    def _generate_ai_response(self, request_info: Dict[str, Any]) -> str:
        """Generate realistic AI service response"""
        path = request_info.get('path', '/')
        method = request_info.get('method', 'GET')
        
        # Generate different responses based on endpoint
        if '/api/v1/coordinator' in path:
            content = self._generate_coordinator_response(method)
        elif '/api/v1/agents' in path:
            content = self._generate_agents_response(method)
        elif '/ollama' in path:
            content = self._generate_ollama_response(method)
        elif '/health' in path:
            content = '{"status": "healthy", "version": "1.0.0"}'
        else:
            content = '{"error": "Not Found", "code": 404}'
            
        response = f"""HTTP/1.1 200 OK\r
Content-Type: application/json\r
Content-Length: {len(content)}\r
Server: SutazAI/1.0\r
\r
{content}"""
        
        return response
        
    def _generate_coordinator_response(self, method: str) -> str:
        """Generate coordinator API response"""
        if method == "POST":
            return """{"status": "success", "task_id": "task_12345", "message": "Task queued for processing"}"""
        else:
            return """{"status": "ready", "active_agents": 5, "queue_length": 2}"""
            
    def _generate_agents_response(self, method: str) -> str:
        """Generate agents API response"""
        return """{
    "agents": [
        {"id": "agent_001", "type": "ai-senior-engineer", "status": "active"},
        {"id": "agent_002", "type": "ai-qa-team-lead", "status": "active"},
        {"id": "agent_003", "type": "ai-system-architect", "status": "idle"}
    ],
    "total": 3
}"""
        
    def _generate_ollama_response(self, method: str) -> str:
        """Generate Ollama API response"""
        if method == "POST":
            return """{"model": "tinyllama", "response": "I am a helpful AI assistant.", "done": true}"""
        else:
            return """{"models": [{"name": "tinyllama", "size": "1.1GB"}]}"""

class HoneypotOrchestrator:
    """Orchestrates multiple honeypots and manages the overall deception infrastructure"""
    
    def __init__(self):
        self.database = HoneypotDatabase()
        self.intelligence_engine = ThreatIntelligenceEngine(self.database)
        self.honeypots: Dict[str, BaseHoneypot] = {}
        self.is_running = False
        self.logger = logging.getLogger("honeypot.orchestrator")
        
    async def start(self):
        """Start all honeypots"""
        self.is_running = True
        self.logger.info("Starting honeypot infrastructure...")
        
        # Define honeypot configurations
        honeypot_configs = [
            # SSH honeypots
            {"type": SSHHoneypot, "id": "ssh_2222", "port": 2222},
            {"type": SSHHoneypot, "id": "ssh_22", "port": 22},
            
            # HTTP/HTTPS honeypots
            {"type": HTTPHoneypot, "id": "http_8080", "port": 8080},
            {"type": HTTPHoneypot, "id": "http_80", "port": 80},
            {"type": HTTPHoneypot, "id": "https_8443", "port": 8443, "ssl": True},
            
            # Database honeypots
            {"type": DatabaseHoneypot, "id": "mysql_3306", "port": 3306, "db_type": "mysql"},
            {"type": DatabaseHoneypot, "id": "postgresql_5432", "port": 5432, "db_type": "postgresql"},
            
            # AI Agent honeypots
            {"type": AIAgentHoneypot, "id": "ai_agent_11434", "port": 11434},
            {"type": AIAgentHoneypot, "id": "ai_agent_8000", "port": 8000},
        ]
        
        # Start honeypots
        for config in honeypot_configs:
            try:
                honeypot_class = config["type"]
                honeypot_id = config["id"]
                port = config["port"]
                
                # Check if port is available
                if not await self._is_port_available(port):
                    self.logger.warning(f"Port {port} is not available, skipping {honeypot_id}")
                    continue
                
                # Create honeypot instance
                if honeypot_class == DatabaseHoneypot:
                    honeypot = honeypot_class(
                        honeypot_id, port, self.database, self.intelligence_engine,
                        config.get("db_type", "mysql")
                    )
                else:
                    honeypot = honeypot_class(
                        honeypot_id, port, self.database, self.intelligence_engine
                    )
                
                # Start honeypot
                await honeypot.start()
                self.honeypots[honeypot_id] = honeypot
                self.logger.info(f"Started honeypot: {honeypot_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to start honeypot {config['id']}: {e}")
        
        # Start monitoring and analysis tasks
        asyncio.create_task(self._monitor_threats())
        asyncio.create_task(self._generate_reports())
        
        self.logger.info(f"Honeypot infrastructure started with {len(self.honeypots)} active honeypots")
        
    async def stop(self):
        """Stop all honeypots"""
        self.is_running = False
        self.logger.info("Stopping honeypot infrastructure...")
        
        # Stop all honeypots
        stop_tasks = []
        for honeypot in self.honeypots.values():
            stop_tasks.append(honeypot.stop())
        
        await asyncio.gather(*stop_tasks, return_exceptions=True)
        self.honeypots.clear()
        
        self.logger.info("Honeypot infrastructure stopped")
        
    async def _is_port_available(self, port: int) -> bool:
        """Check if port is available"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex(('127.0.0.1', port))
                return result != 0  # Port is available if connection failed
        except Exception:
            return False
            
    async def _monitor_threats(self):
        """Monitor for high-priority threats"""
        while self.is_running:
            try:
                # Get recent high-severity events
                events = self.database.get_events(limit=50, hours=1)
                high_threat_events = [e for e in events if e.severity in ['critical', 'high']]
                
                if high_threat_events:
                    await self._process_high_threat_events(high_threat_events)
                
                # Update threat intelligence
                await self._update_threat_intelligence()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Threat monitoring error: {e}")
                await asyncio.sleep(60)
                
    async def _process_high_threat_events(self, events: List[HoneypotEvent]):
        """Process high-threat events"""
        # Group events by source IP
        ip_events = {}
        for event in events:
            if event.source_ip not in ip_events:
                ip_events[event.source_ip] = []
            ip_events[event.source_ip].append(event)
        
        # Analyze each IP
        for source_ip, ip_event_list in ip_events.items():
            if len(ip_event_list) >= 5:  # Multiple attacks from same IP
                await self._handle_persistent_attacker(source_ip, ip_event_list)
                
    async def _handle_persistent_attacker(self, source_ip: str, events: List[HoneypotEvent]):
        """Handle persistent attacker"""
        self.logger.critical(f"PERSISTENT ATTACKER DETECTED: {source_ip} with {len(events)} recent attacks")
        
        # Send critical alert
        if SECURITY_INTEGRATION and security_manager:
            await security_manager.audit.log_event(
                "persistent_attacker_detected",
                "critical",
                "honeypot_orchestrator",
                {
                    "source_ip": source_ip,
                    "attack_count": len(events),
                    "honeypots_targeted": list(set(e.honeypot_id for e in events)),
                    "attack_types": list(set(e.event_type for e in events))
                },
                ip_address=source_ip
            )
            
    async def _update_threat_intelligence(self):
        """Update threat intelligence indicators"""
        # This would fetch from external threat feeds in production
        # For now, maintain basic indicators
        pass
        
    async def _generate_reports(self):
        """Generate periodic security reports"""
        while self.is_running:
            try:
                await asyncio.sleep(3600)  # Generate report every hour
                
                if not self.is_running:
                    break
                    
                report = await self.generate_security_report()
                self.logger.info(f"Security report generated: {report['summary']['total_events']} events in last hour")
                
            except Exception as e:
                self.logger.error(f"Report generation error: {e}")
                
    async def get_status(self) -> Dict[str, Any]:
        """Get honeypot infrastructure status"""
        active_honeypots = []
        for honeypot_id, honeypot in self.honeypots.items():
            active_honeypots.append({
                "id": honeypot_id,
                "type": honeypot.honeypot_type.value,
                "port": honeypot.port,
                "status": "active" if honeypot.is_running else "inactive"
            })
        
        # Get recent statistics
        recent_events = self.database.get_events(limit=1000, hours=24)
        
        return {
            "infrastructure_status": "active" if self.is_running else "inactive",
            "active_honeypots": len(self.honeypots),
            "honeypot_details": active_honeypots,
            "events_last_24h": len(recent_events),
            "unique_attackers_24h": len(set(e.source_ip for e in recent_events)),
            "high_severity_events_24h": len([e for e in recent_events if e.severity in ['critical', 'high']]),
            "database_status": "connected"
        }
        
    async def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report"""
        events = self.database.get_events(limit=1000, hours=24)
        
        # Basic statistics
        total_events = len(events)
        unique_ips = set(e.source_ip for e in events)
        
        # Threat breakdown
        threat_breakdown = {}
        attack_vectors = {}
        honeypot_activity = {}
        
        for event in events:
            # Severity breakdown
            severity = event.severity
            threat_breakdown[severity] = threat_breakdown.get(severity, 0) + 1
            
            # Attack vector breakdown
            if event.attack_vector:
                attack_vectors[event.attack_vector] = attack_vectors.get(event.attack_vector, 0) + 1
            
            # Honeypot activity
            honeypot_type = event.honeypot_type
            honeypot_activity[honeypot_type] = honeypot_activity.get(honeypot_type, 0) + 1
        
        # Top attackers
        ip_counts = {}
        for event in events:
            ip_counts[event.source_ip] = ip_counts.get(event.source_ip, 0) + 1
        
        top_attackers = sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "period": "24 hours",
            "summary": {
                "total_events": total_events,
                "unique_attackers": len(unique_ips),
                "active_honeypots": len(self.honeypots),
                "threat_breakdown": threat_breakdown,
                "attack_vectors": attack_vectors,
                "honeypot_activity": honeypot_activity
            },
            "top_attackers": [
                {"ip": ip, "attempts": count} for ip, count in top_attackers
            ],
            "recommendations": self._generate_security_recommendations(events)
        }
        
    def _generate_security_recommendations(self, events: List[HoneypotEvent]) -> List[str]:
        """Generate security recommendations based on observed attacks"""
        recommendations = []
        
        # Check for common attack patterns
        sql_injection_events = [e for e in events if e.attack_vector == 'sql_injection']
        if sql_injection_events:
            recommendations.append(
                f"SQL injection attempts detected ({len(sql_injection_events)} events). "
                "Ensure all database inputs are properly sanitized and use parameterized queries."
            )
        
        xss_events = [e for e in events if e.attack_vector == 'xss']
        if xss_events:
            recommendations.append(
                f"XSS attempts detected ({len(xss_events)} events). "
                "Implement proper input validation and output encoding."
            )
        
        cmd_injection_events = [e for e in events if e.attack_vector == 'command_injection']
        if cmd_injection_events:
            recommendations.append(
                f"Command injection attempts detected ({len(cmd_injection_events)} events). "
                "Avoid executing system commands with user input."
            )
        
        # Check for brute force attacks
        ssh_events = [e for e in events if e.honeypot_type == 'ssh']
        if len(ssh_events) > 50:
            recommendations.append(
                f"High volume of SSH attacks detected ({len(ssh_events)} events). "
                "Consider implementing fail2ban or similar brute force protection."
            )
        
        # Check for scanning activity
        unique_ips = set(e.source_ip for e in events)
        if len(events) > 100 and len(unique_ips) < 10:
            recommendations.append(
                "Concentrated attack activity detected from few sources. "
                "Consider implementing IP-based rate limiting."
            )
        
        return recommendations

# Global honeypot orchestrator instance
honeypot_orchestrator = HoneypotOrchestrator()