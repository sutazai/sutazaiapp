#!/usr/bin/env python3
"""
Automated Incident Response and Forensics Framework
Implements comprehensive incident detection, response, and forensic analysis
"""

import asyncio
import logging
import json
import os
import time
import hashlib
import subprocess
import shutil
import tempfile
import tarfile
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import redis
import psycopg2
from psycopg2.extras import RealDictCursor
import threading
from concurrent.futures import ThreadPoolExecutor
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import psutil
import docker

class IncidentSeverity(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class IncidentStatus(Enum):
    NEW = "new"
    INVESTIGATING = "investigating"
    CONTAINED = "contained"
    RESOLVED = "resolved"
    CLOSED = "closed"

class IncidentType(Enum):
    MALWARE = "malware"
    DATA_BREACH = "data_breach"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DDOS = "ddos"
    SYSTEM_COMPROMISE = "system_compromise"
    INSIDER_THREAT = "insider_threat"
    PHISHING = "phishing"
    RANSOMWARE = "ransomware"
    APT = "apt"
    DATA_EXFILTRATION = "data_exfiltration"

class ResponseAction(Enum):
    ISOLATE_HOST = "isolate_host"
    BLOCK_IP = "block_ip"
    DISABLE_ACCOUNT = "disable_account"
    COLLECT_FORENSICS = "collect_forensics"
    QUARANTINE_FILE = "quarantine_file"
    RESET_PASSWORDS = "reset_passwords"
    NOTIFY_STAKEHOLDERS = "notify_stakeholders"
    ESCALATE = "escalate"
    MONITOR = "monitor"

@dataclass
class SecurityIncident:
    """Security incident record"""
    incident_id: str
    title: str
    description: str
    incident_type: IncidentType
    severity: IncidentSeverity
    status: IncidentStatus
    source_ip: Optional[str]
    affected_systems: List[str]
    indicators: List[str]
    discovered_at: datetime
    reported_by: str
    assigned_to: Optional[str]
    estimated_impact: str
    containment_actions: List[ResponseAction]
    timeline: List[Dict[str, Any]]
    evidence_collected: List[str]
    forensic_artifacts: List[str]

@dataclass
class ForensicArtifact:
    """Digital forensic artifact"""
    artifact_id: str
    incident_id: str
    artifact_type: str  # memory_dump, disk_image, network_capture, log_file
    source_system: str
    collection_time: datetime
    file_path: str
    file_hash: str
    file_size: int
    chain_of_custody: List[Dict[str, Any]]
    analysis_status: str
    metadata: Dict[str, Any]

class IncidentResponseEngine:
    """Automated incident response and forensics engine"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.redis_client = None
        self.db_connection = None
        self.docker_client = None
        self.active_incidents: Dict[str, SecurityIncident] = {}
        self.forensic_artifacts: Dict[str, ForensicArtifact] = {}
        self.response_playbooks = {}
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=10)
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize incident response components"""
        try:
            # Initialize Redis
            self.redis_client = redis.Redis(
                host=self.config.get('redis_host', 'redis'),
                port=self.config.get('redis_port', 6379),
                password=self.config.get('redis_password'),
                ssl=True,
                decode_responses=True
            )
            
            # Initialize PostgreSQL
            self.db_connection = psycopg2.connect(
                host=self.config.get('postgres_host', 'postgres'),
                port=self.config.get('postgres_port', 5432),
                database=self.config.get('postgres_db', 'sutazai'),
                user=self.config.get('postgres_user', 'sutazai'),
                password=self.config.get('postgres_password'),
                sslmode='require'
            )
            
            # Initialize Docker client
            self.docker_client = docker.from_env()
            
            # Load response playbooks
            self._load_response_playbooks()
            
            # Setup forensic storage
            self._setup_forensic_storage()
            
            # Load active incidents
            self._load_active_incidents()
            
            self.logger.info("Incident Response Engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Incident Response Engine: {e}")
            raise
    
    def _load_response_playbooks(self):
        """Load incident response playbooks"""
        self.response_playbooks = {
            IncidentType.MALWARE: {
                "actions": [
                    ResponseAction.ISOLATE_HOST,
                    ResponseAction.COLLECT_FORENSICS,
                    ResponseAction.QUARANTINE_FILE,
                    ResponseAction.NOTIFY_STAKEHOLDERS
                ],
                "timeline": {
                    "immediate": [ResponseAction.ISOLATE_HOST],
                    "short_term": [ResponseAction.COLLECT_FORENSICS, ResponseAction.QUARANTINE_FILE],
                    "medium_term": [ResponseAction.NOTIFY_STAKEHOLDERS]
                }
            },
            IncidentType.DATA_BREACH: {
                "actions": [
                    ResponseAction.ISOLATE_HOST,
                    ResponseAction.COLLECT_FORENSICS,
                    ResponseAction.RESET_PASSWORDS,
                    ResponseAction.NOTIFY_STAKEHOLDERS,
                    ResponseAction.ESCALATE
                ],
                "timeline": {
                    "immediate": [ResponseAction.ISOLATE_HOST, ResponseAction.COLLECT_FORENSICS],
                    "short_term": [ResponseAction.RESET_PASSWORDS],
                    "medium_term": [ResponseAction.NOTIFY_STAKEHOLDERS, ResponseAction.ESCALATE]
                }
            },
            IncidentType.UNAUTHORIZED_ACCESS: {
                "actions": [
                    ResponseAction.DISABLE_ACCOUNT,
                    ResponseAction.COLLECT_FORENSICS,
                    ResponseAction.RESET_PASSWORDS,
                    ResponseAction.MONITOR
                ],
                "timeline": {
                    "immediate": [ResponseAction.DISABLE_ACCOUNT],
                    "short_term": [ResponseAction.COLLECT_FORENSICS, ResponseAction.RESET_PASSWORDS],
                    "medium_term": [ResponseAction.MONITOR]
                }
            },
            IncidentType.DDOS: {
                "actions": [
                    ResponseAction.BLOCK_IP,
                    ResponseAction.MONITOR,
                    ResponseAction.NOTIFY_STAKEHOLDERS
                ],
                "timeline": {
                    "immediate": [ResponseAction.BLOCK_IP],
                    "short_term": [ResponseAction.MONITOR],
                    "medium_term": [ResponseAction.NOTIFY_STAKEHOLDERS]
                }
            },
            IncidentType.RANSOMWARE: {
                "actions": [
                    ResponseAction.ISOLATE_HOST,
                    ResponseAction.COLLECT_FORENSICS,
                    ResponseAction.NOTIFY_STAKEHOLDERS,
                    ResponseAction.ESCALATE
                ],
                "timeline": {
                    "immediate": [ResponseAction.ISOLATE_HOST],
                    "short_term": [ResponseAction.COLLECT_FORENSICS],
                    "medium_term": [ResponseAction.NOTIFY_STAKEHOLDERS, ResponseAction.ESCALATE]
                }
            }
        }
    
    def _setup_forensic_storage(self):
        """Setup forensic evidence storage"""
        try:
            forensic_path = self.config.get('forensic_storage_path', '/opt/sutazaiapp/forensics')
            os.makedirs(forensic_path, exist_ok=True)
            
            # Create subdirectories
            subdirs = ['memory_dumps', 'disk_images', 'network_captures', 'log_files', 'quarantine']
            for subdir in subdirs:
                os.makedirs(os.path.join(forensic_path, subdir), exist_ok=True)
            
            self.logger.info(f"Forensic storage setup at {forensic_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to setup forensic storage: {e}")
    
    def _load_active_incidents(self):
        """Load active incidents from database"""
        try:
            cursor = self.db_connection.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT * FROM security_incidents 
                WHERE status NOT IN ('resolved', 'closed')
            """)
            
            for row in cursor.fetchall():
                incident = SecurityIncident(
                    incident_id=row['incident_id'],
                    title=row['title'],
                    description=row['description'],
                    incident_type=IncidentType(row['incident_type']),
                    severity=IncidentSeverity(row['severity']),
                    status=IncidentStatus(row['status']),
                    source_ip=row['source_ip'],
                    affected_systems=row['affected_systems'] or [],
                    indicators=row['indicators'] or [],
                    discovered_at=row['discovered_at'],
                    reported_by=row['reported_by'],
                    assigned_to=row['assigned_to'],
                    estimated_impact=row['estimated_impact'] or '',
                    containment_actions=[ResponseAction(a) for a in (row['containment_actions'] or [])],
                    timeline=row['timeline'] or [],
                    evidence_collected=row['evidence_collected'] or [],
                    forensic_artifacts=row['forensic_artifacts'] or []
                )
                self.active_incidents[incident.incident_id] = incident
            
            cursor.close()
            self.logger.info(f"Loaded {len(self.active_incidents)} active incidents")
            
        except Exception as e:
            self.logger.error(f"Failed to load active incidents: {e}")
    
    async def create_incident(self, title: str, description: str, incident_type: IncidentType,
                            severity: IncidentSeverity, source_ip: str = None,
                            affected_systems: List[str] = None, indicators: List[str] = None) -> str:
        """Create new security incident"""
        try:
            incident_id = self._generate_incident_id()
            
            incident = SecurityIncident(
                incident_id=incident_id,
                title=title,
                description=description,
                incident_type=incident_type,
                severity=severity,
                status=IncidentStatus.NEW,
                source_ip=source_ip,
                affected_systems=affected_systems or [],
                indicators=indicators or [],
                discovered_at=datetime.utcnow(),
                reported_by="automated_system",
                assigned_to=None,
                estimated_impact="",
                containment_actions=[],
                timeline=[{
                    "timestamp": datetime.utcnow().isoformat(),
                    "action": "incident_created",
                    "description": "Incident automatically created by security system"
                }],
                evidence_collected=[],
                forensic_artifacts=[]
            )
            
            # Store incident
            self.active_incidents[incident_id] = incident
            await self._store_incident(incident)
            
            # Trigger automated response
            await self._trigger_automated_response(incident)
            
            self.logger.info(f"Created incident: {incident_id} - {title}")
            return incident_id
            
        except Exception as e:
            self.logger.error(f"Failed to create incident: {e}")
            raise
    
    async def _trigger_automated_response(self, incident: SecurityIncident):
        """Trigger automated incident response"""
        try:
            # Get response playbook
            playbook = self.response_playbooks.get(incident.incident_type)
            if not playbook:
                self.logger.warning(f"No playbook found for incident type: {incident.incident_type}")
                return
            
            # Execute immediate actions
            immediate_actions = playbook["timeline"].get("immediate", [])
            for action in immediate_actions:
                await self._execute_response_action(incident, action)
            
            # Schedule short-term actions
            short_term_actions = playbook["timeline"].get("short_term", [])
            if short_term_actions:
                await asyncio.sleep(60)  # Wait 1 minute
                for action in short_term_actions:
                    await self._execute_response_action(incident, action)
            
            # Schedule medium-term actions
            medium_term_actions = playbook["timeline"].get("medium_term", [])
            if medium_term_actions:
                # Schedule for later execution
                asyncio.create_task(self._schedule_delayed_actions(incident, medium_term_actions, 300))  # 5 minutes
            
        except Exception as e:
            self.logger.error(f"Failed to trigger automated response: {e}")
    
    async def _schedule_delayed_actions(self, incident: SecurityIncident, actions: List[ResponseAction], delay: int):
        """Schedule delayed response actions"""
        await asyncio.sleep(delay)
        for action in actions:
            await self._execute_response_action(incident, action)
    
    async def _execute_response_action(self, incident: SecurityIncident, action: ResponseAction):
        """Execute specific response action"""
        try:
            self.logger.info(f"Executing response action: {action.value} for incident {incident.incident_id}")
            
            if action == ResponseAction.ISOLATE_HOST:
                await self._isolate_host(incident)
            elif action == ResponseAction.BLOCK_IP:
                await self._block_ip(incident)
            elif action == ResponseAction.DISABLE_ACCOUNT:
                await self._disable_account(incident)
            elif action == ResponseAction.COLLECT_FORENSICS:
                await self._collect_forensics(incident)
            elif action == ResponseAction.QUARANTINE_FILE:
                await self._quarantine_file(incident)
            elif action == ResponseAction.RESET_PASSWORDS:
                await self._reset_passwords(incident)
            elif action == ResponseAction.NOTIFY_STAKEHOLDERS:
                await self._notify_stakeholders(incident)
            elif action == ResponseAction.ESCALATE:
                await self._escalate_incident(incident)
            elif action == ResponseAction.MONITOR:
                await self._enhance_monitoring(incident)
            
            # Update incident timeline
            incident.timeline.append({
                "timestamp": datetime.utcnow().isoformat(),
                "action": action.value,
                "description": f"Executed response action: {action.value}"
            })
            
            # Update containment actions
            if action not in incident.containment_actions:
                incident.containment_actions.append(action)
            
            # Update incident status
            if incident.status == IncidentStatus.NEW:
                incident.status = IncidentStatus.INVESTIGATING
            
            # Store updated incident
            await self._store_incident(incident)
            
        except Exception as e:
            self.logger.error(f"Failed to execute response action {action.value}: {e}")
            
            # Log failed action in timeline
            incident.timeline.append({
                "timestamp": datetime.utcnow().isoformat(),
                "action": f"{action.value}_failed",
                "description": f"Failed to execute {action.value}: {str(e)}"
            })
    
    async def _isolate_host(self, incident: SecurityIncident):
        """Isolate affected host from network"""
        try:
            for system in incident.affected_systems:
                self.logger.info(f"Isolating host: {system}")
                
                # Add iptables rules to block all traffic except management
                isolation_rules = [
                    ['iptables', '-I', 'INPUT', '1', '-j', 'DROP'],
                    ['iptables', '-I', 'OUTPUT', '1', '-j', 'DROP'],
                    ['iptables', '-I', 'INPUT', '1', '-p', 'tcp', '--dport', '22', '-j', 'ACCEPT'],
                    ['iptables', '-I', 'OUTPUT', '1', '-p', 'tcp', '--sport', '22', '-j', 'ACCEPT']
                ]
                
                for rule in isolation_rules:
                    subprocess.run(rule, check=False)
                
                # Store isolation action
                isolation_data = {
                    "action": "host_isolated",
                    "system": system,
                    "timestamp": datetime.utcnow().isoformat(),
                    "incident_id": incident.incident_id
                }
                
                self.redis_client.lpush("isolation_actions", json.dumps(isolation_data))
                
        except Exception as e:
            self.logger.error(f"Failed to isolate host: {e}")
            raise
    
    async def _block_ip(self, incident: SecurityIncident):
        """Block source IP address"""
        try:
            if incident.source_ip:
                self.logger.info(f"Blocking IP: {incident.source_ip}")
                
                # Add iptables rule to block IP
                subprocess.run([
                    'iptables', '-I', 'INPUT', '1',
                    '-s', incident.source_ip, '-j', 'DROP'
                ], check=False)
                
                # Store in Redis for distributed blocking
                block_data = {
                    "ip": incident.source_ip,
                    "reason": f"Incident {incident.incident_id}: {incident.title}",
                    "blocked_at": datetime.utcnow().isoformat(),
                    "incident_id": incident.incident_id
                }
                
                self.redis_client.setex(
                    f"blocked_ip:{incident.source_ip}",
                    86400,  # 24 hours
                    json.dumps(block_data)
                )
                
        except Exception as e:
            self.logger.error(f"Failed to block IP: {e}")
            raise
    
    async def _disable_account(self, incident: SecurityIncident):
        """Disable compromised user accounts"""
        try:
            # Extract potential usernames from indicators
            usernames = []
            for indicator in incident.indicators:
                if '@' not in indicator and len(indicator.split()) == 1:
                    usernames.append(indicator)
            
            cursor = self.db_connection.cursor()
            for username in usernames:
                self.logger.info(f"Disabling account: {username}")
                
                # Disable user account
                cursor.execute(
                    "UPDATE users SET is_active = false WHERE username = %s",
                    (username,)
                )
                
                # Log account disable action
                cursor.execute("""
                    INSERT INTO security_audit_log (event_type, details, timestamp)
                    VALUES (%s, %s, %s)
                """, (
                    "account_disabled",
                    json.dumps({
                        "username": username,
                        "incident_id": incident.incident_id,
                        "reason": "Incident response - unauthorized access"
                    }),
                    datetime.utcnow()
                ))
            
            self.db_connection.commit()
            cursor.close()
            
        except Exception as e:
            self.logger.error(f"Failed to disable accounts: {e}")
            raise
    
    async def _collect_forensics(self, incident: SecurityIncident):
        """Collect forensic evidence"""
        try:
            self.logger.info(f"Collecting forensics for incident: {incident.incident_id}")
            
            # Collect system information
            system_info = await self._collect_system_forensics(incident)
            if system_info:
                incident.forensic_artifacts.append(system_info.artifact_id)
            
            # Collect network forensics
            network_info = await self._collect_network_forensics(incident)
            if network_info:
                incident.forensic_artifacts.append(network_info.artifact_id)
            
            # Collect log forensics
            log_info = await self._collect_log_forensics(incident)
            if log_info:
                incident.forensic_artifacts.append(log_info.artifact_id)
            
            # Collect memory dumps if critical
            if incident.severity == IncidentSeverity.CRITICAL:
                memory_dump = await self._collect_memory_dump(incident)
                if memory_dump:
                    incident.forensic_artifacts.append(memory_dump.artifact_id)
            
        except Exception as e:
            self.logger.error(f"Failed to collect forensics: {e}")
            raise
    
    async def _collect_system_forensics(self, incident: SecurityIncident) -> Optional[ForensicArtifact]:
        """Collect system forensic information"""
        try:
            artifact_id = self._generate_artifact_id()
            forensic_path = self.config.get('forensic_storage_path', '/opt/sutazaiapp/forensics')
            artifact_path = os.path.join(forensic_path, 'system_info', f"{artifact_id}.json")
            
            os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
            
            # Collect system information
            system_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "incident_id": incident.incident_id,
                "hostname": os.uname().nodename,
                "processes": [proc.info for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time'])],
                "network_connections": [
                    conn._asdict() for conn in psutil.net_connections() 
                    if conn.status == 'ESTABLISHED'
                ],
                "system_stats": {
                    "cpu_percent": psutil.cpu_percent(),
                    "memory": psutil.virtual_memory()._asdict(),
                    "disk": psutil.disk_usage('/')._asdict()
                },
                "environment_variables": dict(os.environ),
                "mounted_filesystems": [mount._asdict() for mount in psutil.disk_partitions()]
            }
            
            # Store system data
            with open(artifact_path, 'w') as f:
                json.dump(system_data, f, indent=2, default=str)
            
            # Calculate file hash
            file_hash = self._calculate_file_hash(artifact_path)
            file_size = os.path.getsize(artifact_path)
            
            # Create forensic artifact record
            artifact = ForensicArtifact(
                artifact_id=artifact_id,
                incident_id=incident.incident_id,
                artifact_type="system_info",
                source_system=os.uname().nodename,
                collection_time=datetime.utcnow(),
                file_path=artifact_path,
                file_hash=file_hash,
                file_size=file_size,
                chain_of_custody=[{
                    "action": "collected",
                    "timestamp": datetime.utcnow().isoformat(),
                    "operator": "automated_system",
                    "hash": file_hash
                }],
                analysis_status="pending",
                metadata={"processes_count": len(system_data["processes"])}
            )
            
            # Store artifact
            self.forensic_artifacts[artifact_id] = artifact
            await self._store_forensic_artifact(artifact)
            
            return artifact
            
        except Exception as e:
            self.logger.error(f"Failed to collect system forensics: {e}")
            return None
    
    async def _collect_network_forensics(self, incident: SecurityIncident) -> Optional[ForensicArtifact]:
        """Collect network forensic information"""
        try:
            artifact_id = self._generate_artifact_id()
            forensic_path = self.config.get('forensic_storage_path', '/opt/sutazaiapp/forensics')
            artifact_path = os.path.join(forensic_path, 'network_captures', f"{artifact_id}.pcap")
            
            os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
            
            # Collect network capture using tcpdump
            self.logger.info("Starting network capture...")
            
            tcpdump_cmd = [
                'timeout', '60',  # Capture for 60 seconds
                'tcpdump', '-i', 'any', '-w', artifact_path,
                '-s', '65535'  # Capture full packets
            ]
            
            if incident.source_ip:
                tcpdump_cmd.extend(['host', incident.source_ip])
            
            result = subprocess.run(tcpdump_cmd, capture_output=True, text=True)
            
            if os.path.exists(artifact_path) and os.path.getsize(artifact_path) > 0:
                file_hash = self._calculate_file_hash(artifact_path)
                file_size = os.path.getsize(artifact_path)
                
                artifact = ForensicArtifact(
                    artifact_id=artifact_id,
                    incident_id=incident.incident_id,
                    artifact_type="network_capture",
                    source_system=os.uname().nodename,
                    collection_time=datetime.utcnow(),
                    file_path=artifact_path,
                    file_hash=file_hash,
                    file_size=file_size,
                    chain_of_custody=[{
                        "action": "collected",
                        "timestamp": datetime.utcnow().isoformat(),
                        "operator": "automated_system",
                        "hash": file_hash
                    }],
                    analysis_status="pending",
                    metadata={"capture_duration": "60s", "filter": incident.source_ip or "all"}
                )
                
                self.forensic_artifacts[artifact_id] = artifact
                await self._store_forensic_artifact(artifact)
                
                return artifact
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to collect network forensics: {e}")
            return None
    
    async def _collect_log_forensics(self, incident: SecurityIncident) -> Optional[ForensicArtifact]:
        """Collect log forensic information"""
        try:
            artifact_id = self._generate_artifact_id()
            forensic_path = self.config.get('forensic_storage_path', '/opt/sutazaiapp/forensics')
            artifact_path = os.path.join(forensic_path, 'log_files', f"{artifact_id}.tar.gz")
            
            os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
            
            # Collect relevant logs
            log_sources = [
                '/var/log/auth.log',
                '/var/log/syslog',
                '/opt/sutazaiapp/logs',
                '/var/log/nginx',
                '/var/log/apache2'
            ]
            
            # Create temporary directory for log collection
            with tempfile.TemporaryDirectory() as temp_dir:
                collected_logs = []
                
                for log_source in log_sources:
                    if os.path.exists(log_source):
                        if os.path.isfile(log_source):
                            # Copy individual log file
                            dest_path = os.path.join(temp_dir, os.path.basename(log_source))
                            shutil.copy2(log_source, dest_path)
                            collected_logs.append(log_source)
                        elif os.path.isdir(log_source):
                            # Copy log directory
                            dest_path = os.path.join(temp_dir, os.path.basename(log_source))
                            shutil.copytree(log_source, dest_path, ignore_errors=True)
                            collected_logs.append(log_source)
                
                # Get logs from Redis
                redis_logs = {
                    'security_events': self.redis_client.lrange('security_events', 0, -1),
                    'rasp_events': self.redis_client.lrange('rasp_events', 0, -1),
                    'threat_alerts': self.redis_client.lrange('threat_alerts', 0, -1)
                }
                
                # Save Redis logs
                redis_log_path = os.path.join(temp_dir, 'redis_logs.json')
                with open(redis_log_path, 'w') as f:
                    json.dump(redis_logs, f, indent=2, default=str)
                collected_logs.append('redis_logs')
                
                # Create compressed archive
                if collected_logs:
                    with tarfile.open(artifact_path, 'w:gz') as tar:
                        tar.add(temp_dir, arcname='logs')
                    
                    file_hash = self._calculate_file_hash(artifact_path)
                    file_size = os.path.getsize(artifact_path)
                    
                    artifact = ForensicArtifact(
                        artifact_id=artifact_id,
                        incident_id=incident.incident_id,
                        artifact_type="log_files",
                        source_system=os.uname().nodename,
                        collection_time=datetime.utcnow(),
                        file_path=artifact_path,
                        file_hash=file_hash,
                        file_size=file_size,
                        chain_of_custody=[{
                            "action": "collected",
                            "timestamp": datetime.utcnow().isoformat(),
                            "operator": "automated_system",
                            "hash": file_hash
                        }],
                        analysis_status="pending",
                        metadata={"log_sources": collected_logs}
                    )
                    
                    self.forensic_artifacts[artifact_id] = artifact
                    await self._store_forensic_artifact(artifact)
                    
                    return artifact
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to collect log forensics: {e}")
            return None
    
    async def _collect_memory_dump(self, incident: SecurityIncident) -> Optional[ForensicArtifact]:
        """Collect memory dump for critical incidents"""
        try:
            artifact_id = self._generate_artifact_id()
            forensic_path = self.config.get('forensic_storage_path', '/opt/sutazaiapp/forensics')
            artifact_path = os.path.join(forensic_path, 'memory_dumps', f"{artifact_id}.mem")
            
            os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
            
            # Use available memory dump tools
            dump_commands = [
                ['cat', '/proc/kcore'],  # Linux kernel memory
                ['dd', 'if=/dev/mem', f'of={artifact_path}', 'bs=1M', 'count=100']  # First 100MB of memory
            ]
            
            memory_collected = False
            for cmd in dump_commands:
                try:
                    if cmd[0] == 'cat':
                        # Skip /proc/kcore as it's usually restricted
                        continue
                    elif cmd[0] == 'dd':
                        # Try to collect limited memory dump
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                        if result.returncode == 0 and os.path.exists(artifact_path):
                            memory_collected = True
                            break
                except Exception:
                    continue
            
            # If direct memory access fails, collect process memory maps
            if not memory_collected:
                process_maps = []
                for proc in psutil.process_iter(['pid', 'name']):
                    try:
                        maps_file = f'/proc/{proc.info["pid"]}/maps'
                        if os.path.exists(maps_file):
                            with open(maps_file, 'r') as f:
                                process_maps.append({
                                    'pid': proc.info['pid'],
                                    'name': proc.info['name'],
                                    'maps': f.read()
                                })
                    except Exception:
                        continue
                
                # Save process maps as alternative
                with open(artifact_path, 'w') as f:
                    json.dump(process_maps, f, indent=2)
                memory_collected = True
            
            if memory_collected and os.path.exists(artifact_path):
                file_hash = self._calculate_file_hash(artifact_path)
                file_size = os.path.getsize(artifact_path)
                
                artifact = ForensicArtifact(
                    artifact_id=artifact_id,
                    incident_id=incident.incident_id,
                    artifact_type="memory_dump",
                    source_system=os.uname().nodename,
                    collection_time=datetime.utcnow(),
                    file_path=artifact_path,
                    file_hash=file_hash,
                    file_size=file_size,
                    chain_of_custody=[{
                        "action": "collected",
                        "timestamp": datetime.utcnow().isoformat(),
                        "operator": "automated_system",
                        "hash": file_hash
                    }],
                    analysis_status="pending",
                    metadata={"collection_method": "process_maps" if not memory_collected else "dd"}
                )
                
                self.forensic_artifacts[artifact_id] = artifact
                await self._store_forensic_artifact(artifact)
                
                return artifact
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to collect memory dump: {e}")
            return None
    
    async def _quarantine_file(self, incident: SecurityIncident):
        """Quarantine malicious files"""
        try:
            quarantine_path = os.path.join(
                self.config.get('forensic_storage_path', '/opt/sutazaiapp/forensics'),
                'quarantine'
            )
            os.makedirs(quarantine_path, exist_ok=True)
            
            # Extract file paths from indicators
            file_paths = []
            for indicator in incident.indicators:
                if '/' in indicator and os.path.exists(indicator):
                    file_paths.append(indicator)
            
            for file_path in file_paths:
                self.logger.info(f"Quarantining file: {file_path}")
                
                # Calculate hash before moving
                file_hash = self._calculate_file_hash(file_path)
                
                # Move file to quarantine
                quarantine_file_path = os.path.join(
                    quarantine_path,
                    f"{incident.incident_id}_{os.path.basename(file_path)}_{file_hash[:8]}"
                )
                
                shutil.move(file_path, quarantine_file_path)
                
                # Log quarantine action
                quarantine_data = {
                    "action": "file_quarantined",
                    "original_path": file_path,
                    "quarantine_path": quarantine_file_path,
                    "file_hash": file_hash,
                    "timestamp": datetime.utcnow().isoformat(),
                    "incident_id": incident.incident_id
                }
                
                self.redis_client.lpush("quarantine_actions", json.dumps(quarantine_data))
                
        except Exception as e:
            self.logger.error(f"Failed to quarantine files: {e}")
            raise
    
    async def _reset_passwords(self, incident: SecurityIncident):
        """Reset passwords for affected accounts"""
        try:
            # Generate new temporary passwords
            import secrets
            import string
            
            cursor = self.db_connection.cursor()
            
            # Extract usernames from indicators
            usernames = []
            for indicator in incident.indicators:
                if '@' not in indicator and len(indicator.split()) == 1:
                    usernames.append(indicator)
            
            for username in usernames:
                # Generate secure temporary password
                temp_password = ''.join(secrets.choice(
                    string.ascii_letters + string.digits + "!@#$%^&*"
                ) for _ in range(16))
                
                # Hash password (using bcrypt in production)
                import bcrypt
                password_hash = bcrypt.hashpw(temp_password.encode('utf-8'), bcrypt.gensalt())
                
                # Update password in database
                cursor.execute("""
                    UPDATE users 
                    SET password_hash = %s, password_reset_required = true
                    WHERE username = %s
                """, (password_hash.decode('utf-8'), username))
                
                # Log password reset
                cursor.execute("""
                    INSERT INTO security_audit_log (event_type, details, timestamp)
                    VALUES (%s, %s, %s)
                """, (
                    "password_reset",
                    json.dumps({
                        "username": username,
                        "incident_id": incident.incident_id,
                        "reason": "Incident response - password reset"
                    }),
                    datetime.utcnow()
                ))
                
                self.logger.info(f"Password reset for user: {username}")
            
            self.db_connection.commit()
            cursor.close()
            
        except Exception as e:
            self.logger.error(f"Failed to reset passwords: {e}")
            raise
    
    async def _notify_stakeholders(self, incident: SecurityIncident):
        """Notify stakeholders about the incident"""
        try:
            # Prepare notification data
            notification_data = {
                "incident_id": incident.incident_id,
                "title": incident.title,
                "severity": incident.severity.name,
                "status": incident.status.value,
                "incident_type": incident.incident_type.value,
                "discovered_at": incident.discovered_at.isoformat(),
                "affected_systems": incident.affected_systems,
                "containment_actions": [action.value for action in incident.containment_actions]
            }
            
            # Send email notification
            await self._send_email_notification(incident, notification_data)
            
            # Send webhook notification
            await self._send_webhook_notification(incident, notification_data)
            
            # Store notification in Redis for dashboard
            self.redis_client.lpush("incident_notifications", json.dumps(notification_data))
            
        except Exception as e:
            self.logger.error(f"Failed to notify stakeholders: {e}")
            raise
    
    async def _send_email_notification(self, incident: SecurityIncident, data: Dict[str, Any]):
        """Send email notification"""
        try:
            email_config = self.config.get('email', {})
            if not email_config.get('enabled', False):
                return
            
            # Prepare email content
            subject = f"Security Incident Alert: {incident.severity.name} - {incident.title}"
            
            body = f"""
Security Incident Report

Incident ID: {incident.incident_id}
Title: {incident.title}
Severity: {incident.severity.name}
Type: {incident.incident_type.value}
Status: {incident.status.value}
Discovered: {incident.discovered_at}

Description:
{incident.description}

Affected Systems:
{', '.join(incident.affected_systems) if incident.affected_systems else 'None specified'}

Containment Actions Taken:
{', '.join([action.value for action in incident.containment_actions])}

Indicators of Compromise:
{', '.join(incident.indicators) if incident.indicators else 'None'}

This is an automated notification from the SutazAI Security Incident Response System.
            """
            
            # Send email
            msg = MIMEMultipart()
            msg['From'] = email_config['from']
            msg['To'] = ', '.join(email_config['security_team'])
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(email_config['smtp_host'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['username'], email_config['password'])
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Email notification sent for incident: {incident.incident_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to send email notification: {e}")
    
    async def _send_webhook_notification(self, incident: SecurityIncident, data: Dict[str, Any]):
        """Send webhook notification"""
        try:
            webhook_url = self.config.get('webhook_url')
            if not webhook_url:
                return
            
            response = requests.post(webhook_url, json=data, timeout=10)
            response.raise_for_status()
            
            self.logger.info(f"Webhook notification sent for incident: {incident.incident_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to send webhook notification: {e}")
    
    async def _escalate_incident(self, incident: SecurityIncident):
        """Escalate incident to higher severity or external teams"""
        try:
            # Auto-escalate based on severity
            if incident.severity == IncidentSeverity.CRITICAL:
                incident.severity = IncidentSeverity.CRITICAL  # Already at highest
                # Notify external incident response team
                await self._notify_external_team(incident)
            elif incident.severity == IncidentSeverity.HIGH:
                incident.severity = IncidentSeverity.CRITICAL
            else:
                incident.severity = IncidentSeverity.HIGH
            
            # Log escalation
            incident.timeline.append({
                "timestamp": datetime.utcnow().isoformat(),
                "action": "escalated",
                "description": f"Incident escalated to {incident.severity.name}"
            })
            
            # Re-notify with escalated severity
            await self._notify_stakeholders(incident)
            
        except Exception as e:
            self.logger.error(f"Failed to escalate incident: {e}")
            raise
    
    async def _notify_external_team(self, incident: SecurityIncident):
        """Notify external incident response team"""
        try:
            external_config = self.config.get('external_team', {})
            if not external_config.get('enabled', False):
                return
            
            # Create external ticket or notification
            self.logger.critical(f"Critical incident requires external team: {incident.incident_id}")
            
            # This would integrate with external systems like:
            # - PagerDuty
            # - ServiceNow
            # - External MSSP
            
        except Exception as e:
            self.logger.error(f"Failed to notify external team: {e}")
    
    async def _enhance_monitoring(self, incident: SecurityIncident):
        """Enhance monitoring for incident indicators"""
        try:
            # Add enhanced monitoring rules
            monitoring_rules = []
            
            if incident.source_ip:
                monitoring_rules.append({
                    "type": "ip_monitoring",
                    "value": incident.source_ip,
                    "duration": "24h"
                })
            
            for indicator in incident.indicators:
                monitoring_rules.append({
                    "type": "indicator_monitoring",
                    "value": indicator,
                    "duration": "24h"
                })
            
            # Store monitoring rules
            for rule in monitoring_rules:
                rule["incident_id"] = incident.incident_id
                rule["created_at"] = datetime.utcnow().isoformat()
                self.redis_client.lpush("enhanced_monitoring", json.dumps(rule))
            
            self.logger.info(f"Enhanced monitoring configured for incident: {incident.incident_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to enhance monitoring: {e}")
            raise
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file"""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            self.logger.error(f"Failed to calculate file hash: {e}")
            return ""
    
    async def _store_incident(self, incident: SecurityIncident):
        """Store incident in database"""
        try:
            cursor = self.db_connection.cursor()
            
            cursor.execute("""
                INSERT INTO security_incidents 
                (incident_id, title, description, incident_type, severity, status,
                 source_ip, affected_systems, indicators, discovered_at, reported_by,
                 assigned_to, estimated_impact, containment_actions, timeline,
                 evidence_collected, forensic_artifacts)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (incident_id) DO UPDATE SET
                status = EXCLUDED.status,
                containment_actions = EXCLUDED.containment_actions,
                timeline = EXCLUDED.timeline,
                evidence_collected = EXCLUDED.evidence_collected,
                forensic_artifacts = EXCLUDED.forensic_artifacts,
                updated_at = CURRENT_TIMESTAMP
            """, (
                incident.incident_id, incident.title, incident.description,
                incident.incident_type.value, incident.severity.value, incident.status.value,
                incident.source_ip, json.dumps(incident.affected_systems),
                json.dumps(incident.indicators), incident.discovered_at,
                incident.reported_by, incident.assigned_to, incident.estimated_impact,
                json.dumps([action.value for action in incident.containment_actions]),
                json.dumps(incident.timeline), json.dumps(incident.evidence_collected),
                json.dumps(incident.forensic_artifacts)
            ))
            
            self.db_connection.commit()
            cursor.close()
            
        except Exception as e:
            self.logger.error(f"Failed to store incident: {e}")
    
    async def _store_forensic_artifact(self, artifact: ForensicArtifact):
        """Store forensic artifact in database"""
        try:
            cursor = self.db_connection.cursor()
            
            cursor.execute("""
                INSERT INTO forensic_artifacts 
                (artifact_id, incident_id, artifact_type, source_system, collection_time,
                 file_path, file_hash, file_size, chain_of_custody, analysis_status, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                artifact.artifact_id, artifact.incident_id, artifact.artifact_type,
                artifact.source_system, artifact.collection_time, artifact.file_path,
                artifact.file_hash, artifact.file_size, json.dumps(artifact.chain_of_custody),
                artifact.analysis_status, json.dumps(artifact.metadata)
            ))
            
            self.db_connection.commit()
            cursor.close()
            
        except Exception as e:
            self.logger.error(f"Failed to store forensic artifact: {e}")
    
    def _generate_incident_id(self) -> str:
        """Generate unique incident ID"""
        timestamp = str(int(time.time() * 1000))
        random_part = hashlib.md5(os.urandom(16)).hexdigest()[:8]
        return f"INC_{timestamp}_{random_part}"
    
    def _generate_artifact_id(self) -> str:
        """Generate unique artifact ID"""
        timestamp = str(int(time.time() * 1000))
        random_part = hashlib.md5(os.urandom(16)).hexdigest()[:8]
        return f"ART_{timestamp}_{random_part}"
    
    async def start_incident_monitoring(self):
        """Start incident monitoring and response"""
        self.running = True
        self.logger.info("Started incident response monitoring")
        
        while self.running:
            try:
                # Monitor for new incidents from threat detection
                threat_alerts = self.redis_client.lrange("threat_alerts", 0, -1)
                self.redis_client.delete("threat_alerts")
                
                for alert_json in threat_alerts:
                    try:
                        alert = json.loads(alert_json)
                        
                        # Create incident from threat alert
                        await self.create_incident(
                            title=f"Threat Detected: {alert.get('threat_type', 'Unknown')}",
                            description=alert.get('description', ''),
                            incident_type=self._map_threat_to_incident_type(alert.get('threat_type')),
                            severity=self._map_threat_to_severity(alert.get('threat_level')),
                            source_ip=alert.get('source_ip'),
                            affected_systems=[alert.get('target_ip')] if alert.get('target_ip') else [],
                            indicators=[alert.get('threat_type', '')]
                        )
                        
                    except Exception as e:
                        self.logger.error(f"Failed to process threat alert: {e}")
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Incident monitoring error: {e}")
                await asyncio.sleep(60)
    
    def _map_threat_to_incident_type(self, threat_type: str) -> IncidentType:
        """Map threat type to incident type"""
        mapping = {
            "malware": IncidentType.MALWARE,
            "ddos": IncidentType.DDOS,
            "brute_force": IncidentType.UNAUTHORIZED_ACCESS,
            "data_exfiltration": IncidentType.DATA_EXFILTRATION,
            "apt": IncidentType.APT,
            "ransomware": IncidentType.RANSOMWARE
        }
        return mapping.get(threat_type, IncidentType.SYSTEM_COMPROMISE)
    
    def _map_threat_to_severity(self, threat_level: str) -> IncidentSeverity:
        """Map threat level to incident severity"""
        mapping = {
            "LOW": IncidentSeverity.LOW,
            "MEDIUM": IncidentSeverity.MEDIUM,
            "HIGH": IncidentSeverity.HIGH,
            "CRITICAL": IncidentSeverity.CRITICAL
        }
        return mapping.get(threat_level, IncidentSeverity.MEDIUM)
    
    def stop_incident_monitoring(self):
        """Stop incident monitoring"""
        self.running = False
        self.logger.info("Stopped incident response monitoring")
    
    def get_incident_statistics(self) -> Dict[str, Any]:
        """Get incident response statistics"""
        total_incidents = len(self.active_incidents)
        
        severity_counts = {}
        status_counts = {}
        type_counts = {}
        
        for incident in self.active_incidents.values():
            severity_counts[incident.severity.name] = severity_counts.get(incident.severity.name, 0) + 1
            status_counts[incident.status.value] = status_counts.get(incident.status.value, 0) + 1
            type_counts[incident.incident_type.value] = type_counts.get(incident.incident_type.value, 0) + 1
        
        return {
            "total_active_incidents": total_incidents,
            "severity_breakdown": severity_counts,
            "status_breakdown": status_counts,
            "type_breakdown": type_counts,
            "forensic_artifacts": len(self.forensic_artifacts),
            "last_updated": datetime.utcnow().isoformat()
        }

# Database schema for incident response
INCIDENT_RESPONSE_SCHEMA = """
-- Security incidents table
CREATE TABLE IF NOT EXISTS security_incidents (
    id SERIAL PRIMARY KEY,
    incident_id VARCHAR(255) UNIQUE NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    incident_type VARCHAR(50) NOT NULL,
    severity INTEGER NOT NULL,
    status VARCHAR(50) NOT NULL,
    source_ip INET,
    affected_systems JSONB DEFAULT '[]',
    indicators JSONB DEFAULT '[]',
    discovered_at TIMESTAMP NOT NULL,
    reported_by VARCHAR(255),
    assigned_to VARCHAR(255),
    estimated_impact TEXT,
    containment_actions JSONB DEFAULT '[]',
    timeline JSONB DEFAULT '[]',
    evidence_collected JSONB DEFAULT '[]',
    forensic_artifacts JSONB DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Forensic artifacts table
CREATE TABLE IF NOT EXISTS forensic_artifacts (
    id SERIAL PRIMARY KEY,
    artifact_id VARCHAR(255) UNIQUE NOT NULL,
    incident_id VARCHAR(255) REFERENCES security_incidents(incident_id),
    artifact_type VARCHAR(50) NOT NULL,
    source_system VARCHAR(255) NOT NULL,
    collection_time TIMESTAMP NOT NULL,
    file_path TEXT NOT NULL,
    file_hash VARCHAR(64),
    file_size BIGINT,
    chain_of_custody JSONB DEFAULT '[]',
    analysis_status VARCHAR(50) DEFAULT 'pending',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_security_incidents_status ON security_incidents(status);
CREATE INDEX IF NOT EXISTS idx_security_incidents_severity ON security_incidents(severity);
CREATE INDEX IF NOT EXISTS idx_security_incidents_type ON security_incidents(incident_type);
CREATE INDEX IF NOT EXISTS idx_forensic_artifacts_incident_id ON forensic_artifacts(incident_id);
CREATE INDEX IF NOT EXISTS idx_forensic_artifacts_type ON forensic_artifacts(artifact_type);
"""

if __name__ == "__main__":
    # Example usage
    config = {
        'redis_host': 'redis',
        'redis_port': 6379,
        'postgres_host': 'postgres',
        'postgres_port': 5432,
        'postgres_db': 'sutazai',
        'postgres_user': 'sutazai',
        'forensic_storage_path': '/opt/sutazaiapp/forensics',
        'email': {
            'enabled': True,
            'smtp_host': 'smtp.gmail.com',
            'smtp_port': 587,
            'from': 'security@sutazai.com',
            'security_team': ['security-team@sutazai.com'],
            'username': 'security@sutazai.com',
            'password': 'secure_password'
        },
        'webhook_url': 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL',
        'external_team': {
            'enabled': True,
            'contact': 'external-ir@security-firm.com'
        }
    }
    
    async def main():
        ir_engine = IncidentResponseEngine(config)
        
        # Create a test incident
        incident_id = await ir_engine.create_incident(
            title="Suspicious Network Activity",
            description="Unusual outbound network connections detected",
            incident_type=IncidentType.SYSTEM_COMPROMISE,
            severity=IncidentSeverity.HIGH,
            source_ip="192.168.1.100",
            affected_systems=["web-server-01"],
            indicators=["outbound_connection", "192.168.1.100", "suspicious_process"]
        )
        
        logger.info(f"Created incident: {incident_id}")
        
        # Start monitoring
        # await ir_engine.start_incident_monitoring()
        
        # Get statistics
        stats = ir_engine.get_incident_statistics()
        logger.info(f"Incident Response Statistics: {stats}")
    
    asyncio.run(main())