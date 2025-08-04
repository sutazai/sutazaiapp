#!/usr/bin/env python3
"""
Advanced Threat Detection and Response System
Implements machine learning-based threat detection with automated response
"""

import asyncio
import logging
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import redis
import psycopg2
from psycopg2.extras import RealDictCursor
import hashlib
import threading
from collections import deque, defaultdict
import pickle
import joblib
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import subprocess
import os

class ThreatLevel(Enum):
    INFO = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5

class ThreatType(Enum):
    MALWARE = "malware"
    BOTNET = "botnet"
    APT = "apt"
    DDOS = "ddos"
    BRUTE_FORCE = "brute_force"
    DATA_EXFILTRATION = "data_exfiltration"
    LATERAL_MOVEMENT = "lateral_movement"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    RECONNAISSANCE = "reconnaissance"
    ANOMALY = "anomaly"

class ResponseAction(Enum):
    MONITOR = "monitor"
    ALERT = "alert"
    BLOCK_IP = "block_ip"
    ISOLATE_HOST = "isolate_host"
    QUARANTINE = "quarantine"
    TERMINATE_PROCESS = "terminate_process"
    COLLECT_FORENSICS = "collect_forensics"
    ESCALATE = "escalate"

@dataclass
class ThreatEvent:
    """Threat event detected by the system"""
    event_id: str
    threat_type: ThreatType
    threat_level: ThreatLevel
    confidence: float
    source_ip: str
    target_ip: str
    timestamp: datetime
    description: str
    indicators: Dict[str, Any]
    raw_data: Dict[str, Any]
    response_actions: List[ResponseAction]
    mitre_tactics: List[str] = None
    mitre_techniques: List[str] = None

@dataclass
class ThreatIntelligence:
    """Threat intelligence indicator"""
    indicator: str
    indicator_type: str  # ip, domain, hash, etc.
    threat_types: List[ThreatType]
    confidence: float
    source: str
    first_seen: datetime
    last_seen: datetime
    tags: List[str] = None

class MLThreatDetector:
    """Machine learning-based threat detector"""
    
    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
    def extract_features(self, event_data: Dict[str, Any]) -> np.array:
        """Extract features from event data for ML model"""
        features = []
        
        # Temporal features
        timestamp = event_data.get('timestamp', datetime.utcnow())
        features.extend([
            timestamp.hour,
            timestamp.day,
            timestamp.weekday(),
        ])
        
        # Network features
        features.extend([
            len(event_data.get('source_ip', '')),
            len(event_data.get('user_agent', '')),
            event_data.get('request_size', 0),
            event_data.get('response_size', 0),
            event_data.get('response_time', 0),
        ])
        
        # Behavioral features
        features.extend([
            event_data.get('request_frequency', 0),
            event_data.get('unique_endpoints', 0),
            event_data.get('error_rate', 0),
            event_data.get('failed_logins', 0),
        ])
        
        # Ensure consistent feature vector size
        while len(features) < 20:
            features.append(0)
        
        return np.array(features[:20])  # Fixed size feature vector
    
    def train_anomaly_detector(self, normal_data: List[Dict[str, Any]]):
        """Train anomaly detection model on normal data"""
        if not normal_data:
            return
        
        features = np.array([self.extract_features(data) for data in normal_data])
        features_scaled = self.scaler.fit_transform(features)
        
        self.isolation_forest.fit(features_scaled)
        self.is_trained = True
        
    def detect_anomaly(self, event_data: Dict[str, Any]) -> Tuple[bool, float]:
        """Detect if event is anomalous"""
        if not self.is_trained:
            return False, 0.0
        
        features = self.extract_features(event_data).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        
        anomaly_score = self.isolation_forest.decision_function(features_scaled)[0]
        is_anomaly = self.isolation_forest.predict(features_scaled)[0] == -1
        
        # Convert to confidence score (0-1)
        confidence = abs(anomaly_score) / 2.0  # Normalize roughly to 0-1
        confidence = min(max(confidence, 0), 1)
        
        return is_anomaly, confidence

class ThreatDetectionEngine:
    """Advanced threat detection and response engine"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.redis_client = None
        self.db_connection = None
        self.ml_detector = MLThreatDetector()
        self.threat_intel: Dict[str, ThreatIntelligence] = {}
        self.active_threats: Dict[str, ThreatEvent] = {}
        self.response_handlers = {}
        self.running = False
        self.event_buffer = deque(maxlen=10000)
        self.ip_reputation_cache = {}
        self._initialize_components()

    def _initialize_components(self):
        """Initialize threat detection components"""
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
            
            # Load threat intelligence
            self._load_threat_intelligence()
            
            # Setup response handlers
            self._setup_response_handlers()
            
            # Train ML models
            self._train_ml_models()
            
            self.logger.info("Threat Detection Engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Threat Detection Engine: {e}")
            raise

    def _load_threat_intelligence(self):
        """Load threat intelligence from various sources"""
        try:
            # Load from database
            cursor = self.db_connection.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT * FROM threat_intelligence 
                WHERE expires_at IS NULL OR expires_at > NOW()
            """)
            
            for row in cursor.fetchall():
                intel = ThreatIntelligence(
                    indicator=row['indicator'],
                    indicator_type=row['indicator_type'],
                    threat_types=[ThreatType(t) for t in row['threat_types']],
                    confidence=row['confidence'],
                    source=row['source'],
                    first_seen=row['first_seen'],
                    last_seen=row['last_seen'],
                    tags=row.get('tags', [])
                )
                self.threat_intel[row['indicator']] = intel
            
            cursor.close()
            
            # Load additional threat feeds
            self._update_threat_feeds()
            
            self.logger.info(f"Loaded {len(self.threat_intel)} threat intelligence indicators")
            
        except Exception as e:
            self.logger.error(f"Failed to load threat intelligence: {e}")

    def _update_threat_feeds(self):
        """Update threat intelligence from external feeds"""
        # This would integrate with threat intelligence feeds
        # Examples: AlienVault OTX, VirusTotal, MISP, etc.
        
        # Placeholder implementation
        sample_intel = [
            ThreatIntelligence(
                "198.51.100.1", "ip", [ThreatType.BOTNET], 0.8,
                "local_feed", datetime.utcnow(), datetime.utcnow()
            ),
            ThreatIntelligence(
                "malicious.example.com", "domain", [ThreatType.MALWARE], 0.9,
                "local_feed", datetime.utcnow(), datetime.utcnow()
            ),
        ]
        
        for intel in sample_intel:
            self.threat_intel[intel.indicator] = intel

    def _setup_response_handlers(self):
        """Setup automated response handlers"""
        self.response_handlers = {
            ResponseAction.MONITOR: self._handle_monitor,
            ResponseAction.ALERT: self._handle_alert,
            ResponseAction.BLOCK_IP: self._handle_block_ip,
            ResponseAction.ISOLATE_HOST: self._handle_isolate_host,
            ResponseAction.QUARANTINE: self._handle_quarantine,
            ResponseAction.TERMINATE_PROCESS: self._handle_terminate_process,
            ResponseAction.COLLECT_FORENSICS: self._handle_collect_forensics,
            ResponseAction.ESCALATE: self._handle_escalate,
        }

    def _train_ml_models(self):
        """Train machine learning models"""
        try:
            # Load historical normal data for training
            cursor = self.db_connection.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT * FROM security_events 
                WHERE event_type = 'normal_activity' 
                AND timestamp > NOW() - INTERVAL '30 days'
                LIMIT 10000
            """)
            
            normal_data = [dict(row) for row in cursor.fetchall()]
            cursor.close()
            
            if normal_data:
                self.ml_detector.train_anomaly_detector(normal_data)
                self.logger.info("ML anomaly detector trained successfully")
            else:
                self.logger.warning("No training data available for ML models")
                
        except Exception as e:
            self.logger.error(f"Failed to train ML models: {e}")

    async def start_detection(self):
        """Start threat detection engine"""
        self.running = True
        
        # Start event processing thread
        processing_thread = threading.Thread(target=self._event_processing_loop)
        processing_thread.daemon = True
        processing_thread.start()
        
        # Start threat intelligence updates
        intel_thread = threading.Thread(target=self._threat_intel_update_loop)
        intel_thread.daemon = True
        intel_thread.start()
        
        # Start threat hunting
        hunting_thread = threading.Thread(target=self._threat_hunting_loop)
        hunting_thread.daemon = True
        hunting_thread.start()
        
        self.logger.info("Threat detection engine started")

    def _event_processing_loop(self):
        """Main event processing loop"""
        while self.running:
            try:
                # Get events from Redis queue
                event_data = self.redis_client.blpop(['security_events', 'rasp_events', 'network_events'], timeout=1)
                
                if event_data:
                    queue_name, event_json = event_data
                    event = json.loads(event_json)
                    
                    # Process the event
                    self._process_event(event)
                
            except Exception as e:
                self.logger.error(f"Event processing error: {e}")
                time.sleep(1)

    def _process_event(self, event_data: Dict[str, Any]):
        """Process individual security event"""
        try:
            # Add to event buffer
            self.event_buffer.append(event_data)
            
            # Analyze event for threats
            threats = self._analyze_event(event_data)
            
            # Handle detected threats
            for threat in threats:
                self._handle_threat(threat)
                
        except Exception as e:
            self.logger.error(f"Event analysis error: {e}")

    def _analyze_event(self, event_data: Dict[str, Any]) -> List[ThreatEvent]:
        """Analyze event for potential threats"""
        threats = []
        
        # Rule-based detection
        rule_threats = self._rule_based_detection(event_data)
        threats.extend(rule_threats)
        
        # Threat intelligence matching
        intel_threats = self._threat_intelligence_matching(event_data)
        threats.extend(intel_threats)
        
        # ML-based anomaly detection
        ml_threats = self._ml_based_detection(event_data)
        threats.extend(ml_threats)
        
        # Behavioral analysis
        behavioral_threats = self._behavioral_analysis(event_data)
        threats.extend(behavioral_threats)
        
        return threats

    def _rule_based_detection(self, event_data: Dict[str, Any]) -> List[ThreatEvent]:
        """Rule-based threat detection"""
        threats = []
        
        # Brute force detection
        if self._detect_brute_force(event_data):
            threat = ThreatEvent(
                event_id=self._generate_event_id(),
                threat_type=ThreatType.BRUTE_FORCE,
                threat_level=ThreatLevel.HIGH,
                confidence=0.9,
                source_ip=event_data.get('source_ip', ''),
                target_ip=event_data.get('target_ip', ''),
                timestamp=datetime.utcnow(),
                description="Brute force attack detected",
                indicators={'failed_attempts': event_data.get('failed_logins', 0)},
                raw_data=event_data,
                response_actions=[ResponseAction.BLOCK_IP, ResponseAction.ALERT],
                mitre_tactics=['TA0006'],  # Credential Access
                mitre_techniques=['T1110']  # Brute Force
            )
            threats.append(threat)
        
        # DDoS detection
        if self._detect_ddos(event_data):
            threat = ThreatEvent(
                event_id=self._generate_event_id(),
                threat_type=ThreatType.DDOS,
                threat_level=ThreatLevel.CRITICAL,
                confidence=0.95,
                source_ip=event_data.get('source_ip', ''),
                target_ip=event_data.get('target_ip', ''),
                timestamp=datetime.utcnow(),
                description="DDoS attack detected",
                indicators={'request_rate': event_data.get('request_rate', 0)},
                raw_data=event_data,
                response_actions=[ResponseAction.BLOCK_IP, ResponseAction.ALERT, ResponseAction.ESCALATE],
                mitre_tactics=['TA0040'],  # Impact
                mitre_techniques=['T1499']  # Endpoint Denial of Service
            )
            threats.append(threat)
        
        return threats

    def _threat_intelligence_matching(self, event_data: Dict[str, Any]) -> List[ThreatEvent]:
        """Match event against threat intelligence"""
        threats = []
        
        # Check source IP
        source_ip = event_data.get('source_ip')
        if source_ip and source_ip in self.threat_intel:
            intel = self.threat_intel[source_ip]
            threat = ThreatEvent(
                event_id=self._generate_event_id(),
                threat_type=intel.threat_types[0] if intel.threat_types else ThreatType.ANOMALY,
                threat_level=self._confidence_to_threat_level(intel.confidence),
                confidence=intel.confidence,
                source_ip=source_ip,
                target_ip=event_data.get('target_ip', ''),
                timestamp=datetime.utcnow(),
                description=f"Known threat indicator: {intel.indicator}",
                indicators={'intel_source': intel.source, 'intel_tags': intel.tags},
                raw_data=event_data,
                response_actions=self._determine_response_actions(intel.threat_types[0], intel.confidence)
            )
            threats.append(threat)
        
        # Check domains, hashes, etc.
        # ... additional threat intelligence matching logic
        
        return threats

    def _ml_based_detection(self, event_data: Dict[str, Any]) -> List[ThreatEvent]:
        """ML-based anomaly detection"""
        threats = []
        
        # Check if event is anomalous
        is_anomaly, confidence = self.ml_detector.detect_anomaly(event_data)
        
        if is_anomaly and confidence > 0.7:
            threat = ThreatEvent(
                event_id=self._generate_event_id(),
                threat_type=ThreatType.ANOMALY,
                threat_level=self._confidence_to_threat_level(confidence),
                confidence=confidence,
                source_ip=event_data.get('source_ip', ''),
                target_ip=event_data.get('target_ip', ''),
                timestamp=datetime.utcnow(),
                description="ML-detected anomalous behavior",
                indicators={'anomaly_score': confidence},
                raw_data=event_data,
                response_actions=[ResponseAction.MONITOR, ResponseAction.ALERT]
            )
            threats.append(threat)
        
        return threats

    def _behavioral_analysis(self, event_data: Dict[str, Any]) -> List[ThreatEvent]:
        """Behavioral analysis for advanced threats"""
        threats = []
        
        # Data exfiltration detection
        if self._detect_data_exfiltration(event_data):
            threat = ThreatEvent(
                event_id=self._generate_event_id(),
                threat_type=ThreatType.DATA_EXFILTRATION,
                threat_level=ThreatLevel.CRITICAL,
                confidence=0.8,
                source_ip=event_data.get('source_ip', ''),
                target_ip=event_data.get('target_ip', ''),
                timestamp=datetime.utcnow(),
                description="Potential data exfiltration detected",
                indicators={'data_volume': event_data.get('data_transferred', 0)},
                raw_data=event_data,
                response_actions=[ResponseAction.ALERT, ResponseAction.COLLECT_FORENSICS, ResponseAction.ESCALATE],
                mitre_tactics=['TA0010'],  # Exfiltration
                mitre_techniques=['T1041']  # Exfiltration Over C2 Channel
            )
            threats.append(threat)
        
        return threats

    def _detect_brute_force(self, event_data: Dict[str, Any]) -> bool:
        """Detect brute force attacks"""
        failed_logins = event_data.get('failed_logins', 0)
        return failed_logins > self.config.get('brute_force_threshold', 10)

    def _detect_ddos(self, event_data: Dict[str, Any]) -> bool:
        """Detect DDoS attacks"""
        request_rate = event_data.get('request_rate', 0)
        return request_rate > self.config.get('ddos_threshold', 1000)

    def _detect_data_exfiltration(self, event_data: Dict[str, Any]) -> bool:
        """Detect data exfiltration"""
        data_transferred = event_data.get('data_transferred', 0)
        return data_transferred > self.config.get('exfiltration_threshold', 100 * 1024 * 1024)  # 100MB

    def _confidence_to_threat_level(self, confidence: float) -> ThreatLevel:
        """Convert confidence score to threat level"""
        if confidence >= 0.9:
            return ThreatLevel.CRITICAL
        elif confidence >= 0.7:
            return ThreatLevel.HIGH
        elif confidence >= 0.5:
            return ThreatLevel.MEDIUM
        elif confidence >= 0.3:
            return ThreatLevel.LOW
        else:
            return ThreatLevel.INFO

    def _determine_response_actions(self, threat_type: ThreatType, confidence: float) -> List[ResponseAction]:
        """Determine appropriate response actions"""
        actions = [ResponseAction.MONITOR, ResponseAction.ALERT]
        
        if confidence >= 0.8:
            if threat_type in [ThreatType.MALWARE, ThreatType.BOTNET]:
                actions.extend([ResponseAction.BLOCK_IP, ResponseAction.ISOLATE_HOST])
            elif threat_type == ThreatType.DDOS:
                actions.extend([ResponseAction.BLOCK_IP, ResponseAction.ESCALATE])
            elif threat_type == ThreatType.DATA_EXFILTRATION:
                actions.extend([ResponseAction.COLLECT_FORENSICS, ResponseAction.ESCALATE])
        
        return actions

    def _handle_threat(self, threat: ThreatEvent):
        """Handle detected threat"""
        # Store threat
        self.active_threats[threat.event_id] = threat
        
        # Log threat
        self.logger.warning(
            f"Threat detected: {threat.threat_type.value} "
            f"(Level: {threat.threat_level.name}, Confidence: {threat.confidence})"
        )
        
        # Store in database
        self._store_threat_event(threat)
        
        # Execute response actions
        for action in threat.response_actions:
            if action in self.response_handlers:
                try:
                    self.response_handlers[action](threat)
                except Exception as e:
                    self.logger.error(f"Response action {action.value} failed: {e}")

    def _store_threat_event(self, threat: ThreatEvent):
        """Store threat event in database"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("""
                INSERT INTO threat_events 
                (event_id, threat_type, threat_level, confidence, source_ip, target_ip, 
                 timestamp, description, indicators, raw_data, response_actions)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                threat.event_id,
                threat.threat_type.value,
                threat.threat_level.value,
                threat.confidence,
                threat.source_ip,
                threat.target_ip,
                threat.timestamp,
                threat.description,
                json.dumps(threat.indicators),
                json.dumps(threat.raw_data),
                json.dumps([action.value for action in threat.response_actions])
            ))
            self.db_connection.commit()
            cursor.close()
            
        except Exception as e:
            self.logger.error(f"Failed to store threat event: {e}")

    # Response handlers
    def _handle_monitor(self, threat: ThreatEvent):
        """Handle monitor response"""
        self.logger.info(f"Monitoring threat: {threat.event_id}")

    def _handle_alert(self, threat: ThreatEvent):
        """Handle alert response"""
        alert_data = {
            "threat_id": threat.event_id,
            "type": threat.threat_type.value,
            "level": threat.threat_level.name,
            "confidence": threat.confidence,
            "source_ip": threat.source_ip,
            "description": threat.description,
            "timestamp": threat.timestamp.isoformat()
        }
        
        # Send to alert queue
        self.redis_client.lpush("threat_alerts", json.dumps(alert_data))
        
        # Send email/webhook notifications
        self._send_notification(threat)

    def _handle_block_ip(self, threat: ThreatEvent):
        """Handle IP blocking response"""
        try:
            # Add to blocked IPs in Redis
            block_data = {
                "reason": f"Threat: {threat.threat_type.value}",
                "confidence": threat.confidence,
                "blocked_at": threat.timestamp.isoformat(),
                "threat_id": threat.event_id
            }
            
            self.redis_client.setex(
                f"blocked_ip:{threat.source_ip}",
                3600,  # 1 hour
                json.dumps(block_data)
            )
            
            # Add iptables rule (if running on host)
            subprocess.run([
                'iptables', '-I', 'INPUT', '1',
                '-s', threat.source_ip, '-j', 'DROP'
            ], check=False)
            
            self.logger.info(f"Blocked IP: {threat.source_ip}")
            
        except Exception as e:
            self.logger.error(f"Failed to block IP {threat.source_ip}: {e}")

    def _handle_isolate_host(self, threat: ThreatEvent):
        """Handle host isolation response"""
        self.logger.warning(f"Host isolation triggered for threat: {threat.event_id}")
        # Implement host isolation logic

    def _handle_quarantine(self, threat: ThreatEvent):
        """Handle quarantine response"""
        self.logger.warning(f"Quarantine triggered for threat: {threat.event_id}")
        # Implement quarantine logic

    def _handle_terminate_process(self, threat: ThreatEvent):
        """Handle process termination response"""
        self.logger.warning(f"Process termination triggered for threat: {threat.event_id}")
        # Implement process termination logic

    def _handle_collect_forensics(self, threat: ThreatEvent):
        """Handle forensics collection response"""
        self.logger.info(f"Collecting forensics for threat: {threat.event_id}")
        
        # Collect relevant data
        forensics_data = {
            "threat_id": threat.event_id,
            "timestamp": threat.timestamp.isoformat(),
            "source_ip": threat.source_ip,
            "raw_event": threat.raw_data,
            "system_info": self._collect_system_info(),
            "network_connections": self._collect_network_connections(),
            "process_list": self._collect_process_list()
        }
        
        # Store forensics data
        self.redis_client.set(
            f"forensics:{threat.event_id}",
            json.dumps(forensics_data),
            ex=86400  # 24 hours
        )

    def _handle_escalate(self, threat: ThreatEvent):
        """Handle escalation response"""
        self.logger.critical(f"Escalating threat: {threat.event_id}")
        
        # Send high-priority notification
        self._send_escalation_notification(threat)

    def _send_notification(self, threat: ThreatEvent):
        """Send threat notification"""
        try:
            # Webhook notification
            webhook_url = self.config.get('webhook_url')
            if webhook_url:
                payload = {
                    "threat_id": threat.event_id,
                    "type": threat.threat_type.value,
                    "level": threat.threat_level.name,
                    "confidence": threat.confidence,
                    "source_ip": threat.source_ip,
                    "description": threat.description,
                    "timestamp": threat.timestamp.isoformat()
                }
                requests.post(webhook_url, json=payload, timeout=5)
            
            # Email notification
            self._send_email_notification(threat)
            
        except Exception as e:
            self.logger.error(f"Failed to send notification: {e}")

    def _send_email_notification(self, threat: ThreatEvent):
        """Send email notification"""
        try:
            email_config = self.config.get('email', {})
            if not email_config.get('enabled', False):
                return
            
            msg = MIMEMultipart()
            msg['From'] = email_config['from']
            msg['To'] = ', '.join(email_config['to'])
            msg['Subject'] = f"Security Threat Detected - {threat.threat_level.name}"
            
            body = f"""
Security Threat Detected

Threat ID: {threat.event_id}
Type: {threat.threat_type.value}
Level: {threat.threat_level.name}
Confidence: {threat.confidence}
Source IP: {threat.source_ip}
Description: {threat.description}
Timestamp: {threat.timestamp}

Raw Data:
{json.dumps(threat.raw_data, indent=2)}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(email_config['smtp_host'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['username'], email_config['password'])
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            self.logger.error(f"Failed to send email notification: {e}")

    def _send_escalation_notification(self, threat: ThreatEvent):
        """Send escalation notification"""
        # Send to incident response team
        self._send_notification(threat)
        
        # Create incident ticket (integrate with ticketing system)
        # ... incident creation logic

    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect system information for forensics"""
        try:
            return {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": dict(psutil.disk_usage('/')._asdict()),
                "boot_time": psutil.boot_time(),
                "users": [user._asdict() for user in psutil.users()]
            }
        except Exception:
            return {}

    def _collect_network_connections(self) -> List[Dict[str, Any]]:
        """Collect network connections for forensics"""
        try:
            connections = []
            for conn in psutil.net_connections():
                connections.append({
                    "fd": conn.fd,
                    "family": conn.family.name if conn.family else None,
                    "type": conn.type.name if conn.type else None,
                    "laddr": conn.laddr._asdict() if conn.laddr else None,
                    "raddr": conn.raddr._asdict() if conn.raddr else None,
                    "status": conn.status,
                    "pid": conn.pid
                })
            return connections
        except Exception:
            return []

    def _collect_process_list(self) -> List[Dict[str, Any]]:
        """Collect process list for forensics"""
        try:
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
                processes.append(proc.info)
            return processes
        except Exception:
            return []

    def _threat_intel_update_loop(self):
        """Update threat intelligence periodically"""
        while self.running:
            try:
                self._update_threat_feeds()
                time.sleep(self.config.get('threat_intel_update_interval', 3600))
            except Exception as e:
                self.logger.error(f"Threat intelligence update error: {e}")
                time.sleep(3600)

    def _threat_hunting_loop(self):
        """Proactive threat hunting"""
        while self.running:
            try:
                self._hunt_for_threats()
                time.sleep(self.config.get('threat_hunting_interval', 1800))  # 30 minutes
            except Exception as e:
                self.logger.error(f"Threat hunting error: {e}")
                time.sleep(1800)

    def _hunt_for_threats(self):
        """Hunt for threats proactively"""
        # Analyze patterns in event buffer
        if len(self.event_buffer) < 100:
            return
        
        # Look for suspicious patterns
        ip_counts = defaultdict(int)
        for event in self.event_buffer:
            source_ip = event.get('source_ip')
            if source_ip:
                ip_counts[source_ip] += 1
        
        # Check for IPs with unusual activity
        for ip, count in ip_counts.items():
            if count > self.config.get('hunting_threshold', 50):
                # Create hunting-based threat
                threat = ThreatEvent(
                    event_id=self._generate_event_id(),
                    threat_type=ThreatType.RECONNAISSANCE,
                    threat_level=ThreatLevel.MEDIUM,
                    confidence=0.6,
                    source_ip=ip,
                    target_ip="",
                    timestamp=datetime.utcnow(),
                    description=f"Suspicious activity detected during threat hunting: {count} events",
                    indicators={"event_count": count},
                    raw_data={"hunting_detection": True},
                    response_actions=[ResponseAction.MONITOR, ResponseAction.ALERT]
                )
                
                self._handle_threat(threat)

    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        timestamp = str(int(time.time() * 1000))
        random_part = hashlib.md5(os.urandom(16)).hexdigest()[:8]
        return f"threat_{timestamp}_{random_part}"

    def stop_detection(self):
        """Stop threat detection engine"""
        self.running = False
        self.logger.info("Threat detection engine stopped")

    def get_active_threats(self) -> List[ThreatEvent]:
        """Get list of active threats"""
        return list(self.active_threats.values())

    def get_threat_statistics(self) -> Dict[str, Any]:
        """Get threat detection statistics"""
        threat_counts = defaultdict(int)
        level_counts = defaultdict(int)
        
        for threat in self.active_threats.values():
            threat_counts[threat.threat_type.value] += 1
            level_counts[threat.threat_level.name] += 1
        
        return {
            "total_active_threats": len(self.active_threats),
            "threat_type_counts": dict(threat_counts),
            "threat_level_counts": dict(level_counts),
            "ml_model_trained": self.ml_detector.is_trained,
            "threat_intel_indicators": len(self.threat_intel)
        }

# Database schema initialization
THREAT_DETECTION_SCHEMA = """
-- Threat events table
CREATE TABLE IF NOT EXISTS threat_events (
    id SERIAL PRIMARY KEY,
    event_id VARCHAR(255) UNIQUE NOT NULL,
    threat_type VARCHAR(100) NOT NULL,
    threat_level VARCHAR(50) NOT NULL,
    confidence FLOAT NOT NULL,
    source_ip INET,
    target_ip INET,
    timestamp TIMESTAMP NOT NULL,
    description TEXT,
    indicators JSONB,
    raw_data JSONB,
    response_actions JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Threat intelligence table
CREATE TABLE IF NOT EXISTS threat_intelligence (
    id SERIAL PRIMARY KEY,
    indicator VARCHAR(255) NOT NULL,
    indicator_type VARCHAR(50) NOT NULL,
    threat_types JSONB NOT NULL,
    confidence FLOAT NOT NULL,
    source VARCHAR(255) NOT NULL,
    first_seen TIMESTAMP NOT NULL,
    last_seen TIMESTAMP NOT NULL,
    tags JSONB,
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_threat_events_timestamp ON threat_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_threat_events_source_ip ON threat_events(source_ip);
CREATE INDEX IF NOT EXISTS idx_threat_intel_indicator ON threat_intelligence(indicator);
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
        'brute_force_threshold': 10,
        'ddos_threshold': 1000,
        'exfiltration_threshold': 100 * 1024 * 1024,
        'threat_intel_update_interval': 3600,
        'threat_hunting_interval': 1800,
        'hunting_threshold': 50,
        'webhook_url': None,
        'email': {
            'enabled': False,
            'smtp_host': 'smtp.gmail.com',
            'smtp_port': 587,
            'from': 'security@sutazai.com',
            'to': ['admin@sutazai.com'],
            'username': '',
            'password': ''
        }
    }
    
    threat_detector = ThreatDetectionEngine(config)
    
    # Start detection
    asyncio.run(threat_detector.start_detection())
    
    print("Threat Detection Engine initialized and running...")
    print(f"Statistics: {threat_detector.get_threat_statistics()}")