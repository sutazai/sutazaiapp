#!/usr/bin/env python3
"""
Alert and Notification System for SutazAI Human Oversight
Provides real-time alerts, notifications, and escalation mechanisms
"""

import asyncio
import json
import logging
import sqlite3
import smtplib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
from app.schemas.message_types import AlertSeverity
import uuid
import aiohttp
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import asyncio_mqtt
import websockets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class AlertCategory(Enum):
    SYSTEM_PERFORMANCE = "system_performance"
    SECURITY_THREAT = "security_threat"
    COMPLIANCE_VIOLATION = "compliance_violation"
    AGENT_MALFUNCTION = "agent_malfunction"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    HUMAN_INTERVENTION_REQUIRED = "human_intervention_required"
    DATA_INTEGRITY = "data_integrity"
    OPERATIONAL_ANOMALY = "operational_anomaly"

class NotificationChannel(Enum):
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    WEBHOOKS = "webhooks"
    DASHBOARD = "dashboard"
    MOBILE_PUSH = "mobile_push"
    TEAMS = "teams"
    PAGERDUTY = "pagerduty"

class EscalationLevel(Enum):
    LEVEL_1 = "level_1"  # First responders
    LEVEL_2 = "level_2"  # Team leads
    LEVEL_3 = "level_3"  # Management
    LEVEL_4 = "level_4"  # Executive

@dataclass
class Alert:
    """Represents a system alert"""
    id: str
    title: str
    description: str
    severity: AlertSeverity
    category: AlertCategory
    agent_id: Optional[str]
    created_at: datetime
    updated_at: datetime
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved: bool = False
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    escalation_level: EscalationLevel = EscalationLevel.LEVEL_1
    escalated_at: Optional[datetime] = None

@dataclass
class NotificationRule:
    """Defines how alerts should be routed and escalated"""
    id: str
    name: str
    conditions: Dict[str, Any]  # Alert matching conditions
    channels: List[NotificationChannel]
    recipients: List[str]
    escalation_rules: List[Dict[str, Any]]
    enabled: bool = True
    created_at: datetime = None

@dataclass
class NotificationTemplate:
    """Template for notification messages"""
    id: str
    channel: NotificationChannel
    severity: AlertSeverity
    subject_template: str
    body_template: str
    metadata: Dict[str, Any] = None

class AlertNotificationSystem:
    """
    Comprehensive alert and notification system for SutazAI oversight
    """
    
    def __init__(self, db_path: str = "/opt/sutazaiapp/backend/oversight/oversight.db",
                 config_path: str = "/opt/sutazaiapp/backend/oversight/alert_config.json"):
        self.db_path = Path(db_path)
        self.config_path = Path(config_path)
        
        # Active alerts and notification rules
        self.active_alerts: Dict[str, Alert] = {}
        self.notification_rules: Dict[str, NotificationRule] = {}
        self.notification_templates: Dict[str, NotificationTemplate] = {}
        
        # Notification handlers
        self.notification_handlers: Dict[NotificationChannel, Callable] = {
            NotificationChannel.EMAIL: self._send_email_notification,
            NotificationChannel.SMS: self._send_sms_notification,
            NotificationChannel.SLACK: self._send_slack_notification,
            NotificationChannel.WEBHOOKS: self._send_webhook_notification,
            NotificationChannel.DASHBOARD: self._send_dashboard_notification,
            NotificationChannel.TEAMS: self._send_teams_notification,
            NotificationChannel.PAGERDUTY: self._send_pagerduty_notification
        }
        
        # Configuration
        self.config = self._load_config()
        
        # Initialize database
        self._init_database()
        
        # Load default rules and templates
        self._load_default_rules()
        self._load_default_templates()
        
        # Background tasks
        self.background_tasks = set()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load notification system configuration"""
        default_config = {
            "email": {
                "smtp_server": "localhost",
                "smtp_port": 587,
                "use_tls": True,
                "username": "",
                "password": "",
                "from_address": "sutazai-alerts@localhost"
            },
            "slack": {
                "webhook_url": "",
                "bot_token": "",
                "default_channel": "#alerts"
            },
            "sms": {
                "provider": "twilio",
                "account_sid": "",
                "auth_token": "",
                "from_number": ""
            },
            "pagerduty": {
                "integration_key": "",
                "api_token": ""
            },
            "teams": {
                "webhook_url": ""
            },
            "escalation": {
                "level_1_timeout": 300,  # 5 minutes
                "level_2_timeout": 900,  # 15 minutes
                "level_3_timeout": 1800, # 30 minutes
                "level_4_timeout": 3600  # 1 hour
            }
        }
        
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    user_config = json.load(f)
                    # Merge with defaults
                    for key, value in user_config.items():
                        if key in default_config and isinstance(value, dict):
                            default_config[key].update(value)
                        else:
                            default_config[key] = value
        except Exception as e:
            logger.warning(f"Could not load config from {self.config_path}: {e}")
        
        return default_config
    
    def _init_database(self):
        """Initialize alert notification database"""
        with sqlite3.connect(self.db_path) as conn:
            # Alerts table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    category TEXT NOT NULL,
                    agent_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    acknowledged INTEGER DEFAULT 0,
                    acknowledged_by TEXT,
                    acknowledged_at TEXT,
                    resolved INTEGER DEFAULT 0,
                    resolved_by TEXT,
                    resolved_at TEXT,
                    metadata TEXT,
                    escalation_level TEXT DEFAULT 'level_1',
                    escalated_at TEXT
                )
            ''')
            
            # Notification rules table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS notification_rules (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    conditions TEXT NOT NULL,
                    channels TEXT NOT NULL,
                    recipients TEXT NOT NULL,
                    escalation_rules TEXT,
                    enabled INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL
                )
            ''')
            
            # Notification history table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS notification_history (
                    id TEXT PRIMARY KEY,
                    alert_id TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    recipient TEXT NOT NULL,
                    sent_at TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    error_message TEXT,
                    metadata TEXT,
                    FOREIGN KEY (alert_id) REFERENCES alerts (id)
                )
            ''')
            
            # Escalation history table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS escalation_history (
                    id TEXT PRIMARY KEY,
                    alert_id TEXT NOT NULL,
                    from_level TEXT NOT NULL,
                    to_level TEXT NOT NULL,
                    escalated_at TEXT NOT NULL,
                    escalated_by TEXT,
                    reason TEXT,
                    FOREIGN KEY (alert_id) REFERENCES alerts (id)
                )
            ''')
            
            conn.commit()
    
    def _load_default_rules(self):
        """Load default notification rules"""
        default_rules = [
            NotificationRule(
                id="critical_system_alerts",
                name="Critical System Alerts",
                conditions={
                    "severity": ["critical"],
                    "category": ["system_performance", "security_threat", "agent_malfunction"]
                },
                channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK, NotificationChannel.PAGERDUTY],
                recipients=["admin@sutazai.com", "ops-team@sutazai.com"],
                escalation_rules=[
                    {"level": "level_2", "timeout": 300},
                    {"level": "level_3", "timeout": 900},
                    {"level": "level_4", "timeout": 1800}
                ],
                created_at=datetime.utcnow()
            ),
            NotificationRule(
                id="compliance_violations",
                name="Compliance Violations",
                conditions={
                    "category": ["compliance_violation"],
                    "severity": ["high", "critical"]
                },
                channels=[NotificationChannel.EMAIL, NotificationChannel.TEAMS],
                recipients=["compliance@sutazai.com", "legal@sutazai.com"],
                escalation_rules=[
                    {"level": "level_3", "timeout": 600}
                ],
                created_at=datetime.utcnow()
            ),
            NotificationRule(
                id="human_intervention_required",
                name="Human Intervention Required",
                conditions={
                    "category": ["human_intervention_required"]
                },
                channels=[NotificationChannel.EMAIL, NotificationChannel.DASHBOARD, NotificationChannel.MOBILE_PUSH],
                recipients=["operators@sutazai.com"],
                escalation_rules=[
                    {"level": "level_2", "timeout": 600}
                ],
                created_at=datetime.utcnow()
            ),
            NotificationRule(
                id="agent_malfunctions",
                name="Agent Malfunctions",
                conditions={
                    "category": ["agent_malfunction"],
                    "severity": ["medium", "high", "critical"]
                },
                channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK],
                recipients=["ai-team@sutazai.com", "ops-team@sutazai.com"],
                escalation_rules=[
                    {"level": "level_2", "timeout": 900}
                ],
                created_at=datetime.utcnow()
            )
        ]
        
        for rule in default_rules:
            self.notification_rules[rule.id] = rule
    
    def _load_default_templates(self):
        """Load default notification templates"""
        default_templates = [
            # Email templates
            NotificationTemplate(
                id="email_critical",
                channel=NotificationChannel.EMAIL,
                severity=AlertSeverity.CRITICAL,
                subject_template="🚨 CRITICAL ALERT: {title}",
                body_template="""
                <h2 style="color: #f44336;">🚨 CRITICAL ALERT</h2>
                <p><strong>Title:</strong> {title}</p>
                <p><strong>Description:</strong> {description}</p>
                <p><strong>Agent:</strong> {agent_id}</p>
                <p><strong>Category:</strong> {category}</p>
                <p><strong>Time:</strong> {created_at}</p>
                <p><strong>Alert ID:</strong> {id}</p>
                
                <p style="color: #f44336;"><strong>IMMEDIATE ACTION REQUIRED</strong></p>
                
                <p>Please acknowledge this alert immediately and take appropriate action.</p>
                <p>Dashboard: <a href="http://localhost:8095">SutazAI Oversight Interface</a></p>
                """
            ),
            NotificationTemplate(
                id="email_high",
                channel=NotificationChannel.EMAIL,
                severity=AlertSeverity.HIGH,
                subject_template="⚠️ HIGH PRIORITY ALERT: {title}",
                body_template="""
                <h2 style="color: #ff9800;">⚠️ HIGH PRIORITY ALERT</h2>
                <p><strong>Title:</strong> {title}</p>
                <p><strong>Description:</strong> {description}</p>
                <p><strong>Agent:</strong> {agent_id}</p>
                <p><strong>Category:</strong> {category}</p>
                <p><strong>Time:</strong> {created_at}</p>
                <p><strong>Alert ID:</strong> {id}</p>
                
                <p>Please review and address this alert promptly.</p>
                <p>Dashboard: <a href="http://localhost:8095">SutazAI Oversight Interface</a></p>
                """
            ),
            # Slack templates
            NotificationTemplate(
                id="slack_critical",
                channel=NotificationChannel.SLACK,
                severity=AlertSeverity.CRITICAL,
                subject_template="🚨 CRITICAL ALERT",
                body_template="""
                🚨 *CRITICAL ALERT* 🚨
                
                *Title:* {title}
                *Description:* {description}
                *Agent:* {agent_id}
                *Category:* {category}
                *Time:* {created_at}
                *Alert ID:* {id}
                
                🔴 *IMMEDIATE ACTION REQUIRED* 🔴
                
                <http://localhost:8095|Open Dashboard>
                """
            ),
            # Teams templates
            NotificationTemplate(
                id="teams_compliance",
                channel=NotificationChannel.TEAMS,
                severity=AlertSeverity.HIGH,
                subject_template="Compliance Violation Detected",
                body_template="""
                {
                    "@type": "MessageCard",
                    "@context": "http://schema.org/extensions",
                    "themeColor": "ff9800",
                    "summary": "Compliance Violation: {title}",
                    "sections": [{
                        "activityTitle": "⚠️ Compliance Violation Detected",
                        "activitySubtitle": "{title}",
                        "facts": [{
                            "name": "Description",
                            "value": "{description}"
                        }, {
                            "name": "Agent",
                            "value": "{agent_id}"
                        }, {
                            "name": "Severity",
                            "value": "{severity}"
                        }, {
                            "name": "Time",
                            "value": "{created_at}"
                        }],
                        "markdown": true
                    }],
                    "potentialAction": [{
                        "@type": "OpenUri",
                        "name": "Open Dashboard",
                        "targets": [{
                            "os": "default",
                            "uri": "http://localhost:8095"
                        }]
                    }]
                }
                """
            )
        ]
        
        for template in default_templates:
            self.notification_templates[f"{template.channel.value}_{template.severity.value}"] = template
    
    async def create_alert(self, title: str, description: str, severity: AlertSeverity,
                          category: AlertCategory, agent_id: Optional[str] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> Alert:
        """Create a new alert"""
        alert = Alert(
            id=str(uuid.uuid4()),
            title=title,
            description=description,
            severity=severity,
            category=category,
            agent_id=agent_id,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        # Store in memory and database
        self.active_alerts[alert.id] = alert
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO alerts 
                    (id, title, description, severity, category, agent_id, 
                     created_at, updated_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (alert.id, alert.title, alert.description, alert.severity.value,
                      alert.category.value, alert.agent_id, alert.created_at.isoformat(),
                      alert.updated_at.isoformat(), json.dumps(alert.metadata)))
                
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing alert in database: {e}")
        
        # Process notifications
        await self._process_alert_notifications(alert)
        
        # Start escalation monitoring
        if alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
            task = asyncio.create_task(self._monitor_escalation(alert))
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)
        
        logger.info(f"Created {alert.severity.value} alert: {alert.title} [{alert.id}]")
        
        return alert
    
    async def _process_alert_notifications(self, alert: Alert):
        """Process notifications for a new alert"""
        matching_rules = self._find_matching_rules(alert)
        
        for rule in matching_rules:
            if not rule.enabled:
                continue
            
            for channel in rule.channels:
                for recipient in rule.recipients:
                    try:
                        await self._send_notification(alert, channel, recipient)
                    except Exception as e:
                        logger.error(f"Error sending notification via {channel.value} to {recipient}: {e}")
    
    def _find_matching_rules(self, alert: Alert) -> List[NotificationRule]:
        """Find notification rules that match the alert"""
        matching_rules = []
        
        for rule in self.notification_rules.values():
            if self._alert_matches_conditions(alert, rule.conditions):
                matching_rules.append(rule)
        
        return matching_rules
    
    def _alert_matches_conditions(self, alert: Alert, conditions: Dict[str, Any]) -> bool:
        """Check if alert matches rule conditions"""
        for condition_key, condition_values in conditions.items():
            if condition_key == "severity":
                if alert.severity.value not in condition_values:
                    return False
            elif condition_key == "category":
                if alert.category.value not in condition_values:
                    return False
            elif condition_key == "agent_id":
                if alert.agent_id not in condition_values:
                    return False
            # Add more condition types as needed
        
        return True
    
    async def _send_notification(self, alert: Alert, channel: NotificationChannel, recipient: str):
        """Send notification via specified channel"""
        try:
            handler = self.notification_handlers.get(channel)
            if handler:
                success = await handler(alert, recipient)
                
                # Log notification attempt
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('''
                        INSERT INTO notification_history 
                        (id, alert_id, channel, recipient, sent_at, success, error_message)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (str(uuid.uuid4()), alert.id, channel.value, recipient,
                          datetime.utcnow().isoformat(), 1 if success else 0,
                          None if success else "Handler returned False"))
                    
                    conn.commit()
            else:
                logger.warning(f"No handler for notification channel: {channel.value}")
                
        except Exception as e:
            logger.error(f"Error sending {channel.value} notification to {recipient}: {e}")
            
            # Log failed notification
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO notification_history 
                    (id, alert_id, channel, recipient, sent_at, success, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (str(uuid.uuid4()), alert.id, channel.value, recipient,
                      datetime.utcnow().isoformat(), 0, str(e)))
                
                conn.commit()
    
    async def _send_email_notification(self, alert: Alert, recipient: str) -> bool:
        """Send email notification"""
        try:
            template = self.notification_templates.get(f"email_{alert.severity.value}")
            if not template:
                template = self.notification_templates.get("email_high")
            
            if not template:
                logger.error("No email template found")
                return False
            
            # Format message
            subject = template.subject_template.format(**asdict(alert))
            body = template.body_template.format(**asdict(alert))
            
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.config['email']['from_address']
            msg['To'] = recipient
            
            # Add body
            html_part = MIMEText(body, 'html')
            msg.attach(html_part)
            
            # Send email
            smtp_config = self.config['email']
            if smtp_config['username'] and smtp_config['password']:
                with smtplib.SMTP(smtp_config['smtp_server'], smtp_config['smtp_port']) as server:
                    if smtp_config['use_tls']:
                        server.starttls()
                    server.login(smtp_config['username'], smtp_config['password'])
                    server.send_message(msg)
                
                logger.info(f"Email notification sent to {recipient} for alert {alert.id}")
                return True
            else:
                logger.warning("Email credentials not configured")
                return False
                
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
            return False
    
    async def _send_sms_notification(self, alert: Alert, recipient: str) -> bool:
        """Send SMS notification"""
        try:
            # SMS integration would go here (Twilio, AWS SNS, etc.)
            # For now, log the attempt
            message = f"SutazAI Alert: {alert.title} - {alert.severity.value.upper()} [{alert.id[:8]}]"
            logger.info(f"SMS notification (simulated) to {recipient}: {message}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending SMS notification: {e}")
            return False
    
    async def _send_slack_notification(self, alert: Alert, recipient: str) -> bool:
        """Send Slack notification"""
        try:
            webhook_url = self.config['slack']['webhook_url']
            if not webhook_url:
                logger.warning("Slack webhook URL not configured")
                return False
            
            template = self.notification_templates.get(f"slack_{alert.severity.value}")
            if not template:
                template = self.notification_templates.get("slack_critical")
            
            if not template:
                logger.error("No Slack template found")
                return False
            
            # Format message
            message = template.body_template.format(**asdict(alert))
            
            payload = {
                "text": message,
                "channel": recipient if recipient.startswith('#') else self.config['slack']['default_channel'],
                "username": "SutazAI Alerts",
                "icon_emoji": ":warning:"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    if response.status == 200:
                        logger.info(f"Slack notification sent for alert {alert.id}")
                        return True
                    else:
                        logger.error(f"Slack API error: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
            return False
    
    async def _send_webhook_notification(self, alert: Alert, recipient: str) -> bool:
        """Send webhook notification"""
        try:
            payload = {
                "alert": asdict(alert),
                "timestamp": datetime.utcnow().isoformat(),
                "source": "sutazai-oversight"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(recipient, json=payload) as response:
                    if response.status < 400:
                        logger.info(f"Webhook notification sent to {recipient} for alert {alert.id}")
                        return True
                    else:
                        logger.error(f"Webhook error: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error sending webhook notification: {e}")
            return False
    
    async def _send_dashboard_notification(self, alert: Alert, recipient: str) -> bool:
        """Send dashboard notification (WebSocket broadcast)"""
        try:
            # This would integrate with the oversight interface WebSocket
            logger.info(f"Dashboard notification queued for alert {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending dashboard notification: {e}")
            return False
    
    async def _send_teams_notification(self, alert: Alert, recipient: str) -> bool:
        """Send Microsoft Teams notification"""
        try:
            webhook_url = self.config['teams']['webhook_url'] or recipient
            if not webhook_url:
                logger.warning("Teams webhook URL not configured")
                return False
            
            template = self.notification_templates.get(f"teams_{alert.category.value}")
            if not template:
                # Use default Teams format
                payload = {
                    "@type": "MessageCard",
                    "@context": "http://schema.org/extensions",
                    "themeColor": "ff9800" if alert.severity == AlertSeverity.HIGH else "f44336",
                    "summary": f"Alert: {alert.title}",
                    "sections": [{
                        "activityTitle": f"🚨 {alert.severity.value.upper()} ALERT",
                        "activitySubtitle": alert.title,
                        "facts": [
                            {"name": "Description", "value": alert.description},
                            {"name": "Agent", "value": alert.agent_id or "System"},
                            {"name": "Category", "value": alert.category.value},
                            {"name": "Time", "value": alert.created_at.strftime("%Y-%m-%d %H:%M:%S")}
                        ],
                        "markdown": True
                    }],
                    "potentialAction": [{
                        "@type": "OpenUri", 
                        "name": "Open Dashboard",
                        "targets": [{"os": "default", "uri": "http://localhost:8095"}]
                    }]
                }
            else:
                # Use template
                payload_str = template.body_template.format(**asdict(alert))
                payload = json.loads(payload_str)
            
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    if response.status == 200:
                        logger.info(f"Teams notification sent for alert {alert.id}")
                        return True
                    else:
                        logger.error(f"Teams API error: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error sending Teams notification: {e}")
            return False
    
    async def _send_pagerduty_notification(self, alert: Alert, recipient: str) -> bool:
        """Send PagerDuty notification"""
        try:
            integration_key = self.config['pagerduty']['integration_key']
            if not integration_key:
                logger.warning("PagerDuty integration key not configured")
                return False
            
            payload = {
                "routing_key": integration_key,
                "event_action": "trigger",
                "dedup_key": alert.id,
                "payload": {
                    "summary": alert.title,
                    "source": "SutazAI Oversight",
                    "severity": "critical" if alert.severity == AlertSeverity.CRITICAL else "error",
                    "component": alert.agent_id or "system",
                    "group": alert.category.value,
                    "class": "ai_system_alert",
                    "custom_details": {
                        "description": alert.description,
                        "alert_id": alert.id,
                        "created_at": alert.created_at.isoformat(),
                        "metadata": alert.metadata
                    }
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post("https://events.pagerduty.com/v2/enqueue", json=payload) as response:
                    if response.status == 202:
                        logger.info(f"PagerDuty notification sent for alert {alert.id}")
                        return True
                    else:
                        logger.error(f"PagerDuty API error: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error sending PagerDuty notification: {e}")
            return False
    
    async def _monitor_escalation(self, alert: Alert):
        """Monitor alert for escalation"""
        escalation_timeouts = self.config['escalation']
        current_level = alert.escalation_level
        
        while not alert.acknowledged and not alert.resolved:
            if current_level == EscalationLevel.LEVEL_1:
                timeout = escalation_timeouts['level_1_timeout']
                next_level = EscalationLevel.LEVEL_2
            elif current_level == EscalationLevel.LEVEL_2:
                timeout = escalation_timeouts['level_2_timeout']
                next_level = EscalationLevel.LEVEL_3
            elif current_level == EscalationLevel.LEVEL_3:
                timeout = escalation_timeouts['level_3_timeout']
                next_level = EscalationLevel.LEVEL_4
            else:
                # Maximum level reached
                break
            
            # Wait for timeout
            await asyncio.sleep(timeout)
            
            # Check if alert is still active
            current_alert = self.active_alerts.get(alert.id)
            if not current_alert or current_alert.acknowledged or current_alert.resolved:
                break
            
            # Escalate
            await self._escalate_alert(alert, next_level)
            current_level = next_level
    
    async def _escalate_alert(self, alert: Alert, to_level: EscalationLevel):
        """Escalate alert to next level"""
        try:
            from_level = alert.escalation_level
            alert.escalation_level = to_level
            alert.escalated_at = datetime.utcnow()
            
            # Update database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    UPDATE alerts 
                    SET escalation_level = ?, escalated_at = ?, updated_at = ?
                    WHERE id = ?
                ''', (to_level.value, alert.escalated_at.isoformat(), 
                          datetime.utcnow().isoformat(), alert.id))
                
                # Log escalation
                conn.execute('''
                    INSERT INTO escalation_history 
                    (id, alert_id, from_level, to_level, escalated_at, reason)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (str(uuid.uuid4()), alert.id, from_level.value, to_level.value,
                      alert.escalated_at.isoformat(), "Automatic escalation due to timeout"))
                
                conn.commit()
            
            # Create escalation alert
            escalation_alert = await self.create_alert(
                title=f"Alert Escalated: {alert.title}",
                description=f"Alert {alert.id} has been escalated to {to_level.value} due to no acknowledgment",
                severity=AlertSeverity.HIGH,
                category=AlertCategory.HUMAN_INTERVENTION_REQUIRED,
                metadata={
                    "original_alert_id": alert.id,
                    "escalated_from": from_level.value,
                    "escalated_to": to_level.value
                }
            )
            
            logger.warning(f"Alert {alert.id} escalated from {from_level.value} to {to_level.value}")
            
        except Exception as e:
            logger.error(f"Error escalating alert {alert.id}: {e}")
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert"""
        try:
            alert = self.active_alerts.get(alert_id)
            if not alert:
                logger.warning(f"Alert {alert_id} not found")
                return False
            
            alert.acknowledged = True
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = datetime.utcnow()
            alert.updated_at = datetime.utcnow()
            
            # Update database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    UPDATE alerts 
                    SET acknowledged = 1, acknowledged_by = ?, acknowledged_at = ?, updated_at = ?
                    WHERE id = ?
                ''', (acknowledged_by, alert.acknowledged_at.isoformat(),
                      alert.updated_at.isoformat(), alert_id))
                
                conn.commit()
            
            logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
            return True
            
        except Exception as e:
            logger.error(f"Error acknowledging alert {alert_id}: {e}")
            return False
    
    async def resolve_alert(self, alert_id: str, resolved_by: str, resolution_note: str = "") -> bool:
        """Resolve an alert"""
        try:
            alert = self.active_alerts.get(alert_id)
            if not alert:
                logger.warning(f"Alert {alert_id} not found")
                return False
            
            alert.resolved = True
            alert.resolved_by = resolved_by
            alert.resolved_at = datetime.utcnow()
            alert.updated_at = datetime.utcnow()
            
            # Update database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    UPDATE alerts 
                    SET resolved = 1, resolved_by = ?, resolved_at = ?, updated_at = ?
                    WHERE id = ?
                ''', (resolved_by, alert.resolved_at.isoformat(),
                      alert.updated_at.isoformat(), alert_id))
                
                conn.commit()
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            logger.info(f"Alert {alert_id} resolved by {resolved_by}")
            return True
            
        except Exception as e:
            logger.error(f"Error resolving alert {alert_id}: {e}")
            return False
    
    async def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())
    
    async def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history"""
        try:
            since = datetime.utcnow() - timedelta(hours=hours)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT id, title, description, severity, category, agent_id,
                           created_at, updated_at, acknowledged, acknowledged_by, acknowledged_at,
                           resolved, resolved_by, resolved_at, metadata, escalation_level, escalated_at
                    FROM alerts 
                    WHERE created_at >= ?
                    ORDER BY created_at DESC
                ''', (since.isoformat(),))
                
                alerts = []
                for row in cursor.fetchall():
                    alerts.append(Alert(
                        id=row[0],
                        title=row[1],
                        description=row[2],
                        severity=AlertSeverity(row[3]),
                        category=AlertCategory(row[4]),
                        agent_id=row[5],
                        created_at=datetime.fromisoformat(row[6]),
                        updated_at=datetime.fromisoformat(row[7]),
                        acknowledged=bool(row[8]),
                        acknowledged_by=row[9],
                        acknowledged_at=datetime.fromisoformat(row[10]) if row[10] else None,
                        resolved=bool(row[11]),
                        resolved_by=row[12],
                        resolved_at=datetime.fromisoformat(row[13]) if row[13] else None,
                        metadata=json.loads(row[14]) if row[14] else {},
                        escalation_level=EscalationLevel(row[15]),
                        escalated_at=datetime.fromisoformat(row[16]) if row[16] else None
                    ))
                
                return alerts
                
        except Exception as e:
            logger.error(f"Error getting alert history: {e}")
            return []
    
    async def start_monitoring(self):
        """Start the alert monitoring system"""
        logger.info("Starting alert notification system")
        
        # Start background monitoring tasks
        tasks = [
            asyncio.create_task(self._monitor_system_health()),
            asyncio.create_task(self._monitor_agent_status()),
            asyncio.create_task(self._monitor_compliance_violations()),
            asyncio.create_task(self._cleanup_old_alerts())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Alert monitoring tasks cancelled")
        except Exception as e:
            logger.error(f"Error in alert monitoring: {e}")
    
    async def _monitor_system_health(self):
        """Monitor overall system health"""
        while True:
            try:
                # This would integrate with system monitoring
                # For now, create sample health checks
                
                # Check memory usage (simulated)
                memory_usage = 85.0  # Simulated percentage
                if memory_usage > 90:
                    await self.create_alert(
                        title="High Memory Usage Detected",
                        description=f"System memory usage at {memory_usage}%",
                        severity=AlertSeverity.HIGH,
                        category=AlertCategory.SYSTEM_PERFORMANCE,
                        metadata={"memory_usage": memory_usage}
                    )
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in system health monitoring: {e}")
                await asyncio.sleep(300)
    
    async def _monitor_agent_status(self):
        """Monitor agent status for anomalies"""
        while True:
            try:
                # Load agent status
                agent_status_path = Path("/opt/sutazaiapp/agents/agent_status.json")
                if agent_status_path.exists():
                    with open(agent_status_path, 'r') as f:
                        status_data = json.load(f)
                        active_agents = status_data.get('active_agents', {})
                        
                        # Check for agent failures
                        for agent_id, agent_info in active_agents.items():
                            if agent_info.get('status') == 'unhealthy':
                                await self.create_alert(
                                    title=f"Agent Health Issue: {agent_id}",
                                    description=f"Agent {agent_id} reporting unhealthy status",
                                    severity=AlertSeverity.MEDIUM,
                                    category=AlertCategory.AGENT_MALFUNCTION,
                                    agent_id=agent_id,
                                    metadata=agent_info
                                )
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in agent status monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _monitor_compliance_violations(self):
        """Monitor for compliance violations"""
        while True:
            try:
                # This would integrate with compliance monitoring
                # For now, simulate compliance checks
                
                await asyncio.sleep(1800)  # Check every 30 minutes
                
            except Exception as e:
                logger.error(f"Error in compliance monitoring: {e}")
                await asyncio.sleep(1800)
    
    async def _cleanup_old_alerts(self):
        """Clean up old resolved alerts"""
        while True:
            try:
                # Remove alerts older than 30 days
                cutoff = datetime.utcnow() - timedelta(days=30)
                
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('''
                        DELETE FROM alerts 
                        WHERE resolved = 1 AND resolved_at < ?
                    ''', (cutoff.isoformat(),))
                    
                    deleted_count = conn.total_changes
                    if deleted_count > 0:
                        logger.info(f"Cleaned up {deleted_count} old alerts")
                    
                    conn.commit()
                
                await asyncio.sleep(86400)  # Run daily
                
            except Exception as e:
                logger.error(f"Error in alert cleanup: {e}")
                await asyncio.sleep(86400)


async def main():
    """Main entry point for standalone alert system"""
    alert_system = AlertNotificationSystem()
    
    # Create some sample alerts for testing
    await alert_system.create_alert(
        title="System Memory Usage High",
        description="System memory usage has exceeded 90%",
        severity=AlertSeverity.HIGH,
        category=AlertCategory.SYSTEM_PERFORMANCE,
        metadata={"memory_usage": 92.5}
    )
    
    await alert_system.create_alert(
        title="Agent Communication Failure",
        description="Agent 'senior-frontend-developer' is not responding",
        severity=AlertSeverity.CRITICAL,
        category=AlertCategory.AGENT_MALFUNCTION,
        agent_id="senior-frontend-developer"
    )
    
    # Start monitoring
    await alert_system.start_monitoring()


if __name__ == "__main__":
    asyncio.run(main())
