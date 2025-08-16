#!/usr/bin/env python3
"""
MCP Automation Alert Manager
Intelligent alerting system with correlation, suppression, and routing
"""

import asyncio
import json
import logging
import os
import smtplib
import time
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import httpx
import yaml
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertState(Enum):
    """Alert states"""
    PENDING = "pending"
    FIRING = "firing"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    ACKNOWLEDGED = "acknowledged"


class NotificationChannel(Enum):
    """Notification channels"""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    PAGERDUTY = "pagerduty"
    LOG = "log"
    PROMETHEUS = "prometheus"


@dataclass
class Alert:
    """Individual alert"""
    id: str
    name: str
    severity: AlertSeverity
    state: AlertState
    component: str
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    fingerprint: str  # For deduplication
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    source: str = "mcp_automation"
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    notification_sent: bool = False
    notification_channels: List[str] = field(default_factory=list)
    correlation_id: Optional[str] = None
    parent_alert_id: Optional[str] = None


@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    condition: str  # Expression to evaluate
    severity: AlertSeverity
    message_template: str
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    notification_channels: List[NotificationChannel] = field(default_factory=list)
    cooldown_minutes: int = 5
    auto_resolve_minutes: Optional[int] = None
    group_by: List[str] = field(default_factory=list)
    threshold: Optional[float] = None
    enabled: bool = True


@dataclass
class AlertGroup:
    """Group of correlated alerts"""
    id: str
    name: str
    alerts: List[Alert] = field(default_factory=list)
    severity: AlertSeverity = AlertSeverity.INFO
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class SuppressionRule:
    """Alert suppression rule"""
    name: str
    conditions: Dict[str, Any]  # Conditions for suppression
    duration_minutes: int
    reason: str
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(hours=1))
    enabled: bool = True


class AlertManager:
    """Intelligent alert management system"""
    
    def __init__(self,
                 config_path: Optional[str] = None,
                 alert_history_size: int = 1000,
                 correlation_window_minutes: int = 5):
        """
        Initialize alert manager
        
        Args:
            config_path: Path to alert configuration
            alert_history_size: Maximum number of alerts to keep in history
            correlation_window_minutes: Time window for alert correlation
        """
        self.config_path = Path(config_path) if config_path else None
        self.alert_history_size = alert_history_size
        self.correlation_window = timedelta(minutes=correlation_window_minutes)
        
        # Alert storage
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_groups: Dict[str, AlertGroup] = {}
        
        # Rules and suppression
        self.alert_rules: Dict[str, AlertRule] = {}
        self.suppression_rules: Dict[str, SuppressionRule] = {}
        
        # Notification tracking
        self.notification_queue: List[Alert] = []
        self.notification_history: Dict[str, datetime] = {}
        
        # Load configuration
        self.config = self._load_config()
        self._load_alert_rules()
        
        # HTTP client for webhooks
        self.http_client = httpx.AsyncClient(timeout=10.0)
        
        # Statistics
        self.stats = defaultdict(int)
        
    def _load_config(self) -> Dict[str, Any]:
        """Load alert configuration"""
        default_config = {
            'notification_channels': {
                'email': {
                    'enabled': False,
                    'smtp_host': 'localhost',
                    'smtp_port': 587,
                    'from_address': 'alerts@sutazaiapp.com',
                    'to_addresses': ['admin@sutazaiapp.com']
                },
                'slack': {
                    'enabled': False,
                    'webhook_url': os.getenv('SLACK_WEBHOOK_URL', '')
                },
                'webhook': {
                    'enabled': True,
                    'url': 'http://localhost:10010/api/v1/alerts'
                },
                'prometheus': {
                    'enabled': True,
                    'pushgateway_url': 'http://localhost:10200/metrics'
                }
            },
            'alert_defaults': {
                'cooldown_minutes': 5,
                'auto_resolve_minutes': 30,
                'max_notifications_per_hour': 10
            },
            'correlation': {
                'enabled': True,
                'window_minutes': 5,
                'min_alerts_for_group': 2
            },
            'suppression': {
                'enabled': True,
                'maintenance_window': {
                    'enabled': False,
                    'start_hour': 2,
                    'end_hour': 4
                }
            }
        }
        
        if self.config_path and self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    default_config.update(loaded_config)
            except Exception as e:
                logger.error(f"Failed to load config from {self.config_path}: {e}")
                
        return default_config
        
    def _load_alert_rules(self):
        """Load predefined alert rules"""
        default_rules = [
            AlertRule(
                name="mcp_server_down",
                condition="mcp_server_up == 0",
                severity=AlertSeverity.CRITICAL,
                message_template="MCP Server {server_name} is down",
                labels={"category": "availability"},
                notification_channels=[NotificationChannel.WEBHOOK, NotificationChannel.LOG]
            ),
            AlertRule(
                name="high_error_rate",
                condition="error_rate > 0.1",
                severity=AlertSeverity.WARNING,
                message_template="High error rate detected: {error_rate:.2%}",
                labels={"category": "reliability"},
                threshold=0.1
            ),
            AlertRule(
                name="sla_violation",
                condition="sla_compliance < 0.95",
                severity=AlertSeverity.ERROR,
                message_template="SLA compliance below threshold: {sla_compliance:.2%}",
                labels={"category": "compliance"},
                threshold=0.95
            ),
            AlertRule(
                name="high_cpu_usage",
                condition="cpu_percent > 80",
                severity=AlertSeverity.WARNING,
                message_template="High CPU usage: {cpu_percent:.1f}%",
                labels={"category": "resources"},
                threshold=80,
                cooldown_minutes=10
            ),
            AlertRule(
                name="high_memory_usage",
                condition="memory_percent > 85",
                severity=AlertSeverity.WARNING,
                message_template="High memory usage: {memory_percent:.1f}%",
                labels={"category": "resources"},
                threshold=85,
                cooldown_minutes=10
            ),
            AlertRule(
                name="disk_space_critical",
                condition="disk_percent > 90",
                severity=AlertSeverity.CRITICAL,
                message_template="Critical disk space on {mount_point}: {disk_percent:.1f}%",
                labels={"category": "resources"},
                threshold=90
            ),
            AlertRule(
                name="automation_failure",
                condition="automation_failed == true",
                severity=AlertSeverity.ERROR,
                message_template="Automation workflow {workflow_name} failed",
                labels={"category": "automation"}
            ),
            AlertRule(
                name="security_violation",
                condition="security_event == true",
                severity=AlertSeverity.EMERGENCY,
                message_template="Security violation detected: {violation_type}",
                labels={"category": "security"},
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.PAGERDUTY]
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.name] = rule
            
    def generate_fingerprint(self, alert_data: Dict[str, Any]) -> str:
        """Generate unique fingerprint for alert deduplication"""
        key_fields = ['name', 'component', 'severity']
        fingerprint_data = {k: alert_data.get(k, '') for k in key_fields}
        return str(hash(json.dumps(fingerprint_data, sort_keys=True)))
        
    async def create_alert(self,
                          name: str,
                          severity: AlertSeverity,
                          component: str,
                          message: str,
                          details: Optional[Dict[str, Any]] = None) -> Alert:
        """
        Create a new alert
        
        Args:
            name: Alert name
            severity: Alert severity
            component: Component that triggered the alert
            message: Alert message
            details: Additional alert details
            
        Returns:
            Created Alert object
        """
        alert_data = {
            'name': name,
            'severity': severity,
            'component': component,
            'message': message,
            'details': details or {}
        }
        
        fingerprint = self.generate_fingerlogger.info(alert_data)
        
        # Check for duplicate alerts
        if fingerprint in self.active_alerts:
            existing_alert = self.active_alerts[fingerprint]
            existing_alert.timestamp = datetime.now()
            logger.debug(f"Alert already exists: {name} for {component}")
            return existing_alert
            
        # Create new alert
        alert_id = f"{name}_{component}_{int(time.time())}"
        alert = Alert(
            id=alert_id,
            name=name,
            severity=severity,
            state=AlertState.PENDING,
            component=component,
            message=message,
            details=details or {},
            timestamp=datetime.now(),
            fingerprint=fingerprint
        )
        
        # Check suppression rules
        if self._is_suppressed(alert):
            alert.state = AlertState.SUPPRESSED
            logger.info(f"Alert suppressed: {alert.name} for {alert.component}")
        else:
            alert.state = AlertState.FIRING
            
        # Store alert
        self.active_alerts[fingerprint] = alert
        self.alert_history.append(alert)
        
        # Maintain history size
        if len(self.alert_history) > self.alert_history_size:
            self.alert_history.pop(0)
            
        # Update statistics
        self.stats[f'alerts_created_{severity.value}'] += 1
        
        # Correlate with other alerts
        await self._correlate_alert(alert)
        
        # Queue for notification if firing
        if alert.state == AlertState.FIRING:
            self.notification_queue.append(alert)
            
        logger.info(f"Alert created: {alert.id} - {alert.message}")
        
        return alert
        
    def _is_suppressed(self, alert: Alert) -> bool:
        """Check if alert should be suppressed"""
        # Check maintenance window
        if self.config['suppression']['maintenance_window']['enabled']:
            current_hour = datetime.now().hour
            start_hour = self.config['suppression']['maintenance_window']['start_hour']
            end_hour = self.config['suppression']['maintenance_window']['end_hour']
            
            if start_hour <= current_hour < end_hour:
                return True
                
        # Check suppression rules
        for rule in self.suppression_rules.values():
            if not rule.enabled or rule.expires_at < datetime.now():
                continue
                
            # Check conditions
            matches = True
            for key, value in rule.conditions.items():
                alert_value = getattr(alert, key, None) or alert.details.get(key)
                if alert_value != value:
                    matches = False
                    break
                    
            if matches:
                logger.debug(f"Alert matches suppression rule: {rule.name}")
                return True
                
        return False
        
    async def _correlate_alert(self, alert: Alert):
        """Correlate alert with existing alerts"""
        if not self.config['correlation']['enabled']:
            return
            
        correlation_window = datetime.now() - self.correlation_window
        related_alerts = []
        
        for existing_alert in self.active_alerts.values():
            if existing_alert.id == alert.id:
                continue
                
            # Check time window
            if existing_alert.timestamp < correlation_window:
                continue
                
            # Check correlation criteria
            if (existing_alert.component == alert.component or
                existing_alert.labels.get('category') == alert.labels.get('category')):
                related_alerts.append(existing_alert)
                
        # Create or update alert group
        if len(related_alerts) >= self.config['correlation']['min_alerts_for_group'] - 1:
            group_id = f"group_{alert.component}_{int(time.time())}"
            
            # Check if group already exists
            for gid, group in self.alert_groups.items():
                if any(a.id in [ra.id for ra in related_alerts] for a in group.alerts):
                    group_id = gid
                    break
                    
            if group_id not in self.alert_groups:
                self.alert_groups[group_id] = AlertGroup(
                    id=group_id,
                    name=f"Alert group for {alert.component}",
                    severity=max([alert.severity] + [a.severity for a in related_alerts],
                               key=lambda s: list(AlertSeverity).index(s))
                )
                
            group = self.alert_groups[group_id]
            group.alerts.append(alert)
            group.updated_at = datetime.now()
            alert.correlation_id = group_id
            
            logger.info(f"Alert correlated to group {group_id}: {alert.id}")
            
    async def resolve_alert(self, alert_id: str):
        """
        Resolve an alert
        
        Args:
            alert_id: ID of the alert to resolve
        """
        resolved_alert = None
        
        for fingerprint, alert in self.active_alerts.items():
            if alert.id == alert_id:
                alert.state = AlertState.RESOLVED
                alert.resolved_at = datetime.now()
                resolved_alert = alert
                del self.active_alerts[fingerprint]
                break
                
        if resolved_alert:
            logger.info(f"Alert resolved: {alert_id}")
            self.stats['alerts_resolved'] += 1
            
            # Send resolution notification
            await self._send_notification(resolved_alert, is_resolution=True)
        else:
            logger.warning(f"Alert not found for resolution: {alert_id}")
            
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """
        Acknowledge an alert
        
        Args:
            alert_id: ID of the alert to acknowledge
            acknowledged_by: User who acknowledged the alert
        """
        for alert in self.active_alerts.values():
            if alert.id == alert_id:
                alert.state = AlertState.ACKNOWLEDGED
                alert.acknowledged_at = datetime.now()
                alert.acknowledged_by = acknowledged_by
                logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
                self.stats['alerts_acknowledged'] += 1
                return
                
        logger.warning(f"Alert not found for acknowledgment: {alert_id}")
        
    async def process_notifications(self):
        """Process queued notifications"""
        while self.notification_queue:
            alert = self.notification_queue.pop(0)
            
            # Check cooldown
            last_notification = self.notification_history.get(alert.fingerprint)
            if last_notification:
                cooldown = timedelta(minutes=self.config['alert_defaults']['cooldown_minutes'])
                if datetime.now() - last_notification < cooldown:
                    logger.debug(f"Alert in cooldown: {alert.id}")
                    continue
                    
            # Send notifications
            await self._send_notification(alert)
            
            # Update history
            self.notification_history[alert.fingerprint] = datetime.now()
            alert.notification_sent = True
            
    async def _send_notification(self, alert: Alert, is_resolution: bool = False):
        """Send alert notification through configured channels"""
        notification_channels = alert.notification_channels or ['log']
        
        for channel in notification_channels:
            try:
                if channel == 'log' or channel == NotificationChannel.LOG.value:
                    self._send_log_notification(alert, is_resolution)
                    
                elif channel == 'webhook' or channel == NotificationChannel.WEBHOOK.value:
                    if self.config['notification_channels']['webhook']['enabled']:
                        await self._send_webhook_notification(alert, is_resolution)
                        
                elif channel == 'slack' or channel == NotificationChannel.SLACK.value:
                    if self.config['notification_channels']['slack']['enabled']:
                        await self._send_slack_notification(alert, is_resolution)
                        
                elif channel == 'email' or channel == NotificationChannel.EMAIL.value:
                    if self.config['notification_channels']['email']['enabled']:
                        await self._send_email_notification(alert, is_resolution)
                        
                elif channel == 'prometheus' or channel == NotificationChannel.PROMETHEUS.value:
                    if self.config['notification_channels']['prometheus']['enabled']:
                        await self._send_prometheus_notification(alert, is_resolution)
                        
            except Exception as e:
                logger.error(f"Failed to send notification via {channel}: {e}")
                
    def _send_log_notification(self, alert: Alert, is_resolution: bool):
        """Send notification to logs"""
        prefix = "RESOLVED" if is_resolution else "ALERT"
        log_level = logging.INFO if is_resolution else {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL,
            AlertSeverity.EMERGENCY: logging.CRITICAL
        }.get(alert.severity, logging.WARNING)
        
        logger.log(log_level, f"{prefix}: [{alert.severity.value.upper()}] "
                            f"{alert.component} - {alert.message}")
                            
    async def _send_webhook_notification(self, alert: Alert, is_resolution: bool):
        """Send notification via webhook"""
        webhook_url = self.config['notification_channels']['webhook']['url']
        
        payload = {
            'event_type': 'alert_resolved' if is_resolution else 'alert_firing',
            'alert': asdict(alert),
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            response = await self.http_client.post(webhook_url, json=payload)
            if response.status_code == 200:
                logger.debug(f"Webhook notification sent for alert {alert.id}")
            else:
                logger.warning(f"Webhook returned status {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
            
    async def _send_slack_notification(self, alert: Alert, is_resolution: bool):
        """Send notification to Slack"""
        webhook_url = self.config['notification_channels']['slack']['webhook_url']
        
        if not webhook_url:
            logger.warning("Slack webhook URL not configured")
            return
            
        color = {
            AlertSeverity.INFO: "#36a64f",
            AlertSeverity.WARNING: "#ff9900",
            AlertSeverity.ERROR: "#ff0000",
            AlertSeverity.CRITICAL: "#ff0000",
            AlertSeverity.EMERGENCY: "#ff0000"
        }.get(alert.severity, "#808080")
        
        payload = {
            'attachments': [{
                'color': color,
                'title': f"{'✅ RESOLVED' if is_resolution else '🚨 ALERT'}: {alert.name}",
                'text': alert.message,
                'fields': [
                    {'title': 'Component', 'value': alert.component, 'short': True},
                    {'title': 'Severity', 'value': alert.severity.value, 'short': True},
                    {'title': 'Time', 'value': alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'), 'short': True}
                ],
                'footer': 'MCP Alert Manager',
                'ts': int(alert.timestamp.timestamp())
            }]
        }
        
        try:
            response = await self.http_client.post(webhook_url, json=payload)
            if response.status_code == 200:
                logger.debug(f"Slack notification sent for alert {alert.id}")
            else:
                logger.warning(f"Slack webhook returned status {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            
    async def _send_email_notification(self, alert: Alert, is_resolution: bool):
        """Send notification via email"""
        email_config = self.config['notification_channels']['email']
        
        try:
            msg = MIMEMultipart()
            msg['From'] = email_config['from_address']
            msg['To'] = ', '.join(email_config['to_addresses'])
            msg['Subject'] = f"{'[RESOLVED]' if is_resolution else '[ALERT]'} {alert.severity.value.upper()}: {alert.name}"
            
            body = f"""
Alert Details:
--------------
Name: {alert.name}
Component: {alert.component}
Severity: {alert.severity.value}
Status: {'Resolved' if is_resolution else 'Firing'}
Time: {alert.timestamp}
Message: {alert.message}

Details:
{json.dumps(alert.details, indent=2)}

--
MCP Alert Manager
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(email_config['smtp_host'], email_config['smtp_port']) as server:
                if email_config.get('smtp_user'):
                    server.starttls()
                    server.login(email_config['smtp_user'], email_config.get('smtp_password', ''))
                server.send_message(msg)
                
            logger.debug(f"Email notification sent for alert {alert.id}")
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            
    async def _send_prometheus_notification(self, alert: Alert, is_resolution: bool):
        """Send alert metrics to Prometheus"""
        # This would integrate with Prometheus AlertManager
        # For now, just log the intent
        logger.debug(f"Would send Prometheus notification for alert {alert.id}")
        
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())
        
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics"""
        active_by_severity = defaultdict(int)
        for alert in self.active_alerts.values():
            active_by_severity[alert.severity.value] += 1
            
        return {
            'active_alerts': len(self.active_alerts),
            'active_by_severity': dict(active_by_severity),
            'total_created': sum(v for k, v in self.stats.items() if k.startswith('alerts_created_')),
            'total_resolved': self.stats.get('alerts_resolved', 0),
            'total_acknowledged': self.stats.get('alerts_acknowledged', 0),
            'alert_groups': len(self.alert_groups),
            'suppressed_alerts': sum(1 for a in self.active_alerts.values() if a.state == AlertState.SUPPRESSED)
        }
        
    async def cleanup(self):
        """Cleanup resources"""
        await self.http_client.aclose()


async def main():
    """Main function for testing"""
    manager = AlertManager()
    
    # Create test alerts
    await manager.create_alert(
        name="test_alert",
        severity=AlertSeverity.WARNING,
        component="test_component",
        message="This is a test alert",
        details={'test_key': 'test_value'}
    )
    
    await manager.create_alert(
        name="critical_alert",
        severity=AlertSeverity.CRITICAL,
        component="database",
        message="Database connection failed",
        details={'error': 'Connection timeout'}
    )
    
    # Process notifications
    await manager.process_notifications()
    
    # Get statistics
    stats = manager.get_alert_statistics()
    logger.info(json.dumps(stats, indent=2))
    
    # Cleanup
    await manager.cleanup()


if __name__ == "__main__":
    asyncio.run(main())