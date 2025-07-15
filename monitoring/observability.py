"""
SutazAI Monitoring and Observability System
Enterprise-grade monitoring, metrics collection, and alerting

This module provides comprehensive monitoring capabilities including
metrics collection, alerting, distributed tracing, and system health monitoring.
"""

import asyncio
import json
import logging
import time
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import psutil
import queue
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import sqlite3
from collections import defaultdict, deque
import numpy as np
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import OpenTelemetry
from opentelemetry import trace, metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class MetricType(Enum):
    """Metric types"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

@dataclass
class Metric:
    """Metric data structure"""
    name: str
    type: MetricType
    value: Union[int, float]
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    description: str = ""

@dataclass
class Alert:
    """Alert data structure"""
    id: str
    title: str
    description: str
    level: AlertLevel
    source: str
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HealthCheck:
    """Health check result"""
    component: str
    status: str  # "healthy", "unhealthy", "warning"
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    response_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class MetricsCollector:
    """Collects and stores system metrics"""
    
    def __init__(self, db_path: str = "/opt/sutazaiapp/data/metrics.db"):
        self.db_path = db_path
        self.metrics_queue = queue.Queue()
        self.running = False
        self.collection_thread = None
        
        # Prometheus metrics
        self.prometheus_metrics = {
            "system_cpu_usage": Gauge("system_cpu_usage_percent", "System CPU usage"),
            "system_memory_usage": Gauge("system_memory_usage_percent", "System memory usage"),
            "system_disk_usage": Gauge("system_disk_usage_percent", "System disk usage"),
            "agi_tasks_total": Counter("agi_tasks_total", "Total AGI tasks", ["status"]),
            "agi_task_duration": Histogram("agi_task_duration_seconds", "AGI task duration"),
            "neural_network_activity": Gauge("neural_network_activity", "Neural network activity level"),
            "api_requests_total": Counter("api_requests_total", "Total API requests", ["method", "endpoint"]),
            "api_request_duration": Histogram("api_request_duration_seconds", "API request duration")
        }
        
        # Initialize database
        self._init_database()
        
        logger.info("Metrics Collector initialized")
    
    def _init_database(self):
        """Initialize metrics database"""
        try:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    value REAL NOT NULL,
                    labels TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    description TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_metrics_name_timestamp 
                ON metrics(name, timestamp)
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("Metrics database initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize metrics database: {e}")
            raise
    
    def start_collection(self):
        """Start metrics collection"""
        if self.running:
            return
        
        self.running = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        
        logger.info("Metrics collection started")
    
    def stop_collection(self):
        """Stop metrics collection"""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        
        logger.info("Metrics collection stopped")
    
    def _collection_loop(self):
        """Main metrics collection loop"""
        while self.running:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Process queued metrics
                self._process_metrics_queue()
                
                time.sleep(10)  # Collect every 10 seconds
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_metric(Metric(
                name="system_cpu_usage",
                type=MetricType.GAUGE,
                value=cpu_percent,
                description="System CPU usage percentage"
            ))
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.record_metric(Metric(
                name="system_memory_usage",
                type=MetricType.GAUGE,
                value=memory.percent,
                description="System memory usage percentage"
            ))
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.record_metric(Metric(
                name="system_disk_usage",
                type=MetricType.GAUGE,
                value=disk_percent,
                description="System disk usage percentage"
            ))
            
            # Network I/O
            net_io = psutil.net_io_counters()
            self.record_metric(Metric(
                name="network_bytes_sent",
                type=MetricType.COUNTER,
                value=net_io.bytes_sent,
                description="Network bytes sent"
            ))
            
            self.record_metric(Metric(
                name="network_bytes_received",
                type=MetricType.COUNTER,
                value=net_io.bytes_recv,
                description="Network bytes received"
            ))
            
            # Update Prometheus metrics
            self.prometheus_metrics["system_cpu_usage"].set(cpu_percent)
            self.prometheus_metrics["system_memory_usage"].set(memory.percent)
            self.prometheus_metrics["system_disk_usage"].set(disk_percent)
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    def _process_metrics_queue(self):
        """Process metrics from the queue"""
        try:
            while not self.metrics_queue.empty():
                try:
                    metric = self.metrics_queue.get_nowait()
                    self._store_metric(metric)
                except queue.Empty:
                    break
                except Exception as e:
                    logger.error(f"Failed to process metric: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to process metrics queue: {e}")
    
    def record_metric(self, metric: Metric):
        """Record a metric"""
        try:
            self.metrics_queue.put(metric)
        except Exception as e:
            logger.error(f"Failed to record metric: {e}")
    
    def _store_metric(self, metric: Metric):
        """Store metric in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO metrics (name, type, value, labels, timestamp, description)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                metric.name,
                metric.type.value,
                metric.value,
                json.dumps(metric.labels),
                metric.timestamp.isoformat(),
                metric.description
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store metric: {e}")
    
    def get_metrics(self, 
                   metric_name: str = None,
                   start_time: datetime = None,
                   end_time: datetime = None,
                   limit: int = 1000) -> List[Dict[str, Any]]:
        """Get metrics from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = "SELECT * FROM metrics WHERE 1=1"
            params = []
            
            if metric_name:
                query += " AND name = ?"
                params.append(metric_name)
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time.isoformat())
            
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time.isoformat())
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            metrics = []
            for row in rows:
                metrics.append({
                    "id": row[0],
                    "name": row[1],
                    "type": row[2],
                    "value": row[3],
                    "labels": json.loads(row[4]) if row[4] else {},
                    "timestamp": row[5],
                    "description": row[6]
                })
            
            conn.close()
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return []

class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self, db_path: str = "/opt/sutazaiapp/data/alerts.db"):
        self.db_path = db_path
        self.alerts = {}
        self.alert_rules = []
        self.notification_channels = []
        
        # Initialize database
        self._init_database()
        
        logger.info("Alert Manager initialized")
    
    def _init_database(self):
        """Initialize alerts database"""
        try:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    level TEXT NOT NULL,
                    source TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolved_at DATETIME,
                    metadata TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_alerts_level_timestamp 
                ON alerts(level, timestamp)
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("Alerts database initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize alerts database: {e}")
            raise
    
    def add_alert_rule(self, rule: Dict[str, Any]):
        """Add an alert rule"""
        self.alert_rules.append(rule)
        logger.info(f"Added alert rule: {rule['name']}")
    
    def add_notification_channel(self, channel: Dict[str, Any]):
        """Add a notification channel"""
        self.notification_channels.append(channel)
        logger.info(f"Added notification channel: {channel['name']}")
    
    def create_alert(self, alert: Alert):
        """Create a new alert"""
        try:
            # Store alert
            self.alerts[alert.id] = alert
            self._store_alert(alert)
            
            # Send notifications
            self._send_notifications(alert)
            
            logger.info(f"Created alert: {alert.title} ({alert.level.value})")
            
        except Exception as e:
            logger.error(f"Failed to create alert: {e}")
    
    def resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        try:
            if alert_id in self.alerts:
                alert = self.alerts[alert_id]
                alert.resolved = True
                alert.resolved_at = datetime.now()
                
                # Update database
                self._update_alert(alert)
                
                logger.info(f"Resolved alert: {alert.title}")
                
        except Exception as e:
            logger.error(f"Failed to resolve alert: {e}")
    
    def _store_alert(self, alert: Alert):
        """Store alert in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO alerts 
                (id, title, description, level, source, timestamp, resolved, resolved_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.id,
                alert.title,
                alert.description,
                alert.level.value,
                alert.source,
                alert.timestamp.isoformat(),
                alert.resolved,
                alert.resolved_at.isoformat() if alert.resolved_at else None,
                json.dumps(alert.metadata)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store alert: {e}")
    
    def _update_alert(self, alert: Alert):
        """Update alert in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE alerts 
                SET resolved = ?, resolved_at = ?, metadata = ?
                WHERE id = ?
            ''', (
                alert.resolved,
                alert.resolved_at.isoformat() if alert.resolved_at else None,
                json.dumps(alert.metadata),
                alert.id
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to update alert: {e}")
    
    def _send_notifications(self, alert: Alert):
        """Send notifications for an alert"""
        for channel in self.notification_channels:
            try:
                if channel["type"] == "email":
                    self._send_email_notification(alert, channel)
                elif channel["type"] == "webhook":
                    self._send_webhook_notification(alert, channel)
                elif channel["type"] == "slack":
                    self._send_slack_notification(alert, channel)
                    
            except Exception as e:
                logger.error(f"Failed to send notification via {channel['name']}: {e}")
    
    def _send_email_notification(self, alert: Alert, channel: Dict[str, Any]):
        """Send email notification"""
        try:
            msg = MIMEMultipart()
            msg['From'] = channel["from_email"]
            msg['To'] = channel["to_email"]
            msg['Subject'] = f"[{alert.level.value.upper()}] {alert.title}"
            
            body = f"""
            Alert: {alert.title}
            Level: {alert.level.value}
            Source: {alert.source}
            Time: {alert.timestamp.isoformat()}
            
            Description:
            {alert.description}
            
            Metadata:
            {json.dumps(alert.metadata, indent=2)}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(channel["smtp_server"], channel["smtp_port"])
            if channel.get("use_tls"):
                server.starttls()
            if channel.get("username") and channel.get("password"):
                server.login(channel["username"], channel["password"])
            
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
    
    def _send_webhook_notification(self, alert: Alert, channel: Dict[str, Any]):
        """Send webhook notification"""
        try:
            payload = {
                "alert_id": alert.id,
                "title": alert.title,
                "description": alert.description,
                "level": alert.level.value,
                "source": alert.source,
                "timestamp": alert.timestamp.isoformat(),
                "metadata": alert.metadata
            }
            
            response = requests.post(
                channel["webhook_url"],
                json=payload,
                headers=channel.get("headers", {}),
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Webhook notification sent successfully")
            else:
                logger.error(f"Webhook notification failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
    
    def _send_slack_notification(self, alert: Alert, channel: Dict[str, Any]):
        """Send Slack notification"""
        try:
            color = {
                AlertLevel.INFO: "good",
                AlertLevel.WARNING: "warning",
                AlertLevel.ERROR: "danger",
                AlertLevel.CRITICAL: "danger"
            }
            
            payload = {
                "text": f"Alert: {alert.title}",
                "attachments": [
                    {
                        "color": color.get(alert.level, "warning"),
                        "fields": [
                            {
                                "title": "Level",
                                "value": alert.level.value,
                                "short": True
                            },
                            {
                                "title": "Source",
                                "value": alert.source,
                                "short": True
                            },
                            {
                                "title": "Time",
                                "value": alert.timestamp.isoformat(),
                                "short": True
                            },
                            {
                                "title": "Description",
                                "value": alert.description,
                                "short": False
                            }
                        ]
                    }
                ]
            }
            
            response = requests.post(
                channel["slack_webhook_url"],
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Slack notification sent successfully")
            else:
                logger.error(f"Slack notification failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
    
    def get_alerts(self, 
                   level: AlertLevel = None,
                   resolved: bool = None,
                   start_time: datetime = None,
                   end_time: datetime = None,
                   limit: int = 100) -> List[Dict[str, Any]]:
        """Get alerts from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = "SELECT * FROM alerts WHERE 1=1"
            params = []
            
            if level:
                query += " AND level = ?"
                params.append(level.value)
            
            if resolved is not None:
                query += " AND resolved = ?"
                params.append(resolved)
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time.isoformat())
            
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time.isoformat())
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            alerts = []
            for row in rows:
                alerts.append({
                    "id": row[0],
                    "title": row[1],
                    "description": row[2],
                    "level": row[3],
                    "source": row[4],
                    "timestamp": row[5],
                    "resolved": bool(row[6]),
                    "resolved_at": row[7],
                    "metadata": json.loads(row[8]) if row[8] else {}
                })
            
            conn.close()
            return alerts
            
        except Exception as e:
            logger.error(f"Failed to get alerts: {e}")
            return []

class HealthMonitor:
    """Monitors system health and components"""
    
    def __init__(self):
        self.health_checks = {}
        self.check_results = defaultdict(deque)
        self.running = False
        self.monitor_thread = None
        
        logger.info("Health Monitor initialized")
    
    def add_health_check(self, name: str, check_func: Callable[[], HealthCheck]):
        """Add a health check"""
        self.health_checks[name] = check_func
        logger.info(f"Added health check: {name}")
    
    def start_monitoring(self):
        """Start health monitoring"""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main health monitoring loop"""
        while self.running:
            try:
                for name, check_func in self.health_checks.items():
                    try:
                        result = check_func()
                        
                        # Store result (keep last 100 results)
                        self.check_results[name].append(result)
                        if len(self.check_results[name]) > 100:
                            self.check_results[name].popleft()
                        
                        # Log unhealthy status
                        if result.status != "healthy":
                            logger.warning(f"Health check {name} status: {result.status} - {result.message}")
                        
                    except Exception as e:
                        logger.error(f"Health check {name} failed: {e}")
                        
                        # Record failure
                        failure_result = HealthCheck(
                            component=name,
                            status="unhealthy",
                            message=f"Health check failed: {e}",
                            response_time=0.0
                        )
                        self.check_results[name].append(failure_result)
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status"""
        try:
            overall_status = "healthy"
            components = {}
            
            for name, results in self.check_results.items():
                if results:
                    latest_result = results[-1]
                    components[name] = {
                        "status": latest_result.status,
                        "message": latest_result.message,
                        "timestamp": latest_result.timestamp.isoformat(),
                        "response_time": latest_result.response_time
                    }
                    
                    # Determine overall status
                    if latest_result.status == "unhealthy":
                        overall_status = "unhealthy"
                    elif latest_result.status == "warning" and overall_status == "healthy":
                        overall_status = "warning"
                else:
                    components[name] = {
                        "status": "unknown",
                        "message": "No health check results",
                        "timestamp": None,
                        "response_time": 0.0
                    }
            
            return {
                "overall_status": overall_status,
                "components": components,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get health status: {e}")
            return {"overall_status": "unknown", "components": {}, "error": str(e)}

class ObservabilitySystem:
    """Main observability system coordinating all monitoring components"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.health_monitor = HealthMonitor()
        
        # Setup default health checks
        self._setup_default_health_checks()
        
        # Setup default alert rules
        self._setup_default_alert_rules()
        
        # Start Prometheus metrics server
        self._start_prometheus_server()
        
        logger.info("Observability System initialized")
    
    def _setup_default_health_checks(self):
        """Setup default health checks"""
        
        def system_health_check():
            start_time = time.time()
            
            # Check system resources
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            disk_percent = psutil.disk_usage('/').percent
            
            response_time = time.time() - start_time
            
            if cpu_percent > 90 or memory_percent > 95 or disk_percent > 95:
                status = "unhealthy"
                message = f"High resource usage: CPU {cpu_percent}%, Memory {memory_percent}%, Disk {disk_percent}%"
            elif cpu_percent > 70 or memory_percent > 80 or disk_percent > 80:
                status = "warning"
                message = f"Moderate resource usage: CPU {cpu_percent}%, Memory {memory_percent}%, Disk {disk_percent}%"
            else:
                status = "healthy"
                message = f"System resources normal: CPU {cpu_percent}%, Memory {memory_percent}%, Disk {disk_percent}%"
            
            return HealthCheck(
                component="system",
                status=status,
                message=message,
                response_time=response_time
            )
        
        self.health_monitor.add_health_check("system", system_health_check)
    
    def _setup_default_alert_rules(self):
        """Setup default alert rules"""
        
        # High CPU usage alert
        self.alert_manager.add_alert_rule({
            "name": "high_cpu_usage",
            "condition": "system_cpu_usage > 80",
            "level": AlertLevel.WARNING,
            "description": "System CPU usage is high"
        })
        
        # High memory usage alert
        self.alert_manager.add_alert_rule({
            "name": "high_memory_usage",
            "condition": "system_memory_usage > 90",
            "level": AlertLevel.CRITICAL,
            "description": "System memory usage is critically high"
        })
        
        # Disk space alert
        self.alert_manager.add_alert_rule({
            "name": "low_disk_space",
            "condition": "system_disk_usage > 85",
            "level": AlertLevel.WARNING,
            "description": "System disk space is running low"
        })
    
    def _start_prometheus_server(self):
        """Start Prometheus metrics server"""
        try:
            start_http_server(8090)
            logger.info("Prometheus metrics server started on port 8090")
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")
    
    def start_monitoring(self):
        """Start all monitoring components"""
        self.metrics_collector.start_collection()
        self.health_monitor.start_monitoring()
        logger.info("Observability monitoring started")
    
    def stop_monitoring(self):
        """Stop all monitoring components"""
        self.metrics_collector.stop_collection()
        self.health_monitor.stop_monitoring()
        logger.info("Observability monitoring stopped")
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get comprehensive system overview"""
        return {
            "health_status": self.health_monitor.get_health_status(),
            "recent_alerts": self.alert_manager.get_alerts(limit=10),
            "system_metrics": self.metrics_collector.get_metrics(limit=50),
            "timestamp": datetime.now().isoformat()
        }

# Global observability instance
_observability_instance = None

def get_observability_system() -> ObservabilitySystem:
    """Get the global observability system instance"""
    global _observability_instance
    if _observability_instance is None:
        _observability_instance = ObservabilitySystem()
    return _observability_instance

def main():
    """Main entry point for testing observability system"""
    obs_system = get_observability_system()
    obs_system.start_monitoring()
    
    try:
        # Keep monitoring running
        while True:
            time.sleep(60)
            overview = obs_system.get_system_overview()
            print(f"System Overview: {json.dumps(overview, indent=2)}")
    except KeyboardInterrupt:
        obs_system.stop_monitoring()

if __name__ == "__main__":
    main()