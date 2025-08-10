#!/usr/bin/env python3
"""
Continuous Performance Monitoring System for SutazAI
===================================================

Real-time performance monitoring with:
- Live metrics collection and analysis
- SLA violation detection and alerting
- Automated performance optimization triggers
- Dynamic resource allocation recommendations
- Real-time dashboard updates
- Intelligent alert filtering and prioritization
"""

import asyncio
import json
import time
import logging
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import queue
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from pathlib import Path
import yaml

# Web framework for dashboard
try:
    from flask import Flask, render_template, jsonify, request
    from flask_socketio import SocketIO, emit
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    logging.warning("Flask not available. Install with: pip install flask flask-socketio")

# Import our forecasting system
try:
    from performance_forecasting_models import PerformanceForecastingSystem, ForecastResult
    FORECASTING_AVAILABLE = True
except ImportError:
    FORECASTING_AVAILABLE = False
    logging.warning("Performance forecasting models not available")

# Metrics collection
import psutil
import docker
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceAlert:
    """Performance alert data structure"""
    id: str
    timestamp: datetime
    severity: str  # 'critical', 'warning', 'info'
    component: str
    metric: str
    current_value: float
    threshold_value: float
    message: str
    acknowledged: bool = False
    resolved: bool = False
    auto_actionable: bool = False
    recommended_actions: List[str] = None

@dataclass
class SLAViolation:
    """SLA violation tracking"""
    component: str
    metric: str
    sla_threshold: float
    actual_value: float
    violation_duration: timedelta
    impact_level: str  # 'low', 'medium', 'high', 'critical'
    first_violation_time: datetime
    last_violation_time: datetime
    violation_count: int = 1

@dataclass
class PerformanceMetric:
    """Performance metric data point"""
    timestamp: datetime
    component: str
    metric_name: str
    value: float
    unit: str
    tags: Dict[str, str] = None

class MetricsCollector:
    """Collects performance metrics from various sources"""
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.collection_interval = 30  # seconds
        self.running = False
        self.metrics_queue = queue.Queue()
    
    async def collect_system_metrics(self) -> List[PerformanceMetric]:
        """Collect system-level performance metrics"""
        metrics = []
        timestamp = datetime.now()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        metrics.append(PerformanceMetric(
            timestamp=timestamp,
            component='system',
            metric_name='cpu_percent',
            value=cpu_percent,
            unit='percent'
        ))
        
        # Memory metrics
        memory = psutil.virtual_memory()
        metrics.append(PerformanceMetric(
            timestamp=timestamp,
            component='system',
            metric_name='memory_percent',
            value=memory.percent,
            unit='percent'
        ))
        
        metrics.append(PerformanceMetric(
            timestamp=timestamp,
            component='system',
            metric_name='memory_used_gb',
            value=memory.used / (1024**3),
            unit='gigabytes'
        ))
        
        # Disk I/O
        try:
            disk_io = psutil.disk_io_counters()
            if disk_io:
                metrics.append(PerformanceMetric(
                    timestamp=timestamp,
                    component='system',
                    metric_name='disk_read_mb_per_sec',
                    value=disk_io.read_bytes / (1024**2),
                    unit='mb_per_second'
                ))
                
                metrics.append(PerformanceMetric(
                    timestamp=timestamp,
                    component='system',
                    metric_name='disk_write_mb_per_sec',
                    value=disk_io.write_bytes / (1024**2),
                    unit='mb_per_second'
                ))
        except Exception as e:
            # Suppressed exception (was bare except)
            logger.debug(f"Suppressed exception: {e}")
            pass
        
        # Network I/O
        try:
            network_io = psutil.net_io_counters()
            if network_io:
                metrics.append(PerformanceMetric(
                    timestamp=timestamp,
                    component='system',
                    metric_name='network_bytes_sent',
                    value=network_io.bytes_sent,
                    unit='bytes'
                ))
                
                metrics.append(PerformanceMetric(
                    timestamp=timestamp,
                    component='system',
                    metric_name='network_bytes_recv',
                    value=network_io.bytes_recv,
                    unit='bytes'
                ))
        except Exception as e:
            # Suppressed exception (was bare except)
            logger.debug(f"Suppressed exception: {e}")
            pass
        
        return metrics
    
    async def collect_container_metrics(self) -> List[PerformanceMetric]:
        """Collect container-level performance metrics"""
        metrics = []
        timestamp = datetime.now()
        
        try:
            containers = self.docker_client.containers.list()
            
            for container in containers:
                try:
                    stats = container.stats(stream=False)
                    
                    # CPU usage
                    cpu_delta = (stats['cpu_stats']['cpu_usage']['total_usage'] - 
                               stats['precpu_stats']['cpu_usage']['total_usage'])
                    system_delta = (stats['cpu_stats']['system_cpu_usage'] - 
                                  stats['precpu_stats']['system_cpu_usage'])
                    
                    if system_delta > 0:
                        cpu_percent = (cpu_delta / system_delta) * 100.0
                        metrics.append(PerformanceMetric(
                            timestamp=timestamp,
                            component=container.name,
                            metric_name='container_cpu_percent',
                            value=cpu_percent,
                            unit='percent',
                            tags={'container_id': container.id[:12]}
                        ))
                    
                    # Memory usage
                    memory_usage = stats['memory_stats']['usage']
                    memory_limit = stats['memory_stats']['limit']
                    
                    if memory_limit > 0:
                        memory_percent = (memory_usage / memory_limit) * 100.0
                        metrics.append(PerformanceMetric(
                            timestamp=timestamp,
                            component=container.name,
                            metric_name='container_memory_percent',
                            value=memory_percent,
                            unit='percent',
                            tags={'container_id': container.id[:12]}
                        ))
                        
                        metrics.append(PerformanceMetric(
                            timestamp=timestamp,
                            component=container.name,
                            metric_name='container_memory_usage_mb',
                            value=memory_usage / (1024**2),
                            unit='megabytes',
                            tags={'container_id': container.id[:12]}
                        ))
                
                except Exception as e:
                    logger.debug(f"Error collecting stats for container {container.name}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error collecting container metrics: {e}")
        
        return metrics
    
    async def collect_agent_metrics(self) -> List[PerformanceMetric]:
        """Collect agent-specific performance metrics"""
        metrics = []
        timestamp = datetime.now()
        
        # Get agent containers
        agent_containers = [c for c in self.docker_client.containers.list() 
                          if 'agent' in c.name.lower()]
        
        for container in agent_containers:
            agent_name = container.name.replace('sutazaiapp-', '')
            
            try:
                # Try to get health check response time
                port = self.get_agent_port(container)
                if port:
                    start_time = time.time()
                    try:
                        response = requests.get(f"http://localhost:{port}/health", timeout=5)
                        response_time = (time.time() - start_time) * 1000  # ms
                        
                        metrics.append(PerformanceMetric(
                            timestamp=timestamp,
                            component=agent_name,
                            metric_name='agent_health_response_time_ms',
                            value=response_time,
                            unit='milliseconds',
                            tags={'port': str(port), 'status_code': str(response.status_code)}
                        ))
                        
                        # Agent availability
                        is_healthy = response.status_code == 200
                        metrics.append(PerformanceMetric(
                            timestamp=timestamp,
                            component=agent_name,
                            metric_name='agent_availability',
                            value=1.0 if is_healthy else 0.0,
                            unit='boolean',
                            tags={'port': str(port)}
                        ))
                        
                    except requests.RequestException:
                        # Agent is not responding
                        metrics.append(PerformanceMetric(
                            timestamp=timestamp,
                            component=agent_name,
                            metric_name='agent_availability',
                            value=0.0,
                            unit='boolean',
                            tags={'port': str(port) if port else 'unknown'}
                        ))
                
            except Exception as e:
                logger.debug(f"Error collecting metrics for agent {agent_name}: {e}")
        
        return metrics
    
    def get_agent_port(self, container) -> Optional[int]:
        """Extract agent port from container configuration"""
        try:
            port_bindings = container.attrs['NetworkSettings']['Ports']
            for internal_port, bindings in port_bindings.items():
                if bindings:
                    return int(bindings[0]['HostPort'])
        except Exception as e:
            # Suppressed exception (was bare except)
            logger.debug(f"Suppressed exception: {e}")
            pass
        return None
    
    async def start_collection(self):
        """Start continuous metrics collection"""
        self.running = True
        logger.info("Starting continuous metrics collection")
        
        while self.running:
            try:
                # Collect all metrics
                all_metrics = []
                
                system_metrics = await self.collect_system_metrics()
                all_metrics.extend(system_metrics)
                
                container_metrics = await self.collect_container_metrics()
                all_metrics.extend(container_metrics)
                
                agent_metrics = await self.collect_agent_metrics()
                all_metrics.extend(agent_metrics)
                
                # Add metrics to queue for processing
                for metric in all_metrics:
                    self.metrics_queue.put(metric)
                
                logger.debug(f"Collected {len(all_metrics)} metrics")
                
                # Wait for next collection interval
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error during metrics collection: {e}")
                await asyncio.sleep(10)  # Short delay before retry
    
    def stop_collection(self):
        """Stop metrics collection"""
        self.running = False

class SLAManager:
    """Manages SLA definitions and violation tracking"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.sla_thresholds = self.load_sla_config()
        self.violations = {}  # component -> SLAViolation
        self.violation_history = []
    
    def load_sla_config(self) -> Dict[str, Any]:
        """Load SLA configuration"""
        default_slas = {
            'system': {
                'cpu_percent': {'threshold': 80, 'comparison': 'max'},
                'memory_percent': {'threshold': 85, 'comparison': 'max'}
            },
            'agents': {
                'agent_health_response_time_ms': {'threshold': 1000, 'comparison': 'max'},
                'agent_availability': {'threshold': 0.95, 'comparison': 'min'}
            },
            'containers': {
                'container_cpu_percent': {'threshold': 90, 'comparison': 'max'},
                'container_memory_percent': {'threshold': 90, 'comparison': 'max'}
            }
        }
        
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    return config.get('sla_thresholds', default_slas)
        except Exception as e:
            logger.warning(f"Failed to load SLA config: {e}")
        
        return default_slas
    
    def check_sla_violation(self, metric: PerformanceMetric) -> Optional[SLAViolation]:
        """Check if a metric violates SLA"""
        
        # Determine SLA category
        category = 'system'
        if metric.component != 'system':
            if 'agent' in metric.metric_name:
                category = 'agents'
            elif 'container' in metric.metric_name:
                category = 'containers'
        
        # Get SLA threshold
        sla_config = self.sla_thresholds.get(category, {}).get(metric.metric_name)
        if not sla_config:
            return None
        
        threshold = sla_config['threshold']
        comparison = sla_config['comparison']
        
        # Check violation
        violation = False
        if comparison == 'max' and metric.value > threshold:
            violation = True
        elif comparison == 'min' and metric.value < threshold:
            violation = True
        
        if not violation:
            # Clear any existing violation for this component/metric
            violation_key = f"{metric.component}:{metric.metric_name}"
            if violation_key in self.violations:
                del self.violations[violation_key]
            return None
        
        # Create or update violation
        violation_key = f"{metric.component}:{metric.metric_name}"
        
        if violation_key in self.violations:
            # Update existing violation
            existing_violation = self.violations[violation_key]
            existing_violation.actual_value = metric.value
            existing_violation.last_violation_time = metric.timestamp
            existing_violation.violation_count += 1
            existing_violation.violation_duration = (
                metric.timestamp - existing_violation.first_violation_time
            )
            return existing_violation
        else:
            # Create new violation
            impact_level = self.determine_impact_level(metric.metric_name, metric.value, threshold)
            
            violation = SLAViolation(
                component=metric.component,
                metric=metric.metric_name,
                sla_threshold=threshold,
                actual_value=metric.value,
                violation_duration=timedelta(0),
                impact_level=impact_level,
                first_violation_time=metric.timestamp,
                last_violation_time=metric.timestamp
            )
            
            self.violations[violation_key] = violation
            self.violation_history.append(violation)
            
            return violation
    
    def determine_impact_level(self, metric_name: str, actual_value: float, 
                             threshold: float) -> str:
        """Determine the impact level of an SLA violation"""
        
        # Calculate how far over threshold we are
        if 'percent' in metric_name:
            over_threshold = (actual_value - threshold) / threshold
        else:
            over_threshold = (actual_value - threshold) / threshold
        
        if over_threshold > 0.5:  # >50% over threshold
            return 'critical'
        elif over_threshold > 0.25:  # >25% over threshold
            return 'high'
        elif over_threshold > 0.1:   # >10% over threshold
            return 'medium'
        else:
            return 'low'
    
    def get_active_violations(self) -> List[SLAViolation]:
        """Get all active SLA violations"""
        return list(self.violations.values())
    
    def get_violation_summary(self) -> Dict[str, Any]:
        """Get summary of SLA violations"""
        active_violations = self.get_active_violations()
        
        return {
            'total_active_violations': len(active_violations),
            'critical_violations': len([v for v in active_violations if v.impact_level == 'critical']),
            'high_violations': len([v for v in active_violations if v.impact_level == 'high']),
            'medium_violations': len([v for v in active_violations if v.impact_level == 'medium']),
            'low_violations': len([v for v in active_violations if v.impact_level == 'low']),
            'violations_by_component': self.group_violations_by_component(active_violations),
            'longest_violation_duration': self.get_longest_violation_duration(active_violations)
        }
    
    def group_violations_by_component(self, violations: List[SLAViolation]) -> Dict[str, int]:
        """Group violations by component"""
        component_counts = {}
        for violation in violations:
            component_counts[violation.component] = component_counts.get(violation.component, 0) + 1
        return component_counts
    
    def get_longest_violation_duration(self, violations: List[SLAViolation]) -> float:
        """Get the longest violation duration in seconds"""
        if not violations:
            return 0.0
        return max(violation.violation_duration.total_seconds() for violation in violations)

class AlertManager:
    """Manages performance alerts and notifications"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alerts = {}  # alert_id -> PerformanceAlert
        self.alert_handlers = []
        self.alert_queue = queue.Queue()
        self.running = False
        
        # Setup alert handlers
        self.setup_alert_handlers()
    
    def setup_alert_handlers(self):
        """Setup alert notification handlers"""
        if self.config.get('email_alerts', {}).get('enabled', False):
            self.alert_handlers.append(self.send_email_alert)
        
        if self.config.get('webhook_alerts', {}).get('enabled', False):
            self.alert_handlers.append(self.send_webhook_alert)
        
        # Always include console logging
        self.alert_handlers.append(self.log_alert)
    
    def create_alert(self, component: str, metric: str, current_value: float,
                    threshold_value: float, severity: str, message: str,
                    recommended_actions: List[str] = None) -> PerformanceAlert:
        """Create a performance alert"""
        
        alert_id = f"{component}:{metric}:{severity}:{int(time.time())}"
        
        alert = PerformanceAlert(
            id=alert_id,
            timestamp=datetime.now(),
            severity=severity,
            component=component,
            metric=metric,
            current_value=current_value,
            threshold_value=threshold_value,
            message=message,
            recommended_actions=recommended_actions or []
        )
        
        self.alerts[alert_id] = alert
        self.alert_queue.put(alert)
        
        return alert
    
    def process_sla_violation(self, violation: SLAViolation):
        """Process SLA violation and create appropriate alerts"""
        
        # Determine severity based on impact level
        severity_mapping = {
            'critical': 'critical',
            'high': 'critical',
            'medium': 'warning',
            'low': 'warning'
        }
        
        severity = severity_mapping.get(violation.impact_level, 'warning')
        
        # Generate message
        message = (f"SLA violation detected for {violation.component}. "
                  f"{violation.metric} is {violation.actual_value:.2f} "
                  f"(threshold: {violation.sla_threshold:.2f})")
        
        # Generate recommended actions
        recommended_actions = self.generate_recommended_actions(violation)
        
        # Create alert
        self.create_alert(
            component=violation.component,
            metric=violation.metric,
            current_value=violation.actual_value,
            threshold_value=violation.sla_threshold,
            severity=severity,
            message=message,
            recommended_actions=recommended_actions
        )
    
    def generate_recommended_actions(self, violation: SLAViolation) -> List[str]:
        """Generate recommended actions for SLA violations"""
        actions = []
        
        if 'cpu' in violation.metric.lower():
            actions.extend([
                "Investigate CPU-intensive processes",
                "Consider horizontal scaling",
                "Review and optimize algorithms",
                "Check for infinite loops or inefficient code"
            ])
        
        elif 'memory' in violation.metric.lower():
            actions.extend([
                "Check for memory leaks",
                "Increase memory limits if needed",
                "Optimize data structures",
                "Implement garbage collection"
            ])
        
        elif 'response_time' in violation.metric.lower():
            actions.extend([
                "Investigate network latency",
                "Optimize database queries",
                "Check for bottlenecks",
                "Consider caching strategies"
            ])
        
        elif 'availability' in violation.metric.lower():
            actions.extend([
                "Check service health",
                "Restart unhealthy containers",
                "Investigate recent deployments",
                "Review error logs"
            ])
        
        return actions
    
    async def start_alert_processing(self):
        """Start processing alerts"""
        self.running = True
        logger.info("Starting alert processing")
        
        while self.running:
            try:
                # Process alerts from queue
                try:
                    alert = self.alert_queue.get(timeout=1)
                    await self.process_alert(alert)
                except queue.Empty:
                    continue
                
            except Exception as e:
                logger.error(f"Error processing alerts: {e}")
                await asyncio.sleep(1)
    
    async def process_alert(self, alert: PerformanceAlert):
        """Process a single alert"""
        logger.info(f"Processing {alert.severity} alert for {alert.component}:{alert.metric}")
        
        # Send notifications via all configured handlers
        for handler in self.alert_handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
    
    async def log_alert(self, alert: PerformanceAlert):
        """Log alert to console/file"""
        logger.warning(f"ALERT [{alert.severity.upper()}] {alert.component}: {alert.message}")
        
        if alert.recommended_actions:
            logger.info(f"Recommended actions for {alert.component}:")
            for action in alert.recommended_actions:
                logger.info(f"  - {action}")
    
    async def send_email_alert(self, alert: PerformanceAlert):
        """Send email alert"""
        email_config = self.config.get('email_alerts', {})
        
        if not email_config.get('enabled', False):
            return
        
        try:
            msg = MimeMultipart()
            msg['From'] = email_config.get('from_email', 'alerts@sutazai.com')
            msg['To'] = email_config.get('to_email', 'admin@sutazai.com')
            msg['Subject'] = f"SutazAI Performance Alert - {alert.severity.upper()}"
            
            body = f"""
Performance Alert Details:

Component: {alert.component}
Metric: {alert.metric}
Severity: {alert.severity.upper()}
Current Value: {alert.current_value:.2f}
Threshold: {alert.threshold_value:.2f}
Timestamp: {alert.timestamp}

Message: {alert.message}

Recommended Actions:
"""
            
            for action in alert.recommended_actions:
                body += f"- {action}\n"
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(email_config.get('smtp_server', 'localhost'))
            if email_config.get('use_tls', False):
                server.starttls()
            if email_config.get('username'):
                server.login(email_config['username'], email_config['password'])
            
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent for {alert.component}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    async def send_webhook_alert(self, alert: PerformanceAlert):
        """Send webhook alert"""
        webhook_config = self.config.get('webhook_alerts', {})
        
        if not webhook_config.get('enabled', False):
            return
        
        try:
            webhook_url = webhook_config.get('url')
            if not webhook_url:
                return
            
            payload = {
                'alert_id': alert.id,
                'timestamp': alert.timestamp.isoformat(),
                'severity': alert.severity,
                'component': alert.component,
                'metric': alert.metric,
                'current_value': alert.current_value,
                'threshold_value': alert.threshold_value,
                'message': alert.message,
                'recommended_actions': alert.recommended_actions
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Webhook alert sent for {alert.component}")
            
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
    
    def stop_alert_processing(self):
        """Stop alert processing"""
        self.running = False

class PerformanceDashboard:
    """Web-based performance monitoring dashboard"""
    
    def __init__(self, monitor_system):
        self.monitor_system = monitor_system
        self.app = None
        self.socketio = None
        
        if FLASK_AVAILABLE:
            self.setup_flask_app()
    
    def setup_flask_app(self):
        """Setup Flask application"""
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'sutazai-performance-monitor'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Routes
        self.app.route('/')(self.dashboard_home)
        self.app.route('/api/metrics')(self.get_metrics_api)
        self.app.route('/api/alerts')(self.get_alerts_api)
        self.app.route('/api/sla-violations')(self.get_sla_violations_api)
        self.app.route('/api/forecast/<metric>')(self.get_forecast_api)
        
        # WebSocket events
        self.socketio.on('connect')(self.handle_connect)
        self.socketio.on('disconnect')(self.handle_disconnect)
    
    def dashboard_home(self):
        """Dashboard home page"""
        return render_template('performance_dashboard.html')
    
    def get_metrics_api(self):
        """API endpoint for current metrics"""
        try:
            # Get recent metrics from database
            metrics = self.monitor_system.get_recent_metrics(hours=1)
            return jsonify({
                'status': 'success',
                'metrics': [asdict(m) for m in metrics],
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    def get_alerts_api(self):
        """API endpoint for active alerts"""
        try:
            alerts = list(self.monitor_system.alert_manager.alerts.values())
            active_alerts = [a for a in alerts if not a.resolved]
            
            return jsonify({
                'status': 'success',
                'alerts': [asdict(a) for a in active_alerts],
                'total_count': len(active_alerts)
            })
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    def get_sla_violations_api(self):
        """API endpoint for SLA violations"""
        try:
            violations = self.monitor_system.sla_manager.get_active_violations()
            summary = self.monitor_system.sla_manager.get_violation_summary()
            
            return jsonify({
                'status': 'success',
                'violations': [asdict(v) for v in violations],
                'summary': summary
            })
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    def get_forecast_api(self, metric):
        """API endpoint for performance forecasts"""
        try:
            if not FORECASTING_AVAILABLE:
                return jsonify({'status': 'error', 'message': 'Forecasting not available'}), 503
            
            forecast = self.monitor_system.forecasting_system.generate_forecast(
                metric=metric,
                horizon_hours=24,
                model_type='ensemble'
            )
            
            if forecast:
                return jsonify({
                    'status': 'success',
                    'forecast': asdict(forecast)
                })
            else:
                return jsonify({'status': 'error', 'message': 'Forecast generation failed'}), 500
                
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    def handle_connect(self):
        """Handle WebSocket connection"""
        logger.info("Dashboard client connected")
        emit('status', {'message': 'Connected to SutazAI Performance Monitor'})
    
    def handle_disconnect(self):
        """Handle WebSocket disconnection"""
        logger.info("Dashboard client disconnected")
    
    def broadcast_metrics(self, metrics: List[PerformanceMetric]):
        """Broadcast metrics to connected clients"""
        if self.socketio:
            self.socketio.emit('metrics_update', {
                'metrics': [asdict(m) for m in metrics],
                'timestamp': datetime.now().isoformat()
            })
    
    def broadcast_alert(self, alert: PerformanceAlert):
        """Broadcast alert to connected clients"""
        if self.socketio:
            self.socketio.emit('new_alert', asdict(alert))
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the dashboard server"""
        if self.socketio:
            logger.info(f"Starting performance dashboard on {host}:{port}")
            self.socketio.run(self.app, host=host, port=port, debug=debug)
        else:
            logger.error("Flask not available, cannot start dashboard")

class ContinuousPerformanceMonitor:
    """Main continuous performance monitoring system"""
    
    def __init__(self, config_path: str = "/opt/sutazaiapp/config/benchmark_config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        
        # Initialize components
        self.metrics_collector = MetricsCollector()
        self.sla_manager = SLAManager(config_path)
        self.alert_manager = AlertManager(self.config.get('alerting', {}))
        
        # Database for metrics storage
        self.db_path = self.config.get('database_path', '/opt/sutazaiapp/data/performance_metrics.db')
        self.init_database()
        
        # Forecasting system
        if FORECASTING_AVAILABLE:
            self.forecasting_system = PerformanceForecastingSystem(self.db_path)
        else:
            self.forecasting_system = None
        
        # Dashboard
        self.dashboard = PerformanceDashboard(self)
        
        # Control flags
        self.running = False
        self.tasks = []
    
    def load_config(self) -> Dict[str, Any]:
        """Load monitoring configuration"""
        default_config = {
            'collection_interval': 30,
            'database_path': '/opt/sutazaiapp/data/performance_metrics.db',
            'dashboard_port': 5000,
            'alerting': {
                'email_alerts': {'enabled': False},
                'webhook_alerts': {'enabled': False}
            },
            'forecasting': {
                'enabled': True,
                'forecast_horizons': [24, 168, 720]  # 1 day, 1 week, 1 month
            }
        }
        
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    default_config.update(user_config)
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
        
        return default_config
    
    def init_database(self):
        """Initialize database for metrics storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                component TEXT,
                metric_name TEXT,
                value REAL,
                unit TEXT,
                tags TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_alerts (
                id TEXT PRIMARY KEY,
                timestamp DATETIME,
                severity TEXT,
                component TEXT,
                metric TEXT,
                current_value REAL,
                threshold_value REAL,
                message TEXT,
                acknowledged INTEGER DEFAULT 0,
                resolved INTEGER DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_metric(self, metric: PerformanceMetric):
        """Store metric in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO performance_metrics 
            (timestamp, component, metric_name, value, unit, tags)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            metric.timestamp,
            metric.component,
            metric.metric_name,
            metric.value,
            metric.unit,
            json.dumps(metric.tags) if metric.tags else None
        ))
        
        conn.commit()
        conn.close()
    
    def get_recent_metrics(self, hours: int = 1) -> List[PerformanceMetric]:
        """Get recent metrics from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT timestamp, component, metric_name, value, unit, tags
            FROM performance_metrics 
            WHERE timestamp > datetime('now', '-{} hours')
            ORDER BY timestamp DESC
        '''.format(hours))
        
        metrics = []
        for row in cursor.fetchall():
            tags = json.loads(row[5]) if row[5] else None
            metric = PerformanceMetric(
                timestamp=datetime.fromisoformat(row[0]),
                component=row[1],
                metric_name=row[2],
                value=row[3],
                unit=row[4],
                tags=tags
            )
            metrics.append(metric)
        
        conn.close()
        return metrics
    
    async def start_monitoring(self):
        """Start continuous monitoring"""
        self.running = True
        logger.info("Starting continuous performance monitoring")
        
        # Start metrics collection
        collection_task = asyncio.create_task(self.metrics_collector.start_collection())
        self.tasks.append(collection_task)
        
        # Start metrics processing
        processing_task = asyncio.create_task(self.process_metrics())
        self.tasks.append(processing_task)
        
        # Start alert processing
        alert_task = asyncio.create_task(self.alert_manager.start_alert_processing())
        self.tasks.append(alert_task)
        
        # Start dashboard (in separate thread)
        if FLASK_AVAILABLE:
            dashboard_thread = threading.Thread(
                target=self.dashboard.run,
                kwargs={'port': self.config.get('dashboard_port', 5000), 'debug': False}
            )
            dashboard_thread.daemon = True
            dashboard_thread.start()
        
        logger.info("All monitoring services started")
        
        # Wait for all tasks
        await asyncio.gather(*self.tasks, return_exceptions=True)
    
    async def process_metrics(self):
        """Process collected metrics"""
        logger.info("Starting metrics processing")
        
        while self.running:
            try:
                # Process metrics from queue
                processed_count = 0
                
                while not self.metrics_collector.metrics_queue.empty():
                    try:
                        metric = self.metrics_collector.metrics_queue.get_nowait()
                        
                        # Store metric
                        self.store_metric(metric)
                        
                        # Check for SLA violations
                        violation = self.sla_manager.check_sla_violation(metric)
                        if violation:
                            self.alert_manager.process_sla_violation(violation)
                        
                        # Broadcast to dashboard
                        if processed_count % 10 == 0:  # Throttle dashboard updates
                            self.dashboard.broadcast_metrics([metric])
                        
                        processed_count += 1
                        
                    except queue.Empty:
                        break
                
                if processed_count > 0:
                    logger.debug(f"Processed {processed_count} metrics")
                
                await asyncio.sleep(1)  # Process metrics every second
                
            except Exception as e:
                logger.error(f"Error processing metrics: {e}")
                await asyncio.sleep(5)
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        logger.info("Stopping continuous performance monitoring")
        
        self.running = False
        self.metrics_collector.stop_collection()
        self.alert_manager.stop_alert_processing()
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()

# Main execution
async def main():
    """Main monitoring function"""
    monitor = ContinuousPerformanceMonitor()
    
    try:
        await monitor.start_monitoring()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        monitor.stop_monitoring()

if __name__ == "__main__":
    asyncio.run(main())