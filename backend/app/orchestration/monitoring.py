"""
SutazAI Orchestration Monitoring System
Comprehensive monitoring, metrics collection, and alerting for
the multi-agent orchestration system.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import psutil
import redis.asyncio as redis
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class Metric:
    name: str
    type: MetricType
    value: float
    timestamp: datetime
    labels: Dict[str, str] = None
    description: str = ""

@dataclass
class Alert:
    id: str
    name: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    source: str
    resolved: bool = False
    metadata: Dict[str, Any] = None

@dataclass
class HealthCheck:
    component: str
    status: str
    last_check: datetime
    response_time: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

class OrchestrationMonitor:
    """
    Comprehensive monitoring system for multi-agent orchestration:
    - Real-time metrics collection
    - Performance monitoring
    - Health checks
    - Alerting system
    - Dashboard data aggregation
    - Resource usage tracking
    """
    
    def __init__(self, redis_url: str = "redis://:redis_password@localhost:6379"):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        
        # Metrics storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alerts: Dict[str, Alert] = {}
        self.health_checks: Dict[str, HealthCheck] = {}
        
        # Monitoring configuration
        self.collection_interval = 10  # seconds
        self.health_check_interval = 30  # seconds
        self.alert_check_interval = 15  # seconds
        self.metrics_retention = 3600  # 1 hour in seconds
        
        # Alert rules
        self.alert_rules = self._initialize_alert_rules()
        
        # Performance baselines
        self.baselines = {
            "agent_response_time": 5.0,  # seconds
            "task_completion_rate": 0.9,  # 90%
            "system_cpu_usage": 0.8,  # 80%
            "system_memory_usage": 0.8,  # 80%
            "workflow_failure_rate": 0.1,  # 10%
            "message_delivery_rate": 0.95  # 95%
        }
        
        # Component references (to be set by orchestrator)
        self.orchestrator = None
        self.task_router = None
        self.workflow_engine = None
        self.message_bus = None
        self.agent_discovery = None
        self.coordinator = None
    
    async def initialize(self):
        """Initialize the monitoring system"""
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            await self.redis_client.ping()
            
            # Start monitoring tasks
            asyncio.create_task(self._metrics_collector())
            asyncio.create_task(self._health_checker())
            asyncio.create_task(self._alert_processor())
            asyncio.create_task(self._data_cleanup())
            
            logger.info("Orchestration monitoring system initialized")
            
        except Exception as e:
            logger.error(f"Monitoring system initialization failed: {e}")
            raise
    
    def set_components(self, **components):
        """Set component references for monitoring"""
        for name, component in components.items():
            setattr(self, name, component)
    
    def _initialize_alert_rules(self) -> Dict[str, Dict]:
        """Initialize alert rules"""
        return {
            "high_cpu_usage": {
                "metric": "system_cpu_percent",
                "condition": "greater_than",
                "threshold": 80.0,
                "severity": AlertSeverity.WARNING,
                "message": "High CPU usage detected: {value}%"
            },
            "critical_cpu_usage": {
                "metric": "system_cpu_percent",
                "condition": "greater_than",
                "threshold": 95.0,
                "severity": AlertSeverity.CRITICAL,
                "message": "Critical CPU usage detected: {value}%"
            },
            "high_memory_usage": {
                "metric": "system_memory_percent",
                "condition": "greater_than",
                "threshold": 85.0,
                "severity": AlertSeverity.WARNING,
                "message": "High memory usage detected: {value}%"
            },
            "agent_failure_rate": {
                "metric": "agent_failure_rate",
                "condition": "greater_than",
                "threshold": 0.2,
                "severity": AlertSeverity.ERROR,
                "message": "High agent failure rate: {value}%"
            },
            "workflow_failure_rate": {
                "metric": "workflow_failure_rate",
                "condition": "greater_than",
                "threshold": 0.15,
                "severity": AlertSeverity.ERROR,
                "message": "High workflow failure rate: {value}%"
            },
            "slow_response_time": {
                "metric": "avg_agent_response_time",
                "condition": "greater_than",
                "threshold": 10.0,
                "severity": AlertSeverity.WARNING,
                "message": "Slow agent response time: {value}s"
            },
            "message_delivery_failure": {
                "metric": "message_delivery_rate",
                "condition": "less_than",
                "threshold": 0.9,
                "severity": AlertSeverity.ERROR,
                "message": "Low message delivery rate: {value}%"
            },
            "no_healthy_agents": {
                "metric": "healthy_agents_count",
                "condition": "less_than",
                "threshold": 1,
                "severity": AlertSeverity.CRITICAL,
                "message": "No healthy agents available"
            }
        }
    
    async def _metrics_collector(self):
        """Main metrics collection loop"""
        while True:
            try:
                await self._collect_system_metrics()
                await self._collect_orchestration_metrics()
                await self._collect_component_metrics()
                
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            await self.record_metric("system_cpu_percent", cpu_percent, MetricType.GAUGE)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            await self.record_metric("system_memory_percent", memory.percent, MetricType.GAUGE)
            await self.record_metric("system_memory_used_gb", memory.used / (1024**3), MetricType.GAUGE)
            await self.record_metric("system_memory_available_gb", memory.available / (1024**3), MetricType.GAUGE)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            await self.record_metric("system_disk_percent", (disk.used / disk.total) * 100, MetricType.GAUGE)
            await self.record_metric("system_disk_free_gb", disk.free / (1024**3), MetricType.GAUGE)
            
            # Network metrics
            network = psutil.net_io_counters()
            await self.record_metric("network_bytes_sent", network.bytes_sent, MetricType.COUNTER)
            await self.record_metric("network_bytes_recv", network.bytes_recv, MetricType.COUNTER)
            
            # Process metrics
            process_count = len(psutil.pids())
            await self.record_metric("system_process_count", process_count, MetricType.GAUGE)
            
        except Exception as e:
            logger.error(f"System metrics collection failed: {e}")
    
    async def _collect_orchestration_metrics(self):
        """Collect orchestration-specific metrics"""
        try:
            # Agent metrics
            if self.agent_discovery:
                agents = await self.agent_discovery.get_discovered_agents()
                healthy_agents = [a for a in agents if a.status == "healthy"]
                
                await self.record_metric("total_agents_count", len(agents), MetricType.GAUGE)
                await self.record_metric("healthy_agents_count", len(healthy_agents), MetricType.GAUGE)
                await self.record_metric("unhealthy_agents_count", len(agents) - len(healthy_agents), MetricType.GAUGE)
                
                if agents:
                    agent_failure_rate = (len(agents) - len(healthy_agents)) / len(agents)
                    await self.record_metric("agent_failure_rate", agent_failure_rate, MetricType.GAUGE)
            
            # Task routing metrics
            if self.task_router:
                queue_status = await self.task_router.get_queue_status()
                await self.record_metric("task_queue_size", queue_status.get("queue_size", 0), MetricType.GAUGE)
                
                metrics = queue_status.get("metrics", {})
                await self.record_metric("tasks_routed_total", metrics.get("tasks_routed", 0), MetricType.COUNTER)
                await self.record_metric("routing_failures_total", metrics.get("routing_failures", 0), MetricType.COUNTER)
                await self.record_metric("avg_routing_time", metrics.get("avg_routing_time", 0), MetricType.GAUGE)
                
                if metrics.get("tasks_routed", 0) > 0:
                    success_rate = 1 - (metrics.get("routing_failures", 0) / metrics.get("tasks_routed", 1))
                    await self.record_metric("routing_success_rate", success_rate, MetricType.GAUGE)
            
            # Workflow metrics
            if self.workflow_engine:
                workflow_metrics = await self.workflow_engine.get_metrics()
                
                await self.record_metric("workflows_executed_total", workflow_metrics.get("workflows_executed", 0), MetricType.COUNTER)
                await self.record_metric("workflows_completed_total", workflow_metrics.get("workflows_completed", 0), MetricType.COUNTER)
                await self.record_metric("workflows_failed_total", workflow_metrics.get("workflows_failed", 0), MetricType.COUNTER)
                await self.record_metric("avg_workflow_execution_time", workflow_metrics.get("avg_execution_time", 0), MetricType.GAUGE)
                await self.record_metric("active_workflows_count", workflow_metrics.get("active_workflows", 0), MetricType.GAUGE)
                
                if workflow_metrics.get("workflows_executed", 0) > 0:
                    failure_rate = workflow_metrics.get("workflows_failed", 0) / workflow_metrics.get("workflows_executed", 1)
                    await self.record_metric("workflow_failure_rate", failure_rate, MetricType.GAUGE)
            
            # Message bus metrics
            if self.message_bus:
                bus_metrics = await self.message_bus.get_metrics()
                
                await self.record_metric("messages_sent_total", bus_metrics.get("messages_sent", 0), MetricType.COUNTER)
                await self.record_metric("messages_received_total", bus_metrics.get("messages_received", 0), MetricType.COUNTER)
                await self.record_metric("messages_failed_total", bus_metrics.get("messages_failed", 0), MetricType.COUNTER)
                
                if bus_metrics.get("messages_sent", 0) > 0:
                    delivery_rate = 1 - (bus_metrics.get("messages_failed", 0) / bus_metrics.get("messages_sent", 1))
                    await self.record_metric("message_delivery_rate", delivery_rate, MetricType.GAUGE)
            
            # Coordination metrics
            if self.coordinator:
                coord_metrics = await self.coordinator.get_coordination_metrics()
                
                await self.record_metric("consensus_sessions_completed", coord_metrics.get("consensus_sessions_completed", 0), MetricType.COUNTER)
                await self.record_metric("consensus_success_rate", coord_metrics.get("consensus_success_rate", 0), MetricType.GAUGE)
                await self.record_metric("avg_consensus_time", coord_metrics.get("avg_consensus_time", 0), MetricType.GAUGE)
                await self.record_metric("resource_allocations_total", coord_metrics.get("resource_allocations", 0), MetricType.COUNTER)
                
        except Exception as e:
            logger.error(f"Orchestration metrics collection failed: {e}")
    
    async def _collect_component_metrics(self):
        """Collect component-specific metrics"""
        try:
            # Calculate response times from health checks
            response_times = [hc.response_time for hc in self.health_checks.values() if hc.response_time > 0]
            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
                await self.record_metric("avg_agent_response_time", avg_response_time, MetricType.GAUGE)
                await self.record_metric("max_agent_response_time", max(response_times), MetricType.GAUGE)
                await self.record_metric("min_agent_response_time", min(response_times), MetricType.GAUGE)
            
        except Exception as e:
            logger.error(f"Component metrics collection failed: {e}")
    
    async def record_metric(self, name: str, value: float, metric_type: MetricType, labels: Dict[str, str] = None):
        """Record a metric value"""
        try:
            metric = Metric(
                name=name,
                type=metric_type,
                value=value,
                timestamp=datetime.now(),
                labels=labels or {}
            )
            
            # Store in memory
            self.metrics[name].append(metric)
            
            # Store in Redis with TTL
            await self.redis_client.setex(
                f"metric:{name}:{int(time.time())}",
                self.metrics_retention,
                json.dumps(asdict(metric), default=str)
            )
            
            # Update latest value
            await self.redis_client.hset(
                "latest_metrics",
                name,
                json.dumps(asdict(metric), default=str)
            )
            
        except Exception as e:
            logger.error(f"Failed to record metric {name}: {e}")
    
    async def _health_checker(self):
        """Perform health checks on components"""
        while True:
            try:
                await self._check_agent_health()
                await self._check_component_health()
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Health checker error: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _check_agent_health(self):
        """Check health of discovered agents"""
        if not self.agent_discovery:
            return
        
        try:
            agents = await self.agent_discovery.get_discovered_agents()
            
            for agent in agents:
                start_time = time.time()
                
                try:
                    # Simulate health check (in production, call actual health endpoint)
                    await asyncio.sleep(0.1)  # Simulate network call
                    
                    response_time = time.time() - start_time
                    
                    health_check = HealthCheck(
                        component=f"agent_{agent.id}",
                        status="healthy" if agent.status == "healthy" else "unhealthy",
                        last_check=datetime.now(),
                        response_time=response_time,
                        metadata={"agent_type": agent.type, "endpoint": agent.endpoint}
                    )
                    
                except Exception as e:
                    health_check = HealthCheck(
                        component=f"agent_{agent.id}",
                        status="error",
                        last_check=datetime.now(),
                        response_time=0.0,
                        error_message=str(e),
                        metadata={"agent_type": agent.type, "endpoint": agent.endpoint}
                    )
                
                self.health_checks[f"agent_{agent.id}"] = health_check
            
        except Exception as e:
            logger.error(f"Agent health check failed: {e}")
    
    async def _check_component_health(self):
        """Check health of orchestration components"""
        components = {
            "orchestrator": self.orchestrator,
            "task_router": self.task_router,
            "workflow_engine": self.workflow_engine,
            "message_bus": self.message_bus,
            "agent_discovery": self.agent_discovery,
            "coordinator": self.coordinator
        }
        
        for name, component in components.items():
            start_time = time.time()
            
            try:
                if component and hasattr(component, 'health_check'):
                    is_healthy = component.health_check()
                elif component:
                    is_healthy = True
                else:
                    is_healthy = False
                
                response_time = time.time() - start_time
                
                health_check = HealthCheck(
                    component=name,
                    status="healthy" if is_healthy else "unhealthy",
                    last_check=datetime.now(),
                    response_time=response_time
                )
                
            except Exception as e:
                health_check = HealthCheck(
                    component=name,
                    status="error",
                    last_check=datetime.now(),
                    response_time=0.0,
                    error_message=str(e)
                )
            
            self.health_checks[name] = health_check
    
    async def _alert_processor(self):
        """Process alerts based on metrics and rules"""
        while True:
            try:
                await self._evaluate_alert_rules()
                await self._cleanup_resolved_alerts()
                
                await asyncio.sleep(self.alert_check_interval)
                
            except Exception as e:
                logger.error(f"Alert processor error: {e}")
                await asyncio.sleep(self.alert_check_interval)
    
    async def _evaluate_alert_rules(self):
        """Evaluate alert rules against current metrics"""
        for rule_name, rule in self.alert_rules.items():
            try:
                metric_name = rule["metric"]
                condition = rule["condition"]
                threshold = rule["threshold"]
                severity = rule["severity"]
                message_template = rule["message"]
                
                # Get latest metric value
                if metric_name in self.metrics and self.metrics[metric_name]:
                    latest_metric = self.metrics[metric_name][-1]
                    value = latest_metric.value
                    
                    # Evaluate condition
                    alert_triggered = False
                    if condition == "greater_than" and value > threshold:
                        alert_triggered = True
                    elif condition == "less_than" and value < threshold:
                        alert_triggered = True
                    elif condition == "equals" and value == threshold:
                        alert_triggered = True
                    
                    if alert_triggered:
                        # Check if alert already exists
                        alert_id = f"{rule_name}_{metric_name}"
                        if alert_id not in self.alerts or self.alerts[alert_id].resolved:
                            await self._create_alert(
                                alert_id=alert_id,
                                name=rule_name,
                                severity=severity,
                                message=message_template.format(value=value),
                                source="monitoring_system",
                                metadata={"metric": metric_name, "value": value, "threshold": threshold}
                            )
                    else:
                        # Resolve alert if it exists
                        alert_id = f"{rule_name}_{metric_name}"
                        if alert_id in self.alerts and not self.alerts[alert_id].resolved:
                            await self._resolve_alert(alert_id)
                
            except Exception as e:
                logger.error(f"Alert rule evaluation failed for {rule_name}: {e}")
    
    async def _create_alert(self, alert_id: str, name: str, severity: AlertSeverity, 
                           message: str, source: str, metadata: Dict[str, Any] = None):
        """Create a new alert"""
        alert = Alert(
            id=alert_id,
            name=name,
            severity=severity,
            message=message,
            timestamp=datetime.now(),
            source=source,
            metadata=metadata or {}
        )
        
        self.alerts[alert_id] = alert
        
        # Store in Redis
        await self.redis_client.hset(
            "active_alerts",
            alert_id,
            json.dumps(asdict(alert), default=str)
        )
        
        # Publish alert
        await self.redis_client.publish(
            "orchestration_alerts",
            json.dumps(asdict(alert), default=str)
        )
        
        logger.warning(f"Alert created: {name} - {message}")
    
    async def _resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        if alert_id in self.alerts:
            self.alerts[alert_id].resolved = True
            
            # Remove from active alerts
            await self.redis_client.hdel("active_alerts", alert_id)
            
            # Store in resolved alerts
            await self.redis_client.hset(
                "resolved_alerts",
                alert_id,
                json.dumps(asdict(self.alerts[alert_id]), default=str)
            )
            
            logger.info(f"Alert resolved: {alert_id}")
    
    async def _cleanup_resolved_alerts(self):
        """Clean up old resolved alerts"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        resolved_to_remove = []
        for alert_id, alert in self.alerts.items():
            if alert.resolved and alert.timestamp < cutoff_time:
                resolved_to_remove.append(alert_id)
        
        for alert_id in resolved_to_remove:
            del self.alerts[alert_id]
    
    async def _data_cleanup(self):
        """Clean up old metrics data"""
        while True:
            try:
                # Clean up old metrics from memory
                cutoff_time = datetime.now() - timedelta(seconds=self.metrics_retention)
                
                for metric_name, metric_deque in self.metrics.items():
                    # Remove old metrics
                    while metric_deque and metric_deque[0].timestamp < cutoff_time:
                        metric_deque.popleft()
                
                await asyncio.sleep(300)  # Clean up every 5 minutes
                
            except Exception as e:
                logger.error(f"Data cleanup error: {e}")
                await asyncio.sleep(300)
    
    # Public API
    
    async def get_metrics(self, metric_name: str = None, 
                         time_range: int = 3600) -> Dict[str, Any]:
        """Get metrics data"""
        if metric_name:
            if metric_name in self.metrics:
                cutoff_time = datetime.now() - timedelta(seconds=time_range)
                recent_metrics = [
                    asdict(m) for m in self.metrics[metric_name]
                    if m.timestamp > cutoff_time
                ]
                return {metric_name: recent_metrics}
            else:
                return {}
        else:
            # Return all metrics
            cutoff_time = datetime.now() - timedelta(seconds=time_range)
            all_metrics = {}
            
            for name, metric_deque in self.metrics.items():
                recent_metrics = [
                    asdict(m) for m in metric_deque
                    if m.timestamp > cutoff_time
                ]
                if recent_metrics:
                    all_metrics[name] = recent_metrics
            
            return all_metrics
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard"""
        # Get latest metrics
        latest_metrics = {}
        for name, metric_deque in self.metrics.items():
            if metric_deque:
                latest_metrics[name] = metric_deque[-1].value
        
        # Get health status
        health_status = {}
        for name, health_check in self.health_checks.items():
            health_status[name] = {
                "status": health_check.status,
                "response_time": health_check.response_time,
                "last_check": health_check.last_check.isoformat(),
                "error_message": health_check.error_message
            }
        
        # Get active alerts
        active_alerts = [
            asdict(alert) for alert in self.alerts.values()
            if not alert.resolved
        ]
        
        # Calculate summary statistics
        summary = {
            "total_agents": latest_metrics.get("total_agents_count", 0),
            "healthy_agents": latest_metrics.get("healthy_agents_count", 0),
            "active_workflows": latest_metrics.get("active_workflows_count", 0),
            "task_queue_size": latest_metrics.get("task_queue_size", 0),
            "system_cpu_usage": latest_metrics.get("system_cpu_percent", 0),
            "system_memory_usage": latest_metrics.get("system_memory_percent", 0),
            "alert_count": len(active_alerts)
        }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "latest_metrics": latest_metrics,
            "health_status": health_status,
            "active_alerts": active_alerts
        }
    
    async def get_alerts(self, resolved: bool = False) -> List[Dict[str, Any]]:
        """Get alerts"""
        alerts = [
            asdict(alert) for alert in self.alerts.values()
            if alert.resolved == resolved
        ]
        return sorted(alerts, key=lambda x: x["timestamp"], reverse=True)
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all components"""
        return {
            name: {
                "status": health_check.status,
                "response_time": health_check.response_time,
                "last_check": health_check.last_check.isoformat(),
                "error_message": health_check.error_message
            }
            for name, health_check in self.health_checks.items()
        }
    
    async def stop(self):
        """Stop the monitoring system"""
        if self.redis_client:
            await self.redis_client.close()
        logger.info("Orchestration monitoring system stopped")

# Singleton instance
orchestration_monitor = OrchestrationMonitor()