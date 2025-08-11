"""
Agent Health Monitoring and Management System
Provides comprehensive health monitoring, performance tracking, and automated recovery.
"""

import asyncio
import json
import logging
import time
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union, Callable, Awaitable
import aiohttp
import psutil
from universal_client import UniversalAgentClient, AgentInfo, AgentStatus
from discovery_service import DiscoveryService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class RecoveryAction(Enum):
    """Automated recovery actions."""
    NONE = "none"
    RESTART = "restart"
    SCALE_UP = "scale_up"
    FAILOVER = "failover"
    NOTIFY_ADMIN = "notify_admin"


@dataclass
class HealthMetric:
    """Individual health metric."""
    name: str
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.now)
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    description: str = ""
    
    def get_status(self) -> HealthStatus:
        """Get health status based on thresholds."""
        if self.threshold_critical is not None and self.value >= self.threshold_critical:
            return HealthStatus.CRITICAL
        elif self.threshold_warning is not None and self.value >= self.threshold_warning:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY


@dataclass
class HealthCheck:
    """Health check result for an agent."""
    agent_id: str
    timestamp: datetime
    status: HealthStatus
    response_time: float
    metrics: Dict[str, HealthMetric] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_metric(self, metric: HealthMetric):
        """Add a health metric."""
        self.metrics[metric.name] = metric
        
        # Update overall status based on metric
        metric_status = metric.get_status()
        if metric_status == HealthStatus.CRITICAL and self.status != HealthStatus.CRITICAL:
            self.status = HealthStatus.CRITICAL
        elif metric_status == HealthStatus.WARNING and self.status == HealthStatus.HEALTHY:
            self.status = HealthStatus.WARNING
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "response_time": self.response_time,
            "metrics": {name: {
                "name": metric.name,
                "value": metric.value,
                "unit": metric.unit,
                "status": metric.get_status().value,
                "description": metric.description
            } for name, metric in self.metrics.items()},
            "errors": self.errors,
            "warnings": self.warnings,
            "metadata": self.metadata
        }


@dataclass
class Alert:
    """Health monitoring alert."""
    id: str
    agent_id: str
    severity: AlertSeverity
    title: str
    description: str
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    recovery_actions: List[RecoveryAction] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
        """Mark alert as resolved."""
        self.resolved = True
        self.resolved_at = datetime.now()


@dataclass
class AgentHealthProfile:
    """Comprehensive health profile for an agent."""
    agent_id: str
    agent_info: AgentInfo
    current_status: HealthStatus = HealthStatus.UNKNOWN
    last_check: Optional[datetime] = None
    last_healthy: Optional[datetime] = None
    uptime_start: Optional[datetime] = None
    
    # Performance metrics
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    success_rate: float = 100.0
    error_count: int = 0
    total_requests: int = 0
    
    # Resource usage
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    network_io: Dict[str, float] = field(default_factory=dict)
    
    # Health history
    health_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    active_alerts: List[Alert] = field(default_factory=list)
    resolved_alerts: List[Alert] = field(default_factory=list)
    
    # Configuration
    check_interval: int = 30  # seconds
    timeout: int = 10  # seconds
    failure_threshold: int = 3
    recovery_threshold: int = 2
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    
    def add_health_check(self, health_check: HealthCheck):
        """Add health check result."""
        self.health_history.append(health_check)
        self.last_check = health_check.timestamp
        self.current_status = health_check.status
        
        if health_check.status == HealthStatus.HEALTHY:
            self.last_healthy = health_check.timestamp
            self.consecutive_successes += 1
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
            self.consecutive_successes = 0
        
        # Update response time
        self.response_times.append(health_check.response_time)
    
    def get_average_response_time(self, window_minutes: int = 10) -> float:
        """Get average response time in the specified window."""
        if not self.response_times:
            return 0.0
        
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        recent_checks = [
            check for check in self.health_history
            if check.timestamp >= cutoff_time
        ]
        
        if not recent_checks:
            return statistics.mean(self.response_times)
        
        return statistics.mean([check.response_time for check in recent_checks])
    
    def get_uptime_percentage(self, window_hours: int = 24) -> float:
        """Calculate uptime percentage in the specified window."""
        if not self.health_history:
            return 0.0
        
        cutoff_time = datetime.now() - timedelta(hours=window_hours)
        recent_checks = [
            check for check in self.health_history
            if check.timestamp >= cutoff_time
        ]
        
        if not recent_checks:
            return 0.0
        
        healthy_checks = [
            check for check in recent_checks
            if check.status == HealthStatus.HEALTHY
        ]
        
        return (len(healthy_checks) / len(recent_checks)) * 100
    
    def is_failing(self) -> bool:
        """Check if agent is in a failing state."""
        return self.consecutive_failures >= self.failure_threshold
    
    def is_recovered(self) -> bool:
        """Check if agent has recovered."""
        return self.consecutive_successes >= self.recovery_threshold


class AgentHealthChecker:
    """Performs health checks on individual agents."""
    
    def __init__(self, universal_client: UniversalAgentClient):
        self.universal_client = universal_client
        self.custom_checks: Dict[str, Callable] = {}
    
    async def check_agent_health(self, agent_info: AgentInfo) -> HealthCheck:
        """Perform comprehensive health check on an agent."""
        start_time = time.time()
        health_check = HealthCheck(
            agent_id=agent_info.id,
            timestamp=datetime.now(),
            status=HealthStatus.HEALTHY,
            response_time=0.0
        )
        
        try:
            # Basic connectivity check
            is_reachable = await self._check_connectivity(agent_info)
            response_time = time.time() - start_time
            health_check.response_time = response_time
            
            if not is_reachable:
                health_check.status = HealthStatus.UNHEALTHY
                health_check.errors.append("Agent not reachable")
                return health_check
            
            # Response time metric
            response_metric = HealthMetric(
                name="response_time",
                value=response_time * 1000,  # Convert to milliseconds
                unit="ms",
                threshold_warning=1000.0,  # 1 second
                threshold_critical=5000.0,  # 5 seconds
                description="HTTP response time"
            )
            health_check.add_metric(response_metric)
            
            # Check agent-specific endpoints
            await self._check_agent_endpoints(agent_info, health_check)
            
            # Check resource usage
            await self._check_resource_usage(agent_info, health_check)
            
            # Custom health checks
            if agent_info.id in self.custom_checks:
                await self.custom_checks[agent_info.id](agent_info, health_check)
            
        except Exception as e:
            health_check.status = HealthStatus.UNHEALTHY
            health_check.errors.append(f"Health check failed: {str(e)}")
            logger.error(f"Health check failed for {agent_info.id}: {str(e)}")
        
        return health_check
    
    async def _check_connectivity(self, agent_info: AgentInfo) -> bool:
        """Check basic connectivity to agent."""
        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(agent_info.health_check_url) as response:
                    return response.status == 200
        except Exception:
            return False
    
    async def _check_agent_endpoints(self, agent_info: AgentInfo, health_check: HealthCheck):
        """Check agent-specific endpoints."""
        try:
            # Check capabilities endpoint
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"{agent_info.endpoint}/capabilities") as response:
                    if response.status == 200:
                        capabilities_data = await response.json()
                        health_check.metadata["capabilities"] = capabilities_data
                    else:
                        health_check.warnings.append("Capabilities endpoint returned non-200 status")
                
                # Check metrics endpoint if available
                try:
                    async with session.get(f"{agent_info.endpoint}/metrics") as response:
                        if response.status == 200:
                            metrics_data = await response.json()
                            await self._process_agent_metrics(metrics_data, health_check)
                except aiohttp.ClientError:
                    # Metrics endpoint not available - not critical
                    pass
                    
        except Exception as e:
            health_check.warnings.append(f"Endpoint check failed: {str(e)}")
    
    async def _process_agent_metrics(self, metrics_data: Dict[str, Any], health_check: HealthCheck):
        """Process metrics returned by agent."""
        for metric_name, metric_info in metrics_data.items():
            if isinstance(metric_info, dict) and "value" in metric_info:
                metric = HealthMetric(
                    name=metric_name,
                    value=float(metric_info["value"]),
                    unit=metric_info.get("unit", ""),
                    threshold_warning=metric_info.get("warning_threshold"),
                    threshold_critical=metric_info.get("critical_threshold"),
                    description=metric_info.get("description", "")
                )
                health_check.add_metric(metric)
    
    async def _check_resource_usage(self, agent_info: AgentInfo, health_check: HealthCheck):
        """Check system resource usage."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_metric = HealthMetric(
                name="cpu_usage",
                value=cpu_percent,
                unit="%",
                threshold_warning=80.0,
                threshold_critical=95.0,
                description="CPU usage percentage"
            )
            health_check.add_metric(cpu_metric)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_metric = HealthMetric(
                name="memory_usage",
                value=memory.percent,
                unit="%",
                threshold_warning=80.0,
                threshold_critical=95.0,
                description="Memory usage percentage"
            )
            health_check.add_metric(memory_metric)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_metric = HealthMetric(
                name="disk_usage",
                value=disk.percent,
                unit="%",
                threshold_warning=80.0,
                threshold_critical=95.0,
                description="Disk usage percentage"
            )
            health_check.add_metric(disk_metric)
            
            # Network I/O
            network = psutil.net_io_counters()
            if network:
                health_check.metadata["network_io"] = {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv
                }
            
        except Exception as e:
            health_check.warnings.append(f"Resource check failed: {str(e)}")
    
    def register_custom_check(self, agent_id: str, check_function: Callable):
        """Register custom health check for specific agent."""
        self.custom_checks[agent_id] = check_function


class AlertManager:
    """Manages health monitoring alerts."""
    
    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.alert_handlers: Dict[AlertSeverity, List[Callable]] = defaultdict(list)
        self.alert_history: deque = deque(maxlen=10000)
        self.notification_channels: List[Callable] = []
        
    def create_alert(
        self,
        agent_id: str,
        severity: AlertSeverity,
        title: str,
        description: str,
        recovery_actions: Optional[List[RecoveryAction]] = None
    ) -> Alert:
        """Create a new alert."""
        alert_id = f"{agent_id}_{int(time.time())}"
        alert = Alert(
            id=alert_id,
            agent_id=agent_id,
            severity=severity,
            title=title,
            description=description,
            recovery_actions=recovery_actions or []
        )
        
        self.alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Trigger alert handlers
        asyncio.create_task(self._process_alert(alert))
        
        logger.warning(f"Alert created: {title} for agent {agent_id}")
        return alert
    
    async def _process_alert(self, alert: Alert):
        """Process alert through handlers."""
        # Execute severity-specific handlers
        for handler in self.alert_handlers[alert.severity]:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {str(e)}")
        
        # Send notifications
        for channel in self.notification_channels:
            try:
                await channel(alert)
            except Exception as e:
                logger.error(f"Notification failed: {str(e)}")
    
        """Resolve an alert."""
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            logger.info(f"Alert resolved: {alert.title}")
    
    def get_active_alerts(self, agent_id: Optional[str] = None) -> List[Alert]:
        """Get active alerts, optionally filtered by agent."""
        active_alerts = [
            alert for alert in self.alerts.values()
            if not alert.resolved
        ]
        
        if agent_id:
            active_alerts = [
                alert for alert in active_alerts
                if alert.agent_id == agent_id
            ]
        
        return sorted(active_alerts, key=lambda x: x.timestamp, reverse=True)
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        active_alerts = self.get_active_alerts()
        resolved_alerts = [alert for alert in self.alerts.values() if alert.resolved]
        
        severity_counts = defaultdict(int)
        for alert in active_alerts:
            severity_counts[alert.severity.value] += 1
        
        return {
            "total_alerts": len(self.alerts),
            "active_alerts": len(active_alerts),
            "resolved_alerts": len(resolved_alerts),
            "severity_breakdown": dict(severity_counts),
            "resolution_rate": len(resolved_alerts) / len(self.alerts) * 100 if self.alerts else 0
        }
    
    def register_alert_handler(self, severity: AlertSeverity, handler: Callable):
        """Register alert handler for specific severity."""
        self.alert_handlers[severity].append(handler)
    
    def register_notification_channel(self, channel: Callable):
        """Register notification channel."""
        self.notification_channels.append(channel)


class RecoveryManager:
    """Manages automated recovery actions."""
    
    def __init__(self, universal_client: UniversalAgentClient):
        self.universal_client = universal_client
        self.recovery_strategies: Dict[str, List[RecoveryAction]] = {}
        self.recovery_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
    def configure_recovery_strategy(self, agent_id: str, actions: List[RecoveryAction]):
        """Configure recovery strategy for an agent."""
        self.recovery_strategies[agent_id] = actions
    
    async def execute_recovery(self, agent_id: str, health_profile: AgentHealthProfile) -> bool:
        """Execute recovery actions for a failing agent."""
        if agent_id not in self.recovery_strategies:
            logger.warning(f"No recovery strategy configured for {agent_id}")
            return False
        
        actions = self.recovery_strategies[agent_id]
        recovery_log = {
            "timestamp": datetime.now().isoformat(),
            "agent_id": agent_id,
            "actions_attempted": [],
            "success": False
        }
        
        for action in actions:
            try:
                success = await self._execute_action(action, agent_id, health_profile)
                recovery_log["actions_attempted"].append({
                    "action": action.value,
                    "success": success,
                    "timestamp": datetime.now().isoformat()
                })
                
                if success:
                    recovery_log["success"] = True
                    logger.info(f"Recovery action {action.value} succeeded for {agent_id}")
                    break
                else:
                    logger.warning(f"Recovery action {action.value} failed for {agent_id}")
                    
            except Exception as e:
                logger.error(f"Recovery action {action.value} failed for {agent_id}: {str(e)}")
                recovery_log["actions_attempted"].append({
                    "action": action.value,
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        self.recovery_history[agent_id].append(recovery_log)
        return recovery_log["success"]
    
    async def _execute_action(
        self,
        action: RecoveryAction,
        agent_id: str,
        health_profile: AgentHealthProfile
    ) -> bool:
        """Execute a specific recovery action."""
        if action == RecoveryAction.RESTART:
            return await self._restart_agent(agent_id, health_profile)
        elif action == RecoveryAction.SCALE_UP:
            return await self._scale_up_agent(agent_id, health_profile)
        elif action == RecoveryAction.FAILOVER:
            return await self._failover_agent(agent_id, health_profile)
        elif action == RecoveryAction.NOTIFY_ADMIN:
            return await self._notify_admin(agent_id, health_profile)
        else:
            return False
    
    async def _restart_agent(self, agent_id: str, health_profile: AgentHealthProfile) -> bool:
        """Restart an agent."""
        try:
            # This would integrate with container orchestration or process management
            logger.info(f"Attempting to restart agent {agent_id}")
            
            # Simulate restart (replace with actual restart logic)
            await asyncio.sleep(2)
            
            # Verify agent is back online
            await asyncio.sleep(5)
            
            # Check if agent responds
            try:
                timeout = aiohttp.ClientTimeout(total=10)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(health_profile.agent_info.health_check_url) as response:
                        return response.status == 200
            except Exception:
                return False
                
        except Exception as e:
            logger.error(f"Failed to restart agent {agent_id}: {str(e)}")
            return False
    
    async def _scale_up_agent(self, agent_id: str, health_profile: AgentHealthProfile) -> bool:
        """Scale up agent resources."""
        try:
            logger.info(f"Attempting to scale up agent {agent_id}")
            
            # This would integrate with container orchestration
            # For now, just log the action
            await asyncio.sleep(1)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to scale up agent {agent_id}: {str(e)}")
            return False
    
    async def _failover_agent(self, agent_id: str, health_profile: AgentHealthProfile) -> bool:
        """Failover to backup agent."""
        try:
            logger.info(f"Attempting failover for agent {agent_id}")
            
            # This would redirect traffic to a backup agent
            await asyncio.sleep(1)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to failover agent {agent_id}: {str(e)}")
            return False
    
    async def _notify_admin(self, agent_id: str, health_profile: AgentHealthProfile) -> bool:
        """Notify administrator."""
        try:
            logger.critical(f"ADMIN NOTIFICATION: Agent {agent_id} requires manual intervention")
            
            # This would send email, Slack message, etc.
            await asyncio.sleep(0.1)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to notify admin for agent {agent_id}: {str(e)}")
            return False
    
    def get_recovery_history(self, agent_id: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Get recovery history."""
        if agent_id:
            return {agent_id: self.recovery_history.get(agent_id, [])}
        return dict(self.recovery_history)


class HealthMonitor:
    """Main health monitoring system."""
    
    def __init__(
        self,
        universal_client: UniversalAgentClient,
        discovery_service: DiscoveryService
    ):
        self.universal_client = universal_client
        self.discovery_service = discovery_service
        self.health_checker = AgentHealthChecker(universal_client)
        self.alert_manager = AlertManager()
        self.recovery_manager = RecoveryManager(universal_client)
        
        self.agent_profiles: Dict[str, AgentHealthProfile] = {}
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
        self.is_running = False
        
        # Configuration
        self.default_check_interval = 30  # seconds
        self.batch_size = 10  # agents to check concurrently
        
        # Setup default alert handlers
        self._setup_default_handlers()
    
    def _setup_default_handlers(self):
        """Setup default alert handlers."""
        async def critical_alert_handler(alert: Alert):
            """Handle critical alerts."""
            agent_id = alert.agent_id
            if agent_id in self.agent_profiles:
                profile = self.agent_profiles[agent_id]
                if profile.is_failing():
                    await self.recovery_manager.execute_recovery(agent_id, profile)
        
        async def warning_alert_handler(alert: Alert):
            """Handle warning alerts."""
            logger.warning(f"Warning alert for {alert.agent_id}: {alert.description}")
        
        self.alert_manager.register_alert_handler(AlertSeverity.CRITICAL, critical_alert_handler)
        self.alert_manager.register_alert_handler(AlertSeverity.WARNING, warning_alert_handler)
    
    async def start(self):
        """Start health monitoring."""
        if self.is_running:
            return
        
        self.is_running = True
        logger.info("Starting Health Monitor...")
        
        # Initialize agent profiles
        agents = self.universal_client.list_agents()
        for agent_info in agents:
            profile = AgentHealthProfile(
                agent_id=agent_info.id,
                agent_info=agent_info
            )
            self.agent_profiles[agent_info.id] = profile
            
            # Configure default recovery strategy
            self.recovery_manager.configure_recovery_strategy(
                agent_info.id,
                [RecoveryAction.RESTART, RecoveryAction.NOTIFY_ADMIN]
            )
        
        # Start monitoring loop
        asyncio.create_task(self.monitoring_loop())
        
        logger.info(f"Health Monitor started for {len(self.agent_profiles)} agents")
    
    async def stop(self):
        """Stop health monitoring."""
        if not self.is_running:
            return
        
        self.is_running = False
        logger.info("Stopping Health Monitor...")
        
        # Cancel all monitoring tasks
        for task in self.monitoring_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self.monitoring_tasks:
            await asyncio.gather(*self.monitoring_tasks.values(), return_exceptions=True)
        
        self.monitoring_tasks.clear()
        logger.info("Health Monitor stopped")
    
    async def monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_running:
            try:
                # Get agents to check
                agents_to_check = list(self.agent_profiles.keys())
                
                # Process in batches
                for i in range(0, len(agents_to_check), self.batch_size):
                    batch = agents_to_check[i:i + self.batch_size]
                    
                    # Check agents in parallel
                    check_tasks = []
                    for agent_id in batch:
                        if agent_id in self.agent_profiles:
                            task = asyncio.create_task(
                                self.check_agent_health(agent_id)
                            )
                            check_tasks.append(task)
                    
                    if check_tasks:
                        await asyncio.gather(*check_tasks, return_exceptions=True)
                
                # Wait before next check cycle
                await asyncio.sleep(self.default_check_interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {str(e)}")
                await asyncio.sleep(5)
    
    async def check_agent_health(self, agent_id: str):
        """Check health of a specific agent."""
        if agent_id not in self.agent_profiles:
            return
        
        profile = self.agent_profiles[agent_id]
        
        try:
            # Perform health check
            health_check = await self.health_checker.check_agent_health(profile.agent_info)
            
            # Update profile
            profile.add_health_check(health_check)
            profile.total_requests += 1
            
            if health_check.status == HealthStatus.HEALTHY:
                profile.success_rate = ((profile.success_rate * (profile.total_requests - 1)) + 100) / profile.total_requests
            else:
                profile.error_count += 1
                profile.success_rate = (profile.success_rate * (profile.total_requests - 1)) / profile.total_requests
            
            # Check for alerts
            await self._check_for_alerts(agent_id, health_check, profile)
            
        except Exception as e:
            logger.error(f"Health check failed for {agent_id}: {str(e)}")
            profile.error_count += 1
    
    async def _check_for_alerts(
        self,
        agent_id: str,
        health_check: HealthCheck,
        profile: AgentHealthProfile
    ):
        """Check if alerts should be generated."""
        # Agent becoming unhealthy
        if (health_check.status in [HealthStatus.CRITICAL, HealthStatus.UNHEALTHY] and
            profile.consecutive_failures == profile.failure_threshold):
            
            self.alert_manager.create_alert(
                agent_id=agent_id,
                severity=AlertSeverity.CRITICAL,
                title=f"Agent {agent_id} is unhealthy",
                description=f"Agent has failed {profile.failure_threshold} consecutive health checks",
                recovery_actions=[RecoveryAction.RESTART, RecoveryAction.NOTIFY_ADMIN]
            )
        
        # High response time
        avg_response_time = profile.get_average_response_time()
        if avg_response_time > 5000:  # 5 seconds
            self.alert_manager.create_alert(
                agent_id=agent_id,
                severity=AlertSeverity.WARNING,
                title=f"High response time for {agent_id}",
                description=f"Average response time is {avg_response_time:.0f}ms"
            )
        
        # Low success rate
        if profile.success_rate < 80 and profile.total_requests > 10:
            self.alert_manager.create_alert(
                agent_id=agent_id,
                severity=AlertSeverity.WARNING,
                title=f"Low success rate for {agent_id}",
                description=f"Success rate is {profile.success_rate:.1f}%"
            )
        
        # High resource usage
        for metric_name, metric in health_check.metrics.items():
            if metric.get_status() == HealthStatus.CRITICAL:
                self.alert_manager.create_alert(
                    agent_id=agent_id,
                    severity=AlertSeverity.ERROR,
                    title=f"High {metric_name} for {agent_id}",
                    description=f"{metric_name} is {metric.value}{metric.unit}"
                )
    
    def get_agent_health_summary(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get health summary for an agent."""
        if agent_id not in self.agent_profiles:
            return None
        
        profile = self.agent_profiles[agent_id]
        latest_check = profile.health_history[-1] if profile.health_history else None
        
        return {
            "agent_id": agent_id,
            "current_status": profile.current_status.value,
            "last_check": profile.last_check.isoformat() if profile.last_check else None,
            "last_healthy": profile.last_healthy.isoformat() if profile.last_healthy else None,
            "uptime_percentage": profile.get_uptime_percentage(),
            "average_response_time": profile.get_average_response_time(),
            "success_rate": profile.success_rate,
            "error_count": profile.error_count,
            "total_requests": profile.total_requests,
            "consecutive_failures": profile.consecutive_failures,
            "active_alerts": len([alert for alert in profile.active_alerts if not alert.resolved]),
            "latest_metrics": latest_check.to_dict() if latest_check else None
        }
    
    def get_system_health_overview(self) -> Dict[str, Any]:
        """Get system-wide health overview."""
        total_agents = len(self.agent_profiles)
        healthy_agents = 0
        warning_agents = 0
        critical_agents = 0
        unknown_agents = 0
        
        total_response_time = 0
        total_success_rate = 0
        total_uptime = 0
        
        for profile in self.agent_profiles.values():
            if profile.current_status == HealthStatus.HEALTHY:
                healthy_agents += 1
            elif profile.current_status == HealthStatus.WARNING:
                warning_agents += 1
            elif profile.current_status == HealthStatus.CRITICAL:
                critical_agents += 1
            else:
                unknown_agents += 1
            
            total_response_time += profile.get_average_response_time()
            total_success_rate += profile.success_rate
            total_uptime += profile.get_uptime_percentage()
        
        active_alerts = self.alert_manager.get_active_alerts()
        
        return {
            "total_agents": total_agents,
            "healthy_agents": healthy_agents,
            "warning_agents": warning_agents,
            "critical_agents": critical_agents,
            "unknown_agents": unknown_agents,
            "system_health_percentage": (healthy_agents / total_agents * 100) if total_agents > 0 else 0,
            "average_response_time": total_response_time / total_agents if total_agents > 0 else 0,
            "average_success_rate": total_success_rate / total_agents if total_agents > 0 else 0,
            "average_uptime": total_uptime / total_agents if total_agents > 0 else 0,
            "active_alerts": len(active_alerts),
            "alert_breakdown": {
                "critical": len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]),
                "error": len([a for a in active_alerts if a.severity == AlertSeverity.ERROR]),
                "warning": len([a for a in active_alerts if a.severity == AlertSeverity.WARNING]),
                "info": len([a for a in active_alerts if a.severity == AlertSeverity.INFO])
            }
        }
    
    def get_health_metrics(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get detailed health metrics."""
        if agent_id:
            return self.get_agent_health_summary(agent_id)
        else:
            return self.get_system_health_overview()


# Example usage and testing
if __name__ == "__main__":
    async def main():
        """Example usage of the Health Monitor."""
        
        # Initialize components
        async with UniversalAgentClient() as client:
            discovery = DiscoveryService(client)
            await discovery.start()
            
            try:
                # Initialize health monitor
                monitor = HealthMonitor(client, discovery)
                await monitor.start()
                
                # Run for a while to collect metrics
                logger.info("Health monitoring started - collecting metrics...")
                await asyncio.sleep(60)  # Monitor for 1 minute
                
                # Show system health overview
                overview = monitor.get_system_health_overview()
                print(f"System Health Overview:")
                print(f"- Total agents: {overview['total_agents']}")
                print(f"- Healthy agents: {overview['healthy_agents']}")
                print(f"- System health: {overview['system_health_percentage']:.1f}%")
                print(f"- Average response time: {overview['average_response_time']:.1f}ms")
                print(f"- Active alerts: {overview['active_alerts']}")
                
                # Show individual agent health
                for agent_id in list(monitor.agent_profiles.keys())[:3]:  # Show first 3
                    summary = monitor.get_agent_health_summary(agent_id)
                    if summary:
                        print(f"\nAgent {agent_id}:")
                        print(f"- Status: {summary['current_status']}")
                        print(f"- Uptime: {summary['uptime_percentage']:.1f}%")
                        print(f"- Success rate: {summary['success_rate']:.1f}%")
                        print(f"- Response time: {summary['average_response_time']:.1f}ms")
                
                # Show alert statistics
                alert_stats = monitor.alert_manager.get_alert_statistics()
                print(f"\nAlert Statistics:")
                print(f"- Total alerts: {alert_stats['total_alerts']}")
                print(f"- Active alerts: {alert_stats['active_alerts']}")
                print(f"- Resolution rate: {alert_stats['resolution_rate']:.1f}%")
                
            finally:
                await monitor.stop()
                await discovery.stop()
    
    # Run the example
