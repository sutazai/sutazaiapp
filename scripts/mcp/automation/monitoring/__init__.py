"""
MCP Automation Monitoring Package
Comprehensive monitoring and health check system for MCP infrastructure
"""

from .metrics_collector import MCPMetricsCollector, MCPServerMetrics, AutomationMetrics
from .health_monitor import MCPHealthMonitor, HealthStatus, HealthCheck, ComponentHealth, SystemHealth
from .alert_manager import AlertManager, Alert, AlertSeverity, AlertState, NotificationChannel
from .dashboard_config import DashboardManager, Dashboard, Panel
from .log_aggregator import LogAggregator, LogEntry, LogLevel, LogSource
from .sla_monitor import SLAMonitor, SLI, SLO, SLAMeasurement, SLAReport, ComplianceStatus

__version__ = "1.0.0"
__author__ = "MCP Automation Team"

__all__ = [
    # Metrics
    "MCPMetricsCollector",
    "MCPServerMetrics",
    "AutomationMetrics",
    
    # Health
    "MCPHealthMonitor",
    "HealthStatus",
    "HealthCheck",
    "ComponentHealth",
    "SystemHealth",
    
    # Alerts
    "AlertManager",
    "Alert",
    "AlertSeverity",
    "AlertState",
    "NotificationChannel",
    
    # Dashboards
    "DashboardManager",
    "Dashboard",
    "Panel",
    
    # Logs
    "LogAggregator",
    "LogEntry",
    "LogLevel",
    "LogSource",
    
    # SLA
    "SLAMonitor",
    "SLI",
    "SLO",
    "SLAMeasurement",
    "SLAReport",
    "ComplianceStatus"
]