#!/usr/bin/env python3
"""
MCP Automation Dashboard Configuration
Grafana dashboard definitions and management
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
import httpx
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Panel:
    """Grafana panel configuration"""
    id: int
    title: str
    type: str  # graph, stat, gauge, table, heatmap, etc.
    datasource: str
    targets: List[Dict[str, Any]]
    gridPos: Dict[str, int]  # x, y, w, h
    options: Dict[str, Any] = field(default_factory=dict)
    fieldConfig: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    transparent: bool = False


@dataclass
class Dashboard:
    """Grafana dashboard configuration"""
    uid: str
    title: str
    tags: List[str]
    panels: List[Panel]
    templating: Dict[str, Any] = field(default_factory=dict)
    annotations: Dict[str, Any] = field(default_factory=dict)
    refresh: str = "10s"
    time: Dict[str, str] = field(default_factory=lambda: {"from": "now-1h", "to": "now"})
    schemaVersion: int = 16
    version: int = 1
    editable: bool = True


class DashboardManager:
    """Manage Grafana dashboards for MCP automation"""
    
    def __init__(self,
                 grafana_url: str = "http://localhost:10201",
                 grafana_api_key: Optional[str] = None):
        """
        Initialize dashboard manager
        
        Args:
            grafana_url: Grafana base URL
            grafana_api_key: Grafana API key for authentication
        """
        self.grafana_url = grafana_url
        self.grafana_api_key = grafana_api_key or os.getenv('GRAFANA_API_KEY')
        
        # HTTP client
        headers = {}
        if self.grafana_api_key:
            headers['Authorization'] = f'Bearer {self.grafana_api_key}'
        self.http_client = httpx.AsyncClient(base_url=grafana_url, headers=headers, timeout=30.0)
        
        # Dashboard definitions
        self.dashboards = self._create_dashboards()
        
    def _create_dashboards(self) -> Dict[str, Dashboard]:
        """Create dashboard definitions"""
        dashboards = {}
        
        # MCP Server Overview Dashboard
        dashboards['mcp_overview'] = self._create_mcp_overview_dashboard()
        
        # Automation Performance Dashboard
        dashboards['automation_performance'] = self._create_automation_performance_dashboard()
        
        # System Health Dashboard
        dashboards['system_health'] = self._create_system_health_dashboard()
        
        # Alert Dashboard
        dashboards['alerts'] = self._create_alert_dashboard()
        
        # SLA Compliance Dashboard
        dashboards['sla_compliance'] = self._create_sla_dashboard()
        
        return dashboards
        
    def _create_mcp_overview_dashboard(self) -> Dashboard:
        """Create MCP Server Overview dashboard"""
        panels = [
            # Server Status Panel
            Panel(
                id=1,
                title="MCP Server Status",
                type="stat",
                datasource="Prometheus",
                targets=[{
                    "expr": "mcp_server_up",
                    "legendFormat": "{{server_name}}",
                    "refId": "A"
                }],
                gridPos={"x": 0, "y": 0, "w": 24, "h": 4},
                options={
                    "orientation": "horizontal",
                    "textMode": "name",
                    "colorMode": "background",
                    "graphMode": "none",
                    "justifyMode": "center"
                },
                fieldConfig={
                    "defaults": {
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {"color": "red", "value": 0},
                                {"color": "green", "value": 1}
                            ]
                        },
                        "mappings": [
                            {"type": "value", "value": "0", "text": "Down"},
                            {"type": "value", "value": "1", "text": "Up"}
                        ]
                    }
                }
            ),
            
            # Request Rate Panel
            Panel(
                id=2,
                title="MCP Request Rate",
                type="graph",
                datasource="Prometheus",
                targets=[{
                    "expr": "rate(mcp_server_requests_total[5m])",
                    "legendFormat": "{{server_name}} - {{method}}",
                    "refId": "A"
                }],
                gridPos={"x": 0, "y": 4, "w": 12, "h": 8},
                options={
                    "legend": {
                        "displayMode": "list",
                        "placement": "bottom"
                    }
                },
                fieldConfig={
                    "defaults": {
                        "unit": "reqps",
                        "custom": {
                            "axisLabel": "Requests/sec",
                            "drawStyle": "line",
                            "lineInterpolation": "smooth",
                            "fillOpacity": 10
                        }
                    }
                }
            ),
            
            # Error Rate Panel
            Panel(
                id=3,
                title="MCP Error Rate",
                type="graph",
                datasource="Prometheus",
                targets=[{
                    "expr": "rate(mcp_server_errors_total[5m])",
                    "legendFormat": "{{server_name}} - {{error_type}}",
                    "refId": "A"
                }],
                gridPos={"x": 12, "y": 4, "w": 12, "h": 8},
                options={
                    "legend": {
                        "displayMode": "list",
                        "placement": "bottom"
                    }
                },
                fieldConfig={
                    "defaults": {
                        "unit": "errors/sec",
                        "custom": {
                            "axisLabel": "Errors/sec",
                            "drawStyle": "line",
                            "lineInterpolation": "smooth",
                            "fillOpacity": 10
                        },
                        "color": {
                            "mode": "palette-classic",
                            "seriesBy": "last"
                        }
                    }
                }
            ),
            
            # Latency Heatmap
            Panel(
                id=4,
                title="MCP Server Latency Heatmap",
                type="heatmap",
                datasource="Prometheus",
                targets=[{
                    "expr": "sum(rate(mcp_server_latency_seconds_bucket[5m])) by (le, server_name)",
                    "format": "heatmap",
                    "legendFormat": "{{le}}",
                    "refId": "A"
                }],
                gridPos={"x": 0, "y": 12, "w": 24, "h": 8},
                options={
                    "calculate": False,
                    "cellGap": 2,
                    "color": {
                        "scheme": "Spectral",
                        "mode": "scheme"
                    }
                }
            ),
            
            # Resource Usage
            Panel(
                id=5,
                title="MCP Server Resource Usage",
                type="graph",
                datasource="Prometheus",
                targets=[
                    {
                        "expr": "mcp_server_memory_bytes / 1024 / 1024",
                        "legendFormat": "{{server_name}} - Memory",
                        "refId": "A"
                    },
                    {
                        "expr": "mcp_server_cpu_percent",
                        "legendFormat": "{{server_name}} - CPU",
                        "refId": "B"
                    }
                ],
                gridPos={"x": 0, "y": 20, "w": 24, "h": 8},
                options={
                    "legend": {
                        "displayMode": "table",
                        "placement": "right"
                    }
                },
                fieldConfig={
                    "defaults": {
                        "custom": {
                            "drawStyle": "line",
                            "lineInterpolation": "smooth",
                            "fillOpacity": 10
                        }
                    },
                    "overrides": [
                        {
                            "matcher": {"id": "byRegexp", "options": ".*Memory.*"},
                            "properties": [
                                {"id": "unit", "value": "mbytes"},
                                {"id": "custom.axisLabel", "value": "Memory (MB)"}
                            ]
                        },
                        {
                            "matcher": {"id": "byRegexp", "options": ".*CPU.*"},
                            "properties": [
                                {"id": "unit", "value": "percent"},
                                {"id": "custom.axisLabel", "value": "CPU (%)"}
                            ]
                        }
                    ]
                }
            )
        ]
        
        return Dashboard(
            uid="mcp-overview",
            title="MCP Server Overview",
            tags=["mcp", "automation", "overview"],
            panels=panels,
            templating={
                "list": [
                    {
                        "name": "server",
                        "type": "query",
                        "datasource": "Prometheus",
                        "query": "label_values(mcp_server_up, server_name)",
                        "refresh": 1,
                        "multi": True,
                        "includeAll": True
                    }
                ]
            }
        )
        
    def _create_automation_performance_dashboard(self) -> Dashboard:
        """Create Automation Performance dashboard"""
        panels = [
            # Execution Summary
            Panel(
                id=1,
                title="Automation Execution Summary",
                type="stat",
                datasource="Prometheus",
                targets=[
                    {
                        "expr": "sum(rate(mcp_automation_executions_total[1h]))",
                        "legendFormat": "Total Executions",
                        "refId": "A"
                    },
                    {
                        "expr": "sum(rate(mcp_automation_executions_total{status='success'}[1h])) / sum(rate(mcp_automation_executions_total[1h])) * 100",
                        "legendFormat": "Success Rate",
                        "refId": "B"
                    }
                ],
                gridPos={"x": 0, "y": 0, "w": 24, "h": 4},
                options={
                    "orientation": "horizontal",
                    "textMode": "value_and_name",
                    "colorMode": "value",
                    "graphMode": "area"
                },
                fieldConfig={
                    "defaults": {
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {"color": "red", "value": 0},
                                {"color": "yellow", "value": 80},
                                {"color": "green", "value": 95}
                            ]
                        }
                    },
                    "overrides": [
                        {
                            "matcher": {"id": "byName", "options": "Success Rate"},
                            "properties": [{"id": "unit", "value": "percent"}]
                        }
                    ]
                }
            ),
            
            # Execution Timeline
            Panel(
                id=2,
                title="Automation Execution Timeline",
                type="graph",
                datasource="Prometheus",
                targets=[
                    {
                        "expr": "sum(rate(mcp_automation_executions_total{status='success'}[5m])) by (workflow_name)",
                        "legendFormat": "{{workflow_name}} - Success",
                        "refId": "A"
                    },
                    {
                        "expr": "sum(rate(mcp_automation_executions_total{status='failed'}[5m])) by (workflow_name)",
                        "legendFormat": "{{workflow_name}} - Failed",
                        "refId": "B"
                    }
                ],
                gridPos={"x": 0, "y": 4, "w": 24, "h": 8},
                options={
                    "legend": {
                        "displayMode": "list",
                        "placement": "bottom"
                    }
                }
            ),
            
            # Queue and Active Tasks
            Panel(
                id=3,
                title="Queue and Active Tasks",
                type="graph",
                datasource="Prometheus",
                targets=[
                    {
                        "expr": "mcp_automation_queue_size",
                        "legendFormat": "{{workflow_name}} - Queue",
                        "refId": "A"
                    },
                    {
                        "expr": "mcp_automation_active_tasks",
                        "legendFormat": "{{workflow_name}} - Active",
                        "refId": "B"
                    }
                ],
                gridPos={"x": 0, "y": 12, "w": 12, "h": 8},
                options={
                    "legend": {
                        "displayMode": "list",
                        "placement": "bottom"
                    }
                }
            ),
            
            # Execution Duration
            Panel(
                id=4,
                title="Execution Duration",
                type="graph",
                datasource="Prometheus",
                targets=[{
                    "expr": "histogram_quantile(0.95, sum(rate(mcp_automation_duration_seconds_bucket[5m])) by (workflow_name, le))",
                    "legendFormat": "{{workflow_name}} - p95",
                    "refId": "A"
                }],
                gridPos={"x": 12, "y": 12, "w": 12, "h": 8},
                fieldConfig={
                    "defaults": {
                        "unit": "s",
                        "custom": {
                            "axisLabel": "Duration (seconds)"
                        }
                    }
                }
            )
        ]
        
        return Dashboard(
            uid="automation-performance",
            title="Automation Performance",
            tags=["automation", "performance", "mcp"],
            panels=panels
        )
        
    def _create_system_health_dashboard(self) -> Dashboard:
        """Create System Health dashboard"""
        panels = [
            # System Overview
            Panel(
                id=1,
                title="System Health Overview",
                type="gauge",
                datasource="Prometheus",
                targets=[
                    {
                        "expr": "100 - mcp_system_cpu_percent",
                        "legendFormat": "CPU Available",
                        "refId": "A"
                    },
                    {
                        "expr": "100 - mcp_system_memory_percent",
                        "legendFormat": "Memory Available",
                        "refId": "B"
                    }
                ],
                gridPos={"x": 0, "y": 0, "w": 12, "h": 8},
                options={
                    "showThresholdLabels": True,
                    "showThresholdMarkers": True
                },
                fieldConfig={
                    "defaults": {
                        "unit": "percent",
                        "min": 0,
                        "max": 100,
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {"color": "red", "value": 0},
                                {"color": "yellow", "value": 20},
                                {"color": "green", "value": 50}
                            ]
                        }
                    }
                }
            ),
            
            # Disk Usage
            Panel(
                id=2,
                title="Disk Usage",
                type="bar-gauge",
                datasource="Prometheus",
                targets=[{
                    "expr": "mcp_system_disk_percent",
                    "legendFormat": "{{mount_point}}",
                    "refId": "A"
                }],
                gridPos={"x": 12, "y": 0, "w": 12, "h": 8},
                options={
                    "orientation": "horizontal",
                    "displayMode": "basic",
                    "showUnfilled": True
                },
                fieldConfig={
                    "defaults": {
                        "unit": "percent",
                        "min": 0,
                        "max": 100,
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {"color": "green", "value": 0},
                                {"color": "yellow", "value": 70},
                                {"color": "red", "value": 90}
                            ]
                        }
                    }
                }
            ),
            
            # Resource Trends
            Panel(
                id=3,
                title="Resource Usage Trends",
                type="graph",
                datasource="Prometheus",
                targets=[
                    {
                        "expr": "mcp_system_cpu_percent",
                        "legendFormat": "CPU %",
                        "refId": "A"
                    },
                    {
                        "expr": "mcp_system_memory_percent",
                        "legendFormat": "Memory %",
                        "refId": "B"
                    }
                ],
                gridPos={"x": 0, "y": 8, "w": 24, "h": 8},
                fieldConfig={
                    "defaults": {
                        "unit": "percent",
                        "custom": {
                            "axisLabel": "Usage %",
                            "drawStyle": "line",
                            "lineInterpolation": "smooth",
                            "fillOpacity": 10
                        }
                    }
                }
            ),
            
            # Network I/O
            Panel(
                id=4,
                title="Network I/O",
                type="graph",
                datasource="Prometheus",
                targets=[
                    {
                        "expr": "rate(mcp_system_network_sent_bytes[5m])",
                        "legendFormat": "Sent",
                        "refId": "A"
                    },
                    {
                        "expr": "rate(mcp_system_network_received_bytes[5m])",
                        "legendFormat": "Received",
                        "refId": "B"
                    }
                ],
                gridPos={"x": 0, "y": 16, "w": 24, "h": 8},
                fieldConfig={
                    "defaults": {
                        "unit": "binBps",
                        "custom": {
                            "axisLabel": "Bytes/sec",
                            "drawStyle": "line",
                            "lineInterpolation": "smooth",
                            "fillOpacity": 10
                        }
                    }
                }
            )
        ]
        
        return Dashboard(
            uid="system-health",
            title="System Health",
            tags=["system", "health", "resources"],
            panels=panels
        )
        
    def _create_alert_dashboard(self) -> Dashboard:
        """Create Alert dashboard"""
        panels = [
            # Alert Summary
            Panel(
                id=1,
                title="Active Alerts",
                type="table",
                datasource="Prometheus",
                targets=[{
                    "expr": "ALERTS{alertstate='firing'}",
                    "format": "table",
                    "instant": True,
                    "refId": "A"
                }],
                gridPos={"x": 0, "y": 0, "w": 24, "h": 8},
                options={
                    "showHeader": True,
                    "sortBy": [{"displayName": "Severity", "desc": True}]
                }
            ),
            
            # Alert Trends
            Panel(
                id=2,
                title="Alert Trends",
                type="graph",
                datasource="Prometheus",
                targets=[{
                    "expr": "sum(ALERTS) by (alertname, severity)",
                    "legendFormat": "{{alertname}} - {{severity}}",
                    "refId": "A"
                }],
                gridPos={"x": 0, "y": 8, "w": 24, "h": 8}
            )
        ]
        
        return Dashboard(
            uid="alerts",
            title="Alerts",
            tags=["alerts", "monitoring"],
            panels=panels
        )
        
    def _create_sla_dashboard(self) -> Dashboard:
        """Create SLA Compliance dashboard"""
        panels = [
            # SLA Overview
            Panel(
                id=1,
                title="SLA Compliance Overview",
                type="stat",
                datasource="Prometheus",
                targets=[{
                    "expr": "mcp_automation_sla_compliance_ratio * 100",
                    "legendFormat": "{{workflow_name}}",
                    "refId": "A"
                }],
                gridPos={"x": 0, "y": 0, "w": 24, "h": 4},
                options={
                    "orientation": "horizontal",
                    "textMode": "value_and_name",
                    "colorMode": "background",
                    "graphMode": "area"
                },
                fieldConfig={
                    "defaults": {
                        "unit": "percent",
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {"color": "red", "value": 0},
                                {"color": "yellow", "value": 95},
                                {"color": "green", "value": 99}
                            ]
                        }
                    }
                }
            ),
            
            # SLA Trends
            Panel(
                id=2,
                title="SLA Compliance Trends",
                type="graph",
                datasource="Prometheus",
                targets=[{
                    "expr": "mcp_automation_sla_compliance_ratio * 100",
                    "legendFormat": "{{workflow_name}}",
                    "refId": "A"
                }],
                gridPos={"x": 0, "y": 4, "w": 24, "h": 12},
                fieldConfig={
                    "defaults": {
                        "unit": "percent",
                        "min": 0,
                        "max": 100,
                        "custom": {
                            "axisLabel": "Compliance %",
                            "drawStyle": "line",
                            "lineInterpolation": "smooth",
                            "fillOpacity": 10
                        },
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {"color": "red", "value": 0},
                                {"color": "yellow", "value": 95},
                                {"color": "green", "value": 99}
                            ]
                        }
                    }
                }
            )
        ]
        
        return Dashboard(
            uid="sla-compliance",
            title="SLA Compliance",
            tags=["sla", "compliance", "automation"],
            panels=panels
        )
        
    async def deploy_dashboard(self, dashboard_key: str) -> bool:
        """
        Deploy a dashboard to Grafana
        
        Args:
            dashboard_key: Key of the dashboard to deploy
            
        Returns:
            True if successful, False otherwise
        """
        if dashboard_key not in self.dashboards:
            logger.error(f"Dashboard not found: {dashboard_key}")
            return False
            
        dashboard = self.dashboards[dashboard_key]
        
        # Convert dashboard to JSON
        dashboard_json = {
            "dashboard": asdict(dashboard),
            "overwrite": True,
            "message": f"Updated by MCP Dashboard Manager at {datetime.now()}"
        }
        
        try:
            response = await self.http_client.post(
                "/api/dashboards/db",
                json=dashboard_json
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Dashboard deployed successfully: {dashboard.title} (UID: {result.get('uid')})")
                return True
            else:
                logger.error(f"Failed to deploy dashboard: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error deploying dashboard: {e}")
            return False
            
    async def deploy_all_dashboards(self):
        """Deploy all dashboards to Grafana"""
        logger.info("Deploying all dashboards to Grafana...")
        
        for key in self.dashboards:
            success = await self.deploy_dashboard(key)
            if not success:
                logger.warning(f"Failed to deploy dashboard: {key}")
                
    def export_dashboard(self, dashboard_key: str, output_path: str):
        """
        Export dashboard to JSON file
        
        Args:
            dashboard_key: Key of the dashboard to export
            output_path: Path to save the dashboard JSON
        """
        if dashboard_key not in self.dashboards:
            logger.error(f"Dashboard not found: {dashboard_key}")
            return
            
        dashboard = self.dashboards[dashboard_key]
        dashboard_dict = asdict(dashboard)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(dashboard_dict, f, indent=2, default=str)
            
        logger.info(f"Dashboard exported to: {output_file}")
        
    def export_all_dashboards(self, output_dir: str):
        """Export all dashboards to JSON files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for key, dashboard in self.dashboards.items():
            self.export_dashboard(key, output_path / f"{key}.json")
            
    async def cleanup(self):
        """Cleanup resources"""
        await self.http_client.aclose()


async def main():
    """Main function for testing"""
    manager = DashboardManager()
    
    # Export dashboards to files
    manager.export_all_dashboards("/opt/sutazaiapp/monitoring/grafana/dashboards/mcp")
    
    # Deploy to Grafana (if available)
    # await manager.deploy_all_dashboards()
    
    # Cleanup
    await manager.cleanup()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())