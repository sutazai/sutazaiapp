#!/usr/bin/env python3
"""
MCP Automation Metrics Collector
Collects and exposes Prometheus metrics for MCP automation infrastructure
"""

import asyncio
import json
import logging
import os
import psutil
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

from prometheus_client import (
    Counter, Gauge, Histogram, Summary, Info,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
    push_to_gateway
)
from prometheus_client.core import GaugeMetricFamily, CounterMetricFamily

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics to collect"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    INFO = "info"


@dataclass
class MCPServerMetrics:
    """Metrics for individual MCP server"""
    name: str
    status: str  # healthy, degraded, failed
    uptime: float
    request_count: int
    error_count: int
    latency_ms: float
    memory_mb: float
    cpu_percent: float
    last_check: datetime
    version: str
    capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AutomationMetrics:
    """Metrics for automation workflows"""
    workflow_name: str
    executions_total: int
    executions_successful: int
    executions_failed: int
    average_duration_seconds: float
    last_execution: Optional[datetime]
    queue_size: int
    active_tasks: int
    error_rate: float
    sla_compliance: float


class MCPMetricsCollector:
    """Prometheus metrics collector for MCP automation"""
    
    def __init__(self, 
                 mcp_servers_path: str = "/opt/sutazaiapp/scripts/mcp/wrappers",
                 push_gateway: Optional[str] = None,
                 collection_interval: int = 30):
        """
        Initialize MCP metrics collector
        
        Args:
            mcp_servers_path: Path to MCP server wrappers
            push_gateway: Optional Prometheus push gateway URL
            collection_interval: Metrics collection interval in seconds
        """
        self.mcp_servers_path = Path(mcp_servers_path)
        self.push_gateway = push_gateway
        self.collection_interval = collection_interval
        self.registry = CollectorRegistry()
        
        # Initialize metrics
        self._init_metrics()
        
        # Server status tracking
        self.server_metrics: Dict[str, MCPServerMetrics] = {}
        self.automation_metrics: Dict[str, AutomationMetrics] = {}
        
        # Collection statistics
        self.collection_stats = {
            'collections_total': 0,
            'collections_failed': 0,
            'last_collection': None,
            'collection_duration': 0
        }
        
    def _init_metrics(self):
        """Initialize Prometheus metrics"""
        # MCP Server metrics
        self.mcp_server_up = Gauge(
            'mcp_server_up',
            'MCP server availability (1=up, 0=down)',
            ['server_name'],
            registry=self.registry
        )
        
        self.mcp_server_requests = Counter(
            'mcp_server_requests_total',
            'Total number of requests to MCP server',
            ['server_name', 'method'],
            registry=self.registry
        )
        
        self.mcp_server_errors = Counter(
            'mcp_server_errors_total',
            'Total number of errors from MCP server',
            ['server_name', 'error_type'],
            registry=self.registry
        )
        
        self.mcp_server_latency = Histogram(
            'mcp_server_latency_seconds',
            'MCP server response latency',
            ['server_name', 'operation'],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
        )
        
        self.mcp_server_memory = Gauge(
            'mcp_server_memory_bytes',
            'Memory usage of MCP server',
            ['server_name'],
            registry=self.registry
        )
        
        self.mcp_server_cpu = Gauge(
            'mcp_server_cpu_percent',
            'CPU usage percentage of MCP server',
            ['server_name'],
            registry=self.registry
        )
        
        # Automation workflow metrics
        self.automation_executions = Counter(
            'mcp_automation_executions_total',
            'Total number of automation executions',
            ['workflow_name', 'status'],
            registry=self.registry
        )
        
        self.automation_duration = Summary(
            'mcp_automation_duration_seconds',
            'Duration of automation workflow execution',
            ['workflow_name'],
            registry=self.registry
        )
        
        self.automation_queue_size = Gauge(
            'mcp_automation_queue_size',
            'Number of tasks in automation queue',
            ['workflow_name'],
            registry=self.registry
        )
        
        self.automation_active_tasks = Gauge(
            'mcp_automation_active_tasks',
            'Number of currently active automation tasks',
            ['workflow_name'],
            registry=self.registry
        )
        
        self.automation_sla_compliance = Gauge(
            'mcp_automation_sla_compliance_ratio',
            'SLA compliance ratio for automation workflows',
            ['workflow_name'],
            registry=self.registry
        )
        
        # System metrics
        self.system_cpu = Gauge(
            'mcp_system_cpu_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.system_memory = Gauge(
            'mcp_system_memory_percent',
            'System memory usage percentage',
            registry=self.registry
        )
        
        self.system_disk = Gauge(
            'mcp_system_disk_percent',
            'System disk usage percentage',
            ['mount_point'],
            registry=self.registry
        )
        
        self.system_network_sent = Counter(
            'mcp_system_network_sent_bytes',
            'Total bytes sent over network',
            registry=self.registry
        )
        
        self.system_network_recv = Counter(
            'mcp_system_network_received_bytes',
            'Total bytes received over network',
            registry=self.registry
        )
        
        # Collection metrics
        self.collector_runs = Counter(
            'mcp_collector_runs_total',
            'Total number of metric collection runs',
            registry=self.registry
        )
        
        self.collector_errors = Counter(
            'mcp_collector_errors_total',
            'Total number of collection errors',
            ['error_type'],
            registry=self.registry
        )
        
        self.collector_duration = Histogram(
            'mcp_collector_duration_seconds',
            'Duration of metric collection',
            registry=self.registry
        )
        
    async def collect_mcp_server_metrics(self, server_name: str) -> Optional[MCPServerMetrics]:
        """
        Collect metrics for a specific MCP server
        
        Args:
            server_name: Name of the MCP server
            
        Returns:
            MCPServerMetrics or None if collection failed
        """
        try:
            wrapper_path = self.mcp_servers_path / f"{server_name}.sh"
            
            if not wrapper_path.exists():
                logger.warning(f"MCP server wrapper not found: {wrapper_path}")
                return None
                
            # Check server health
            health_result = subprocess.run(
                [str(wrapper_path), "--selfcheck"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            is_healthy = health_result.returncode == 0
            
            # Get process metrics if server is running
            memory_mb = 0
            cpu_percent = 0
            
            try:
                # Find process by name pattern
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    if server_name in str(proc.info.get('cmdline', [])):
                        process = psutil.Process(proc.info['pid'])
                        memory_mb = process.memory_info().rss / 1024 / 1024
                        cpu_percent = process.cpu_percent(interval=1)
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
                
            # Get version info
            version = "unknown"
            try:
                version_result = subprocess.run(
                    [str(wrapper_path), "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if version_result.returncode == 0:
                    version = version_result.stdout.strip()
            except:
                pass
                
            metrics = MCPServerMetrics(
                name=server_name,
                status="healthy" if is_healthy else "failed",
                uptime=time.time(),  # Will be calculated based on previous state
                request_count=0,  # Will be updated from logs
                error_count=0,  # Will be updated from logs
                latency_ms=0,  # Will be updated from actual measurements
                memory_mb=memory_mb,
                cpu_percent=cpu_percent,
                last_check=datetime.now(),
                version=version
            )
            
            # Update Prometheus metrics
            self.mcp_server_up.labels(server_name=server_name).set(1 if is_healthy else 0)
            self.mcp_server_memory.labels(server_name=server_name).set(memory_mb * 1024 * 1024)
            self.mcp_server_cpu.labels(server_name=server_name).set(cpu_percent)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting metrics for {server_name}: {e}")
            self.collector_errors.labels(error_type='mcp_server_collection').inc()
            return None
            
    async def collect_automation_metrics(self) -> Dict[str, AutomationMetrics]:
        """
        Collect metrics for automation workflows
        
        Returns:
            Dictionary of automation metrics by workflow name
        """
        try:
            # This would integrate with actual automation system
            # For now, return example metrics
            workflows = {
                'mcp_update': AutomationMetrics(
                    workflow_name='mcp_update',
                    executions_total=100,
                    executions_successful=95,
                    executions_failed=5,
                    average_duration_seconds=45.5,
                    last_execution=datetime.now() - timedelta(minutes=30),
                    queue_size=2,
                    active_tasks=1,
                    error_rate=0.05,
                    sla_compliance=0.98
                ),
                'health_check': AutomationMetrics(
                    workflow_name='health_check',
                    executions_total=500,
                    executions_successful=498,
                    executions_failed=2,
                    average_duration_seconds=2.5,
                    last_execution=datetime.now() - timedelta(minutes=5),
                    queue_size=0,
                    active_tasks=0,
                    error_rate=0.004,
                    sla_compliance=0.996
                )
            }
            
            # Update Prometheus metrics
            for name, metrics in workflows.items():
                self.automation_executions.labels(
                    workflow_name=name,
                    status='success'
                ).inc(metrics.executions_successful)
                
                self.automation_executions.labels(
                    workflow_name=name,
                    status='failed'
                ).inc(metrics.executions_failed)
                
                self.automation_queue_size.labels(workflow_name=name).set(metrics.queue_size)
                self.automation_active_tasks.labels(workflow_name=name).set(metrics.active_tasks)
                self.automation_sla_compliance.labels(workflow_name=name).set(metrics.sla_compliance)
                
            return workflows
            
        except Exception as e:
            logger.error(f"Error collecting automation metrics: {e}")
            self.collector_errors.labels(error_type='automation_collection').inc()
            return {}
            
    def collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.system_cpu.set(cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.system_memory.set(memory.percent)
            
            # Disk metrics
            for partition in psutil.disk_partitions():
                if partition.mountpoint:
                    usage = psutil.disk_usage(partition.mountpoint)
                    self.system_disk.labels(mount_point=partition.mountpoint).set(usage.percent)
                    
            # Network metrics
            net_io = psutil.net_io_counters()
            self.system_network_sent.inc(net_io.bytes_sent)
            self.system_network_recv.inc(net_io.bytes_recv)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            self.collector_errors.labels(error_type='system_collection').inc()
            
    async def collect_all_metrics(self):
        """Collect all metrics"""
        start_time = time.time()
        
        try:
            self.collector_runs.inc()
            
            # Collect MCP server metrics
            mcp_servers = [
                'filesystem', 'github', 'postgres', 'browser', 'docker',
                'kubernetes', 'gitlab', 'slack', 'google-drive', 'linear',
                'sentry', 'aws', 'google-cloud', 'azure', 'terraform',
                'extended-memory', 'browser-use'
            ]
            
            tasks = []
            for server in mcp_servers:
                tasks.append(self.collect_mcp_server_metrics(server))
                
            server_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(server_results):
                if isinstance(result, MCPServerMetrics):
                    self.server_metrics[result.name] = result
                elif isinstance(result, Exception):
                    logger.error(f"Error collecting metrics for {mcp_servers[i]}: {result}")
                    
            # Collect automation metrics
            self.automation_metrics = await self.collect_automation_metrics()
            
            # Collect system metrics
            self.collect_system_metrics()
            
            # Update collection statistics
            duration = time.time() - start_time
            self.collector_duration.observe(duration)
            self.collection_stats['collections_total'] += 1
            self.collection_stats['last_collection'] = datetime.now()
            self.collection_stats['collection_duration'] = duration
            
            logger.info(f"Metrics collection completed in {duration:.2f}s")
            
        except Exception as e:
            logger.error(f"Error during metrics collection: {e}")
            self.collector_errors.labels(error_type='general').inc()
            self.collection_stats['collections_failed'] += 1
            
    async def start_collection_loop(self):
        """Start continuous metrics collection"""
        logger.info(f"Starting metrics collection loop with {self.collection_interval}s interval")
        
        while True:
            try:
                await self.collect_all_metrics()
                
                # Push to gateway if configured
                if self.push_gateway:
                    try:
                        push_to_gateway(
                            self.push_gateway,
                            job='mcp_metrics_collector',
                            registry=self.registry
                        )
                        logger.debug(f"Pushed metrics to gateway: {self.push_gateway}")
                    except Exception as e:
                        logger.error(f"Failed to push metrics to gateway: {e}")
                        
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                
            await asyncio.sleep(self.collection_interval)
            
    def get_metrics(self) -> bytes:
        """
        Get metrics in Prometheus format
        
        Returns:
            Metrics in Prometheus text format
        """
        return generate_latest(self.registry)
        
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get human-readable metrics summary
        
        Returns:
            Dictionary with metrics summary
        """
        healthy_servers = sum(1 for m in self.server_metrics.values() if m.status == 'healthy')
        total_servers = len(self.server_metrics)
        
        successful_workflows = sum(m.executions_successful for m in self.automation_metrics.values())
        total_workflows = sum(m.executions_total for m in self.automation_metrics.values())
        
        return {
            'collection_stats': self.collection_stats,
            'mcp_servers': {
                'total': total_servers,
                'healthy': healthy_servers,
                'unhealthy': total_servers - healthy_servers,
                'health_percentage': (healthy_servers / total_servers * 100) if total_servers > 0 else 0
            },
            'automation': {
                'total_executions': total_workflows,
                'successful_executions': successful_workflows,
                'success_rate': (successful_workflows / total_workflows * 100) if total_workflows > 0 else 0,
                'active_workflows': len(self.automation_metrics)
            },
            'system': {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': {p.mountpoint: psutil.disk_usage(p.mountpoint).percent 
                              for p in psutil.disk_partitions() if p.mountpoint}
            },
            'timestamp': datetime.now().isoformat()
        }


async def main():
    """Main function for testing"""
    collector = MCPMetricsCollector(
        push_gateway="http://localhost:10200/metrics/job/mcp_collector"
    )
    
    # Run one collection cycle
    await collector.collect_all_metrics()
    
    # Print summary
    summary = collector.get_metrics_summary()
    print(json.dumps(summary, indent=2, default=str))
    
    # Get Prometheus metrics
    metrics = collector.get_metrics()
    print("\nPrometheus Metrics Sample:")
    print(metrics.decode('utf-8')[:1000])


if __name__ == "__main__":
    asyncio.run(main())