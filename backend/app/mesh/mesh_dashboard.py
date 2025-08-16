"""
Service Mesh Dashboard - Real-time visualization and monitoring
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

from app.mesh.service_mesh import get_mesh, ServiceState
from app.mesh.distributed_tracing import get_tracer

logger = logging.getLogger(__name__)

@dataclass
class ServiceMetrics:
    """Metrics for a service"""
    service_name: str
    instance_count: int
    healthy_count: int
    unhealthy_count: int
    degraded_count: int
    request_rate: float  # requests per second
    error_rate: float  # percentage
    p50_latency: float  # milliseconds
    p95_latency: float  # milliseconds
    p99_latency: float  # milliseconds
    circuit_breakers_open: int
    active_connections: int
    
    @property
    def health_percentage(self) -> float:
        if self.instance_count == 0:
            return 0.0
        return (self.healthy_count / self.instance_count) * 100

@dataclass
class MeshMetrics:
    """Overall mesh metrics"""
    total_services: int
    total_instances: int
    healthy_instances: int
    unhealthy_instances: int
    degraded_instances: int
    total_requests: int
    failed_requests: int
    avg_latency: float
    circuit_breakers_open: int
    active_traces: int
    
    @property
    def health_score(self) -> float:
        if self.total_instances == 0:
            return 100.0
        return (self.healthy_instances / self.total_instances) * 100
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 100.0
        return ((self.total_requests - self.failed_requests) / self.total_requests) * 100

class MeshDashboard:
    """Service mesh dashboard for monitoring and visualization"""
    
    def __init__(self):
        self.mesh = None
        self.tracer = get_tracer("mesh-dashboard")
        self.metrics_history: List[MeshMetrics] = []
        self.service_metrics_history: Dict[str, List[ServiceMetrics]] = {}
        self.max_history_size = 1000
        self.collection_interval = 10  # seconds
        self._collection_task = None
        
    async def initialize(self):
        """Initialize dashboard"""
        self.mesh = await get_mesh()
        # Start metrics collection
        self._collection_task = asyncio.create_task(self._collect_metrics_loop())
        logger.info("Mesh dashboard initialized")
    
    async def shutdown(self):
        """Shutdown dashboard"""
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
    
    async def _collect_metrics_loop(self):
        """Background task to collect metrics"""
        while True:
            try:
                await self._collect_metrics()
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _collect_metrics(self):
        """Collect current metrics"""
        topology = await self.mesh.get_service_topology()
        
        # Calculate mesh metrics
        total_instances = topology["total_instances"]
        healthy_instances = topology["healthy_instances"]
        unhealthy_instances = 0
        degraded_instances = 0
        
        for service_data in topology["services"].values():
            for instance in service_data["instances"]:
                if instance["state"] == "unhealthy":
                    unhealthy_instances += 1
                elif instance["state"] == "degraded":
                    degraded_instances += 1
        
        circuit_breakers_open = sum(
            1 for state in topology["circuit_breakers"].values()
            if state == "open"
        )
        
        # Get trace metrics
        active_traces = len(self.tracer.collector.traces)
        
        # Calculate real metrics from traces
        total_requests = 0
        failed_requests = 0
        latencies = []
        
        # Analyze recent traces for metrics
        recent_traces = self.tracer.collector.search_traces(limit=100)
        for trace in recent_traces:
            if 'spans' in trace:
                for span in trace['spans']:
                    total_requests += 1
                    if span.get('status') == 'ERROR':
                        failed_requests += 1
                    if 'duration' in span:
                        latencies.append(span['duration'])
        
        # Calculate average latency
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        
        mesh_metrics = MeshMetrics(
            total_services=len(topology["services"]),
            total_instances=total_instances,
            healthy_instances=healthy_instances,
            unhealthy_instances=unhealthy_instances,
            degraded_instances=degraded_instances,
            total_requests=total_requests,
            failed_requests=failed_requests,
            avg_latency=avg_latency * 1000,  # Convert to milliseconds
            circuit_breakers_open=circuit_breakers_open,
            active_traces=active_traces
        )
        
        # Add to history
        self.metrics_history.append(mesh_metrics)
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history.pop(0)
        
        # Collect per-service metrics
        for service_name, service_data in topology["services"].items():
            healthy = service_data["healthy"]
            total = service_data["total"]
            
            # Calculate service-specific metrics from traces
            service_traces = self.tracer.collector.search_traces(
                service_name=service_name,
                limit=50
            )
            
            service_requests = 0
            service_errors = 0
            service_latencies = []
            
            for trace in service_traces:
                if 'spans' in trace:
                    for span in trace['spans']:
                        if span.get('service_name') == service_name:
                            service_requests += 1
                            if span.get('status') == 'ERROR':
                                service_errors += 1
                            if 'duration' in span:
                                service_latencies.append(span['duration'] * 1000)  # Convert to ms
            
            # Calculate latency percentiles
            p50_latency = 0.0
            p95_latency = 0.0
            p99_latency = 0.0
            
            if service_latencies:
                service_latencies.sort()
                p50_idx = int(len(service_latencies) * 0.50)
                p95_idx = int(len(service_latencies) * 0.95)
                p99_idx = int(len(service_latencies) * 0.99)
                
                p50_latency = service_latencies[min(p50_idx, len(service_latencies)-1)]
                p95_latency = service_latencies[min(p95_idx, len(service_latencies)-1)]
                p99_latency = service_latencies[min(p99_idx, len(service_latencies)-1)]
            
            # Calculate rates based on collection interval
            request_rate = service_requests / self.collection_interval if service_requests > 0 else 0.0
            error_rate = (service_errors / service_requests * 100) if service_requests > 0 else 0.0
            
            service_metrics = ServiceMetrics(
                service_name=service_name,
                instance_count=total,
                healthy_count=healthy,
                unhealthy_count=sum(1 for i in service_data["instances"] if i["state"] == "unhealthy"),
                degraded_count=sum(1 for i in service_data["instances"] if i["state"] == "degraded"),
                request_rate=request_rate,
                error_rate=error_rate,
                p50_latency=p50_latency,
                p95_latency=p95_latency,
                p99_latency=p99_latency,
                circuit_breakers_open=sum(1 for i in service_data["instances"] if i["circuit_breaker"]),
                active_connections=sum(i["connections"] for i in service_data["instances"])
            )
            
            if service_name not in self.service_metrics_history:
                self.service_metrics_history[service_name] = []
            
            self.service_metrics_history[service_name].append(service_metrics)
            if len(self.service_metrics_history[service_name]) > self.max_history_size:
                self.service_metrics_history[service_name].pop(0)
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get complete dashboard data"""
        topology = await self.mesh.get_service_topology()
        
        # Get latest metrics
        latest_metrics = self.metrics_history[-1] if self.metrics_history else None
        
        # Get service dependencies
        dependencies = self.tracer.collector.get_service_dependencies()
        
        # Get recent traces
        recent_traces = self.tracer.collector.search_traces(limit=10)
        
        # Build service graph
        service_graph = self._build_service_graph(topology, dependencies)
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mesh_metrics": asdict(latest_metrics) if latest_metrics else None,
            "services": self._format_services(topology),
            "service_graph": service_graph,
            "recent_traces": recent_traces,
            "alerts": self._get_alerts(topology, latest_metrics),
            "metrics_history": self._format_metrics_history()
        }
    
    def _build_service_graph(self, topology: Dict[str, Any], dependencies: Dict[str, List[str]]) -> Dict[str, Any]:
        """Build service dependency graph for visualization"""
        nodes = []
        edges = []
        
        # Create nodes for each service
        for service_name, service_data in topology["services"].items():
            node = {
                "id": service_name,
                "label": service_name,
                "instances": service_data["total"],
                "healthy": service_data["healthy"],
                "status": self._get_service_status(service_data)
            }
            nodes.append(node)
        
        # Create edges for dependencies
        for source, targets in dependencies.items():
            for target in targets:
                edge = {
                    "source": source,
                    "target": target,
                    "label": "calls"
                }
                edges.append(edge)
        
        return {
            "nodes": nodes,
            "edges": edges
        }
    
    def _get_service_status(self, service_data: Dict[str, Any]) -> str:
        """Determine service status"""
        if service_data["total"] == 0:
            return "unknown"
        
        health_ratio = service_data["healthy"] / service_data["total"]
        
        if health_ratio >= 0.8:
            return "healthy"
        elif health_ratio >= 0.5:
            return "degraded"
        else:
            return "unhealthy"
    
    def _format_services(self, topology: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format service data for dashboard"""
        services = []
        
        for service_name, service_data in topology["services"].items():
            # Get latest metrics for service
            service_metrics = None
            if service_name in self.service_metrics_history and self.service_metrics_history[service_name]:
                service_metrics = self.service_metrics_history[service_name][-1]
            
            service_info = {
                "name": service_name,
                "instances": service_data["instances"],
                "total_instances": service_data["total"],
                "healthy_instances": service_data["healthy"],
                "status": self._get_service_status(service_data),
                "metrics": asdict(service_metrics) if service_metrics else None
            }
            services.append(service_info)
        
        return services
    
    def _get_alerts(self, topology: Dict[str, Any], metrics: Optional[MeshMetrics]) -> List[Dict[str, Any]]:
        """Generate alerts based on current state"""
        alerts = []
        
        # Check overall health
        if metrics and metrics.health_score < 50:
            alerts.append({
                "severity": "critical",
                "message": f"Mesh health critically low: {metrics.health_score:.1f}%",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        elif metrics and metrics.health_score < 80:
            alerts.append({
                "severity": "warning",
                "message": f"Mesh health degraded: {metrics.health_score:.1f}%",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        
        # Check circuit breakers
        if metrics and metrics.circuit_breakers_open > 0:
            alerts.append({
                "severity": "warning",
                "message": f"{metrics.circuit_breakers_open} circuit breakers open",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        
        # Check individual services
        for service_name, service_data in topology["services"].items():
            if service_data["total"] > 0:
                health_ratio = service_data["healthy"] / service_data["total"]
                if health_ratio == 0:
                    alerts.append({
                        "severity": "critical",
                        "message": f"Service {service_name} has no healthy instances",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                elif health_ratio < 0.5:
                    alerts.append({
                        "severity": "warning",
                        "message": f"Service {service_name} is unhealthy ({service_data['healthy']}/{service_data['total']} healthy)",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
        
        return alerts
    
    def _format_metrics_history(self) -> Dict[str, Any]:
        """Format metrics history for charts"""
        if not self.metrics_history:
            return {}
        
        # Get last hour of data
        time_series = {
            "timestamps": [],
            "health_score": [],
            "total_instances": [],
            "healthy_instances": [],
            "circuit_breakers": []
        }
        
        for metrics in self.metrics_history[-60:]:  # Last 60 samples (10 minutes if 10s interval)
            time_series["timestamps"].append(time.time())
            time_series["health_score"].append(metrics.health_score)
            time_series["total_instances"].append(metrics.total_instances)
            time_series["healthy_instances"].append(metrics.healthy_instances)
            time_series["circuit_breakers"].append(metrics.circuit_breakers_open)
        
        return time_series
    
    async def get_service_details(self, service_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific service"""
        topology = await self.mesh.get_service_topology()
        
        if service_name not in topology["services"]:
            return None
        
        service_data = topology["services"][service_name]
        
        # Get service traces
        service_traces = self.tracer.collector.search_traces(
            service_name=service_name,
            limit=20
        )
        
        # Get metrics history
        metrics_history = []
        if service_name in self.service_metrics_history:
            for metrics in self.service_metrics_history[service_name][-60:]:
                metrics_history.append(asdict(metrics))
        
        return {
            "service_name": service_name,
            "current_state": service_data,
            "metrics_history": metrics_history,
            "recent_traces": service_traces,
            "dependencies": self.tracer.collector.get_service_dependencies().get(service_name, [])
        }

# Global dashboard instance
_dashboard: Optional[MeshDashboard] = None

async def get_dashboard() -> MeshDashboard:
    """Get or create dashboard instance"""
    global _dashboard
    if _dashboard is None:
        _dashboard = MeshDashboard()
        await _dashboard.initialize()
    return _dashboard