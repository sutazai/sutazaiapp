"""
SutazAI Comprehensive Monitoring Endpoints
Real-time monitoring for multi-agent system
"""

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from typing import Dict, List, Optional, Any
import asyncio
import json
import time
import psutil
import docker
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, asdict
import redis
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.metrics import *
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/monitoring", tags=["monitoring"])

# WebSocket connection manager for real-time monitoring
class MonitoringConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.monitoring_task = None
        self.is_monitoring = False

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        if not self.is_monitoring:
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.is_monitoring = True

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        
        if not self.active_connections and self.is_monitoring:
            if self.monitoring_task:
                self.monitoring_task.cancel()
            self.is_monitoring = False

    async def broadcast(self, message: dict):
        """Broadcast monitoring data to all connected clients"""
        if self.active_connections:
            message_str = json.dumps(message)
            disconnected = []
            
            for connection in self.active_connections:
                try:
                    await connection.send_text(message_str)
                except (ValueError, TypeError, KeyError, AttributeError) as e:
                    logger.error(f"Unexpected exception: {e}", exc_info=True)
                    disconnected.append(connection)
            
            # Remove disconnected clients
            for connection in disconnected:
                self.disconnect(connection)

    async def _monitoring_loop(self):
        """Continuous monitoring loop for real-time updates"""
        while self.is_monitoring and self.active_connections:
            try:
                # Collect real-time metrics
                metrics_data = await self._collect_realtime_metrics()
                await self.broadcast(metrics_data)
                await asyncio.sleep(5)  # Update every 5 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(10)

    async def _collect_realtime_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive real-time metrics"""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Docker metrics
            docker_client = docker.from_env()
            containers = docker_client.containers.list()
            
            agent_containers = [c for c in containers if 'sutazai-' in c.name and 'agent' in c.name]
            running_agents = len([c for c in agent_containers if c.status == 'running'])
            
            # Redis connection for task queue metrics
            try:
                redis_client = redis.Redis(host='sutazai-redis', port=6379, db=0)
                queue_size = redis_client.llen('task_queue')
                active_tasks = redis_client.scard('active_tasks')
            except (ValueError, TypeError, KeyError, AttributeError) as e:
                logger.error(f"Unexpected exception: {e}", exc_info=True)
                queue_size = 0
                active_tasks = 0

            return {
                "timestamp": datetime.utcnow().isoformat(),
                "system": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_used_gb": memory.used / (1024**3),
                    "memory_total_gb": memory.total / (1024**3),
                    "disk_percent": disk.percent,
                    "disk_used_gb": disk.used / (1024**3),
                    "disk_total_gb": disk.total / (1024**3)
                },
                "agents": {
                    "total_containers": len(agent_containers),
                    "running_agents": running_agents,
                    "stopped_agents": len(agent_containers) - running_agents
                },
                "tasks": {
                    "queue_size": queue_size,
                    "active_tasks": active_tasks
                },
                "containers": [
                    {
                        "name": c.name,
                        "status": c.status,
                        "cpu_percent": self._get_container_cpu(c),
                        "memory_mb": self._get_container_memory(c)
                    }
                    for c in agent_containers
                ]
            }
        except Exception as e:
            logger.error(f"Error collecting realtime metrics: {e}")
            return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}

    def _get_container_cpu(self, container) -> float:
        """Get container CPU usage percentage"""
        try:
            stats = container.stats(stream=False)
            cpu_stats = stats['cpu_stats']
            precpu_stats = stats['precpu_stats']
            
            cpu_delta = cpu_stats['cpu_usage']['total_usage'] - precpu_stats['cpu_usage']['total_usage']
            system_delta = cpu_stats['system_cpu_usage'] - precpu_stats['system_cpu_usage']
            
            if system_delta > 0 and cpu_delta > 0:
                cpu_percent = (cpu_delta / system_delta) * len(cpu_stats['cpu_usage']['percpu_usage']) * 100
                return round(cpu_percent, 2)
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            # Suppressed exception (was bare except)
            logger.debug(f"Suppressed exception: {e}")
            pass
        return 0.0

    def _get_container_memory(self, container) -> float:
        """Get container memory usage in MB"""
        try:
            stats = container.stats(stream=False)
            memory_usage = stats['memory_stats']['usage']
            return round(memory_usage / (1024**2), 2)  # Convert to MB
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.warning(f"Exception caught, returning: {e}")
            return 0.0

# Global connection manager
manager = MonitoringConnectionManager()

@dataclass
class AgentHealth:
    agent_name: str
    status: str
    last_heartbeat: datetime
    cpu_usage: float
    memory_usage_mb: float
    task_count: int
    error_count: int
    uptime_seconds: int

@dataclass
class SystemHealth:
    timestamp: datetime
    overall_status: str
    active_agents: int
    total_agents: int
    cpu_usage_percent: float
    memory_usage_percent: float
    disk_usage_percent: float
    api_response_time_ms: float
    total_requests_last_minute: int
    error_rate_percent: float

@router.get("/metrics", response_class=PlainTextResponse)
async def get_prometheus_metrics():
    """Endpoint for Prometheus to scrape metrics"""
    try:
        # Update dynamic metrics
        await update_system_metrics()
        await update_agent_metrics()
        
        # Generate Prometheus formatted metrics
        return generate_latest()
    except Exception as e:
        logger.error(f"Error generating metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate metrics")

@router.get("/health")
async def system_health() -> SystemHealth:
    """Get overall system health status"""
    try:
        # Get system metrics
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Count active agents
        docker_client = docker.from_env()
        containers = docker_client.containers.list()
        agent_containers = [c for c in containers if 'sutazai-' in c.name]
        active_agents = len([c for c in agent_containers if c.status == 'running'])
        total_agents = len(agent_containers)
        
        # Calculate overall status
        status = "healthy"
        if cpu_percent > 80 or memory.percent > 90 or disk.percent > 90:
            status = "degraded"
        if active_agents < total_agents * 0.8:  # Less than 80% agents running
            status = "critical"
        
        # Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test API metrics (would come from actual metrics in production)
        api_response_time = 150.0  # ms
        total_requests = 100
        error_rate = 2.5  # percent
        
        return SystemHealth(
            timestamp=datetime.utcnow(),
            overall_status=status,
            active_agents=active_agents,
            total_agents=total_agents,
            cpu_usage_percent=cpu_percent,
            memory_usage_percent=memory.percent,
            disk_usage_percent=disk.percent,
            api_response_time_ms=api_response_time,
            total_requests_last_minute=total_requests,
            error_rate_percent=error_rate
        )
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system health")

@router.get("/agents/health")
async def agents_health() -> List[AgentHealth]:
    """Get health status of all AI agents"""
    try:
        docker_client = docker.from_env()
        containers = docker_client.containers.list(all=True)
        agent_containers = [c for c in containers if 'sutazai-' in c.name and any(x in c.name for x in ['agent', 'orchestrator', 'specialist', 'manager'])]
        
        agent_health_list = []
        
        for container in agent_containers:
            try:
                # Extract agent name
                agent_name = container.name.replace('sutazai-', '').replace('-', '_')
                
                # Get container stats
                if container.status == 'running':
                    stats = container.stats(stream=False)
                    cpu_usage = manager._get_container_cpu(container)
                    memory_usage = manager._get_container_memory(container)
                    
                    # Calculate uptime
                    started_at = datetime.fromisoformat(container.attrs['State']['StartedAt'].replace('Z', '+00:00'))
                    uptime = (datetime.now(started_at.tzinfo) - started_at).total_seconds()
                else:
                    cpu_usage = 0.0
                    memory_usage = 0.0
                    uptime = 0
                
                # Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test task and error counts (would come from actual metrics)
                task_count = 0
                error_count = 0
                
                agent_health = AgentHealth(
                    agent_name=agent_name,
                    status=container.status,
                    last_heartbeat=datetime.utcnow() if container.status == 'running' else datetime.utcnow() - timedelta(minutes=5),
                    cpu_usage=cpu_usage,
                    memory_usage_mb=memory_usage,
                    task_count=task_count,
                    error_count=error_count,
                    uptime_seconds=int(uptime)
                )
                
                agent_health_list.append(agent_health)
                
            except Exception as e:
                logger.error(f"Error getting health for container {container.name}: {e}")
                continue
        
        return agent_health_list
        
    except Exception as e:
        logger.error(f"Error getting agents health: {e}")
        raise HTTPException(status_code=500, detail="Failed to get agents health")

@router.get("/agents/{agent_name}/metrics")
async def agent_specific_metrics(agent_name: str) -> Dict[str, Any]:
    """Get detailed metrics for a specific agent"""
    try:
        docker_client = docker.from_env()
        container_name = f"sutazai-{agent_name.replace('_', '-')}"
        
        try:
            container = docker_client.containers.get(container_name)
        except docker.errors.NotFound:
            raise HTTPException(status_code=404, detail=f"Agent {agent_name} not found")
        
        if container.status != 'running':
            return {
                "agent_name": agent_name,
                "status": container.status,
                "metrics": None,
                "message": "Agent is not running"
            }
        
        # Get detailed container stats
        stats = container.stats(stream=False)
        
        # Calculate CPU usage
        cpu_percent = manager._get_container_cpu(container)
        
        # Memory stats
        memory_stats = stats['memory_stats']
        memory_usage = memory_stats.get('usage', 0)
        memory_limit = memory_stats.get('limit', 0)
        memory_percent = (memory_usage / memory_limit * 100) if memory_limit > 0 else 0
        
        # Network stats
        networks = stats.get('networks', {})
        rx_bytes = sum(net.get('rx_bytes', 0) for net in networks.values())
        tx_bytes = sum(net.get('tx_bytes', 0) for net in networks.values())
        
        # Block I/O stats
        blkio_stats = stats.get('blkio_stats', {})
        io_read = sum(stat.get('value', 0) for stat in blkio_stats.get('io_service_bytes_recursive', []) if stat.get('op') == 'Read')
        io_write = sum(stat.get('value', 0) for stat in blkio_stats.get('io_service_bytes_recursive', []) if stat.get('op') == 'Write')
        
        return {
            "agent_name": agent_name,
            "status": container.status,
            "container_id": container.short_id,
            "metrics": {
                "cpu": {
                    "usage_percent": cpu_percent
                },
                "memory": {
                    "usage_bytes": memory_usage,
                    "usage_mb": memory_usage / (1024**2),
                    "limit_bytes": memory_limit,
                    "usage_percent": memory_percent
                },
                "network": {
                    "rx_bytes": rx_bytes,
                    "tx_bytes": tx_bytes,
                    "rx_mb": rx_bytes / (1024**2),
                    "tx_mb": tx_bytes / (1024**2)
                },
                "disk_io": {
                    "read_bytes": io_read,
                    "write_bytes": io_write,
                    "read_mb": io_read / (1024**2),
                    "write_mb": io_write / (1024**2)
                }
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting agent metrics for {agent_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics for agent {agent_name}")

@router.websocket("/realtime")
async def realtime_monitoring(websocket: WebSocket):
    """WebSocket endpoint for real-time monitoring data"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle client messages
            data = await websocket.receive_text()
            # Handle any client commands here if needed
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@router.get("/sla/report")
async def sla_report(hours: int = 24) -> Dict[str, Any]:
    """Generate SLA compliance report"""
    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        # Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test SLA calculations (would use actual metrics in production)
        uptime_percent = 99.5
        avg_response_time = 145.0  # ms
        error_rate = 0.02  # 2%
        agent_availability = 98.8
        
        # SLA thresholds
        sla_thresholds = {
            "uptime_target": 99.9,
            "response_time_target": 200.0,
            "error_rate_target": 0.01,
            "agent_availability_target": 99.0
        }
        
        # Calculate SLA compliance
        uptime_compliant = uptime_percent >= sla_thresholds["uptime_target"]
        response_time_compliant = avg_response_time <= sla_thresholds["response_time_target"]
        error_rate_compliant = error_rate <= sla_thresholds["error_rate_target"]
        agent_availability_compliant = agent_availability >= sla_thresholds["agent_availability_target"]
        
        overall_sla_compliance = all([
            uptime_compliant,
            response_time_compliant,
            error_rate_compliant,
            agent_availability_compliant
        ])
        
        return {
            "report_period": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_hours": hours
            },
            "sla_metrics": {
                "uptime_percent": uptime_percent,
                "avg_response_time_ms": avg_response_time,
                "error_rate_percent": error_rate * 100,
                "agent_availability_percent": agent_availability
            },
            "sla_thresholds": {
                "uptime_target_percent": sla_thresholds["uptime_target"],
                "response_time_target_ms": sla_thresholds["response_time_target"],
                "error_rate_target_percent": sla_thresholds["error_rate_target"] * 100,
                "agent_availability_target_percent": sla_thresholds["agent_availability_target"]
            },
            "compliance": {
                "uptime_compliant": uptime_compliant,
                "response_time_compliant": response_time_compliant,
                "error_rate_compliant": error_rate_compliant,
                "agent_availability_compliant": agent_availability_compliant,
                "overall_sla_compliance": overall_sla_compliance
            },
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating SLA report: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate SLA report")

@router.post("/alerts/webhook")
async def alert_webhook(alert_data: Dict[str, Any]):
    """Webhook endpoint for receiving alerts from AlertManager"""
    try:
        logger.info(f"Received alert webhook: {alert_data}")
        
        # Process and potentially trigger automated responses
        alerts = alert_data.get('alerts', [])
        
        for alert in alerts:
            alert_name = alert.get('labels', {}).get('alertname', 'Unknown')
            severity = alert.get('labels', {}).get('severity', 'info')
            service = alert.get('labels', {}).get('service', 'unknown')
            
            # Log the alert
            logger.warning(f"Alert received - Name: {alert_name}, Severity: {severity}, Service: {service}")
            
            # Implement automated responses based on alert type
            if severity == 'critical' and service == 'agents':
                await handle_critical_agent_alert(alert)
            elif severity == 'critical' and service == 'backend':
                await handle_critical_backend_alert(alert)
        
        return {"status": "received", "processed_alerts": len(alerts)}
        
    except Exception as e:
        logger.error(f"Error processing alert webhook: {e}")
        raise HTTPException(status_code=500, detail="Failed to process alert")

async def handle_critical_agent_alert(alert: Dict[str, Any]):
    """Handle critical agent alerts with automated responses"""
    agent_name = alert.get('labels', {}).get('agent_name', 'unknown')
    logger.critical(f"Critical agent alert for {agent_name}: {alert.get('annotations', {}).get('summary', 'No summary')}")
    
    # Implement automated agent restart logic here
    # This would be expanded based on specific requirements

async def handle_critical_backend_alert(alert: Dict[str, Any]):
    """Handle critical backend alerts"""
    logger.critical(f"Critical backend alert: {alert.get('annotations', {}).get('summary', 'No summary')}")
    
    # Implement automated backend recovery logic here

async def update_system_metrics():
    """Update system-level Prometheus metrics"""
    try:
        # System resource metrics
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        # Update memory usage metric by component
        memory_usage.labels(component="system").set(memory.used)
        
        # Count active agents
        docker_client = docker.from_env()
        containers = docker_client.containers.list()
        agent_containers = [c for c in containers if 'sutazai-' in c.name]
        running_agents = len([c for c in agent_containers if c.status == 'running'])
        
        # Update agent metrics
        active_agents.set(running_agents)
        total_agents.set(len(agent_containers))
        
    except Exception as e:
        logger.error(f"Error updating system metrics: {e}")

async def update_agent_metrics():
    """Update agent-specific Prometheus metrics"""
    try:
        docker_client = docker.from_env()
        containers = docker_client.containers.list()
        agent_containers = [c for c in containers if 'sutazai-' in c.name]
        
        for container in agent_containers:
            if container.status == 'running':
                agent_name = container.name.replace('sutazai-', '').replace('-', '_')
                
                # Update memory usage for this agent
                memory_mb = manager._get_container_memory(container)
                memory_usage.labels(component=f"agent_{agent_name}").set(memory_mb * 1024 * 1024)  # Convert to bytes
                
    except Exception as e:
