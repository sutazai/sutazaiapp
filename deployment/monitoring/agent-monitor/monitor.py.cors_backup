"""
SutazAI Comprehensive Agent Monitoring Service
Real-time monitoring and analysis of 46+ AI agents
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import os
import signal
import sys

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import docker
import psutil
import redis
import requests
from prometheus_client import start_http_server, Counter, Gauge, Histogram, Summary
import schedule
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/monitor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Prometheus metrics for the monitoring service itself
monitor_requests = Counter('monitor_requests_total', 'Total monitoring requests', ['endpoint'])
monitor_errors = Counter('monitor_errors_total', 'Total monitoring errors', ['error_type'])
agent_discovery_time = Histogram('agent_discovery_seconds', 'Time to discover agents')
websocket_connections = Gauge('monitor_websocket_connections', 'Active WebSocket connections')

class SutazAIAgentMonitor:
    """Comprehensive monitoring service for SutazAI multi-agent system"""
    
    def __init__(self):
        self.app = FastAPI(title="SutazAI Agent Monitor", version="1.0.0")
        self.setup_routes()
        self.setup_middleware()
        
        # Configuration
        self.prometheus_url = os.getenv('PROMETHEUS_URL', 'http://prometheus:9090')
        self.grafana_url = os.getenv('GRAFANA_URL', 'http://grafana:3000')
        self.jaeger_url = os.getenv('JAEGER_URL', 'http://jaeger:16686')
        self.loki_url = os.getenv('LOKI_URL', 'http://loki:3100')
        self.backend_url = os.getenv('BACKEND_URL', 'http://sutazai-backend:8000')
        
        # Docker client
        try:
            self.docker_client = docker.from_env()
            logger.info("Docker client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            self.docker_client = None
        
        # Redis client for task queue monitoring
        try:
            self.redis_client = redis.Redis(host='sutazai-redis', port=6379, db=0, decode_responses=True)
            self.redis_client.ping()
            logger.info("Redis client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Redis client: {e}")
            self.redis_client = None
        
        # WebSocket connections
        self.active_connections: List[WebSocket] = []
        
        # Monitoring data
        self.agent_data = {}
        self.system_metrics = {}
        self.alert_history = []
        
        # Background tasks
        self.monitoring_active = True
        self.background_tasks = []
        
    def setup_middleware(self):
        """Setup FastAPI middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/")
        async def root():
            return HTMLResponse(content=self.get_dashboard_html())
        
        @self.app.get("/health")
        async def health():
            monitor_requests.labels(endpoint='health').inc()
            return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
        
        @self.app.get("/agents/status")
        async def agents_status():
            monitor_requests.labels(endpoint='agents_status').inc()
            return await self.get_agents_status()
        
        @self.app.get("/system/metrics")
        async def system_metrics():
            monitor_requests.labels(endpoint='system_metrics').inc()
            return await self.get_system_metrics()
        
        @self.app.get("/orchestration/workflows")
        async def orchestration_workflows():
            monitor_requests.labels(endpoint='orchestration_workflows').inc()
            return await self.get_orchestration_metrics()
        
        @self.app.get("/alerts/active")
        async def active_alerts():
            monitor_requests.labels(endpoint='active_alerts').inc()
            return await self.get_active_alerts()
        
        @self.app.get("/reports/generate")
        async def generate_report():
            monitor_requests.labels(endpoint='generate_report').inc()
            return await self.generate_monitoring_report()
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.websocket_handler(websocket)
    
    async def get_agents_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all AI agents"""
        try:
            if not self.docker_client:
                return {"error": "Docker client not available"}
            
            with agent_discovery_time.time():
                containers = self.docker_client.containers.list(all=True)
                agent_containers = [c for c in containers if 'sutazai-' in c.name and 
                                 any(keyword in c.name for keyword in ['agent', 'orchestrator', 'specialist', 'manager'])]
            
            agents_status = []
            
            for container in agent_containers:
                try:
                    agent_name = container.name.replace('sutazai-', '').replace('-', '_')
                    
                    # Basic container info
                    status_info = {
                        "name": agent_name,
                        "container_name": container.name,
                        "status": container.status,
                        "image": container.image.tags[0] if container.image.tags else "unknown",
                        "created": container.attrs['Created'],
                        "started": container.attrs['State'].get('StartedAt', ''),
                    }
                    
                    # Resource usage (only for running containers)
                    if container.status == 'running':
                        try:
                            stats = container.stats(stream=False)
                            
                            # CPU calculation
                            cpu_stats = stats['cpu_stats']
                            precpu_stats = stats['precpu_stats']
                            cpu_delta = cpu_stats['cpu_usage']['total_usage'] - precpu_stats['cpu_usage']['total_usage']
                            system_delta = cpu_stats['system_cpu_usage'] - precpu_stats['system_cpu_usage']
                            
                            if system_delta > 0 and cpu_delta > 0:
                                cpu_percent = (cpu_delta / system_delta) * len(cpu_stats['cpu_usage']['percpu_usage']) * 100
                            else:
                                cpu_percent = 0.0
                            
                            # Memory usage
                            memory_stats = stats['memory_stats']
                            memory_usage = memory_stats.get('usage', 0)
                            memory_limit = memory_stats.get('limit', 0)
                            memory_percent = (memory_usage / memory_limit * 100) if memory_limit > 0 else 0
                            
                            # Network I/O
                            networks = stats.get('networks', {})
                            rx_bytes = sum(net.get('rx_bytes', 0) for net in networks.values())
                            tx_bytes = sum(net.get('tx_bytes', 0) for net in networks.values())
                            
                            status_info.update({
                                "cpu_percent": round(cpu_percent, 2),
                                "memory_usage_bytes": memory_usage,
                                "memory_usage_mb": round(memory_usage / (1024**2), 2),
                                "memory_percent": round(memory_percent, 2),
                                "network_rx_bytes": rx_bytes,
                                "network_tx_bytes": tx_bytes,
                                "uptime_seconds": int((datetime.utcnow() - datetime.fromisoformat(
                                    container.attrs['State']['StartedAt'].replace('Z', '+00:00').replace('+00:00', '')
                                )).total_seconds()) if container.attrs['State'].get('StartedAt') else 0
                            })
                            
                        except Exception as e:
                            logger.error(f"Error getting stats for {container.name}: {e}")
                            status_info.update({
                                "cpu_percent": 0.0,
                                "memory_usage_mb": 0.0,
                                "memory_percent": 0.0,
                                "error": f"Stats unavailable: {str(e)}"
                            })
                    else:
                        status_info.update({
                            "cpu_percent": 0.0,
                            "memory_usage_mb": 0.0,
                            "memory_percent": 0.0,
                            "uptime_seconds": 0
                        })
                    
                    agents_status.append(status_info)
                    
                except Exception as e:
                    logger.error(f"Error processing container {container.name}: {e}")
                    monitor_errors.labels(error_type='container_processing').inc()
                    continue
            
            # Summary statistics
            total_agents = len(agents_status)
            running_agents = len([a for a in agents_status if a['status'] == 'running'])
            stopped_agents = len([a for a in agents_status if a['status'] in ['exited', 'stopped']])
            
            return {
                "summary": {
                    "total_agents": total_agents,
                    "running_agents": running_agents,
                    "stopped_agents": stopped_agents,
                    "healthy_percentage": round((running_agents / total_agents * 100) if total_agents > 0 else 0, 2)
                },
                "agents": agents_status,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting agents status: {e}")
            monitor_errors.labels(error_type='agents_status').inc()
            return {"error": str(e)}
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        try:
            # System resource metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_per_core = psutil.cpu_percent(interval=1, percpu=True)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network I/O
            network = psutil.net_io_counters()
            
            # Load averages (Unix systems)
            try:
                load_avg = os.getloadavg()
            except:
                load_avg = [0.0, 0.0, 0.0]
            
            # Docker system info
            docker_info = {}
            if self.docker_client:
                try:
                    docker_info = self.docker_client.info()
                except Exception as e:
                    logger.error(f"Error getting Docker info: {e}")
            
            # Redis metrics
            redis_info = {}
            if self.redis_client:
                try:
                    redis_info = self.redis_client.info()
                    
                    # Task queue metrics
                    queue_sizes = {}
                    for key in self.redis_client.scan_iter(match="*queue*"):
                        try:
                            queue_sizes[key] = self.redis_client.llen(key)
                        except:
                            queue_sizes[key] = 0
                    
                    redis_info['queue_sizes'] = queue_sizes
                    
                except Exception as e:
                    logger.error(f"Error getting Redis info: {e}")
            
            return {
                "system": {
                    "cpu_percent": cpu_percent,
                    "cpu_per_core": cpu_per_core,
                    "load_average": {
                        "1min": load_avg[0],
                        "5min": load_avg[1],
                        "15min": load_avg[2]
                    },
                    "memory": {
                        "total_gb": round(memory.total / (1024**3), 2),
                        "available_gb": round(memory.available / (1024**3), 2),
                        "used_gb": round(memory.used / (1024**3), 2),
                        "percent": memory.percent,
                        "free_gb": round(memory.free / (1024**3), 2)
                    },
                    "disk": {
                        "total_gb": round(disk.total / (1024**3), 2),
                        "used_gb": round(disk.used / (1024**3), 2),
                        "free_gb": round(disk.free / (1024**3), 2),
                        "percent": round(disk.used / disk.total * 100, 2)
                    },
                    "network": {
                        "bytes_sent": network.bytes_sent,
                        "bytes_recv": network.bytes_recv,
                        "packets_sent": network.packets_sent,
                        "packets_recv": network.packets_recv
                    }
                },
                "docker": {
                    "containers_running": docker_info.get('ContainersRunning', 0),
                    "containers_total": docker_info.get('Containers', 0),
                    "images": docker_info.get('Images', 0),
                    "server_version": docker_info.get('ServerVersion', 'unknown')
                },
                "redis": {
                    "connected_clients": redis_info.get('connected_clients', 0),
                    "used_memory_mb": round(redis_info.get('used_memory', 0) / (1024**2), 2),
                    "keyspace_hits": redis_info.get('keyspace_hits', 0),
                    "keyspace_misses": redis_info.get('keyspace_misses', 0),
                    "queue_sizes": redis_info.get('queue_sizes', {})
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            monitor_errors.labels(error_type='system_metrics').inc()
            return {"error": str(e)}
    
    async def get_orchestration_metrics(self) -> Dict[str, Any]:
        """Get orchestration and workflow metrics"""
        try:
            # Get orchestration containers
            orchestration_containers = []
            if self.docker_client:
                containers = self.docker_client.containers.list(all=True)
                orchestration_containers = [c for c in containers if any(keyword in c.name for keyword in 
                                          ['orchestr', 'coordinator', 'workflow', 'task-assignment'])]
            
            # Task queue analysis
            workflow_metrics = {}
            if self.redis_client:
                try:
                    # Get all workflow-related keys
                    workflow_keys = list(self.redis_client.scan_iter(match="workflow:*"))
                    task_keys = list(self.redis_client.scan_iter(match="task:*"))
                    
                    workflow_metrics = {
                        "active_workflows": len(workflow_keys),
                        "total_tasks": len(task_keys),
                        "queued_tasks": self.redis_client.llen('task_queue') if self.redis_client.exists('task_queue') else 0,
                        "processing_tasks": self.redis_client.scard('processing_tasks') if self.redis_client.exists('processing_tasks') else 0,
                        "completed_tasks": self.redis_client.scard('completed_tasks') if self.redis_client.exists('completed_tasks') else 0,
                        "failed_tasks": self.redis_client.scard('failed_tasks') if self.redis_client.exists('failed_tasks') else 0
                    }
                except Exception as e:
                    logger.error(f"Error getting workflow metrics from Redis: {e}")
            
            # Orchestration container status
            orchestration_status = []
            for container in orchestration_containers:
                try:
                    orchestration_status.append({
                        "name": container.name,
                        "status": container.status,
                        "uptime": self._get_container_uptime(container)
                    })
                except Exception as e:
                    logger.error(f"Error getting orchestration container status: {e}")
            
            return {
                "orchestration_containers": orchestration_status,
                "workflow_metrics": workflow_metrics,
                "coordination_health": {
                    "active_orchestrators": len([c for c in orchestration_status if c['status'] == 'running']),
                    "total_orchestrators": len(orchestration_status)
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting orchestration metrics: {e}")
            monitor_errors.labels(error_type='orchestration_metrics').inc()
            return {"error": str(e)}
    
    async def get_active_alerts(self) -> Dict[str, Any]:
        """Get active alerts from Prometheus AlertManager"""
        try:
            # Query AlertManager API
            alerts_url = f"{self.prometheus_url.replace('9090', '9093')}/api/v1/alerts"
            
            try:
                response = requests.get(alerts_url, timeout=10)
                if response.status_code == 200:
                    alerts_data = response.json()
                    active_alerts = alerts_data.get('data', [])
                else:
                    active_alerts = []
            except Exception as e:
                logger.error(f"Error fetching alerts from AlertManager: {e}")
                active_alerts = []
            
            # Categorize alerts by severity
            critical_alerts = [a for a in active_alerts if a.get('labels', {}).get('severity') == 'critical']
            warning_alerts = [a for a in active_alerts if a.get('labels', {}).get('severity') == 'warning']
            info_alerts = [a for a in active_alerts if a.get('labels', {}).get('severity') == 'info']
            
            return {
                "summary": {
                    "total_alerts": len(active_alerts),
                    "critical_alerts": len(critical_alerts),
                    "warning_alerts": len(warning_alerts),
                    "info_alerts": len(info_alerts)
                },
                "alerts": {
                    "critical": critical_alerts,
                    "warning": warning_alerts,
                    "info": info_alerts
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting active alerts: {e}")
            monitor_errors.labels(error_type='active_alerts').inc()
            return {"error": str(e)}
    
    async def generate_monitoring_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring report"""
        try:
            # Gather all monitoring data
            agents_status = await self.get_agents_status()
            system_metrics = await self.get_system_metrics()
            orchestration_metrics = await self.get_orchestration_metrics()
            active_alerts = await self.get_active_alerts()
            
            # Calculate report metrics
            report_time = datetime.utcnow()
            
            # System health score (0-100)
            health_score = self._calculate_health_score(agents_status, system_metrics, active_alerts)
            
            # Recommendations
            recommendations = self._generate_recommendations(agents_status, system_metrics, active_alerts)
            
            report = {
                "report_metadata": {
                    "generated_at": report_time.isoformat(),
                    "report_type": "comprehensive_monitoring",
                    "version": "1.0.0"
                },
                "executive_summary": {
                    "overall_health_score": health_score,
                    "total_agents": agents_status.get('summary', {}).get('total_agents', 0),
                    "running_agents": agents_status.get('summary', {}).get('running_agents', 0),
                    "critical_alerts": active_alerts.get('summary', {}).get('critical_alerts', 0),
                    "system_cpu_usage": system_metrics.get('system', {}).get('cpu_percent', 0),
                    "system_memory_usage": system_metrics.get('system', {}).get('memory', {}).get('percent', 0)
                },
                "detailed_analysis": {
                    "agents_status": agents_status,
                    "system_metrics": system_metrics,
                    "orchestration_metrics": orchestration_metrics,
                    "active_alerts": active_alerts
                },
                "recommendations": recommendations,
                "next_report_scheduled": (report_time + timedelta(hours=24)).isoformat()
            }
            
            # Save report to file
            report_filename = f"/app/data/monitoring_report_{report_time.strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_filename, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Monitoring report generated: {report_filename}")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating monitoring report: {e}")
            monitor_errors.labels(error_type='generate_report').inc()
            return {"error": str(e)}
    
    def _calculate_health_score(self, agents_status: Dict, system_metrics: Dict, active_alerts: Dict) -> float:
        """Calculate overall system health score (0-100)"""
        try:
            score = 100.0
            
            # Agent health impact (40% of score)
            agent_summary = agents_status.get('summary', {})
            total_agents = agent_summary.get('total_agents', 1)
            running_agents = agent_summary.get('running_agents', 0)
            agent_health = (running_agents / total_agents) * 40 if total_agents > 0 else 0
            
            # System resource impact (30% of score)
            system = system_metrics.get('system', {})
            cpu_usage = system.get('cpu_percent', 0)
            memory_usage = system.get('memory', {}).get('percent', 0)
            disk_usage = system.get('disk', {}).get('percent', 0)
            
            # Penalize high resource usage
            resource_penalty = 0
            if cpu_usage > 80:
                resource_penalty += (cpu_usage - 80) / 5
            if memory_usage > 85:
                resource_penalty += (memory_usage - 85) / 3
            if disk_usage > 90:
                resource_penalty += (disk_usage - 90) / 2
            
            resource_health = max(0, 30 - resource_penalty)
            
            # Alert impact (30% of score)
            alert_summary = active_alerts.get('summary', {})
            critical_alerts = alert_summary.get('critical_alerts', 0)
            warning_alerts = alert_summary.get('warning_alerts', 0)
            
            alert_penalty = (critical_alerts * 10) + (warning_alerts * 3)
            alert_health = max(0, 30 - alert_penalty)
            
            final_score = agent_health + resource_health + alert_health
            return round(max(0, min(100, final_score)), 2)
            
        except Exception as e:
            logger.error(f"Error calculating health score: {e}")
            return 0.0
    
    def _generate_recommendations(self, agents_status: Dict, system_metrics: Dict, active_alerts: Dict) -> List[str]:
        """Generate actionable recommendations based on monitoring data"""
        recommendations = []
        
        try:
            # Agent recommendations
            agent_summary = agents_status.get('summary', {})
            if agent_summary.get('running_agents', 0) < agent_summary.get('total_agents', 0):
                stopped_count = agent_summary.get('total_agents', 0) - agent_summary.get('running_agents', 0)
                recommendations.append(f"Restart {stopped_count} stopped agent(s) to restore full system functionality")
            
            # System resource recommendations
            system = system_metrics.get('system', {})
            if system.get('cpu_percent', 0) > 80:
                recommendations.append("High CPU usage detected - consider scaling up resources or optimizing agent workloads")
            
            if system.get('memory', {}).get('percent', 0) > 85:
                recommendations.append("High memory usage detected - consider adding more RAM or optimizing memory usage")
            
            if system.get('disk', {}).get('percent', 0) > 90:
                recommendations.append("Disk space running low - clean up old logs and data or expand storage")
            
            # Alert recommendations
            alert_summary = active_alerts.get('summary', {})
            if alert_summary.get('critical_alerts', 0) > 0:
                recommendations.append("Address critical alerts immediately to prevent system degradation")
            
            if alert_summary.get('warning_alerts', 0) > 5:
                recommendations.append("Multiple warning alerts active - investigate underlying issues")
            
            # Default recommendation if everything looks good
            if not recommendations:
                recommendations.append("System is operating within normal parameters - continue monitoring")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations.append("Unable to generate recommendations due to monitoring error")
        
        return recommendations
    
    def _get_container_uptime(self, container) -> int:
        """Get container uptime in seconds"""
        try:
            if container.status != 'running':
                return 0
            
            started_at = container.attrs['State'].get('StartedAt', '')
            if started_at:
                start_time = datetime.fromisoformat(started_at.replace('Z', '+00:00').replace('+00:00', ''))
                return int((datetime.utcnow() - start_time).total_seconds())
        except:
            pass
        return 0
    
    async def websocket_handler(self, websocket: WebSocket):
        """Handle WebSocket connections for real-time monitoring"""
        await websocket.accept()
        self.active_connections.append(websocket)
        websocket_connections.set(len(self.active_connections))
        
        try:
            while True:
                # Send real-time monitoring data every 5 seconds
                monitoring_data = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "agents_summary": await self._get_agents_summary(),
                    "system_summary": await self._get_system_summary(),
                    "alerts_summary": await self._get_alerts_summary()
                }
                
                await websocket.send_text(json.dumps(monitoring_data))
                await asyncio.sleep(5)
                
        except WebSocketDisconnect:
            self.active_connections.remove(websocket)
            websocket_connections.set(len(self.active_connections))
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
                websocket_connections.set(len(self.active_connections))
    
    async def _get_agents_summary(self) -> Dict[str, Any]:
        """Get lightweight agents summary for real-time updates"""
        if not self.docker_client:
            return {"error": "Docker client not available"}
        
        try:
            containers = self.docker_client.containers.list(all=True)
            agent_containers = [c for c in containers if 'sutazai-' in c.name and 
                             any(keyword in c.name for keyword in ['agent', 'orchestrator', 'specialist', 'manager'])]
            
            running = len([c for c in agent_containers if c.status == 'running'])
            total = len(agent_containers)
            
            return {
                "total_agents": total,
                "running_agents": running,
                "health_percentage": round((running / total * 100) if total > 0 else 0, 1)
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def _get_system_summary(self) -> Dict[str, Any]:
        """Get lightweight system summary for real-time updates"""
        try:
            cpu = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            return {
                "cpu_percent": round(cpu, 1),
                "memory_percent": round(memory.percent, 1),
                "memory_used_gb": round(memory.used / (1024**3), 2)
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def _get_alerts_summary(self) -> Dict[str, Any]:
        """Get lightweight alerts summary for real-time updates"""
        try:
            # This would integrate with AlertManager in production
            return {
                "critical_alerts": 0,
                "warning_alerts": 0,
                "total_alerts": 0
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_dashboard_html(self) -> str:
        """Generate HTML dashboard for web interface"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>SutazAI Agent Monitor</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .metric-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric-title { font-size: 18px; font-weight: bold; margin-bottom: 10px; color: #2c3e50; }
        .metric-value { font-size: 24px; font-weight: bold; color: #27ae60; }
        .status-good { color: #27ae60; }
        .status-warning { color: #f39c12; }
        .status-critical { color: #e74c3c; }
        .realtime-data { background: #ecf0f1; padding: 10px; border-radius: 4px; margin-top: 10px; font-family: monospace; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 SutazAI Multi-Agent Monitoring Dashboard</h1>
            <p>Real-time monitoring of 46+ AI agents and system components</p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-title">Agent Status</div>
                <div class="metric-value" id="agent-status">Loading...</div>
                <div class="realtime-data" id="agent-data"></div>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">System Resources</div>
                <div class="metric-value" id="system-status">Loading...</div>
                <div class="realtime-data" id="system-data"></div>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">Active Alerts</div>
                <div class="metric-value" id="alert-status">Loading...</div>
                <div class="realtime-data" id="alert-data"></div>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">Workflow Status</div>
                <div class="metric-value" id="workflow-status">Loading...</div>
                <div class="realtime-data" id="workflow-data"></div>
            </div>
        </div>
        
        <div style="margin-top: 30px; text-align: center; color: #7f8c8d;">
            <p>🔄 Real-time updates via WebSocket • Last updated: <span id="last-update">Never</span></p>
        </div>
    </div>

    <script>
        // WebSocket connection for real-time updates
        const ws = new WebSocket(`ws://${window.location.host}/ws`);
        
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            updateDashboard(data);
            document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
        };
        
        ws.onerror = function(error) {
            console.error('WebSocket error:', error);
            document.getElementById('last-update').textContent = 'Connection Error';
        };
        
        function updateDashboard(data) {
            // Update agent status
            const agentData = data.agents_summary;
            const agentStatus = agentData.running_agents + '/' + agentData.total_agents + ' Running';
            document.getElementById('agent-status').textContent = agentStatus;
            document.getElementById('agent-status').className = 'metric-value ' + 
                (agentData.health_percentage > 90 ? 'status-good' : 
                 agentData.health_percentage > 70 ? 'status-warning' : 'status-critical');
            document.getElementById('agent-data').textContent = 
                `Health: ${agentData.health_percentage}%`;
            
            // Update system status
            const systemData = data.system_summary;
            document.getElementById('system-status').textContent = 
                `CPU: ${systemData.cpu_percent}% | RAM: ${systemData.memory_percent}%`;
            document.getElementById('system-status').className = 'metric-value ' + 
                (systemData.cpu_percent < 70 && systemData.memory_percent < 80 ? 'status-good' : 
                 systemData.cpu_percent < 85 && systemData.memory_percent < 90 ? 'status-warning' : 'status-critical');
            document.getElementById('system-data').textContent = 
                `Memory Used: ${systemData.memory_used_gb} GB`;
            
            // Update alert status
            const alertData = data.alerts_summary;
            const totalAlerts = alertData.total_alerts;
            document.getElementById('alert-status').textContent = totalAlerts + ' Active';
            document.getElementById('alert-status').className = 'metric-value ' + 
                (totalAlerts === 0 ? 'status-good' : 
                 alertData.critical_alerts === 0 ? 'status-warning' : 'status-critical');
            document.getElementById('alert-data').textContent = 
                `Critical: ${alertData.critical_alerts} | Warning: ${alertData.warning_alerts}`;
            
            // Update workflow status
            document.getElementById('workflow-status').textContent = 'Active';
            document.getElementById('workflow-status').className = 'metric-value status-good';
            document.getElementById('workflow-data').textContent = 'Orchestration systems operational';
        }
    </script>
</body>
</html>
        """
    
    async def start_background_tasks(self):
        """Start background monitoring tasks"""
        # Schedule automated report generation
        schedule.every().day.at("06:00").do(self.scheduled_report_generation)
        
        # Start scheduler in background thread
        def run_scheduler():
            while self.monitoring_active:
                schedule.run_pending()
                time.sleep(60)
        
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        self.background_tasks.append(scheduler_thread)
        
        logger.info("Background monitoring tasks started")
    
    def scheduled_report_generation(self):
        """Scheduled task for automated report generation"""
        try:
            # This would be called by the scheduler
            asyncio.run(self.generate_monitoring_report())
            logger.info("Scheduled monitoring report generated")
        except Exception as e:
            logger.error(f"Error in scheduled report generation: {e}")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.monitoring_active = False
        sys.exit(0)

# Initialize and run the monitoring service
if __name__ == "__main__":
    # Create monitor instance
    monitor = SutazAIAgentMonitor()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, monitor.signal_handler)
    signal.signal(signal.SIGTERM, monitor.signal_handler)
    
    # Start Prometheus metrics server
    start_http_server(8889)  # Prometheus metrics on port 8889
    logger.info("Prometheus metrics server started on port 8889")
    
    # Start background tasks
    asyncio.run(monitor.start_background_tasks())
    
    # Start the main FastAPI application
    logger.info("Starting SutazAI Agent Monitor on port 8888")
    uvicorn.run(
        monitor.app,
        host="0.0.0.0",
        port=int(os.getenv('WEBSOCKET_PORT', 8888)),
        log_level="info"
    )