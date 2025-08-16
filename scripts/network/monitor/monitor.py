#!/usr/bin/env python3
"""
MCP Network Monitoring Dashboard
Real-time monitoring of MCP service network connectivity and health
"""

import asyncio
import json
import time
import socket
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import structlog

logger = structlog.get_logger()

class NetworkMonitor:
    def __init__(self):
        self.services = [
            {"name": "postgres", "host": "mcp-postgres", "port": 11100},
            {"name": "files", "host": "mcp-files", "port": 11101},
            {"name": "http", "host": "mcp-http", "port": 11102},
            {"name": "ddg", "host": "mcp-ddg", "port": 11103},
            {"name": "github", "host": "mcp-github", "port": 11104},
            {"name": "memory", "host": "mcp-memory", "port": 11105}
        ]
        self.consul_host = "mcp-consul-agent"
        self.consul_port = 8500
        self.haproxy_host = "mcp-load-balancer"
        self.haproxy_port = 8080
        self.health_history: Dict[str, List[Dict]] = {}
        self.network_stats = {}
        
    async def check_tcp_connectivity(self, host: str, port: int, timeout: float = 5.0) -> bool:
        """Check TCP connectivity to a service"""
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), 
                timeout=timeout
            )
            writer.close()
            await writer.wait_closed()
            return True
        except Exception:
            return False
    
    async def check_http_health(self, service: Dict[str, Any]) -> Dict[str, Any]:
        """Check HTTP health of a service"""
        health_data = {
            "service": service["name"],
            "timestamp": datetime.now().isoformat(),
            "tcp_reachable": False,
            "http_healthy": False,
            "response_time": None,
            "error": None
        }
        
        try:
            # First check TCP connectivity
            health_data["tcp_reachable"] = await self.check_tcp_connectivity(
                service["host"], service["port"]
            )
            
            if not health_data["tcp_reachable"]:
                health_data["error"] = "TCP connection failed"
                return health_data
            
            # Check HTTP health endpoint
            start_time = time.time()
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"http://{service['host']}:{service['port']}/health")
                health_data["response_time"] = time.time() - start_time
                
                if response.status_code == 200:
                    response_data = response.json()
                    health_data["http_healthy"] = response_data.get("status") == "healthy"
                    health_data["response_data"] = response_data
                else:
                    health_data["error"] = f"HTTP {response.status_code}"
                    
        except httpx.TimeoutException:
            health_data["error"] = "HTTP timeout"
            health_data["response_time"] = 10.0
        except Exception as e:
            health_data["error"] = str(e)
            
        return health_data
    
    async def check_consul_health(self) -> Dict[str, Any]:
        """Check Consul service discovery health"""
        consul_health = {
            "consul_reachable": False,
            "services_registered": 0,
            "services": [],
            "error": None
        }
        
        try:
            # Check TCP connectivity to Consul
            consul_health["consul_reachable"] = await self.check_tcp_connectivity(
                self.consul_host, self.consul_port
            )
            
            if not consul_health["consul_reachable"]:
                consul_health["error"] = "Consul TCP connection failed"
                return consul_health
            
            # Query Consul for registered services
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"http://{self.consul_host}:{self.consul_port}/v1/agent/services")
                
                if response.status_code == 200:
                    services = response.json()
                    mcp_services = [s for s in services.values() if 'mcp' in s.get('Tags', [])]
                    consul_health["services_registered"] = len(mcp_services)
                    consul_health["services"] = mcp_services
                else:
                    consul_health["error"] = f"Consul API returned {response.status_code}"
                    
        except Exception as e:
            consul_health["error"] = str(e)
            
        return consul_health
    
    async def check_haproxy_health(self) -> Dict[str, Any]:
        """Check HAProxy load balancer health"""
        haproxy_health = {
            "haproxy_reachable": False,
            "backends_healthy": 0,
            "backends_total": 0,
            "stats": None,
            "error": None
        }
        
        try:
            # Check TCP connectivity to HAProxy
            haproxy_health["haproxy_reachable"] = await self.check_tcp_connectivity(
                self.haproxy_host, self.haproxy_port
            )
            
            if not haproxy_health["haproxy_reachable"]:
                haproxy_health["error"] = "HAProxy TCP connection failed"
                return haproxy_health
            
            # Get HAProxy stats
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"http://{self.haproxy_host}:{self.haproxy_port}/stats?stats;csv")
                
                if response.status_code == 200:
                    # Parse CSV stats
                    lines = response.text.strip().split('\n')
                    if len(lines) > 1:
                        headers = lines[0].split(',')
                        stats = []
                        for line in lines[1:]:
                            if line.strip():
                                values = line.split(',')
                                if len(values) >= len(headers):
                                    stat = dict(zip(headers, values))
                                    stats.append(stat)
                        
                        # Count healthy backends
                        backends = [s for s in stats if s.get('type') == '2']  # Backend servers
                        haproxy_health["backends_total"] = len(backends)
                        haproxy_health["backends_healthy"] = len([b for b in backends if b.get('status') == 'UP'])
                        haproxy_health["stats"] = stats
                        
        except Exception as e:
            haproxy_health["error"] = str(e)
            
        return haproxy_health
    
    async def collect_network_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive network metrics"""
        start_time = time.time()
        
        # Check all MCP services
        service_health = {}
        for service in self.services:
            health = await self.check_http_health(service)
            service_health[service["name"]] = health
            
            # Store health history (keep last 100 checks)
            if service["name"] not in self.health_history:
                self.health_history[service["name"]] = []
            
            self.health_history[service["name"]].append(health)
            if len(self.health_history[service["name"]]) > 100:
                self.health_history[service["name"]] = self.health_history[service["name"]][-100:]
        
        # Check infrastructure components
        consul_health = await self.check_consul_health()
        haproxy_health = await self.check_haproxy_health()
        
        # Calculate summary statistics
        total_services = len(self.services)
        healthy_services = len([h for h in service_health.values() if h["http_healthy"]])
        tcp_reachable = len([h for h in service_health.values() if h["tcp_reachable"]])
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "collection_time": time.time() - start_time,
            "summary": {
                "total_services": total_services,
                "healthy_services": healthy_services,
                "tcp_reachable": tcp_reachable,
                "health_percentage": (healthy_services / total_services * 100) if total_services > 0 else 0,
                "consul_healthy": consul_health["consul_reachable"],
                "haproxy_healthy": haproxy_health["haproxy_reachable"]
            },
            "services": service_health,
            "consul": consul_health,
            "haproxy": haproxy_health
        }
        
        self.network_stats = metrics
        return metrics
    
    def get_service_health_trend(self, service_name: str, minutes: int = 30) -> Dict[str, Any]:
        """Get health trend for a service over time"""
        if service_name not in self.health_history:
            return {"error": f"No health history for service {service_name}"}
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_checks = []
        
        for check in self.health_history[service_name]:
            check_time = datetime.fromisoformat(check["timestamp"])
            if check_time >= cutoff_time:
                recent_checks.append(check)
        
        if not recent_checks:
            return {"error": "No recent health checks"}
        
        # Calculate statistics
        total_checks = len(recent_checks)
        healthy_checks = len([c for c in recent_checks if c["http_healthy"]])
        tcp_failures = len([c for c in recent_checks if not c["tcp_reachable"]])
        avg_response_time = sum(c["response_time"] for c in recent_checks if c["response_time"]) / total_checks
        
        return {
            "service": service_name,
            "period_minutes": minutes,
            "total_checks": total_checks,
            "healthy_checks": healthy_checks,
            "health_percentage": (healthy_checks / total_checks * 100) if total_checks > 0 else 0,
            "tcp_failures": tcp_failures,
            "average_response_time": avg_response_time,
            "recent_checks": recent_checks[-10:]  # Last 10 checks
        }

# FastAPI application
app = FastAPI(title="MCP Network Monitor", version="1.0.0")
monitor = NetworkMonitor()

# Background task to collect metrics
async def metrics_collector():
    """Background task that collects metrics every 30 seconds"""
    while True:
        try:
            await monitor.collect_network_metrics()
            logger.info("Network metrics collected")
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
        await asyncio.sleep(30)

# Start metrics collection on startup
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(metrics_collector())

@app.get("/")
async def root():
    """Network monitoring dashboard"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MCP Network Monitor</title>
        <meta http-equiv="refresh" content="30">
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .status-healthy { color: green; }
            .status-unhealthy { color: red; }
            .status-warning { color: orange; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            .metric-card { 
                display: inline-block; 
                margin: 10px; 
                padding: 15px; 
                border: 1px solid #ddd; 
                border-radius: 5px; 
                min-width: 200px;
            }
        </style>
    </head>
    <body>
        <h1>MCP Network Monitor</h1>
        <p>Auto-refresh every 30 seconds | <a href="/metrics">Raw Metrics</a> | <a href="/health">Health Summary</a></p>
        
        <div id="summary"></div>
        <div id="services"></div>
        <div id="infrastructure"></div>
        
        <script>
            async function updateDashboard() {
                try {
                    const response = await fetch('/metrics');
                    const data = await response.json();
                    
                    // Update summary
                    const summary = data.summary;
                    document.getElementById('summary').innerHTML = `
                        <div class="metric-card">
                            <h3>Service Health</h3>
                            <p>${summary.healthy_services}/${summary.total_services} services healthy (${summary.health_percentage.toFixed(1)}%)</p>
                        </div>
                        <div class="metric-card">
                            <h3>Network Connectivity</h3>
                            <p>${summary.tcp_reachable}/${summary.total_services} services reachable</p>
                        </div>
                        <div class="metric-card">
                            <h3>Infrastructure</h3>
                            <p>Consul: ${summary.consul_healthy ? 'Healthy' : 'Unhealthy'}</p>
                            <p>HAProxy: ${summary.haproxy_healthy ? 'Healthy' : 'Unhealthy'}</p>
                        </div>
                    `;
                    
                    // Update services table
                    let servicesHTML = '<h2>MCP Services</h2><table><tr><th>Service</th><th>TCP</th><th>HTTP</th><th>Response Time</th><th>Error</th></tr>';
                    for (const [name, service] of Object.entries(data.services)) {
                        const tcpClass = service.tcp_reachable ? 'status-healthy' : 'status-unhealthy';
                        const httpClass = service.http_healthy ? 'status-healthy' : 'status-unhealthy';
                        const responseTime = service.response_time ? `${(service.response_time * 1000).toFixed(0)}ms` : '-';
                        const error = service.error || '-';
                        
                        servicesHTML += `<tr>
                            <td>${name}</td>
                            <td class="${tcpClass}">${service.tcp_reachable ? 'OK' : 'FAIL'}</td>
                            <td class="${httpClass}">${service.http_healthy ? 'HEALTHY' : 'UNHEALTHY'}</td>
                            <td>${responseTime}</td>
                            <td>${error}</td>
                        </tr>`;
                    }
                    servicesHTML += '</table>';
                    document.getElementById('services').innerHTML = servicesHTML;
                    
                } catch (error) {
                    console.error('Failed to update dashboard:', error);
                }
            }
            
            // Update dashboard on load and every 30 seconds
            updateDashboard();
            setInterval(updateDashboard, 30000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/metrics")
async def get_metrics():
    """Get current network metrics"""
    if not monitor.network_stats:
        await monitor.collect_network_metrics()
    return monitor.network_stats

@app.get("/health")
async def health_summary():
    """Get health summary"""
    if not monitor.network_stats:
        return JSONResponse(
            status_code=503,
            content={"status": "no_data", "message": "No metrics collected yet"}
        )
    
    summary = monitor.network_stats["summary"]
    if summary["health_percentage"] >= 80:
        status = "healthy"
    elif summary["health_percentage"] >= 60:
        status = "degraded"
    else:
        status = "unhealthy"
    
    return {
        "status": status,
        "summary": summary,
        "timestamp": monitor.network_stats["timestamp"]
    }

@app.get("/service/{service_name}/trend")
async def get_service_trend(service_name: str, minutes: int = 30):
    """Get health trend for a specific service"""
    return monitor.get_service_health_trend(service_name, minutes)

@app.post("/collect")
async def force_collect():
    """Force metrics collection"""
    await monitor.collect_network_metrics()
    return {"status": "collected", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")