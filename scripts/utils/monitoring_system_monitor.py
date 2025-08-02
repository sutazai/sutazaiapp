#!/usr/bin/env python3
"""
Comprehensive System Monitoring and Alerting Service
"""

from fastapi import FastAPI
import uvicorn
from datetime import datetime
import json
import psutil
import requests
from typing import Dict, List
import asyncio

app = FastAPI(title="SutazAI System Monitor", version="1.0")

# Service endpoints to monitor
SERVICES = {
    "streamlit": "http://localhost:8501",
    "backend": "http://localhost:8000/health",
    "crewai": "http://localhost:8102/health",
    "autogpt": "http://localhost:8103/health", 
    "agentgpt": "http://localhost:8104/health",
    "privategpt": "http://localhost:8105/health",
    "llamaindex": "http://localhost:8106/health",
    "pytorch": "http://localhost:8085/health",
    "tensorflow": "http://localhost:8086/health",
    "jax": "http://localhost:8087/health",
    "gpt_engineer": "http://localhost:8088/health",
    "aider": "http://localhost:8089/health",
    "documind": "http://localhost:8090/health",
    "browser_use": "http://localhost:8091/health",
    "skyvern": "http://localhost:8092/health"
}

@app.get("/")
async def root():
    return {"service": "System Monitor", "status": "active", "timestamp": datetime.now().isoformat()}

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "system_monitor", "port": 8093}

@app.get("/system")
async def system_metrics():
    try:
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "cpu": {
                "usage_percent": cpu_percent,
                "count": psutil.cpu_count(),
                "load_avg": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            },
            "memory": {
                "total_gb": round(memory.total / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "usage_percent": memory.percent
            },
            "disk": {
                "total_gb": round(disk.total / (1024**3), 2),
                "used_gb": round(disk.used / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "usage_percent": round((disk.used / disk.total) * 100, 2)
            },
            "processes": len(psutil.pids()),
            "uptime": psutil.boot_time()
        }
        
        return {"system_metrics": metrics, "status": "collected"}
    except Exception as e:
        return {"error": str(e), "service": "System Monitor"}

@app.get("/services")
async def service_status():
    try:
        service_health = {}
        
        for service_name, url in SERVICES.items():
            try:
                response = requests.get(url, timeout=5)
                service_health[service_name] = {
                    "status": "healthy" if response.status_code == 200 else "unhealthy",
                    "response_time": response.elapsed.total_seconds(),
                    "status_code": response.status_code,
                    "last_check": datetime.now().isoformat()
                }
            except Exception as e:
                service_health[service_name] = {
                    "status": "down",
                    "error": str(e),
                    "last_check": datetime.now().isoformat()
                }
        
        # Calculate overall health
        healthy_count = sum(1 for s in service_health.values() if s["status"] == "healthy")
        total_count = len(service_health)
        overall_health = (healthy_count / total_count) * 100 if total_count > 0 else 0
        
        return {
            "services": service_health,
            "summary": {
                "total_services": total_count,
                "healthy_services": healthy_count,
                "overall_health_percent": round(overall_health, 2)
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e), "service": "System Monitor"}

@app.get("/alerts")
async def get_alerts():
    try:
        alerts = []
        
        # System resource alerts
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        if cpu_percent > 80:
            alerts.append({
                "type": "warning",
                "category": "system",
                "message": f"High CPU usage: {cpu_percent}%",
                "severity": "medium",
                "timestamp": datetime.now().isoformat()
            })
        
        if memory.percent > 85:
            alerts.append({
                "type": "warning", 
                "category": "system",
                "message": f"High memory usage: {memory.percent}%",
                "severity": "high",
                "timestamp": datetime.now().isoformat()
            })
        
        if (disk.used / disk.total) * 100 > 90:
            alerts.append({
                "type": "critical",
                "category": "system", 
                "message": f"Disk space low: {round((disk.used / disk.total) * 100, 2)}%",
                "severity": "critical",
                "timestamp": datetime.now().isoformat()
            })
        
        # Service health alerts
        for service_name, url in SERVICES.items():
            try:
                response = requests.get(url, timeout=5)
                if response.status_code != 200:
                    alerts.append({
                        "type": "error",
                        "category": "service",
                        "message": f"Service {service_name} is unhealthy",
                        "severity": "high",
                        "timestamp": datetime.now().isoformat()
                    })
            except:
                alerts.append({
                    "type": "error",
                    "category": "service", 
                    "message": f"Service {service_name} is down",
                    "severity": "critical",
                    "timestamp": datetime.now().isoformat()
                })
        
        return {
            "alerts": alerts,
            "alert_count": len(alerts),
            "severity_summary": {
                "critical": len([a for a in alerts if a["severity"] == "critical"]),
                "high": len([a for a in alerts if a["severity"] == "high"]), 
                "medium": len([a for a in alerts if a["severity"] == "medium"])
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e), "service": "System Monitor"}

@app.get("/dashboard")
async def monitoring_dashboard():
    try:
        # Get all monitoring data
        system_data = await system_metrics()
        service_data = await service_status()
        alert_data = await get_alerts()
        
        dashboard = {
            "system": system_data.get("system_metrics", {}),
            "services": service_data.get("summary", {}),
            "alerts": alert_data.get("severity_summary", {}),
            "timestamp": datetime.now().isoformat(),
            "status": "operational"
        }
        
        return {"dashboard": dashboard, "status": "generated"}
    except Exception as e:
        return {"error": str(e), "service": "System Monitor"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8093)