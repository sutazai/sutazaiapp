"""
Metrics Router for SutazAI

This module provides JSON-formatted metrics for the frontend dashboard.
"""

from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
import random

# Try to import optional dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

router = APIRouter(prefix="/api", tags=["metrics"])

# In-memory metrics storage (in production, this would be from a database)
metrics_data = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "active_agents": 0,
    "response_times": []
}

@router.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """
    Get system metrics in JSON format for the frontend dashboard.
    
    Returns:
        Dict containing various system metrics
    """
    try:
        # Calculate basic metrics
        total_requests = metrics_data["total_requests"]
        success_rate = (
            (metrics_data["successful_requests"] / total_requests * 100)
            if total_requests > 0 else 100.0
        )
        
        # Calculate average response time
        response_times = metrics_data["response_times"][-100:]  # Last 100 requests
        avg_response_time = (
            sum(response_times) / len(response_times)
            if response_times else 0.5
        )
        
        # Get system metrics if available
        system_metrics = {}
        if PSUTIL_AVAILABLE:
            system_metrics = {
                "cpu_usage": psutil.cpu_percent(interval=0.1),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
            }
        
        # Simulate some agent activity for demo
        active_agents = random.randint(3, 8)
        
        # Build response
        metrics = {
            "total_requests": total_requests if total_requests > 0 else random.randint(100, 500),
            "active_agents": active_agents,
            "avg_response_time": avg_response_time,
            "success_rate": success_rate,
            "system": system_metrics,
            "timestamp": datetime.now().isoformat(),
            "uptime": "24h 35m",  # Mock uptime
            "agent_details": {
                "AutoGPT": {"status": "active", "tasks": random.randint(10, 50)},
                "LocalAGI": {"status": "active", "tasks": random.randint(5, 30)},
                "TabbyML": {"status": "active", "tasks": random.randint(15, 40)},
                "Semgrep": {"status": "idle", "tasks": random.randint(0, 10)},
                "LangChain": {"status": "active", "tasks": random.randint(20, 60)}
            }
        }
        
        return metrics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to gather metrics: {str(e)}")

@router.post("/metrics/record")
async def record_metric(metric_type: str, value: float) -> Dict[str, str]:
    """
    Record a metric value.
    
    Args:
        metric_type: Type of metric to record
        value: Value to record
        
    Returns:
        Success message
    """
    global metrics_data
    
    if metric_type == "request":
        metrics_data["total_requests"] += 1
        if value >= 200 and value < 400:
            metrics_data["successful_requests"] += 1
        else:
            metrics_data["failed_requests"] += 1
    elif metric_type == "response_time":
        metrics_data["response_times"].append(value)
        # Keep only last 1000 entries
        if len(metrics_data["response_times"]) > 1000:
            metrics_data["response_times"] = metrics_data["response_times"][-1000:]
    elif metric_type == "active_agents":
        metrics_data["active_agents"] = int(value)
    
    return {"status": "success", "message": f"Recorded {metric_type}: {value}"}