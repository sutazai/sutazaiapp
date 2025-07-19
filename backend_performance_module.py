#!/usr/bin/env python3
"""
Performance Monitoring Module for SutazAI Backend
Provides real system metrics and monitoring data
"""

import psutil
import time
import json
from datetime import datetime, timedelta
from collections import deque
import threading
import logging

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    def __init__(self):
        self.metrics_history = deque(maxlen=300)  # 5 minutes of data at 1s intervals
        self.api_calls = deque(maxlen=1000)
        self.model_usage = {}
        self.agent_status = {}
        self.alerts = []
        self.start_time = time.time()
        self._lock = threading.Lock()
        
        # Start background monitoring
        self.monitoring_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self.monitoring_thread.start()
    
    def _monitor_system(self):
        """Background thread to collect system metrics"""
        while True:
            try:
                metrics = {
                    "timestamp": datetime.now().isoformat(),
                    "cpu": {
                        "percent": psutil.cpu_percent(interval=0.1),
                        "count": psutil.cpu_count(),
                        "freq": psutil.cpu_freq().current if psutil.cpu_freq() else 0
                    },
                    "memory": {
                        "percent": psutil.virtual_memory().percent,
                        "used_gb": psutil.virtual_memory().used / (1024**3),
                        "total_gb": psutil.virtual_memory().total / (1024**3)
                    },
                    "processes": len(psutil.pids()),
                    "network": {
                        "bytes_sent": psutil.net_io_counters().bytes_sent,
                        "bytes_recv": psutil.net_io_counters().bytes_recv
                    }
                }
                
                with self._lock:
                    self.metrics_history.append(metrics)
                    
            except Exception as e:
                logger.error(f"Error monitoring system: {e}")
            
            time.sleep(1)
    
    def record_api_call(self, endpoint: str, duration: float, status: int, error: str = None):
        """Record an API call for metrics"""
        with self._lock:
            self.api_calls.append({
                "timestamp": datetime.now().isoformat(),
                "endpoint": endpoint,
                "duration": duration,
                "status": status,
                "error": error
            })
    
    def record_model_usage(self, model: str, tokens: int, duration: float):
        """Record model usage statistics"""
        with self._lock:
            if model not in self.model_usage:
                self.model_usage[model] = {
                    "calls": 0,
                    "total_tokens": 0,
                    "total_duration": 0,
                    "last_used": None
                }
            
            self.model_usage[model]["calls"] += 1
            self.model_usage[model]["total_tokens"] += tokens
            self.model_usage[model]["total_duration"] += duration
            self.model_usage[model]["last_used"] = datetime.now().isoformat()
    
    def update_agent_status(self, agent_name: str, status: str, tasks_completed: int = 0):
        """Update agent status"""
        with self._lock:
            self.agent_status[agent_name] = {
                "status": status,
                "tasks_completed": tasks_completed,
                "last_updated": datetime.now().isoformat()
            }
    
    def get_performance_summary(self):
        """Get current performance summary"""
        with self._lock:
            # Get latest metrics
            latest_metrics = self.metrics_history[-1] if self.metrics_history else None
            
            # Calculate API metrics
            recent_calls = [c for c in self.api_calls 
                          if datetime.fromisoformat(c["timestamp"]) > datetime.now() - timedelta(minutes=1)]
            
            error_calls = [c for c in recent_calls if c["status"] >= 400]
            avg_duration = sum(c["duration"] for c in recent_calls) / len(recent_calls) if recent_calls else 0
            
            # Count active models
            active_models = [m for m, stats in self.model_usage.items() 
                           if stats["last_used"] and 
                           datetime.fromisoformat(stats["last_used"]) > datetime.now() - timedelta(minutes=5)]
            
            # Total tokens processed
            total_tokens = sum(stats["total_tokens"] for stats in self.model_usage.values())
            
            return {
                "system": {
                    "cpu_usage": latest_metrics["cpu"]["percent"] if latest_metrics else 0,
                    "memory_usage": latest_metrics["memory"]["percent"] if latest_metrics else 0,
                    "processes": latest_metrics["processes"] if latest_metrics else 0,
                    "uptime_hours": (time.time() - self.start_time) / 3600
                },
                "api": {
                    "total_requests": len(self.api_calls),
                    "requests_per_minute": len(recent_calls),
                    "error_rate": (len(error_calls) / len(recent_calls) * 100) if recent_calls else 0,
                    "average_response_time": round(avg_duration, 3)
                },
                "models": {
                    "active_models": len(active_models),
                    "total_tokens": total_tokens,
                    "model_details": self.model_usage
                },
                "agents": {
                    "active_agents": len([a for a in self.agent_status.values() if a["status"] == "active"]),
                    "total_tasks": sum(a.get("tasks_completed", 0) for a in self.agent_status.values()),
                    "agent_details": self.agent_status
                }
            }
    
    def get_performance_alerts(self):
        """Get current performance alerts"""
        alerts = []
        
        with self._lock:
            if self.metrics_history:
                latest = self.metrics_history[-1]
                
                # High CPU alert
                if latest["cpu"]["percent"] > 80:
                    alerts.append({
                        "level": "warning",
                        "message": f"High CPU usage: {latest['cpu']['percent']}%",
                        "timestamp": latest["timestamp"]
                    })
                
                # High memory alert
                if latest["memory"]["percent"] > 85:
                    alerts.append({
                        "level": "warning",
                        "message": f"High memory usage: {latest['memory']['percent']}%",
                        "timestamp": latest["timestamp"]
                    })
                
                # API errors alert
                recent_errors = [c for c in self.api_calls[-100:] if c["status"] >= 500]
                if len(recent_errors) > 5:
                    alerts.append({
                        "level": "error",
                        "message": f"Multiple API errors detected: {len(recent_errors)} in last 100 calls",
                        "timestamp": datetime.now().isoformat()
                    })
        
        return {"alerts": alerts, "status": "healthy" if not alerts else "warning"}

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

# Initialize with some agent data
performance_monitor.update_agent_status("ChatBot", "active", 10)
performance_monitor.update_agent_status("Reasoning Engine", "ready", 0)
performance_monitor.update_agent_status("Knowledge Manager", "ready", 0)