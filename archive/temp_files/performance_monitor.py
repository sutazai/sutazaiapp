#!/usr/bin/env python3
"""
SutazAI Performance Monitor
Real-time performance tracking and metrics collection system
"""

import time
import psutil
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import deque
import json
import logging

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Comprehensive performance monitoring system"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history = deque(maxlen=max_history)
        self.real_time_metrics = {}
        self.agent_metrics = {}
        self.system_metrics = {}
        self.api_metrics = {}
        self.model_metrics = {}
        
        # Performance thresholds
        self.thresholds = {
            "cpu_warning": 70.0,
            "cpu_critical": 90.0,
            "memory_warning": 80.0,
            "memory_critical": 95.0,
            "response_time_warning": 2.0,
            "response_time_critical": 5.0,
            "error_rate_warning": 0.05,
            "error_rate_critical": 0.15
        }
        
        self.monitoring_active = False
        self.collection_interval = 5  # seconds
        self._monitor_task = None
        
    async def start_monitoring(self):
        """Start continuous performance monitoring"""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Performance monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                system_data = self._collect_system_metrics()
                
                # Collect API metrics
                api_data = self._collect_api_metrics()
                
                # Collect model metrics
                model_data = self._collect_model_metrics()
                
                # Combine all metrics
                timestamp = datetime.now()
                metrics = {
                    "timestamp": timestamp.isoformat(),
                    "system": system_data,
                    "api": api_data,
                    "models": model_data,
                    "agents": self._get_agent_summary()
                }
                
                # Store in history
                self.metrics_history.append(metrics)
                self.real_time_metrics = metrics
                
                # Check for alerts
                self._check_performance_alerts(metrics)
                
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.collection_interval)
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system-level performance metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            # Network metrics
            network = psutil.net_io_counters()
            
            # Process count
            process_count = len(psutil.pids())
            
            # Load average (Unix systems)
            try:
                load_avg = psutil.getloadavg()
            except AttributeError:
                load_avg = [0.0, 0.0, 0.0]  # Windows doesn't have load average
            
            return {
                "cpu": {
                    "percent": cpu_percent,
                    "count": cpu_count,
                    "frequency": cpu_freq.current if cpu_freq else 0,
                    "load_average": load_avg
                },
                "memory": {
                    "total_gb": memory.total / (1024**3),
                    "available_gb": memory.available / (1024**3),
                    "used_gb": memory.used / (1024**3),
                    "percent": memory.percent,
                    "swap_total_gb": swap.total / (1024**3),
                    "swap_used_gb": swap.used / (1024**3),
                    "swap_percent": swap.percent
                },
                "disk": {
                    "total_gb": disk.total / (1024**3),
                    "used_gb": disk.used / (1024**3),
                    "free_gb": disk.free / (1024**3),
                    "percent": (disk.used / disk.total) * 100,
                    "io_read_mb": disk_io.read_bytes / (1024**2) if disk_io else 0,
                    "io_write_mb": disk_io.write_bytes / (1024**2) if disk_io else 0
                },
                "network": {
                    "bytes_sent_mb": network.bytes_sent / (1024**2),
                    "bytes_recv_mb": network.bytes_recv / (1024**2),
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv
                },
                "processes": process_count
            }
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {}
    
    def _collect_api_metrics(self) -> Dict[str, Any]:
        """Collect API performance metrics"""
        # This will be populated by API calls
        return {
            "total_requests": getattr(self, 'total_requests', 0),
            "successful_requests": getattr(self, 'successful_requests', 0),
            "failed_requests": getattr(self, 'failed_requests', 0),
            "average_response_time": getattr(self, 'avg_response_time', 0.0),
            "requests_per_minute": getattr(self, 'requests_per_minute', 0),
            "error_rate": getattr(self, 'error_rate', 0.0),
            "endpoints": getattr(self, 'endpoint_metrics', {})
        }
    
    def _collect_model_metrics(self) -> Dict[str, Any]:
        """Collect AI model performance metrics"""
        return {
            "active_models": getattr(self, 'active_models', []),
            "model_response_times": getattr(self, 'model_response_times', {}),
            "model_usage_count": getattr(self, 'model_usage_count', {}),
            "model_error_rates": getattr(self, 'model_error_rates', {}),
            "total_tokens_processed": getattr(self, 'total_tokens', 0)
        }
    
    def _get_agent_summary(self) -> Dict[str, Any]:
        """Get summary of agent performance"""
        return {
            "total_agents": len(self.agent_metrics),
            "active_agents": sum(1 for agent in self.agent_metrics.values() 
                               if agent.get('status') == 'active'),
            "average_response_time": self._calculate_average_agent_response_time(),
            "total_tasks_completed": sum(agent.get('tasks_completed', 0) 
                                       for agent in self.agent_metrics.values()),
            "total_tasks_failed": sum(agent.get('tasks_failed', 0) 
                                    for agent in self.agent_metrics.values())
        }
    
    def _calculate_average_agent_response_time(self) -> float:
        """Calculate average response time across all agents"""
        response_times = [agent.get('avg_response_time', 0) 
                         for agent in self.agent_metrics.values() 
                         if agent.get('avg_response_time', 0) > 0]
        return sum(response_times) / len(response_times) if response_times else 0.0
    
    def _check_performance_alerts(self, metrics: Dict[str, Any]):
        """Check for performance alerts and warnings"""
        alerts = []
        
        # Check system metrics
        system = metrics.get("system", {})
        
        # CPU alerts
        cpu_percent = system.get("cpu", {}).get("percent", 0)
        if cpu_percent > self.thresholds["cpu_critical"]:
            alerts.append({"type": "critical", "metric": "cpu", "value": cpu_percent})
        elif cpu_percent > self.thresholds["cpu_warning"]:
            alerts.append({"type": "warning", "metric": "cpu", "value": cpu_percent})
        
        # Memory alerts
        memory_percent = system.get("memory", {}).get("percent", 0)
        if memory_percent > self.thresholds["memory_critical"]:
            alerts.append({"type": "critical", "metric": "memory", "value": memory_percent})
        elif memory_percent > self.thresholds["memory_warning"]:
            alerts.append({"type": "warning", "metric": "memory", "value": memory_percent})
        
        # API response time alerts
        avg_response_time = metrics.get("api", {}).get("average_response_time", 0)
        if avg_response_time > self.thresholds["response_time_critical"]:
            alerts.append({"type": "critical", "metric": "response_time", "value": avg_response_time})
        elif avg_response_time > self.thresholds["response_time_warning"]:
            alerts.append({"type": "warning", "metric": "response_time", "value": avg_response_time})
        
        # Error rate alerts
        error_rate = metrics.get("api", {}).get("error_rate", 0)
        if error_rate > self.thresholds["error_rate_critical"]:
            alerts.append({"type": "critical", "metric": "error_rate", "value": error_rate})
        elif error_rate > self.thresholds["error_rate_warning"]:
            alerts.append({"type": "warning", "metric": "error_rate", "value": error_rate})
        
        if alerts:
            logger.warning(f"Performance alerts: {alerts}")
    
    # Methods for external systems to record metrics
    
    def record_api_request(self, endpoint: str, response_time: float, success: bool):
        """Record API request metrics"""
        if not hasattr(self, 'total_requests'):
            self.total_requests = 0
            self.successful_requests = 0
            self.failed_requests = 0
            self.response_times = []
            self.endpoint_metrics = {}
            self.requests_per_minute = 0
            self.last_minute_requests = deque(maxlen=60)
        
        self.total_requests += 1
        self.response_times.append(response_time)
        self.response_times = self.response_times[-100:]  # Keep last 100
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        # Update endpoint-specific metrics
        if endpoint not in self.endpoint_metrics:
            self.endpoint_metrics[endpoint] = {
                "requests": 0,
                "successes": 0,
                "failures": 0,
                "response_times": []
            }
        
        ep_metrics = self.endpoint_metrics[endpoint]
        ep_metrics["requests"] += 1
        ep_metrics["response_times"].append(response_time)
        ep_metrics["response_times"] = ep_metrics["response_times"][-50:]  # Keep last 50
        
        if success:
            ep_metrics["successes"] += 1
        else:
            ep_metrics["failures"] += 1
        
        # Calculate derived metrics
        self.avg_response_time = sum(self.response_times) / len(self.response_times)
        self.error_rate = self.failed_requests / self.total_requests if self.total_requests > 0 else 0
        
        # Track requests per minute
        now = time.time()
        self.last_minute_requests.append(now)
        # Count requests in the last minute
        minute_ago = now - 60
        recent_requests = [req_time for req_time in self.last_minute_requests if req_time > minute_ago]
        self.requests_per_minute = len(recent_requests)
    
    def record_model_usage(self, model: str, response_time: float, tokens: int, success: bool):
        """Record model usage metrics"""
        if not hasattr(self, 'model_response_times'):
            self.model_response_times = {}
            self.model_usage_count = {}
            self.model_error_rates = {}
            self.total_tokens = 0
            self.active_models = set()
        
        self.active_models.add(model)
        self.total_tokens += tokens
        
        # Initialize model metrics if needed
        if model not in self.model_response_times:
            self.model_response_times[model] = []
            self.model_usage_count[model] = 0
            self.model_error_rates[model] = {"successes": 0, "failures": 0}
        
        # Record metrics
        self.model_response_times[model].append(response_time)
        self.model_response_times[model] = self.model_response_times[model][-50:]  # Keep last 50
        self.model_usage_count[model] += 1
        
        if success:
            self.model_error_rates[model]["successes"] += 1
        else:
            self.model_error_rates[model]["failures"] += 1
    
    def record_agent_metrics(self, agent_id: str, metrics: Dict[str, Any]):
        """Record agent-specific metrics"""
        self.agent_metrics[agent_id] = {
            **metrics,
            "last_updated": datetime.now().isoformat()
        }
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current real-time metrics"""
        return self.real_time_metrics
    
    def get_metrics_history(self, minutes: int = 60) -> List[Dict[str, Any]]:
        """Get metrics history for specified time period"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        filtered_history = []
        for metric in self.metrics_history:
            try:
                metric_time = datetime.fromisoformat(metric["timestamp"])
                if metric_time >= cutoff_time:
                    filtered_history.append(metric)
            except (KeyError, ValueError):
                continue
        
        return filtered_history
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        current = self.real_time_metrics
        
        if not current:
            return {"status": "No metrics available", "monitoring_active": self.monitoring_active}
        
        system = current.get("system", {})
        api = current.get("api", {})
        models = current.get("models", {})
        agents = current.get("agents", {})
        
        # Determine overall health status
        health_status = "healthy"
        
        # Check system health
        cpu_percent = system.get("cpu", {}).get("percent", 0)
        memory_percent = system.get("memory", {}).get("percent", 0)
        
        if (cpu_percent > self.thresholds["cpu_critical"] or 
            memory_percent > self.thresholds["memory_critical"]):
            health_status = "critical"
        elif (cpu_percent > self.thresholds["cpu_warning"] or 
              memory_percent > self.thresholds["memory_warning"]):
            health_status = "warning"
        
        # Check API health
        error_rate = api.get("error_rate", 0)
        avg_response_time = api.get("average_response_time", 0)
        
        if (error_rate > self.thresholds["error_rate_critical"] or 
            avg_response_time > self.thresholds["response_time_critical"]):
            health_status = "critical"
        elif (error_rate > self.thresholds["error_rate_warning"] or 
              avg_response_time > self.thresholds["response_time_warning"]):
            if health_status == "healthy":
                health_status = "warning"
        
        return {
            "overall_health": health_status,
            "monitoring_active": self.monitoring_active,
            "last_updated": current.get("timestamp"),
            "system_summary": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "disk_percent": system.get("disk", {}).get("percent", 0),
                "process_count": system.get("processes", 0)
            },
            "api_summary": {
                "total_requests": api.get("total_requests", 0),
                "error_rate": error_rate,
                "average_response_time": avg_response_time,
                "requests_per_minute": api.get("requests_per_minute", 0)
            },
            "model_summary": {
                "active_models": len(models.get("active_models", [])),
                "total_tokens_processed": models.get("total_tokens_processed", 0)
            },
            "agent_summary": {
                "total_agents": agents.get("total_agents", 0),
                "active_agents": agents.get("active_agents", 0),
                "tasks_completed": agents.get("total_tasks_completed", 0),
                "tasks_failed": agents.get("total_tasks_failed", 0)
            }
        }


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


# Decorator for automatic API metrics collection
def track_performance(endpoint_name: str = None):
    """Decorator to automatically track API performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                response_time = time.time() - start_time
                endpoint = endpoint_name or func.__name__
                performance_monitor.record_api_request(endpoint, response_time, success)
        
        return wrapper
    return decorator