#!/usr/bin/env python3
"""
SutazAI Performance Metrics
Tracks and analyzes agent performance metrics
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import time
import logging

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Tracks and analyzes agent performance metrics"""
    
    def __init__(self):
        self.metrics = {}
        self.response_times = {}
        self.task_completion_rates = {}
        self.resource_usage = {}
        self.initialized = False
    
    def initialize(self) -> None:
        """Initialize performance metrics system"""
        self.initialized = True
        logger.info("Performance metrics system initialized")
    
    def record_task_start(self, agent_id: str, task_id: str) -> None:
        """Record when an agent starts a task"""
        if agent_id not in self.metrics:
            self.metrics[agent_id] = {
                "tasks_started": 0,
                "tasks_completed": 0,
                "tasks_failed": 0,
                "active_tasks": {},
                "average_response_time": 0.0
            }
        
        self.metrics[agent_id]["tasks_started"] += 1
        self.metrics[agent_id]["active_tasks"][task_id] = {
            "started_at": time.time(),
            "status": "running"
        }
    
    def record_task_completion(self, agent_id: str, task_id: str, success: bool = True) -> None:
        """Record when an agent completes a task"""
        if agent_id in self.metrics and task_id in self.metrics[agent_id]["active_tasks"]:
            task_info = self.metrics[agent_id]["active_tasks"][task_id]
            completion_time = time.time() - task_info["started_at"]
            
            # Update metrics
            if success:
                self.metrics[agent_id]["tasks_completed"] += 1
            else:
                self.metrics[agent_id]["tasks_failed"] += 1
            
            # Update response times
            if agent_id not in self.response_times:
                self.response_times[agent_id] = []
            
            self.response_times[agent_id].append(completion_time)
            
            # Calculate average response time
            recent_times = self.response_times[agent_id][-10:]  # Last 10 tasks
            self.metrics[agent_id]["average_response_time"] = sum(recent_times) / len(recent_times)
            
            # Remove from active tasks
            del self.metrics[agent_id]["active_tasks"][task_id]
    
    def record_resource_usage(self, agent_id: str, cpu_percent: float, memory_mb: float) -> None:
        """Record resource usage for an agent"""
        if agent_id not in self.resource_usage:
            self.resource_usage[agent_id] = {
                "cpu_history": [],
                "memory_history": [],
                "average_cpu": 0.0,
                "average_memory": 0.0
            }
        
        usage = self.resource_usage[agent_id]
        usage["cpu_history"].append({
            "value": cpu_percent,
            "timestamp": datetime.now().isoformat()
        })
        usage["memory_history"].append({
            "value": memory_mb,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 100 entries
        usage["cpu_history"] = usage["cpu_history"][-100:]
        usage["memory_history"] = usage["memory_history"][-100:]
        
        # Calculate averages
        usage["average_cpu"] = sum(entry["value"] for entry in usage["cpu_history"]) / len(usage["cpu_history"])
        usage["average_memory"] = sum(entry["value"] for entry in usage["memory_history"]) / len(usage["memory_history"])
    
    def get_agent_metrics(self, agent_id: str) -> Dict[str, Any]:
        """Get performance metrics for specific agent"""
        if agent_id not in self.metrics:
            return {"error": f"No metrics found for agent {agent_id}"}
        
        agent_metrics = self.metrics[agent_id].copy()
        
        # Add resource usage if available
        if agent_id in self.resource_usage:
            agent_metrics["resource_usage"] = self.resource_usage[agent_id]
        
        # Calculate completion rate
        total_tasks = agent_metrics["tasks_completed"] + agent_metrics["tasks_failed"]
        if total_tasks > 0:
            agent_metrics["completion_rate"] = agent_metrics["tasks_completed"] / total_tasks
        else:
            agent_metrics["completion_rate"] = 0.0
        
        return agent_metrics
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get overall system performance metrics"""
        total_agents = len(self.metrics)
        total_tasks_completed = sum(agent["tasks_completed"] for agent in self.metrics.values())
        total_tasks_failed = sum(agent["tasks_failed"] for agent in self.metrics.values())
        
        active_tasks = sum(len(agent["active_tasks"]) for agent in self.metrics.values())
        
        average_response_time = 0.0
        if self.metrics:
            response_times = [agent["average_response_time"] for agent in self.metrics.values() 
                            if agent["average_response_time"] > 0]
            if response_times:
                average_response_time = sum(response_times) / len(response_times)
        
        return {
            "total_agents": total_agents,
            "active_tasks": active_tasks,
            "tasks_completed": total_tasks_completed,
            "tasks_failed": total_tasks_failed,
            "average_response_time": average_response_time,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_top_performers(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get top performing agents"""
        agent_performance = []
        
        for agent_id, metrics in self.metrics.items():
            total_tasks = metrics["tasks_completed"] + metrics["tasks_failed"]
            completion_rate = metrics["tasks_completed"] / total_tasks if total_tasks > 0 else 0
            
            agent_performance.append({
                "agent_id": agent_id,
                "completion_rate": completion_rate,
                "average_response_time": metrics["average_response_time"],
                "total_tasks": total_tasks
            })
        
        # Sort by completion rate, then by response time
        agent_performance.sort(key=lambda x: (x["completion_rate"], -x["average_response_time"]), reverse=True)
        
        return agent_performance[:limit]
    
    def cleanup(self) -> None:
        """Cleanup metrics system"""
        self.metrics.clear()
        self.response_times.clear()
        self.task_completion_rates.clear()
        self.resource_usage.clear()
        self.initialized = False