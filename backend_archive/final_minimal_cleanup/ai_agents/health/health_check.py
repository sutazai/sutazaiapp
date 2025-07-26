#!/usr/bin/env python3
"""
SutazAI Health Check System
Monitors agent health and system status
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import time
import logging

logger = logging.getLogger(__name__)


class HealthStatus:
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class HealthCheck:
    """Monitors agent health and system status"""
    
    def __init__(self):
        self.agent_health = {}
        self.system_health = {}
        self.health_history = []
        self.thresholds = {
            "response_time": 5.0,  # seconds
            "error_rate": 0.1,     # 10%
            "memory_usage": 0.8,   # 80%
            "cpu_usage": 0.9       # 90%
        }
        self.initialized = False
    
    def initialize(self) -> None:
        """Initialize health check system"""
        self.initialized = True
        logger.info("Health check system initialized")
    
    def register_agent(self, agent_id: str, agent_info: Dict[str, Any] = None) -> None:
        """Register an agent for health monitoring"""
        self.agent_health[agent_id] = {
            "id": agent_id,
            "status": HealthStatus.UNKNOWN,
            "last_heartbeat": None,
            "response_times": [],
            "error_count": 0,
            "success_count": 0,
            "resource_usage": {},
            "info": agent_info or {},
            "registered_at": datetime.now().isoformat()
        }
    
    def update_agent_heartbeat(self, agent_id: str) -> None:
        """Update agent heartbeat"""
        if agent_id in self.agent_health:
            self.agent_health[agent_id]["last_heartbeat"] = datetime.now().isoformat()
            self._update_agent_status(agent_id)
    
    def record_agent_response(self, agent_id: str, response_time: float, success: bool = True) -> None:
        """Record agent response time and success/failure"""
        if agent_id not in self.agent_health:
            self.register_agent(agent_id)
        
        agent = self.agent_health[agent_id]
        
        # Update response times (keep last 10)
        agent["response_times"].append(response_time)
        agent["response_times"] = agent["response_times"][-10:]
        
        # Update counters
        if success:
            agent["success_count"] += 1
        else:
            agent["error_count"] += 1
        
        # Update status based on new data
        self._update_agent_status(agent_id)
    
    def update_agent_resources(self, agent_id: str, cpu_percent: float, memory_percent: float) -> None:
        """Update agent resource usage"""
        if agent_id in self.agent_health:
            self.agent_health[agent_id]["resource_usage"] = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "updated_at": datetime.now().isoformat()
            }
            self._update_agent_status(agent_id)
    
    def _update_agent_status(self, agent_id: str) -> None:
        """Update agent health status based on all metrics"""
        if agent_id not in self.agent_health:
            return
        
        agent = self.agent_health[agent_id]
        status_factors = []
        
        # Check heartbeat
        if agent["last_heartbeat"]:
            last_heartbeat = datetime.fromisoformat(agent["last_heartbeat"])
            time_since_heartbeat = (datetime.now() - last_heartbeat).total_seconds()
            
            if time_since_heartbeat > 300:  # 5 minutes
                status_factors.append(HealthStatus.CRITICAL)
            elif time_since_heartbeat > 120:  # 2 minutes
                status_factors.append(HealthStatus.WARNING)
            else:
                status_factors.append(HealthStatus.HEALTHY)
        else:
            status_factors.append(HealthStatus.UNKNOWN)
        
        # Check response times
        if agent["response_times"]:
            avg_response_time = sum(agent["response_times"]) / len(agent["response_times"])
            if avg_response_time > self.thresholds["response_time"]:
                status_factors.append(HealthStatus.CRITICAL)
            elif avg_response_time > self.thresholds["response_time"] * 0.7:
                status_factors.append(HealthStatus.WARNING)
            else:
                status_factors.append(HealthStatus.HEALTHY)
        
        # Check error rate
        total_requests = agent["success_count"] + agent["error_count"]
        if total_requests > 0:
            error_rate = agent["error_count"] / total_requests
            if error_rate > self.thresholds["error_rate"]:
                status_factors.append(HealthStatus.CRITICAL)
            elif error_rate > self.thresholds["error_rate"] * 0.5:
                status_factors.append(HealthStatus.WARNING)
            else:
                status_factors.append(HealthStatus.HEALTHY)
        
        # Check resource usage
        resources = agent.get("resource_usage", {})
        if resources:
            cpu = resources.get("cpu_percent", 0)
            memory = resources.get("memory_percent", 0)
            
            if cpu > self.thresholds["cpu_usage"] * 100 or memory > self.thresholds["memory_usage"] * 100:
                status_factors.append(HealthStatus.CRITICAL)
            elif cpu > self.thresholds["cpu_usage"] * 70 or memory > self.thresholds["memory_usage"] * 70:
                status_factors.append(HealthStatus.WARNING)
            else:
                status_factors.append(HealthStatus.HEALTHY)
        
        # Determine overall status (worst case)
        if HealthStatus.CRITICAL in status_factors:
            agent["status"] = HealthStatus.CRITICAL
        elif HealthStatus.WARNING in status_factors:
            agent["status"] = HealthStatus.WARNING
        elif HealthStatus.HEALTHY in status_factors:
            agent["status"] = HealthStatus.HEALTHY
        else:
            agent["status"] = HealthStatus.UNKNOWN
        
        agent["last_status_update"] = datetime.now().isoformat()
    
    def get_agent_health(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get health status for specific agent"""
        return self.agent_health.get(agent_id)
    
    def get_all_agents_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health status for all agents"""
        return self.agent_health.copy()
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get overall system health summary"""
        if not self.agent_health:
            return {
                "overall_status": HealthStatus.UNKNOWN,
                "total_agents": 0,
                "healthy_agents": 0,
                "warning_agents": 0,
                "critical_agents": 0,
                "timestamp": datetime.now().isoformat()
            }
        
        status_counts = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.WARNING: 0,
            HealthStatus.CRITICAL: 0,
            HealthStatus.UNKNOWN: 0
        }
        
        for agent in self.agent_health.values():
            status_counts[agent["status"]] += 1
        
        # Determine overall system status
        if status_counts[HealthStatus.CRITICAL] > 0:
            overall_status = HealthStatus.CRITICAL
        elif status_counts[HealthStatus.WARNING] > 0:
            overall_status = HealthStatus.WARNING
        elif status_counts[HealthStatus.HEALTHY] > 0:
            overall_status = HealthStatus.HEALTHY
        else:
            overall_status = HealthStatus.UNKNOWN
        
        return {
            "overall_status": overall_status,
            "total_agents": len(self.agent_health),
            "healthy_agents": status_counts[HealthStatus.HEALTHY],
            "warning_agents": status_counts[HealthStatus.WARNING],
            "critical_agents": status_counts[HealthStatus.CRITICAL],
            "unknown_agents": status_counts[HealthStatus.UNKNOWN],
            "timestamp": datetime.now().isoformat()
        }
    
    def get_unhealthy_agents(self) -> List[Dict[str, Any]]:
        """Get list of agents with health issues"""
        unhealthy = []
        for agent_id, health in self.agent_health.items():
            if health["status"] in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                unhealthy.append(health)
        return unhealthy
    
    def set_health_thresholds(self, thresholds: Dict[str, float]) -> None:
        """Update health check thresholds"""
        self.thresholds.update(thresholds)
    
    def cleanup(self) -> None:
        """Cleanup health check system"""
        self.agent_health.clear()
        self.system_health.clear()
        self.health_history.clear()
        self.initialized = False