"""
Health Check Module
Monitors agent health and system status
"""

import logging
import threading
import time
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class HealthCheck:
    """Health monitoring system for agents and services"""
    
    def __init__(self):
        self.agents = {}
        self.running = False
        self.monitor_thread = None
        self.check_interval = 30  # seconds
        
    def start(self):
        """Start health monitoring"""
        if self.running:
            return
            
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Health monitoring started")
        
    def stop(self):
        """Stop health monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Health monitoring stopped")
        
    def register_agent_checks(self, agents: Dict[str, Any]):
        """Register agents for health monitoring"""
        self.agents.update(agents)
        logger.info(f"Registered {len(agents)} agents for health monitoring")
        
    def check_agent_health(self, agent_id: str, agent_instance: Any) -> Dict[str, Any]:
        """Check health of a specific agent"""
        try:
            if not hasattr(agent_instance, 'initialized'):
                return {"status": "unknown", "reason": "Agent not properly initialized"}
                
            if not agent_instance.initialized:
                return {"status": "stopped", "reason": "Agent not initialized"}
                
            # Check heartbeat if available
            if hasattr(agent_instance, 'get_heartbeat'):
                last_heartbeat = agent_instance.get_heartbeat()
                if last_heartbeat:
                    heartbeat_time = datetime.fromisoformat(last_heartbeat)
                    time_diff = (datetime.now() - heartbeat_time).total_seconds()
                    
                    if time_diff > 300:  # 5 minutes
                        return {
                            "status": "unhealthy", 
                            "reason": f"No heartbeat for {time_diff:.0f} seconds",
                            "last_heartbeat": last_heartbeat
                        }
                    else:
                        return {
                            "status": "healthy",
                            "last_heartbeat": last_heartbeat,
                            "heartbeat_age_seconds": time_diff
                        }
                        
            return {"status": "healthy", "reason": "Agent is initialized and responding"}
            
        except Exception as e:
            logger.error(f"Health check failed for agent {agent_id}: {e}")
            return {"status": "error", "reason": str(e)}
            
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health"""
        total_agents = len(self.agents)
        healthy_agents = 0
        unhealthy_agents = 0
        
        agent_statuses = {}
        
        for agent_id, agent_instance in self.agents.items():
            health = self.check_agent_health(agent_id, agent_instance)
            agent_statuses[agent_id] = health
            
            if health["status"] == "healthy":
                healthy_agents += 1
            else:
                unhealthy_agents += 1
                
        overall_status = "healthy" if unhealthy_agents == 0 else "degraded" if healthy_agents > 0 else "critical"
        
        return {
            "overall_status": overall_status,
            "total_agents": total_agents,
            "healthy_agents": healthy_agents,
            "unhealthy_agents": unhealthy_agents,
            "agent_details": agent_statuses,
            "timestamp": datetime.now().isoformat()
        }
        
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.running:
            try:
                health_report = self.get_system_health()
                
                if health_report["overall_status"] != "healthy":
                    logger.warning(f"System health degraded: {health_report['unhealthy_agents']} unhealthy agents")
                    
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                time.sleep(10)  # Shorter sleep on error