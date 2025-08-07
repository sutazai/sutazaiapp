"""
Agent Hibernation Manager - Intelligent agent sleep/wake policies for power optimization
"""

import time
import threading
import logging
import psutil
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)

class HibernationState(Enum):
    """Agent hibernation states"""
    ACTIVE = "active"           # Agent is running normally
    IDLE = "idle"              # Agent is idle but not hibernated
    HIBERNATING = "hibernating" # Agent is being hibernated
    HIBERNATED = "hibernated"   # Agent is in hibernation
    WAKING = "waking"          # Agent is being woken up
    FAILED = "failed"          # Hibernation/wake failed

@dataclass
class HibernationPolicy:
    """Policy defining when and how agents should hibernate"""
    name: str
    idle_threshold_minutes: int = 30        # Time before considering hibernation
    hibernate_threshold_minutes: int = 60   # Time before actual hibernation
    max_hibernation_hours: int = 8          # Maximum hibernation time
    cpu_threshold_percent: float = 1.0      # CPU usage threshold
    memory_threshold_mb: int = 100          # Memory usage threshold
    priority_score: int = 1                 # Higher = more important, hibernated last
    wake_conditions: List[str] = field(default_factory=list)  # Conditions to wake agent
    enabled: bool = True

@dataclass
class AgentMetrics:
    """Metrics for an agent related to hibernation decisions"""
    agent_id: str
    last_activity: datetime
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    task_queue_size: int = 0
    priority_score: int = 1
    hibernation_count: int = 0
    total_hibernation_time: timedelta = field(default_factory=lambda: timedelta(0))
    last_hibernation: Optional[datetime] = None
    wake_reason: Optional[str] = None

@dataclass
class HibernationEvent:
    """Event record for hibernation actions"""
    agent_id: str
    action: str  # "hibernate", "wake", "failed_hibernate", "failed_wake"
    timestamp: datetime
    reason: str
    metrics_before: Dict[str, Any]
    metrics_after: Optional[Dict[str, Any]] = None
    power_saved_w: Optional[float] = None

class AgentHibernationManager:
    """Manages agent hibernation for power optimization"""
    
    def __init__(self, agent_manager=None):
        """
        Initialize hibernation manager
        
        Args:
            agent_manager: Reference to the main agent manager
        """
        self.agent_manager = agent_manager
        self._hibernation_policies: Dict[str, HibernationPolicy] = {}
        self._agent_metrics: Dict[str, AgentMetrics] = {}
        self._hibernated_agents: Dict[str, Dict[str, Any]] = {}
        self._hibernation_history: List[HibernationEvent] = []
        
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Power estimation constants
        self._agent_base_power_w = 2.0    # Base power per active agent
        self._hibernated_power_w = 0.1    # Power consumption when hibernated
        
        self._setup_default_policies()
    
    def _setup_default_policies(self) -> None:
        """Setup default hibernation policies"""
        
        # Conservative policy - for important agents
        self.add_policy(HibernationPolicy(
            name="conservative",
            idle_threshold_minutes=60,
            hibernate_threshold_minutes=120,
            max_hibernation_hours=4,
            cpu_threshold_percent=0.5,
            priority_score=10,
            wake_conditions=["task_queued", "high_system_load", "manual_wake"]
        ))
        
        # Balanced policy - for standard agents
        self.add_policy(HibernationPolicy(
            name="balanced",
            idle_threshold_minutes=30,
            hibernate_threshold_minutes=60,
            max_hibernation_hours=6,
            cpu_threshold_percent=1.0,
            priority_score=5,
            wake_conditions=["task_queued", "system_load", "scheduled_wake"]
        ))
        
        # Aggressive policy - for low-priority agents
        self.add_policy(HibernationPolicy(
            name="aggressive",
            idle_threshold_minutes=15,
            hibernate_threshold_minutes=30,
            max_hibernation_hours=12,
            cpu_threshold_percent=2.0,
            priority_score=1,
            wake_conditions=["task_queued", "manual_wake"]
        ))
    
    def add_policy(self, policy: HibernationPolicy) -> None:
        """Add a hibernation policy"""
        self._hibernation_policies[policy.name] = policy
        logger.info(f"Added hibernation policy: {policy.name}")
    
    def remove_policy(self, policy_name: str) -> bool:
        """Remove a hibernation policy"""
        if policy_name in self._hibernation_policies:
            del self._hibernation_policies[policy_name]
            logger.info(f"Removed hibernation policy: {policy_name}")
            return True
        return False
    
    def assign_policy_to_agent(self, agent_id: str, policy_name: str) -> bool:
        """Assign a hibernation policy to an agent"""
        if policy_name not in self._hibernation_policies:
            logger.error(f"Policy '{policy_name}' not found")
            return False
        
        if agent_id not in self._agent_metrics:
            self._agent_metrics[agent_id] = AgentMetrics(
                agent_id=agent_id,
                last_activity=datetime.now()
            )
        
        # Store policy assignment in agent metrics
        metrics = self._agent_metrics[agent_id]
        metrics.priority_score = self._hibernation_policies[policy_name].priority_score
        
        logger.info(f"Assigned policy '{policy_name}' to agent {agent_id}")
        return True
    
    def start_monitoring(self) -> None:
        """Start hibernation monitoring"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Agent hibernation monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop hibernation monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
        
        # Wake all hibernated agents
        with self._lock:
            hibernated_agents = list(self._hibernated_agents.keys())
        
        for agent_id in hibernated_agents:
            self.wake_agent(agent_id, "shutdown")
        
        logger.info("Agent hibernation monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self._monitoring:
            try:
                self._update_agent_metrics()
                self._evaluate_hibernation_candidates()
                self._evaluate_wake_candidates()
                self._cleanup_old_events()
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in hibernation monitoring loop: {e}")
                time.sleep(60)
    
    def _update_agent_metrics(self) -> None:
        """Update metrics for all agents"""
        if not self.agent_manager:
            return
        
        try:
            active_agents = self.agent_manager.get_active_agents()
            
            for agent_id in active_agents:
                if agent_id not in self._agent_metrics:
                    self._agent_metrics[agent_id] = AgentMetrics(
                        agent_id=agent_id,
                        last_activity=datetime.now()
                    )
                
                metrics = self._agent_metrics[agent_id]
                
                # Get agent status and metrics
                try:
                    agent_status = self.agent_manager.get_agent_status(agent_id)
                    agent_metrics = self.agent_manager.get_agent_metrics(agent_id)
                    
                    # Update CPU and memory usage
                    metrics.cpu_usage_percent = agent_metrics.get('cpu_percent', 0.0)
                    metrics.memory_usage_mb = agent_metrics.get('memory_percent', 0.0) * 1024  # Rough estimate
                    
                    # Check if agent had recent activity
                    if (agent_metrics.get('execution_count', 0) > 0 or 
                        agent_status.get('health', {}).get('status') == 'active'):
                        metrics.last_activity = datetime.now()
                        
                except Exception as e:
                    logger.warning(f"Error updating metrics for agent {agent_id}: {e}")
                    
        except Exception as e:
            logger.error(f"Error updating agent metrics: {e}")
    
    def _evaluate_hibernation_candidates(self) -> None:
        """Evaluate which agents should be hibernated"""
        current_time = datetime.now()
        
        for agent_id, metrics in self._agent_metrics.items():
            # Skip already hibernated agents
            if agent_id in self._hibernated_agents:
                continue
            
            # Find applicable policy
            policy = self._find_policy_for_agent(agent_id)
            if not policy or not policy.enabled:
                continue
            
            # Check if agent meets hibernation criteria
            idle_time = (current_time - metrics.last_activity).total_seconds() / 60
            
            if (idle_time >= policy.hibernate_threshold_minutes and
                metrics.cpu_usage_percent <= policy.cpu_threshold_percent and
                metrics.memory_usage_mb <= policy.memory_threshold_mb and
                metrics.task_queue_size == 0):  # No pending tasks
                
                self._hibernate_agent(agent_id, policy, "idle_timeout")
    
    def _evaluate_wake_candidates(self) -> None:
        """Evaluate which hibernated agents should be woken up"""
        current_time = datetime.now()
        
        for agent_id, hibernation_info in list(self._hibernated_agents.items()):
            hibernation_start = hibernation_info['hibernation_start']
            policy = hibernation_info['policy']
            
            # Check maximum hibernation time
            hibernation_duration = (current_time - hibernation_start).total_seconds() / 3600
            if hibernation_duration >= policy.max_hibernation_hours:
                self.wake_agent(agent_id, "max_hibernation_time")
                continue
            
            # Check wake conditions
            if self._should_wake_agent(agent_id, policy):
                self.wake_agent(agent_id, "wake_condition_met")
    
    def _find_policy_for_agent(self, agent_id: str) -> Optional[HibernationPolicy]:
        """Find the best hibernation policy for an agent"""
        if agent_id not in self._agent_metrics:
            return None
        
        metrics = self._agent_metrics[agent_id]
        
        # Find policy based on agent priority
        best_policy = None
        for policy in self._hibernation_policies.values():
            if policy.enabled:
                if best_policy is None or policy.priority_score == metrics.priority_score:
                    best_policy = policy
        
        return best_policy or self._hibernation_policies.get("balanced")
    
    def _should_wake_agent(self, agent_id: str, policy: HibernationPolicy) -> bool:
        """Check if an agent should be woken up"""
        # Check system load
        if "high_system_load" in policy.wake_conditions:
            cpu_percent = psutil.cpu_percent(interval=1.0)
            if cpu_percent > 80:
                return True
        
        if "system_load" in policy.wake_conditions:
            cpu_percent = psutil.cpu_percent(interval=1.0)
            if cpu_percent > 50:
                return True
        
        # Check if agent has pending tasks (if we can query this)
        if "task_queued" in policy.wake_conditions:
            try:
                if self.agent_manager:
                    # This would need to be implemented in the agent manager
                    # For now, we'll assume no pending tasks
                    pass
            except:
                pass
        
        # Check scheduled wake times (simple implementation)
        if "scheduled_wake" in policy.wake_conditions:
            current_hour = datetime.now().hour
            # Wake during business hours (8 AM - 6 PM)
            if 8 <= current_hour <= 18:
                return True
        
        return False
    
    def _hibernate_agent(self, agent_id: str, policy: HibernationPolicy, reason: str) -> bool:
        """Hibernate an agent"""
        try:
            if not self.agent_manager:
                logger.error("Agent manager not available for hibernation")
                return False
            
            # Get current metrics
            metrics_before = self._get_agent_metrics_snapshot(agent_id)
            
            # Pause the agent
            self.agent_manager.pause_agent(agent_id)
            
            # Store hibernation info
            hibernation_info = {
                'hibernation_start': datetime.now(),
                'policy': policy,
                'reason': reason,
                'metrics_before': metrics_before
            }
            
            with self._lock:
                self._hibernated_agents[agent_id] = hibernation_info
            
            # Update agent metrics
            if agent_id in self._agent_metrics:
                self._agent_metrics[agent_id].hibernation_count += 1
                self._agent_metrics[agent_id].last_hibernation = datetime.now()
            
            # Calculate power saved
            power_saved = self._agent_base_power_w - self._hibernated_power_w
            
            # Record event
            event = HibernationEvent(
                agent_id=agent_id,
                action="hibernate",
                timestamp=datetime.now(),
                reason=reason,
                metrics_before=metrics_before,
                power_saved_w=power_saved
            )
            
            with self._lock:
                self._hibernation_history.append(event)
            
            logger.info(f"Hibernated agent {agent_id} (reason: {reason}, power saved: {power_saved:.1f}W)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to hibernate agent {agent_id}: {e}")
            
            # Record failure event
            event = HibernationEvent(
                agent_id=agent_id,
                action="failed_hibernate",
                timestamp=datetime.now(),
                reason=f"Error: {str(e)}",
                metrics_before={}
            )
            
            with self._lock:
                self._hibernation_history.append(event)
            
            return False
    
    def wake_agent(self, agent_id: str, reason: str) -> bool:
        """Wake up a hibernated agent"""
        try:
            if agent_id not in self._hibernated_agents:
                logger.warning(f"Agent {agent_id} is not hibernated")
                return False
            
            if not self.agent_manager:
                logger.error("Agent manager not available for wake")
                return False
            
            hibernation_info = self._hibernated_agents[agent_id]
            
            # Resume the agent
            self.agent_manager.resume_agent(agent_id)
            
            # Calculate hibernation duration
            hibernation_duration = datetime.now() - hibernation_info['hibernation_start']
            
            # Update agent metrics
            if agent_id in self._agent_metrics:
                self._agent_metrics[agent_id].total_hibernation_time += hibernation_duration
                self._agent_metrics[agent_id].wake_reason = reason
                self._agent_metrics[agent_id].last_activity = datetime.now()
            
            # Get current metrics
            metrics_after = self._get_agent_metrics_snapshot(agent_id)
            
            # Record event
            event = HibernationEvent(
                agent_id=agent_id,
                action="wake",
                timestamp=datetime.now(),
                reason=reason,
                metrics_before=hibernation_info['metrics_before'],
                metrics_after=metrics_after
            )
            
            with self._lock:
                self._hibernation_history.append(event)
                del self._hibernated_agents[agent_id]
            
            logger.info(f"Woke agent {agent_id} (reason: {reason}, hibernated for: {hibernation_duration})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to wake agent {agent_id}: {e}")
            
            # Record failure event
            event = HibernationEvent(
                agent_id=agent_id,
                action="failed_wake",
                timestamp=datetime.now(),
                reason=f"Error: {str(e)}",
                metrics_before={}
            )
            
            with self._lock:
                self._hibernation_history.append(event)
            
            return False
    
    def _get_agent_metrics_snapshot(self, agent_id: str) -> Dict[str, Any]:
        """Get a snapshot of agent metrics"""
        try:
            if self.agent_manager:
                status = self.agent_manager.get_agent_status(agent_id)
                metrics = self.agent_manager.get_agent_metrics(agent_id)
                return {
                    "status": status,
                    "metrics": metrics,
                    "timestamp": datetime.now().isoformat()
                }
        except:
            pass
        
        return {"timestamp": datetime.now().isoformat()}
    
    def _cleanup_old_events(self) -> None:
        """Clean up old hibernation events"""
        cutoff_time = datetime.now() - timedelta(days=7)  # Keep 7 days of history
        
        with self._lock:
            self._hibernation_history = [
                event for event in self._hibernation_history 
                if event.timestamp > cutoff_time
            ]
    
    def get_hibernation_stats(self) -> Dict[str, Any]:
        """Get hibernation statistics"""
        with self._lock:
            hibernated_count = len(self._hibernated_agents)
            total_agents = len(self._agent_metrics)
            history = self._hibernation_history.copy()
        
        # Calculate power savings
        total_power_saved = sum(
            event.power_saved_w or 0.0 
            for event in history 
            if event.action == "hibernate" and event.power_saved_w
        )
        
        # Count successful vs failed operations
        hibernation_attempts = len([e for e in history if e.action in ["hibernate", "failed_hibernate"]])
        successful_hibernations = len([e for e in history if e.action == "hibernate"])
        
        wake_attempts = len([e for e in history if e.action in ["wake", "failed_wake"]])
        successful_wakes = len([e for e in history if e.action == "wake"])
        
        # Calculate hibernation efficiency
        hibernation_success_rate = (successful_hibernations / hibernation_attempts) if hibernation_attempts > 0 else 0.0
        wake_success_rate = (successful_wakes / wake_attempts) if wake_attempts > 0 else 0.0
        
        return {
            "currently_hibernated": hibernated_count,
            "total_agents": total_agents,
            "hibernation_ratio": hibernated_count / total_agents if total_agents > 0 else 0.0,
            "total_power_saved_w": total_power_saved,
            "hibernation_success_rate": hibernation_success_rate,
            "wake_success_rate": wake_success_rate,
            "total_hibernation_events": len(history),
            "policies_available": len(self._hibernation_policies),
            "monitoring_active": self._monitoring
        }
    
    def get_agent_hibernation_info(self, agent_id: str) -> Dict[str, Any]:
        """Get hibernation information for a specific agent"""
        info = {
            "agent_id": agent_id,
            "is_hibernated": agent_id in self._hibernated_agents,
            "hibernation_info": None,
            "metrics": None
        }
        
        if agent_id in self._hibernated_agents:
            hibernation_info = self._hibernated_agents[agent_id]
            hibernation_duration = datetime.now() - hibernation_info['hibernation_start']
            
            info["hibernation_info"] = {
                "hibernation_start": hibernation_info['hibernation_start'].isoformat(),
                "hibernation_duration_minutes": hibernation_duration.total_seconds() / 60,
                "reason": hibernation_info['reason'],
                "policy": hibernation_info['policy'].name
            }
        
        if agent_id in self._agent_metrics:
            metrics = self._agent_metrics[agent_id]
            info["metrics"] = {
                "last_activity": metrics.last_activity.isoformat(),
                "hibernation_count": metrics.hibernation_count,
                "total_hibernation_time_hours": metrics.total_hibernation_time.total_seconds() / 3600,
                "priority_score": metrics.priority_score,
                "last_wake_reason": metrics.wake_reason
            }
        
        return info
    
    def force_wake_all(self) -> int:
        """Force wake all hibernated agents"""
        with self._lock:
            hibernated_agents = list(self._hibernated_agents.keys())
        
        woken_count = 0
        for agent_id in hibernated_agents:
            if self.wake_agent(agent_id, "force_wake_all"):
                woken_count += 1
        
        logger.info(f"Force woke {woken_count} agents")
        return woken_count
    
    def export_hibernation_data(self, filename: str) -> None:
        """Export hibernation data to JSON file"""
        with self._lock:
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "stats": self.get_hibernation_stats(),
                "policies": {name: {
                    "idle_threshold_minutes": policy.idle_threshold_minutes,
                    "hibernate_threshold_minutes": policy.hibernate_threshold_minutes,
                    "max_hibernation_hours": policy.max_hibernation_hours,
                    "priority_score": policy.priority_score,
                    "enabled": policy.enabled
                } for name, policy in self._hibernation_policies.items()},
                "hibernated_agents": {
                    agent_id: {
                        "hibernation_start": info['hibernation_start'].isoformat(),
                        "reason": info['reason'],
                        "policy": info['policy'].name
                    } for agent_id, info in self._hibernated_agents.items()
                },
                "agent_metrics": {
                    agent_id: {
                        "hibernation_count": metrics.hibernation_count,
                        "total_hibernation_time_hours": metrics.total_hibernation_time.total_seconds() / 3600,
                        "priority_score": metrics.priority_score
                    } for agent_id, metrics in self._agent_metrics.items()
                }
            }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Hibernation data exported to {filename}")

# Global hibernation manager instance
_global_hibernation_manager: Optional[AgentHibernationManager] = None

def get_hibernation_manager(agent_manager=None) -> AgentHibernationManager:
    """Get or create global hibernation manager instance"""
    global _global_hibernation_manager
    if _global_hibernation_manager is None:
        _global_hibernation_manager = AgentHibernationManager(agent_manager)
    return _global_hibernation_manager