#!/usr/bin/env python3
"""
Compatibility module for BaseAgentV2 imports
Provides a centralized import location for all agents
"""

try:
    # Import from the actual location
    from agents.core.base_agent_v2 import BaseAgentV2, BaseAgent, AgentStatus, AgentMetrics, TaskResult
except ImportError:
    try:
        # Fallback import path
        from core.base_agent_v2 import BaseAgentV2, BaseAgent, AgentStatus, AgentMetrics, TaskResult
    except ImportError:
        # Last resort - define minimal stub
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("Could not import BaseAgentV2, using minimal stub implementation")
        
        class BaseAgentV2:
            def __init__(self, *args, **kwargs):
                self.agent_name = kwargs.get('name', 'unknown')
                self.logger = logging.getLogger(self.__class__.__name__)
                
            async def process_task(self, task):
                return {"status": "success", "message": f"Processed by {self.agent_name}"}
                
            def run(self):
                self.logger.info(f"{self.agent_name} running in stub mode")
        
        BaseAgent = BaseAgentV2
        
        class AgentStatus:
            ACTIVE = "active"
            
        class AgentMetrics:
            pass
            
        class TaskResult:
            pass

# Export all necessary components
__all__ = ['BaseAgentV2', 'BaseAgent', 'AgentStatus', 'AgentMetrics', 'TaskResult']