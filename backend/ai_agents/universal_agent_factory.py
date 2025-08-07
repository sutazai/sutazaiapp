"""
Universal Agent Factory Extension

Auto-generated extension for universal agent support.
"""

from typing import Dict, Any, Optional
from backend.ai_agents.universal_agent_adapter import get_universal_adapter


class UniversalAgentFactory:
    """Factory for creating universal agent instances."""
    
    def __init__(self):
        self.adapter = get_universal_adapter()
        self.agent_registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load agent registry."""
        import json
        try:
            with open("agents/agent_registry.json", "r") as f:
                return json.load(f)
        except Exception:
            return {"agents": {}}
    
    def create_agent(self, agent_name: str, provider: str = "ollama") -> Optional[Any]:
        """Create an agent instance.
        
        Args:
            agent_name: Name of the agent
            provider: Model provider to use
            
        Returns:
            Agent instance or None
        """
        agent = self.adapter.get_agent(agent_name)
        if not agent:
            return None
        
        # Return agent configuration for now
        # In production, this would create actual agent instances
        return {
            "name": agent.name,
            "system_prompt": agent.system_prompt,
            "capabilities": agent.capabilities,
            "model_config": agent.model_config,
            "provider": provider
        }
    
    def list_available_agents(self) -> List[str]:
        """List all available agents."""
        return list(self.adapter.agents.keys())


# Global factory instance
universal_agent_factory = UniversalAgentFactory()
