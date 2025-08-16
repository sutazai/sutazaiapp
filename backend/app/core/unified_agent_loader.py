"""
Unified Agent Configuration Loader
Loads agent configurations from the consolidated registry
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class UnifiedAgentLoader:
    """Loads and manages agent configurations from unified registry"""
    
    def __init__(self, config_dir: str = "/opt/sutazaiapp/config/agents"):
        """Initialize the unified agent loader"""
        self.config_dir = Path(config_dir)
        self.registry_path = self.config_dir / "registry.yaml"
        self.capabilities_path = self.config_dir / "capabilities.yaml"
        self.runtime_path = self.config_dir / "runtime" / "status.json"
        
        # Cache for loaded data
        self._registry = None
        self._capabilities = None
        self._runtime_status = None
        
        # Legacy file paths for backward compatibility
        self.legacy_paths = {
            "agent_registry": Path("/opt/sutazaiapp/agents/agent_registry.json"),
            "agent_status": Path("/opt/sutazaiapp/agents/agent_status.json"),
            "collective": Path("/opt/sutazaiapp/agents/collective_intelligence.json")
        }
    
    def load_registry(self, force_reload: bool = False) -> Dict[str, Any]:
        """Load the unified agent registry"""
        if self._registry and not force_reload:
            return self._registry
        
        try:
            # Try to load new unified registry
            if self.registry_path.exists():
                with open(self.registry_path, 'r') as f:
                    self._registry = yaml.safe_load(f)
                    logger.info(f"Loaded unified registry with {len(self._registry.get('agents', {}))} agents")
                    return self._registry
            else:
                logger.warning("Unified registry not found, falling back to legacy files")
                return self._load_legacy_registry()
        except Exception as e:
            logger.error(f"Error loading registry: {e}")
            return self._load_legacy_registry()
    
    def _load_legacy_registry(self) -> Dict[str, Any]:
        """Load from legacy files for backward compatibility"""
        registry = {
            "version": "legacy",
            "agents": {},
            "statistics": {}
        }
        
        # Try loading legacy agent_registry.json
        if self.legacy_paths["agent_registry"].exists():
            try:
                with open(self.legacy_paths["agent_registry"], 'r') as f:
                    legacy_data = json.load(f)
                    if "agents" in legacy_data:
                        registry["agents"] = legacy_data["agents"]
                        logger.info(f"Loaded {len(registry['agents'])} agents from legacy registry")
            except Exception as e:
                logger.error(f"Error loading legacy registry: {e}")
        
        return registry
    
    def load_capabilities(self) -> Dict[str, Any]:
        """Load the capabilities definitions"""
        if self._capabilities:
            return self._capabilities
        
        try:
            if self.capabilities_path.exists():
                with open(self.capabilities_path, 'r') as f:
                    self._capabilities = yaml.safe_load(f)
                    logger.info(f"Loaded {len(self._capabilities.get('capabilities', {}))} capability definitions")
            else:
                # Generate capabilities from registry if file doesn't exist
                self._capabilities = self._generate_capabilities()
        except Exception as e:
            logger.error(f"Error loading capabilities: {e}")
            self._capabilities = {"capabilities": {}}
        
        return self._capabilities
    
    def _generate_capabilities(self) -> Dict[str, Any]:
        """Generate capabilities from registry data"""
        registry = self.load_registry()
        capabilities = {
            "version": "generated",
            "capabilities": {}
        }
        
        # Collect all unique capabilities
        all_caps = set()
        for agent in registry.get("agents", {}).values():
            all_caps.update(agent.get("capabilities", []))
        
        for cap in sorted(all_caps):
            capabilities["capabilities"][cap] = {
                "name": cap.replace("_", " ").title(),
                "agents": []
            }
        
        return capabilities
    
    def load_runtime_status(self) -> Dict[str, Any]:
        """Load runtime status of agents"""
        try:
            if self.runtime_path.exists():
                with open(self.runtime_path, 'r') as f:
                    self._runtime_status = json.load(f)
            else:
                # Try legacy agent_status.json
                if self.legacy_paths["agent_status"].exists():
                    with open(self.legacy_paths["agent_status"], 'r') as f:
                        legacy_status = json.load(f)
                        self._runtime_status = {
                            "agents": legacy_status.get("active_agents", {})
                        }
        except Exception as e:
            logger.error(f"Error loading runtime status: {e}")
            self._runtime_status = {"agents": {}}
        
        return self._runtime_status
    
    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific agent configuration"""
        registry = self.load_registry()
        return registry.get("agents", {}).get(agent_id)
    
    def get_agents_by_capability(self, capability: str) -> List[Dict[str, Any]]:
        """Get all agents with a specific capability"""
        registry = self.load_registry()
        agents = []
        
        for agent_id, agent in registry.get("agents", {}).items():
            if capability in agent.get("capabilities", []):
                agents.append(agent)
        
        return agents
    
    def get_agents_by_type(self, agent_type: str) -> List[Dict[str, Any]]:
        """Get all agents of a specific type"""
        registry = self.load_registry()
        agents = []
        
        for agent_id, agent in registry.get("agents", {}).items():
            if agent.get("type") == agent_type:
                agents.append(agent)
        
        return agents
    
    def get_active_agents(self) -> List[Dict[str, Any]]:
        """Get all currently active/healthy agents"""
        registry = self.load_registry()
        runtime = self.load_runtime_status()
        active_agents = []
        
        for agent_id, agent in registry.get("agents", {}).items():
            # Check runtime status
            if agent_id in runtime.get("agents", {}):
                runtime_info = runtime["agents"][agent_id]
                if runtime_info.get("status") == "healthy":
                    # Merge runtime info into agent data
                    agent_with_runtime = agent.copy()
                    agent_with_runtime["runtime"] = runtime_info
                    active_agents.append(agent_with_runtime)
        
        return active_agents
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get statistics about the agent ecosystem"""
        registry = self.load_registry()
        runtime = self.load_runtime_status()
        
        stats = registry.get("statistics", {}).copy()
        
        # Update with current runtime stats
        active_count = sum(
            1 for agent_id in runtime.get("agents", {})
            if runtime["agents"][agent_id].get("status") == "healthy"
        )
        
        stats.update({
            "total_registered": len(registry.get("agents", {})),
            "currently_active": active_count,
            "health_percentage": (active_count / len(registry.get("agents", {})) * 100) 
                                if registry.get("agents") else 0
        })
        
        return stats
    
    def search_agents(self, query: str) -> List[Dict[str, Any]]:
        """Search for agents by name or description"""
        registry = self.load_registry()
        query_lower = query.lower()
        matching_agents = []
        
        for agent_id, agent in registry.get("agents", {}).items():
            # Search in name, description, and capabilities
            if (query_lower in agent.get("name", "").lower() or
                query_lower in agent.get("description", "").lower() or
                any(query_lower in cap for cap in agent.get("capabilities", []))):
                matching_agents.append(agent)
        
        return matching_agents

# Singleton instance for easy import
agent_loader = UnifiedAgentLoader()

# Convenience functions
def get_agent(agent_id: str) -> Optional[Dict[str, Any]]:
    """Get a specific agent configuration"""
    return agent_loader.get_agent(agent_id)

def get_active_agents() -> List[Dict[str, Any]]:
    """Get all currently active agents"""
    return agent_loader.get_active_agents()

def get_agents_by_capability(capability: str) -> List[Dict[str, Any]]:
    """Get agents with specific capability"""
    return agent_loader.get_agents_by_capability(capability)

def search_agents(query: str) -> List[Dict[str, Any]]:
    """Search for agents"""
    return agent_loader.search_agents(query)