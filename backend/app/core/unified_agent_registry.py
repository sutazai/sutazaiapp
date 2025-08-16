#!/usr/bin/env python3
"""
Unified Agent Registry - Single Source of Truth for All Agents
Consolidates container agents, Claude agents, and provides intelligent routing
"""

import json
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class AgentCapability:
    """Represents a specific capability of an agent"""
    name: str
    category: str
    priority: int = 5
    keywords: List[str] = field(default_factory=list)

@dataclass
class UnifiedAgent:
    """Unified representation of any agent type"""
    id: str
    name: str
    type: str  # 'claude', 'container', 'external'
    description: str
    capabilities: List[str]
    priority: int = 5
    deployment_info: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def matches_requirements(self, requirements: List[str]) -> Tuple[bool, float]:
        """Check if agent matches requirements and return match score"""
        if not requirements:
            return True, 1.0
            
        matched = 0
        for req in requirements:
            req_lower = req.lower()
            # Check capabilities
            if any(req_lower in cap.lower() for cap in self.capabilities):
                matched += 1
                continue
            # Check description
            if req_lower in self.description.lower():
                matched += 0.5
                continue
            # Check name
            if req_lower in self.name.lower():
                matched += 0.3
                
        score = matched / len(requirements) if requirements else 0
        return score > 0.3, score

class UnifiedAgentRegistry:
    """Single source of truth for all agent configurations"""
    
    def __init__(self):
        self.agents: Dict[str, UnifiedAgent] = {}
        # Fixed paths to real, existing directories and files
        self.claude_agents_path = Path("/opt/sutazaiapp/.claude/agents")
        self.container_registry_path = Path("/opt/sutazaiapp/agents/agent_registry.json")
        self.config_path = Path("/opt/sutazaiapp/config/agents/unified_agent_registry.json")
        self._load_all_agents()
        
    def _load_all_agents(self):
        """Load and consolidate all agent configurations"""
        # Load Claude agents
        self._load_claude_agents()
        # Load container agents
        self._load_container_agents()
        # Remove duplicates
        self._consolidate_duplicates()
        logger.info(f"Loaded {len(self.agents)} unified agents")
        
    def _load_claude_agents(self):
        """Load all Claude agents from .claude/agents directory"""
        if not self.claude_agents_path.exists():
            logger.info(f"Claude agents path does not exist yet: {self.claude_agents_path}")
            # Create the directory if it doesn't exist
            try:
                self.claude_agents_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created Claude agents directory: {self.claude_agents_path}")
            except Exception as e:
                logger.error(f"Failed to create Claude agents directory: {e}")
            return
            
        # Check if directory is accessible
        if not self.claude_agents_path.is_dir():
            logger.error(f"Claude agents path exists but is not a directory: {self.claude_agents_path}")
            return
            
        agent_files = list(self.claude_agents_path.glob("*.md"))
        if not agent_files:
            logger.info("No Claude agent files found in directory")
            return
            
        logger.info(f"Found {len(agent_files)} Claude agent files to load")
        
        for agent_file in agent_files:
            try:
                content = agent_file.read_text()
                agent_name = agent_file.stem
                
                # Parse agent from markdown
                agent = self._parse_claude_agent(agent_name, content)
                if agent:
                    self.agents[f"claude_{agent_name}"] = agent
                    logger.debug(f"Loaded Claude agent: {agent_name}")
                    
            except Exception as e:
                logger.error(f"Failed to load Claude agent {agent_file}: {e}")
                
    def _parse_claude_agent(self, name: str, content: str) -> Optional[UnifiedAgent]:
        """Parse Claude agent from markdown content"""
        try:
            # Extract description from content
            lines = content.split('\n')
            description = ""
            capabilities = []
            
            # Simple parsing - can be enhanced
            for line in lines:
                if line.startswith("Use this agent when"):
                    description = line
                elif "capability" in line.lower() or "specialized" in line.lower():
                    # Extract capabilities from description
                    if "orchestrat" in line.lower():
                        capabilities.append("orchestration")
                    if "code" in line.lower():
                        capabilities.append("code_generation")
                    if "test" in line.lower():
                        capabilities.append("testing")
                    if "deploy" in line.lower():
                        capabilities.append("deployment")
                    if "security" in line.lower():
                        capabilities.append("security_analysis")
                    if "monitor" in line.lower():
                        capabilities.append("monitoring")
                    if "optim" in line.lower():
                        capabilities.append("optimization")
                    if "automat" in line.lower():
                        capabilities.append("automation")
                        
            # If no description found, use first non-empty line
            if not description:
                for line in lines:
                    if line.strip() and not line.startswith('#'):
                        description = line.strip()[:500]  # Limit length
                        break
                        
            return UnifiedAgent(
                id=f"claude_{name}",
                name=name,
                type="claude",
                description=description or f"Claude agent for {name.replace('-', ' ')}",
                capabilities=capabilities or ["general"],
                deployment_info={
                    "method": "task_tool",
                    "agent_file": f"/opt/sutazaiapp/.claude/agents/{name}.md"  # Use absolute path
                }
            )
        except Exception as e:
            logger.error(f"Failed to parse Claude agent {name}: {e}")
            return None
            
    def _load_container_agents(self):
        """Load container-based agents from registry"""
        if not self.container_registry_path.exists():
            logger.warning(f"Container registry not found: {self.container_registry_path}")
            return
            
        try:
            with open(self.container_registry_path) as f:
                registry = json.load(f)
                
            for agent_id, agent_data in registry.get("agents", {}).items():
                # Skip if already loaded as Claude agent (prefer Claude version)
                if f"claude_{agent_id}" in self.agents:
                    continue
                
                # Validate config_path if provided
                config_path = agent_data.get("config_path")
                validated_config = None
                
                if config_path:
                    # Check if config file actually exists
                    full_config_path = Path("/opt/sutazaiapp/agents") / config_path
                    if full_config_path.exists():
                        validated_config = str(full_config_path)
                    else:
                        # Check in configs directory
                        alt_path = Path("/opt/sutazaiapp/agents/configs") / Path(config_path).name
                        if alt_path.exists():
                            validated_config = str(alt_path)
                        else:
                            logger.debug(f"Config file not found for {agent_id}: {config_path}")
                    
                agent = UnifiedAgent(
                    id=f"container_{agent_id}",
                    name=agent_data.get("name", agent_id),
                    type="container",
                    description=agent_data.get("description", ""),
                    capabilities=agent_data.get("capabilities", []),
                    deployment_info={
                        "method": "docker",
                        "config_path": validated_config,  # Only include if file exists
                        "container_name": f"sutazai-{agent_id}",
                        "original_config": config_path  # Keep original for reference
                    }
                )
                self.agents[agent.id] = agent
                
        except Exception as e:
            logger.error(f"Failed to load container agents: {e}")
            
    def _consolidate_duplicates(self):
        """Remove duplicate agents, preferring Claude agents"""
        duplicates = []
        seen_names = {}
        
        for agent_id, agent in self.agents.items():
            normalized_name = agent.name.lower().replace('-', '_').replace(' ', '_')
            
            if normalized_name in seen_names:
                # Keep Claude agent over container agent
                if agent.type == "claude" and self.agents[seen_names[normalized_name]].type == "container":
                    duplicates.append(seen_names[normalized_name])
                    seen_names[normalized_name] = agent_id
                else:
                    duplicates.append(agent_id)
            else:
                seen_names[normalized_name] = agent_id
                
        # Remove duplicates
        for dup_id in duplicates:
            del self.agents[dup_id]
            
        if duplicates:
            logger.info(f"Removed {len(duplicates)} duplicate agents")
            
    def find_best_agent(self, task_description: str, required_capabilities: List[str] = None) -> Optional[UnifiedAgent]:
        """Find the best agent for a given task"""
        if not self.agents:
            return None
            
        requirements = required_capabilities or []
        
        # Add keywords from task description
        keywords = []
        task_lower = task_description.lower()
        
        # Common capability keywords
        capability_keywords = {
            "orchestrat": ["orchestration", "coordination", "multi-agent"],
            "code": ["code_generation", "development", "programming"],
            "test": ["testing", "qa", "validation"],
            "deploy": ["deployment", "release", "production"],
            "security": ["security_analysis", "pentesting", "audit"],
            "monitor": ["monitoring", "observability", "metrics"],
            "optim": ["optimization", "performance", "efficiency"],
            "automat": ["automation", "workflow", "pipeline"],
            "backend": ["backend", "api", "server"],
            "frontend": ["frontend", "ui", "react", "streamlit"],
            "data": ["data", "analysis", "pipeline", "etl"],
            "ai": ["ai", "ml", "machine_learning", "model"],
            "infrast": ["infrastructure", "devops", "docker", "kubernetes"]
        }
        
        for keyword, caps in capability_keywords.items():
            if keyword in task_lower:
                requirements.extend(caps)
                
        # Find agents with matching capabilities
        candidates = []
        for agent in self.agents.values():
            matches, score = agent.matches_requirements(requirements)
            if matches:
                candidates.append((agent, score))
                
        if not candidates:
            # If no specific match, return a general-purpose agent
            if "claude_ai-agent-orchestrator" in self.agents:
                return self.agents["claude_ai-agent-orchestrator"]
            elif "claude_complex-problem-solver" in self.agents:
                return self.agents["claude_complex-problem-solver"]
            # Last resort - return any Claude agent
            for agent in self.agents.values():
                if agent.type == "claude":
                    return agent
                    
        # Sort by score and priority
        candidates.sort(key=lambda x: (x[1], x[0].priority), reverse=True)
        return candidates[0][0]
        
    def get_agent(self, agent_id: str) -> Optional[UnifiedAgent]:
        """Get a specific agent by ID"""
        return self.agents.get(agent_id)
        
    def list_agents(self, agent_type: str = None, capabilities: List[str] = None) -> List[UnifiedAgent]:
        """List agents filtered by type or capabilities"""
        results = []
        
        for agent in self.agents.values():
            # Filter by type
            if agent_type and agent.type != agent_type:
                continue
                
            # Filter by capabilities
            if capabilities:
                if not any(cap in agent.capabilities for cap in capabilities):
                    continue
                    
            results.append(agent)
            
        return results
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        stats = {
            "total_agents": len(self.agents),
            "claude_agents": len([a for a in self.agents.values() if a.type == "claude"]),
            "container_agents": len([a for a in self.agents.values() if a.type == "container"]),
            "capabilities": {},
            "types": {}
        }
        
        # Count capabilities
        for agent in self.agents.values():
            for cap in agent.capabilities:
                stats["capabilities"][cap] = stats["capabilities"].get(cap, 0) + 1
                
            stats["types"][agent.type] = stats["types"].get(agent.type, 0) + 1
            
        return stats
    
    def save_registry(self) -> bool:
        """Save the consolidated registry to a JSON file"""
        try:
            # Ensure config directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert agents to serializable format
            registry_data = {
                "version": "2.0",
                "last_updated": str(Path.cwd()),  # Using cwd as timestamp placeholder
                "agents": {}
            }
            
            for agent_id, agent in self.agents.items():
                registry_data["agents"][agent_id] = {
                    "id": agent.id,
                    "name": agent.name,
                    "type": agent.type,
                    "description": agent.description,
                    "capabilities": agent.capabilities,
                    "priority": agent.priority,
                    "deployment_info": agent.deployment_info,
                    "metadata": agent.metadata
                }
            
            # Write to file
            with open(self.config_path, 'w') as f:
                json.dump(registry_data, f, indent=2)
                
            logger.info(f"Saved unified registry with {len(self.agents)} agents to {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
            return False
    
    def load_saved_registry(self) -> bool:
        """Load a previously saved registry from JSON file"""
        try:
            if not self.config_path.exists():
                logger.info(f"No saved registry found at {self.config_path}")
                return False
                
            with open(self.config_path) as f:
                registry_data = json.load(f)
                
            # Clear current agents
            self.agents.clear()
            
            # Load agents from saved data
            for agent_id, agent_data in registry_data.get("agents", {}).items():
                agent = UnifiedAgent(
                    id=agent_data["id"],
                    name=agent_data["name"],
                    type=agent_data["type"],
                    description=agent_data["description"],
                    capabilities=agent_data["capabilities"],
                    priority=agent_data.get("priority", 5),
                    deployment_info=agent_data.get("deployment_info", {}),
                    metadata=agent_data.get("metadata", {})
                )
                self.agents[agent_id] = agent
                
            logger.info(f"Loaded {len(self.agents)} agents from saved registry")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load saved registry: {e}")
            return False

# Singleton instance
_registry_instance = None

def get_registry() -> UnifiedAgentRegistry:
    """Get the singleton registry instance"""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = UnifiedAgentRegistry()
    return _registry_instance