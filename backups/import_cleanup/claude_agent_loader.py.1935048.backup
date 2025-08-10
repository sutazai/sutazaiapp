"""
Claude Agent Loader Module

This module handles loading and managing Claude AI agent configurations
from the .claude/agents directory and integrating them into the system.
"""

import os
import json
import yaml
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ClaudeAgent:
    """Represents a Claude AI agent configuration."""
    name: str
    description: str
    model: str
    system_prompt: str
    metadata: Dict[str, Any]


class ClaudeAgentLoader:
    """Loader for Claude AI agent configurations."""
    
    def __init__(self, agents_dir: str = ".claude/agents"):
        """Initialize the Claude agent loader.
        
        Args:
            agents_dir: Directory containing Claude agent definitions
        """
        self.agents_dir = Path(agents_dir)
        self.agents: Dict[str, ClaudeAgent] = {}
        self._load_agents()
    
    def _load_agents(self) -> None:
        """Load all agent configurations from the agents directory."""
        if not self.agents_dir.exists():
            logger.warning(f"Agents directory {self.agents_dir} does not exist")
            return
        
        for agent_file in self.agents_dir.glob("*.md"):
            try:
                agent = self._parse_agent_file(agent_file)
                if agent:
                    self.agents[agent.name] = agent
                    logger.info(f"Loaded Claude agent: {agent.name}")
            except Exception as e:
                logger.error(f"Failed to load agent from {agent_file}: {e}")
    
    def _parse_agent_file(self, file_path: Path) -> Optional[ClaudeAgent]:
        """Parse a Claude agent definition file.
        
        Args:
            file_path: Path to the agent definition file
            
        Returns:
            ClaudeAgent object or None if parsing fails
        """
        try:
            content = file_path.read_text()
            
            # Split frontmatter and content
            parts = content.split('---', 2)
            if len(parts) < 3:
                logger.error(f"Invalid agent file format: {file_path}")
                return None
            
            # Parse YAML frontmatter
            frontmatter = yaml.safe_load(parts[1])
            system_prompt = parts[2].strip()
            
            return ClaudeAgent(
                name=frontmatter.get('name', file_path.stem),
                description=frontmatter.get('description', ''),
                model=frontmatter.get('model', 'sonnet'),
                system_prompt=system_prompt,
                metadata=frontmatter
            )
        except Exception as e:
            logger.error(f"Error parsing agent file {file_path}: {e}")
            return None
    
    def get_agent(self, name: str) -> Optional[ClaudeAgent]:
        """Get a specific agent by name.
        
        Args:
            name: Name of the agent
            
        Returns:
            ClaudeAgent object or None if not found
        """
        return self.agents.get(name)
    
    def list_agents(self) -> List[str]:
        """List all available agent names.
        
        Returns:
            List of agent names
        """
        return list(self.agents.keys())
    
    def get_all_agents(self) -> Dict[str, ClaudeAgent]:
        """Get all loaded agents.
        
        Returns:
            Dictionary of agent name to ClaudeAgent object
        """
        return self.agents.copy()
    
    def export_agent_config(self, agent_name: str, format: str = "json") -> Optional[str]:
        """Export an agent configuration in specified format.
        
        Args:
            agent_name: Name of the agent to export
            format: Export format ('json' or 'yaml')
            
        Returns:
            Exported configuration string or None if agent not found
        """
        agent = self.get_agent(agent_name)
        if not agent:
            return None
        
        config = {
            "name": agent.name,
            "description": agent.description,
            "model": agent.model,
            "system_prompt": agent.system_prompt,
            "metadata": agent.metadata
        }
        
        if format == "json":
            return json.dumps(config, indent=2)
        elif format == "yaml":
            return yaml.dump(config, default_flow_style=False)
        else:
            logger.error(f"Unsupported export format: {format}")
            return None
    
    def create_agent_service_config(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Create a service configuration for an agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Service configuration dictionary or None if agent not found
        """
        agent = self.get_agent(agent_name)
        if not agent:
            return None
        
        return {
            "service_name": f"sutazai-{agent.name}",
            "agent_config": {
                "name": agent.name,
                "description": agent.description,
                "model": agent.model,
                "system_prompt": agent.system_prompt,
                "capabilities": self._extract_capabilities(agent.description),
                "environment": {
                    "AGENT_TYPE": "claude",
                    "AGENT_NAME": agent.name
                }
            }
        }
    
    def _extract_capabilities(self, description: str) -> List[str]:
        """Extract capabilities from agent description.
        
        Args:
            description: Agent description text
            
        Returns:
            List of capabilities
        """
        capabilities = []
        
        # Common capability keywords to look for
        capability_keywords = [
            "security", "code", "test", "deploy", "monitor", "analyze",
            "optimize", "automate", "integrate", "validate", "scan",
            "build", "manage", "orchestrate", "configure", "debug"
        ]
        
        description_lower = description.lower()
        for keyword in capability_keywords:
            if keyword in description_lower:
                capabilities.append(keyword)
        
        return capabilities


# Singleton instance
_agent_loader: Optional[ClaudeAgentLoader] = None


def get_claude_agent_loader() -> ClaudeAgentLoader:
    """Get the singleton Claude agent loader instance.
    
    Returns:
        ClaudeAgentLoader instance
    """
    global _agent_loader
    if _agent_loader is None:
        _agent_loader = ClaudeAgentLoader()
    return _agent_loader