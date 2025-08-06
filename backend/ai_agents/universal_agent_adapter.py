"""
Universal Agent Adapter Module

This module provides a universal adapter that converts Claude agent configurations
This enables independence from proprietary AI services.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Protocol
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

from .claude_agent_loader import ClaudeAgent, get_claude_agent_loader

logger = logging.getLogger(__name__)


class ModelProvider(Protocol):
    """Protocol for model providers."""
    
    def generate(self, prompt: str, system_prompt: str, **kwargs) -> str:
        """Generate a response from the model."""
        ...
    
    def stream(self, prompt: str, system_prompt: str, **kwargs) -> Any:
        """Stream responses from the model."""
        ...


@dataclass
class UniversalAgent:
    """Universal agent that can work with any model provider."""
    
    name: str
    description: str
    system_prompt: str
    capabilities: List[str]
    model_config: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary format."""
        return {
            "name": self.name,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "capabilities": self.capabilities,
            "model_config": self.model_config,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }


class UniversalAgentAdapter:
    """Adapter to convert Claude agents to universal format."""
    
    def __init__(self):
        """Initialize the universal agent adapter."""
        self.claude_loader = get_claude_agent_loader()
        self.agents: Dict[str, UniversalAgent] = {}
        self._convert_all_agents()
    
    def _convert_all_agents(self) -> None:
        """Convert all Claude agents to universal format."""
        claude_agents = self.claude_loader.get_all_agents()
        
        for name, claude_agent in claude_agents.items():
            universal_agent = self._convert_agent(claude_agent)
            self.agents[name] = universal_agent
            logger.info(f"Converted agent {name} to universal format")
    
    def _convert_agent(self, claude_agent: ClaudeAgent) -> UniversalAgent:
        """Convert a Claude agent to universal format.
        
        Args:
            claude_agent: Claude agent to convert
            
        Returns:
            UniversalAgent instance
        """
        # Extract capabilities from description
        capabilities = self._extract_capabilities(claude_agent.description)
        
        # Create model-agnostic configuration
        model_config = {
            "temperature": 0.7,
            "max_tokens": 4096,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "preferred_models": [
                "ollama/tinyllama:latest"
            ]
        }
        
        # Adjust parameters based on agent type
        if "security" in claude_agent.name or "security" in capabilities:
            model_config["temperature"] = 0.3  # More deterministic for security
        elif "creative" in capabilities or "generate" in capabilities:
            model_config["temperature"] = 0.9  # More creative
        
        return UniversalAgent(
            name=claude_agent.name,
            description=claude_agent.description,
            system_prompt=claude_agent.system_prompt,
            capabilities=capabilities,
            model_config=model_config,
            metadata={
                "original_model": claude_agent.model,
                "source": "claude",
                **claude_agent.metadata
            }
        )
    
    def _extract_capabilities(self, description: str) -> List[str]:
        """Extract capabilities from agent description."""
        capabilities = []
        
        # Define capability mappings
        capability_patterns = {
            "security_analysis": ["security", "vulnerabilit", "scan", "audit"],
            "code_generation": ["generate", "create", "implement", "build"],
            "testing": ["test", "validate", "verify", "check"],
            "deployment": ["deploy", "release", "publish", "distribute"],
            "monitoring": ["monitor", "track", "observe", "watch"],
            "optimization": ["optimize", "improve", "enhance", "refactor"],
            "automation": ["automate", "workflow", "pipeline", "orchestrate"],
            "integration": ["integrate", "connect", "bridge", "interface"],
            "analysis": ["analyze", "examine", "investigate", "assess"],
            "documentation": ["document", "describe", "explain", "annotate"]
        }
        
        description_lower = description.lower()
        
        for capability, keywords in capability_patterns.items():
            if any(keyword in description_lower for keyword in keywords):
                capabilities.append(capability)
        
        return capabilities
    
    def get_agent(self, name: str) -> Optional[UniversalAgent]:
        """Get a universal agent by name."""
        return self.agents.get(name)
    
    def export_for_ollama(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Export agent configuration for Ollama.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Ollama-compatible configuration
        """
        agent = self.get_agent(agent_name)
        if not agent:
            return None
        
        return {
            "name": f"sutazai_{agent.name}",
            "modelfile": self._create_ollama_modelfile(agent),
            "config": {
                "temperature": agent.model_config["temperature"],
                "num_predict": agent.model_config["max_tokens"],
                "top_p": agent.model_config["top_p"]
            }
        }
    
    def _create_ollama_modelfile(self, agent: UniversalAgent) -> str:
        """Create an Ollama Modelfile for the agent."""
        modelfile = f"""FROM gpt-oss:latest

SYSTEM {agent.system_prompt}

PARAMETER temperature {agent.model_config['temperature']}
PARAMETER num_predict {agent.model_config['max_tokens']}
PARAMETER top_p {agent.model_config['top_p']}

# Agent: {agent.name}
# Capabilities: {', '.join(agent.capabilities)}
# Description: {agent.description}
"""
        return modelfile
    
    def _another_method(self, agent_name: str):
        """Helper method implementation."""
        agent = self.get_agent(agent_name)
        if not agent:
            return None
        
        return {
            "model_name": f"sutazai/{agent.name}",
                "model": agent.model_config["preferred_models"][0],
                "temperature": agent.model_config["temperature"],
                "max_tokens": agent.model_config["max_tokens"],
                "top_p": agent.model_config["top_p"],
                "frequency_penalty": agent.model_config["frequency_penalty"],
                "presence_penalty": agent.model_config["presence_penalty"],
                "metadata": {
                    "agent_name": agent.name,
                    "capabilities": agent.capabilities,
                    "system_prompt": agent.system_prompt
                }
            }
    
    def create_docker_service(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Create a Docker service configuration for the agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Docker service configuration
        """
        agent = self.get_agent(agent_name)
        if not agent:
            return None
        
        return {
            "service_name": f"sutazai-agent-{agent.name}",
            "image": "sutazai/universal-agent:latest",
            "environment": {
                "AGENT_NAME": agent.name,
                "AGENT_CAPABILITIES": ",".join(agent.capabilities),
                "MODEL_PROVIDER": "ollama",
                "MODEL_NAME": "gpt-oss:latest",
                "MODEL_TEMPERATURE": str(agent.model_config["temperature"]),
                "MODEL_MAX_TOKENS": str(agent.model_config["max_tokens"]),
                "SYSTEM_PROMPT": agent.system_prompt
            },
            "labels": {
                "sutazai.agent": "true",
                "sutazai.agent.name": agent.name,
                "sutazai.agent.type": "universal",
                "sutazai.agent.capabilities": ",".join(agent.capabilities)
            },
            "volumes": [
                "./agents/data:/app/data",
                "./agents/logs:/app/logs"
            ],
            "networks": ["sutazai-network"],
            "restart": "unless-stopped"
        }
    
    def save_all_configurations(self, output_dir: str = "agents/configs") -> None:
        """Save all agent configurations to disk.
        
        Args:
            output_dir: Directory to save configurations
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for name, agent in self.agents.items():
            # Save universal config
            universal_path = output_path / f"{name}_universal.json"
            with open(universal_path, 'w') as f:
                json.dump(agent.to_dict(), f, indent=2)
            
            # Save Ollama config
            ollama_config = self.export_for_ollama(name)
            if ollama_config:
                ollama_path = output_path / f"{name}_ollama.json"
                with open(ollama_path, 'w') as f:
                    json.dump(ollama_config, f, indent=2)
                
                # Save Modelfile
                modelfile_path = output_path / f"{name}.modelfile"
                with open(modelfile_path, 'w') as f:
                    f.write(ollama_config["modelfile"])
            
        
        logger.info(f"Saved all agent configurations to {output_path}")


# Singleton instance
_adapter: Optional[UniversalAgentAdapter] = None


def get_universal_adapter() -> UniversalAgentAdapter:
    """Get the singleton universal agent adapter instance."""
    global _adapter
    if _adapter is None:
        _adapter = UniversalAgentAdapter()
    return _adapter