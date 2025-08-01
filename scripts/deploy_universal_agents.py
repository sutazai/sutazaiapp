#!/usr/bin/env python3
"""
Deploy Universal Agents Script

This script deploys all Claude AI agent configurations as universal agents
"""

import os
import sys
import json
import yaml
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.ai_agents.universal_agent_adapter import get_universal_adapter
from backend.ai_agents.claude_agent_loader import get_claude_agent_loader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UniversalAgentDeployer:
    """Deploy universal agents across the codebase."""
    
    def __init__(self, base_dir: str = "."):
        """Initialize the deployer.
        
        Args:
            base_dir: Base directory of the project
        """
        self.base_dir = Path(base_dir)
        self.adapter = get_universal_adapter()
        self.deployment_configs = []
    
    def deploy_all_agents(self) -> None:
        """Deploy all agents across the codebase."""
        logger.info("Starting universal agent deployment...")
        
        # 1. Create agent configurations
        self._create_agent_configs()
        
        # 2. Create Docker services
        self._create_docker_services()
        
        # 3. Create Ollama models
        self._create_ollama_models()
        
        
        # 5. Update backend integration
        self._update_backend_integration()
        
        # 6. Create deployment documentation
        self._create_deployment_docs()
        
        logger.info("Universal agent deployment completed!")
    
    def _create_agent_configs(self) -> None:
        """Create agent configuration files."""
        logger.info("Creating agent configurations...")
        
        # Save all configurations
        self.adapter.save_all_configurations(
            str(self.base_dir / "agents" / "configs")
        )
        
        # Create agent registry
        registry = {
            "agents": {},
            "version": "1.0.0",
            "provider": "universal"
        }
        
        for name, agent in self.adapter.agents.items():
            registry["agents"][name] = {
                "name": agent.name,
                "description": agent.description,
                "capabilities": agent.capabilities,
                "config_path": f"configs/{name}_universal.json"
            }
        
        registry_path = self.base_dir / "agents" / "agent_registry.json"
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)
        
        logger.info(f"Created agent registry at {registry_path}")
    
    def _create_docker_services(self) -> None:
        """Create Docker service configurations."""
        logger.info("Creating Docker service configurations...")
        
        services = {}
        
        for name, agent in self.adapter.agents.items():
            service_config = self.adapter.create_docker_service(name)
            if service_config:
                service_name = service_config.pop("service_name")
                services[service_name] = service_config
        
        # Create docker-compose file for universal agents
        docker_compose = {
            "version": "3.8",
            "services": services,
            "networks": {
                "sutazai-network": {
                    "external": True
                }
            }
        }
        
        compose_path = self.base_dir / "docker-compose-universal-agents.yml"
        with open(compose_path, 'w') as f:
            yaml.dump(docker_compose, f, default_flow_style=False)
        
        logger.info(f"Created Docker compose file at {compose_path}")
    
    def _create_ollama_models(self) -> None:
        """Create Ollama model configurations."""
        logger.info("Creating Ollama model configurations...")
        
        ollama_dir = self.base_dir / "ollama" / "models"
        ollama_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a script to build all Ollama models
        build_script = ["#!/bin/bash", "", "# Build all Ollama models for SutazAI agents", ""]
        
        for name in self.adapter.agents.keys():
            ollama_config = self.adapter.export_for_ollama(name)
            if ollama_config:
                # Save Modelfile
                modelfile_path = ollama_dir / f"{name}.modelfile"
                with open(modelfile_path, 'w') as f:
                    f.write(ollama_config["modelfile"])
                
                # Add to build script
                build_script.append(f"echo 'Building model for {name}...'")
                build_script.append(f"ollama create sutazai_{name} -f {modelfile_path}")
                build_script.append("")
        
        build_script.append("echo 'All models built successfully!'")
        
        # Save build script
        build_script_path = ollama_dir / "build_all_models.sh"
        with open(build_script_path, 'w') as f:
            f.write("\n".join(build_script))
        
        # Make script executable
        build_script_path.chmod(0o755)
        
        logger.info(f"Created Ollama models in {ollama_dir}")
    
        
        
            "model_list": [],
                "drop_params": False,
                "set_verbose": True
            }
        }
        
        for name in self.adapter.agents.keys():
            if agent_config:
        
        with open(config_path, 'w') as f:
        
    
    def _update_backend_integration(self) -> None:
        """Update backend to integrate universal agents."""
        logger.info("Updating backend integration...")
        
        # Create agent factory extension
        factory_extension = '''"""
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
'''
        
        factory_path = self.base_dir / "backend" / "ai_agents" / "universal_agent_factory.py"
        with open(factory_path, 'w') as f:
            f.write(factory_extension)
        
        logger.info(f"Created universal agent factory at {factory_path}")
    
    def _create_deployment_docs(self) -> None:
        """Create deployment documentation."""
        logger.info("Creating deployment documentation...")
        
        docs = [
            "# Universal Agent Deployment Guide",
            "",
            "This guide explains how to use the deployed universal agents that are independent of Claude.",
            "",
            "## Available Agents",
            ""
        ]
        
        for name, agent in self.adapter.agents.items():
            docs.append(f"### {name}")
            docs.append(f"**Description:** {agent.description[:200]}...")
            docs.append(f"**Capabilities:** {', '.join(agent.capabilities)}")
            docs.append("")
        
        docs.extend([
            "## Usage Examples",
            "",
            "### Using with Ollama",
            "```bash",
            "# Build the agent model",
            "cd ollama/models",
            "./build_all_models.sh",
            "",
            "# Use the agent",
            "ollama run sutazai_semgrep-security-analyzer",
            "```",
            "",
            "```python",
            "",
            "response = completion(",
            '    model="sutazai/semgrep-security-analyzer",',
            '    messages=[{"role": "user", "content": "Analyze this code for security issues"}]',
            ")",
            "```",
            "",
            "### Using with Docker",
            "```bash",
            "# Start all universal agents",
            "docker-compose -f docker-compose-universal-agents.yml up -d",
            "```",
            "",
            "## Independence from Claude",
            "",
            "These agents are now completely independent and can run on:",
            "- Local Ollama models",
            "- Any OpenAI-compatible API",
            "- Self-hosted LLMs",
            "",
            "No external API keys or proprietary services required!"
        ])
        
        docs_path = self.base_dir / "docs" / "UNIVERSAL_AGENTS.md"
        docs_path.parent.mkdir(exist_ok=True)
        with open(docs_path, 'w') as f:
            f.write("\n".join(docs))
        
        logger.info(f"Created deployment documentation at {docs_path}")


def main():
    """Main deployment function."""
    logger.info("=== SutazAI Universal Agent Deployment ===")
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Create deployer and run deployment
    deployer = UniversalAgentDeployer()
    deployer.deploy_all_agents()
    
    logger.info("Deployment complete! Your agents are now independent of Claude.")
    logger.info("Check docs/UNIVERSAL_AGENTS.md for usage instructions.")


if __name__ == "__main__":
    main()