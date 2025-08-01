#!/usr/bin/env python3
"""
Universal Agent System Startup Script
=====================================

This script initializes and starts the complete SutazAI Universal Agent System.
It sets up all core components, creates initial agents, and provides a management
interface for the autonomous agent infrastructure.

Features:
- Complete system initialization
- Agent creation and management
- Workflow execution capabilities
- Health monitoring and status reporting
- Integration with existing SutazAI services
"""

import asyncio
import json
import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Add the project root to Python path
sys.path.insert(0, '/opt/sutazaiapp')

from backend.ai_agents.core import UniversalAgentSystem
from backend.ai_agents.core.base_agent import AgentCapability


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/logs/universal_agents.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("universal_agent_startup")


class UniversalAgentSystemManager:
    """
    Manager for the Universal Agent System
    
    Handles initialization, configuration, and lifecycle management
    of the complete agent infrastructure.
    """
    
    def __init__(self):
        self.system: UniversalAgentSystem = None
        self.config = self._load_configuration()
        self.running = False
        self.initial_agents = []
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _load_configuration(self) -> Dict[str, Any]:
        """Load system configuration"""
        config_file = Path("/opt/sutazaiapp/config/universal_agents.json")
        
        default_config = {
            "redis": {
                "url": "redis://localhost:6379",
                "namespace": "sutazai"
            },
            "ollama": {
                "url": "http://localhost:11434",
                "default_model": "codellama"
            },
            "system": {
                "max_concurrent_workflows": 10,
                "max_concurrent_tasks": 50,
                "heartbeat_interval": 30,
                "health_check_interval": 60
            },
            "initial_agents": [
                {
                    "id": "orchestrator-001",
                    "type": "orchestrator",
                    "name": "Master Orchestrator",
                    "config": {
                        "model": "llama2",
                        "max_concurrent_tasks": 10
                    }
                },
                {
                    "id": "code-generator-001",
                    "type": "code_generator",
                    "name": "Primary Code Generator",
                    "config": {
                        "model": "codellama",
                        "max_concurrent_tasks": 5
                    }
                },
                {
                    "id": "generic-agent-001",
                    "type": "generic",
                    "name": "Universal Assistant",
                    "config": {
                        "model": "llama2",
                        "max_concurrent_tasks": 8
                    }
                }
            ],
            "logging": {
                "level": "INFO",
                "file": "/opt/sutazaiapp/logs/universal_agents.log"
            }
        }
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                
                # Merge with defaults
                self._deep_update(default_config, loaded_config)
                logger.info(f"Loaded configuration from {config_file}")
                
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}. Using defaults.")
        else:
            # Create default config file
            try:
                config_file.parent.mkdir(parents=True, exist_ok=True)
                with open(config_file, 'w') as f:
                    json.dump(default_config, f, indent=2)
                logger.info(f"Created default configuration at {config_file}")
            except Exception as e:
                logger.warning(f"Failed to create config file: {e}")
        
        return default_config
    
    def _deep_update(self, base_dict: dict, update_dict: dict):
        """Deep update dictionary"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating shutdown")
        self.running = False
    
    async def initialize_system(self) -> bool:
        """Initialize the universal agent system"""
        try:
            logger.info("Initializing Universal Agent System")
            
            # Create system instance
            self.system = UniversalAgentSystem(
                redis_url=self.config["redis"]["url"],
                ollama_url=self.config["ollama"]["url"],
                namespace=self.config["redis"]["namespace"]
            )
            
            # Initialize all components
            success = await self.system.initialize()
            if not success:
                logger.error("Failed to initialize system")
                return False
            
            logger.info("Universal Agent System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            return False
    
    async def create_initial_agents(self) -> List[str]:
        """Create initial set of agents"""
        created_agents = []
        
        for agent_spec in self.config["initial_agents"]:
            try:
                agent_id = agent_spec["id"]
                agent_type = agent_spec["type"]
                agent_name = agent_spec["name"]
                agent_config = agent_spec.get("config", {})
                
                # Set default model configuration
                if "model_config" not in agent_config:
                    agent_config["model_config"] = {
                        "model": agent_config.get("model", self.config["ollama"]["default_model"]),
                        "ollama_url": self.config["ollama"]["url"]
                    }
                
                # Set default Redis configuration
                if "redis_config" not in agent_config:
                    agent_config["redis_config"] = {
                        "url": self.config["redis"]["url"]
                    }
                
                # Set name and description
                agent_config["name"] = agent_name
                agent_config["description"] = f"Initial {agent_type} agent"
                
                # Create agent
                agent = await self.system.create_agent(agent_id, agent_type, agent_config)
                created_agents.append(agent_id)
                
                logger.info(f"Created initial agent: {agent_id} ({agent_type})")
                
            except Exception as e:
                logger.error(f"Failed to create initial agent {agent_spec.get('id', 'unknown')}: {e}")
        
        return created_agents
    
    async def run_health_checks(self):
        """Run periodic health checks"""
        while self.running:
            try:
                # Get system status
                status = self.system.get_system_status()
                
                # Log health summary
                registry_stats = status.get("registry_stats", {})
                active_agents = registry_stats.get("active_agents", 0)
                unhealthy_agents = registry_stats.get("unhealthy_agents", 0)
                
                logger.info(f"Health Check - Active Agents: {active_agents}, Unhealthy: {unhealthy_agents}")
                
                # Check for critical issues
                if unhealthy_agents > active_agents * 0.5:  # More than 50% unhealthy
                    logger.warning("Critical: More than 50% of agents are unhealthy")
                
                await asyncio.sleep(self.config["system"]["health_check_interval"])
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(60)
    
    async def run_system_monitor(self):
        """Monitor system performance and statistics"""
        while self.running:
            try:
                status = self.system.get_system_status()
                
                # Log detailed statistics every 5 minutes
                logger.info(f"System Status: {json.dumps(status, indent=2)}")
                
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"System monitor error: {e}")
                await asyncio.sleep(300)
    
    async def start_demo_workflow(self):
        """Start a demonstration workflow"""
        try:
            # Create a simple demo workflow
            workflow_spec = {
                "name": "Universal Agent System Demo",
                "description": "Demonstrates the capabilities of the universal agent system",
                "tasks": [
                    {
                        "id": "demo-task-1",
                        "name": "analyze_system_status",
                        "description": "Analyze the current system status and provide insights",
                        "task_type": "analyze",
                        "priority": 3,
                        "required_capabilities": ["reasoning"],
                        "input_data": {
                            "analysis_type": "system_status",
                            "content": "Universal Agent System initial deployment"
                        },
                        "dependencies": [],
                        "max_retries": 2,
                        "timeout_seconds": 300
                    },
                    {
                        "id": "demo-task-2",
                        "name": "generate_welcome_message",
                        "description": "Generate a welcome message for the agent system",
                        "task_type": "generate_code",
                        "priority": 3,
                        "required_capabilities": ["code_generation"],
                        "input_data": {
                            "specification": "Create a Python function that returns a welcome message for the SutazAI Universal Agent System",
                            "language": "python",
                            "code_type": "function"
                        },
                        "dependencies": ["analyze_system_status"],
                        "max_retries": 2,
                        "timeout_seconds": 300
                    }
                ]
            }
            
            # Create and start workflow
            workflow_id = await self.system.create_workflow(workflow_spec)
            success = await self.system.execute_workflow(workflow_id)
            
            if success:
                logger.info(f"Started demo workflow: {workflow_id}")
            else:
                logger.error("Failed to start demo workflow")
                
        except Exception as e:
            logger.error(f"Demo workflow error: {e}")
    
    async def interactive_mode(self):
        """Run interactive management interface"""
        logger.info("Starting interactive mode. Type 'help' for commands.")
        
        while self.running:
            try:
                # In a real implementation, you'd use aioconsole for async input
                # For now, we'll just wait and check periodically
                await asyncio.sleep(1)
                
                # Here you could implement commands like:
                # - status: Show system status
                # - create-agent: Create new agent
                # - list-agents: List all agents
                # - create-workflow: Create new workflow
                # - shutdown: Graceful shutdown
                
            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                break
            except Exception as e:
                logger.error(f"Interactive mode error: {e}")
    
    async def run(self):
        """Run the universal agent system"""
        try:
            logger.info("Starting Universal Agent System Manager")
            
            # Initialize system
            success = await self.initialize_system()
            if not success:
                logger.error("Failed to initialize system. Exiting.")
                return 1
            
            # Create initial agents
            initial_agents = await self.create_initial_agents()
            logger.info(f"Created {len(initial_agents)} initial agents")
            
            # Start background tasks
            self.running = True
            background_tasks = [
                asyncio.create_task(self.run_health_checks()),
                asyncio.create_task(self.run_system_monitor()),
            ]
            
            # Start demo workflow after a short delay
            await asyncio.sleep(5)
            await self.start_demo_workflow()
            
            # Run interactive mode
            await self.interactive_mode()
            
            # Cleanup
            logger.info("Shutting down Universal Agent System")
            self.running = False
            
            # Cancel background tasks
            for task in background_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*background_tasks, return_exceptions=True)
            
            # Shutdown system
            await self.system.shutdown()
            
            logger.info("Universal Agent System shutdown complete")
            return 0
            
        except Exception as e:
            logger.error(f"System error: {e}")
            return 1


async def main():
    """Main entry point"""
    # Ensure log directory exists
    log_dir = Path("/opt/sutazaiapp/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create and run system manager
    manager = UniversalAgentSystemManager()
    return await manager.run()


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)