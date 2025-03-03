#!/usr/bin/env python3.11
"""
Start Script for Supreme AI Orchestrator

This script initializes and starts the Supreme AI Orchestrator system.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

import toml

from core_system.orchestrator.supreme_ai import SupremeAIOrchestrator, OrchestratorConfig
from core_system.orchestrator.exceptions import ConfigError

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/orchestrator.log')
    ]
)

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from TOML file"""
    try:
        with open(config_path, 'r') as f:
            return toml.load(f)
    except Exception as e:
        raise ConfigError(f"Failed to load config from {config_path}: {e}")

def validate_config(config: dict) -> None:
    """Validate configuration values"""
    required_sections = ['primary_server', 'secondary_server', 'orchestrator']
    for section in required_sections:
        if section not in config:
            raise ConfigError(f"Missing required config section: {section}")

    # Validate server configurations
    for server in ['primary_server', 'secondary_server']:
        required_fields = ['id', 'host', 'port', 'sync_port', 'api_key']
        for field in required_fields:
            if field not in config[server]:
                raise ConfigError(f"Missing required field '{field}' in {server} config")

    # Validate orchestrator settings
    required_fields = ['sync_interval', 'max_agents', 'task_timeout']
    for field in required_fields:
        if field not in config['orchestrator']:
            raise ConfigError(f"Missing required field '{field}' in orchestrator config")

def setup_environment() -> None:
    """Setup the environment for the orchestrator"""
    # Ensure required directories exist
    directories = ['logs', 'data', 'config/certs']
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

async def main():
    """Main entry point for starting the orchestrator"""
    try:
        # Load and validate configuration
        config_path = 'config/orchestrator.toml'
        config = load_config(config_path)
        validate_config(config)

        # Setup environment
        setup_environment()

        # Create orchestrator configuration
        orchestrator_config = OrchestratorConfig(
            primary_server=f"http://{config['primary_server']['host']}:{config['primary_server']['port']}",
            secondary_server=f"http://{config['secondary_server']['host']}:{config['secondary_server']['port']}",
            sync_interval=config['orchestrator']['sync_interval'],
            max_agents=config['orchestrator']['max_agents'],
            task_timeout=config['orchestrator']['task_timeout']
        )

        # Initialize and start orchestrator
        orchestrator = SupremeAIOrchestrator(orchestrator_config)
        
        logger.info("Starting Supreme AI Orchestrator...")
        await orchestrator.start()

        # Keep the script running
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("Shutting down orchestrator...")
        if 'orchestrator' in locals():
            await orchestrator.stop()
    except Exception as e:
        logger.error(f"Error starting orchestrator: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 