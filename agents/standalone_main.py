#!/usr/bin/env python3
"""
Standardized main entry point for all SutazAI agents
This file provides a unified startup mechanism that automatically
detects and runs the appropriate agent based on the environment.
"""

import os
import sys
import logging
from agents.core.utils import get_agent_name
from pathlib import Path

# Add current directory and parent to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir.parent))

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

## get_agent_name centralized in agents.core.utils

def import_agent_class(agent_name):
    """Dynamically import the agent class"""
    try:
        # Try to import from the specific agent directory
        agent_dir = current_dir / agent_name
        if agent_dir.exists() and (agent_dir / 'app.py').exists():
            sys.path.insert(0, str(agent_dir))
            
            # Import the app module
            import app
            
            # Find the agent class (should end with 'Agent')
            for attr_name in dir(app):
                attr = getattr(app, attr_name)
                if (isinstance(attr, type) and 
                    attr_name.endswith('Agent') and 
                    attr_name != 'BaseAgent' and
                    attr_name != 'BaseAgent'):
                    logger.info(f"Found agent class: {attr_name}")
                    return attr
        
        # Fallback to base agent
        logger.warning(f"Could not find specific agent class for {agent_name}, using base agent")
        from agents.core.base_agent import BaseAgentV2
        return BaseAgentV2
        
    except Exception as e:
        logger.error(f"Error importing agent class: {e}")
        # Final fallback to base agent
        from agents.core.base_agent import BaseAgent
        return BaseAgent

def main():
    """Main entry point"""
    try:
        agent_name = get_agent_name()
        logger.info(f"Starting agent: {agent_name}")
        
        # Import the appropriate agent class
        AgentClass = import_agent_class(agent_name)
        
        # Create and run the agent
        agent = AgentClass()
        agent.run()
        
    except KeyboardInterrupt:
        logger.info("Agent stopped by user")
    except Exception as e:
        logger.error(f"Fatal error starting agent: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
