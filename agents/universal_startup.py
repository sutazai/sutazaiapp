#!/usr/bin/env python3
"""
Universal startup script for SutazAI agents
Detects and runs the appropriate agent file
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
from agents.core.utils import get_agent_name

## get_agent_name centralized in agents.core.utils

def load_agent_module():
    """Try to load the agent module from various sources"""
    import importlib.util
    
    # List of possible agent files to try
    agent_files = [
        'main.py',
        'app.py', 
        'agent.py',
        f'{get_agent_name()}.py',
        f'{get_agent_name()}_agent.py'
    ]
    
    for filename in agent_files:
        filepath = Path(filename)
        if filepath.exists():
            logger.info(f"Found agent file: {filename}")
            
            # Load the module
            spec = importlib.util.spec_from_file_location("agent_module", filepath)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
    
    # If container_startup.py exists, use it
    if Path('container_startup.py').exists():
        logger.info("Using container_startup.py")
        import container_startup
        return container_startup
    
    return None

def get_app_from_module(module):
    """Extract FastAPI app from module"""
    # Look for 'app' variable
    if hasattr(module, 'app') and isinstance(module.app, FastAPI):
        return module.app
    
    # Look for any FastAPI instance
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, FastAPI):
            return attr
    
    return None

def create_default_app():
    """Create a default FastAPI app if no agent module is found"""
    agent_name = get_agent_name()
    app = FastAPI(
        title=f"SutazAI Agent - {agent_name}",
        version="1.0.0"
    )
    
    @app.get("/health")
    async def health_check():
        return {
            "agent_name": agent_name,
            "status": "active",
            "healthy": True,
            "message": "Default health check"
        }
    
    @app.get("/status")
    async def get_status():
        return {
            "agent_name": agent_name,
            "status": "active",
            "message": "Default agent running"
        }
    
    @app.post("/task")
    async def process_task(task_data: dict):
        return {
            "status": "completed",
            "result": {"message": "Task processed by default handler"}
        }
    
    @app.get("/")
    async def root():
        return {
            "message": f"SutazAI Agent - {agent_name}",
            "version": "1.0.0",
            "agent": agent_name
        }
    
    return app

if __name__ == "__main__":
    # Try to load agent module
    module = load_agent_module()
    
    if module:
        # Try to get app from module
        app = get_app_from_module(module)
        
        if app:
            # Run the app directly
            port = int(os.getenv("PORT", "8080"))
            host = os.getenv("HOST", "0.0.0.0")
            
            logger.info(f"Starting agent {get_agent_name()} with FastAPI app on {host}:{port}")
            
            uvicorn.run(
                app,
                host=host,
                port=port,
                reload=False,
                log_level=os.getenv("LOG_LEVEL", "info").lower()
            )
        else:
            # Try to run module as script
            if hasattr(module, '__name__'):
                logger.info(f"Running module {module.__name__} as script")
                # Module should handle its own execution
            else:
                logger.warning("No FastAPI app found in module, creating default")
                app = create_default_app()
                
                port = int(os.getenv("PORT", "8080"))
                host = os.getenv("HOST", "0.0.0.0")
                
                uvicorn.run(
                    app,
                    host=host,
                    port=port,
                    reload=False,
                    log_level=os.getenv("LOG_LEVEL", "info").lower()
                )
    else:
        # No module found, create default app
        logger.warning("No agent module found, creating default app")
        app = create_default_app()
        
        port = int(os.getenv("PORT", "8080"))
        host = os.getenv("HOST", "0.0.0.0")
        
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=False,
            log_level=os.getenv("LOG_LEVEL", "info").lower()
        )
