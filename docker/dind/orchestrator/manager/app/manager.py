#!/usr/bin/env python3
"""
MCP Container Orchestrator Manager
Manages MCP containers within Docker-in-Docker environment
"""

import asyncio
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Optional

import docker
import structlog
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Prometheus metrics
CONTAINER_COUNT = Gauge('mcp_containers_total', 'Total number of MCP containers')
CONTAINER_STARTS = Counter('mcp_container_starts_total', 'Total container starts')
CONTAINER_STOPS = Counter('mcp_container_stops_total', 'Total container stops')
CONTAINER_ERRORS = Counter('mcp_container_errors_total', 'Total container errors')
OPERATION_DURATION = Histogram('mcp_operation_duration_seconds', 'Operation duration')

class MCPContainer(BaseModel):
    name: str
    image: str
    ports: Dict[str, str] = {}
    environment: Dict[str, str] = {}
    volumes: Dict[str, str] = {}
    restart_policy: str = "unless-stopped"
    
class MCPOrchestrator:
    def __init__(self):
        self.docker_client = None
        self.running_containers: Dict[str, docker.models.containers.Container] = {}
        self.shutdown_event = asyncio.Event()
        
    async def initialize(self):
        """Initialize Docker client connection"""
        try:
            docker_host = os.getenv('DOCKER_HOST', 'unix:///var/run/docker.sock')
            self.docker_client = docker.DockerClient(base_url=docker_host)
            
            # Test connection
            self.docker_client.ping()
            logger.info("Connected to Docker-in-Docker daemon", docker_host=docker_host)
            
            # Load existing MCP containers
            await self._discover_existing_containers()
            
        except Exception as e:
            logger.error("Failed to initialize Docker client", error=str(e))
            raise
    
    async def _discover_existing_containers(self):
        """Discover existing MCP containers"""
        try:
            containers = self.docker_client.containers.list(
                filters={'label': 'mcp.managed=true'}
            )
            
            for container in containers:
                self.running_containers[container.name] = container
                logger.info("Discovered existing MCP container", 
                          name=container.name, status=container.status)
                
            CONTAINER_COUNT.set(len(self.running_containers))
            
        except Exception as e:
            logger.error("Failed to discover existing containers", error=str(e))
    
    async def deploy_mcp_container(self, mcp_config: MCPContainer) -> Dict[str, str]:
        """Deploy a new MCP container"""
        with OPERATION_DURATION.time():
            try:
                # Check if container already exists
                if mcp_config.name in self.running_containers:
                    container = self.running_containers[mcp_config.name]
                    if container.status == 'running':
                        return {"status": "already_running", "container_id": container.id}
                    else:
                        # Remove stopped container
                        container.remove()
                        del self.running_containers[mcp_config.name]
                
                # Prepare container configuration
                container_config = {
                    'image': mcp_config.image,
                    'name': mcp_config.name,
                    'ports': mcp_config.ports,
                    'environment': mcp_config.environment,
                    'volumes': mcp_config.volumes,
                    'restart_policy': {'Name': mcp_config.restart_policy},
                    'labels': {
                        'mcp.managed': 'true',
                        'mcp.deployed_at': datetime.utcnow().isoformat(),
                        'mcp.manager': 'sutazai-mcp-orchestrator'
                    },
                    'network': 'sutazai-dind-internal',
                    'detach': True
                }
                
                # Create and start container
                container = self.docker_client.containers.run(**container_config)
                self.running_containers[mcp_config.name] = container
                
                CONTAINER_STARTS.inc()
                CONTAINER_COUNT.set(len(self.running_containers))
                
                logger.info("MCP container deployed successfully", 
                          name=mcp_config.name, container_id=container.id)
                
                return {
                    "status": "deployed", 
                    "container_id": container.id,
                    "name": mcp_config.name
                }
                
            except Exception as e:
                CONTAINER_ERRORS.inc()
                logger.error("Failed to deploy MCP container", 
                           name=mcp_config.name, error=str(e))
                raise HTTPException(status_code=500, detail=f"Deployment failed: {str(e)}")
    
    async def stop_mcp_container(self, container_name: str) -> Dict[str, str]:
        """Stop and remove MCP container"""
        with OPERATION_DURATION.time():
            try:
                if container_name not in self.running_containers:
                    raise HTTPException(status_code=404, detail="Container not found")
                
                container = self.running_containers[container_name]
                container.stop(timeout=30)
                container.remove()
                
                del self.running_containers[container_name]
                
                CONTAINER_STOPS.inc()
                CONTAINER_COUNT.set(len(self.running_containers))
                
                logger.info("MCP container stopped successfully", name=container_name)
                
                return {"status": "stopped", "name": container_name}
                
            except Exception as e:
                CONTAINER_ERRORS.inc()
                logger.error("Failed to stop MCP container", 
                           name=container_name, error=str(e))
                raise HTTPException(status_code=500, detail=f"Stop failed: {str(e)}")
    
    async def list_containers(self) -> List[Dict[str, str]]:
        """List all managed MCP containers"""
        try:
            containers = []
            for name, container in self.running_containers.items():
                container.reload()  # Refresh status
                containers.append({
                    "name": name,
                    "id": container.id,
                    "status": container.status,
                    "image": container.image.tags[0] if container.image.tags else "unknown",
                    "created": container.attrs['Created'],
                    "ports": container.ports
                })
            return containers
            
        except Exception as e:
            logger.error("Failed to list containers", error=str(e))
            raise HTTPException(status_code=500, detail=f"List failed: {str(e)}")
    
    async def cleanup_orphaned_containers(self) -> Dict[str, int]:
        """Clean up orphaned MCP containers"""
        try:
            all_containers = self.docker_client.containers.list(
                all=True, filters={'label': 'mcp.managed=true'}
            )
            
            cleaned = 0
            for container in all_containers:
                if container.status in ['exited', 'dead']:
                    container.remove()
                    cleaned += 1
                    logger.info("Cleaned orphaned container", name=container.name)
            
            # Refresh running containers list
            await self._discover_existing_containers()
            
            return {"cleaned": cleaned, "running": len(self.running_containers)}
            
        except Exception as e:
            logger.error("Failed to cleanup orphaned containers", error=str(e))
            raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

# Global orchestrator instance
orchestrator = MCPOrchestrator()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    await orchestrator.initialize()
    logger.info("MCP Orchestrator Manager started")
    
    yield
    
    # Shutdown
    logger.info("MCP Orchestrator Manager shutting down")

# FastAPI application
app = FastAPI(
    title="MCP Container Orchestrator",
    description="Docker-in-Docker MCP container management API",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        orchestrator.docker_client.ping()
        return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
    except Exception:
        raise HTTPException(status_code=503, detail="Docker daemon unreachable")

@app.get("/containers")
async def list_containers():
    """List all MCP containers"""
    return await orchestrator.list_containers()

@app.post("/containers")
async def deploy_container(mcp_config: MCPContainer, background_tasks: BackgroundTasks):
    """Deploy a new MCP container"""
    return await orchestrator.deploy_mcp_container(mcp_config)

@app.delete("/containers/{container_name}")
async def stop_container(container_name: str):
    """Stop and remove MCP container"""
    return await orchestrator.stop_mcp_container(container_name)

@app.post("/cleanup")
async def cleanup_orphaned():
    """Clean up orphaned containers"""
    return await orchestrator.cleanup_orphaned_containers()

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

@app.get("/status")
async def get_status():
    """Get orchestrator status"""
    return {
        "running_containers": len(orchestrator.running_containers),
        "docker_info": orchestrator.docker_client.info(),
        "uptime": datetime.utcnow().isoformat()
    }

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info("Received shutdown signal", signal=signum)
    orchestrator.shutdown_event.set()

if __name__ == "__main__":
    import uvicorn
    
    # Setup signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8081,
        log_level="info",
        access_log=True
    )