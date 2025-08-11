#!/usr/bin/env python3
"""Health Monitor Service for SutazAI - Migrated to FastAPI"""

import os
import time
import docker
import logging
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SutazAI Health Monitor", version="1.0.0")
client = docker.from_env()

class ContainerInfo(BaseModel):
    name: str
    status: str
    health: str

class HealthResponse(BaseModel):
    status: str

class MetricsResponse(BaseModel):
    total_containers: int
    healthy: int
    unhealthy: int
    containers: List[ContainerInfo]

def get_container_health() -> List[ContainerInfo]:
    """Get health status of all containers"""
    containers = []
    try:
        for container in client.containers.list(all=True):
            if container.name.startswith('sutazai-'):
                containers.append(ContainerInfo(
                    name=container.name,
                    status=container.status,
                    health=container.attrs.get('State', {}).get('Health', {}).get('Status', 'none')
                ))
    except Exception as e:
        logger.error(f"Error getting container health: {e}")
    return containers

@app.get('/health', response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return HealthResponse(status='healthy')

@app.get('/containers', response_model=List[ContainerInfo])
async def containers():
    """Get container status"""
    return get_container_health()

@app.get('/metrics', response_model=MetricsResponse)
async def metrics():
    """Get system metrics"""
    containers = get_container_health()
    healthy = sum(1 for c in containers if c.health == 'healthy')
    unhealthy = sum(1 for c in containers if c.health == 'unhealthy')
    
    return MetricsResponse(
        total_containers=len(containers),
        healthy=healthy,
        unhealthy=unhealthy,
        containers=containers
    )

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8090, log_level="info")