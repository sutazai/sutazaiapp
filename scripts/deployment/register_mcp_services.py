#!/usr/bin/env python3
"""
Register MCP services in the service mesh
"""
import asyncio
import httpx
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MCP service configurations based on actual running containers
MCP_SERVICES = [
    {"name": "mcp-files", "port": 10300, "tags": ["mcp", "filesystem"]},
    {"name": "mcp-github", "port": 10301, "tags": ["mcp", "github", "api"]},
    {"name": "mcp-http", "port": 10302, "tags": ["mcp", "http", "fetch"]},
    {"name": "mcp-ddg", "port": 10303, "tags": ["mcp", "search", "duckduckgo"]},
    {"name": "mcp-language-server", "port": 10304, "tags": ["mcp", "lsp"]},
    {"name": "mcp-ssh", "port": 10305, "tags": ["mcp", "ssh", "remote"]},
    {"name": "mcp-ultimatecoder", "port": 10306, "tags": ["mcp", "coding", "ai"]},
    {"name": "mcp-context7", "port": 10307, "tags": ["mcp", "context"]},
    {"name": "mcp-compass", "port": 10308, "tags": ["mcp", "discovery"]},
    {"name": "mcp-knowledge-graph", "port": 10309, "tags": ["mcp", "graph", "knowledge"]},
    {"name": "mcp-memory-bank", "port": 10310, "tags": ["mcp", "memory", "storage"]},
    {"name": "mcp-nx", "port": 10311, "tags": ["mcp", "nx", "monorepo"]},
]

async def register_service(service_config):
    """Register a single MCP service with the mesh"""
    try:
        async with httpx.AsyncClient() as client:
            # Register with mesh v2 API
            response = await client.post(
                "http://localhost:10010/api/v1/mesh/v2/register",
                json={
                    "service_name": service_config["name"],
                    "address": "localhost",
                    "port": service_config["port"],
                    "tags": service_config["tags"],
                    "metadata": {
                        "type": "mcp",
                        "protocol": "stdio",
                        "wrapper": f"/opt/sutazaiapp/backend/mcp_wrappers/{service_config['name'].replace('mcp-', '')}_wrapper.sh"
                    }
                },
                timeout=10.0
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"✓ Registered {service_config['name']}: {result['message']}")
                return True
            else:
                logger.error(f"✗ Failed to register {service_config['name']}: {response.status_code} - {response.text}")
                return False
                
    except Exception as e:
        logger.error(f"✗ Error registering {service_config['name']}: {e}")
        return False

async def main():
    """Register all MCP services"""
    logger.info("=" * 60)
    logger.info("MCP SERVICE MESH REGISTRATION")
    logger.info("=" * 60)
    
    # First check mesh health
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:10010/api/v1/mesh/v2/health")
            if response.status_code == 200:
                health = response.json()
                logger.info(f"Mesh status: {health['status']}")
                logger.info(f"Total services: {health['queue_stats']['total_services']}")
            else:
                logger.error(f"Failed to check mesh health: {response.status_code}")
                return 1
    except Exception as e:
        logger.error(f"Cannot connect to mesh: {e}")
        return 1
    
    # Register all MCP services
    results = []
    for service in MCP_SERVICES:
        result = await register_service(service)
        results.append(result)
        await asyncio.sleep(0.5)  # Small delay between registrations
    
    # Summary
    successful = sum(1 for r in results if r)
    failed = len(results) - successful
    
    logger.info("=" * 60)
    logger.info("REGISTRATION SUMMARY")
    logger.info(f"Total: {len(results)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info("=" * 60)
    
    # Check final mesh status
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:10010/api/v1/mesh/v2/topology")
            if response.status_code == 200:
                topology = response.json()
                logger.info(f"Total instances in mesh: {topology['total_instances']}")
                logger.info(f"Healthy instances: {topology['healthy_instances']}")
    except Exception as e:
        logger.error(f"Failed to get topology: {e}")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)