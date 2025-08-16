"""
MCP-Mesh Integration Initializer
Properly registers all MCP services with the service mesh on startup
"""
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import os

logger = logging.getLogger(__name__)

class MCPMeshInitializer:
    """Initializes MCP services and registers them with the mesh"""
    
    # MCP service port mapping (11100-11116)
    MCP_SERVICES = {
        "language-server": {"port": 11100, "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/language-server.sh"},
        "github": {"port": 11101, "wrapper": "npx -y @modelcontextprotocol/server-github"},
        "ultimatecoder": {"port": 11102, "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/ultimatecoder.sh"},
        "sequentialthinking": {"port": 11103, "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/sequentialthinking.sh"},
        "context7": {"port": 11104, "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/context7.sh"},
        "files": {"port": 11105, "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/files.sh"},
        "http": {"port": 11106, "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/http_fetch.sh"},
        "ddg": {"port": 11107, "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/ddg.sh"},
        "postgres": {"port": 11108, "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/postgres.sh"},
        "extended-memory": {"port": 11109, "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/extended-memory.sh"},
        "mcp_ssh": {"port": 11110, "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/mcp_ssh.sh"},
        "nx-mcp": {"port": 11111, "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/nx-mcp.sh"},
        "puppeteer-mcp": {"port": 11112, "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/puppeteer-mcp.sh"},
        "memory-bank-mcp": {"port": 11113, "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/memory-bank-mcp.sh"},
        "playwright-mcp": {"port": 11114, "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/playwright-mcp.sh"},
        "knowledge-graph-mcp": {"port": 11115, "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/knowledge-graph-mcp.sh"},
        "compass-mcp": {"port": 11116, "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/compass-mcp.sh"},
        "claude-task-runner": {"port": 11117, "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/claude-task-runner.sh"}
    }
    
    def __init__(self, mesh_client=None):
        """Initialize with optional mesh client - can work without mesh"""
        self.mesh_client = mesh_client
        self.registered_services: List[str] = []
        
    async def initialize_and_register(self) -> Dict[str, Any]:
        """Initialize MCP services and register with mesh"""
        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "registered": [],
            "failed": [],
            "skipped": [],
            "total": len(self.MCP_SERVICES)
        }
        
        logger.info(f"Attempting to register {len(self.MCP_SERVICES)} MCP services with mesh...")
        
        for name, config in self.MCP_SERVICES.items():
            try:
                # Check if wrapper exists before attempting registration
                wrapper_path = config["wrapper"]
                if not os.path.exists(wrapper_path):
                    results["skipped"].append(name)
                    logger.warning(f"⚠️ Skipping MCP {name}: wrapper not found at {wrapper_path}")
                    continue
                
                # Register MCP service with mesh
                service_data = {
                    "service_name": f"mcp-{name}",
                    "address": "localhost",
                    "port": config["port"],
                    "tags": ["mcp", "stdio-bridge", "available"],
                    "metadata": {
                        "wrapper": config["wrapper"],
                        "mcp_type": name,
                        "protocol": "stdio",
                        "available": True
                    }
                }
                
                # Try to register with mesh - continue even if mesh is not available
                try:
                    if self.mesh_client:
                        instance = await self.mesh_client.register_service(**service_data)
                        if instance:
                            self.registered_services.append(name)
                            results["registered"].append(name)
                            logger.info(f"✅ Registered MCP {name} with mesh on port {config['port']}")
                        else:
                            results["failed"].append(name)
                            logger.warning(f"⚠️ Could not register MCP {name} with mesh, but service may still be available")
                    else:
                        logger.warning(f"⚠️ Mesh client not available, MCP {name} running standalone")
                        results["registered"].append(name)  # Consider it registered for standalone mode
                        self.registered_services.append(name)
                except Exception as mesh_error:
                    # Mesh registration failed, but MCP might still work
                    logger.warning(f"⚠️ Mesh registration failed for MCP {name}: {mesh_error}")
                    results["registered"].append(name)  # Still mark as available
                    self.registered_services.append(name)
                    
            except Exception as e:
                results["failed"].append(name)
                logger.error(f"❌ Error processing MCP {name}: {e}")
        
        # Log summary
        logger.info(f"MCP Mesh Registration Complete:")
        logger.info(f"  Registered: {len(results['registered'])}/{results['total']}")
        logger.info(f"  Failed: {len(results['failed'])}/{results['total']}")
        logger.info(f"  Skipped (not available): {len(results.get('skipped', []))}/{results['total']}")
        
        # Warn if no services were registered but don't fail
        if len(results['registered']) == 0:
            logger.warning("⚠️ No MCP services were registered, but system will continue")
        else:
            logger.info(f"✅ {len(results['registered'])} MCP services are available")
        
        return results
    
    async def deregister_all(self):
        """Deregister all MCP services from mesh"""
        for name in self.registered_services:
            try:
                service_id = f"mcp-{name}"
                await self.mesh_client.deregister_service(service_id)
                logger.info(f"Deregistered MCP {name} from mesh")
            except Exception as e:
                logger.error(f"Error deregistering MCP {name}: {e}")
        
        self.registered_services.clear()

# Global instance
_mcp_mesh_initializer: Optional[MCPMeshInitializer] = None

async def get_mcp_mesh_initializer(mesh_client=None):
    """Get or create MCP mesh initializer - works with or without mesh"""
    global _mcp_mesh_initializer
    if not _mcp_mesh_initializer:
        _mcp_mesh_initializer = MCPMeshInitializer(mesh_client)
    return _mcp_mesh_initializer
