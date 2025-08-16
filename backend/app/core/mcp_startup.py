"""
MCP Startup Integration
Integrates MCP-mesh initialization into the main application startup
"""
import asyncio
import logging
from typing import Optional

from ..mesh.service_mesh import get_service_mesh
from ..mesh.mcp_bridge import get_mcp_bridge
from ..mesh.mcp_initializer import MCPMeshInitializer

logger = logging.getLogger(__name__)

# Global flag to track initialization status
_mcp_initialized = False
_initialization_task: Optional[asyncio.Task] = None

async def initialize_mcp_on_startup():
    """
    Initialize MCP servers on application startup
    This should be called from the FastAPI startup event
    """
    global _mcp_initialized, _initialization_task
    
    if _mcp_initialized:
        logger.info("MCP services already initialized")
        return
    
    try:
        logger.info("Starting MCP-Mesh integration on application startup...")
        
        # Get service mesh instance
        mesh = await get_service_mesh()
        
        # Get MCP bridge
        bridge = await get_mcp_bridge(mesh)
        
        # Initialize all MCP services
        results = await bridge.initialize()
        
        # Log results
        started = len(results.get('started', []))
        failed = len(results.get('failed', []))
        
        if failed == 0:
            logger.info(f"✅ Successfully initialized all {started} MCP services")
            _mcp_initialized = True
        else:
            logger.warning(f"⚠️ Initialized {started} MCP services, {failed} failed")
            _mcp_initialized = True  # Partial success still marks as initialized
        
        # Log individual service status
        for service in results.get('started', []):
            logger.info(f"  ✓ MCP service started: {service}")
        
        for service in results.get('failed', []):
            logger.error(f"  ✗ MCP service failed: {service}")
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to initialize MCP services: {e}")
        _mcp_initialized = False
        raise

async def initialize_mcp_background():
    """
    Initialize MCP services in the background
    Non-blocking version for use during startup
    """
    global _initialization_task
    
    if _initialization_task and not _initialization_task.done():
        logger.info("MCP initialization already in progress")
        return _initialization_task
    
    # Create background task
    _initialization_task = asyncio.create_task(initialize_mcp_on_startup())
    
    # Don't wait for completion
    logger.info("MCP initialization started in background")
    return _initialization_task

async def shutdown_mcp_services():
    """
    Shutdown all MCP services gracefully
    This should be called from the FastAPI shutdown event
    """
    global _mcp_initialized
    
    if not _mcp_initialized:
        logger.info("MCP services not initialized, nothing to shutdown")
        return
    
    try:
        logger.info("Shutting down MCP services...")
        
        # Get service mesh and bridge
        mesh = await get_service_mesh()
        bridge = await get_mcp_bridge(mesh)
        
        # Shutdown all services
        await bridge.shutdown()
        
        _mcp_initialized = False
        logger.info("MCP services shutdown complete")
        
    except Exception as e:
        logger.error(f"Error during MCP shutdown: {e}")

def is_mcp_initialized() -> bool:
    """Check if MCP services are initialized"""
    return _mcp_initialized

async def wait_for_mcp_initialization(timeout: float = 60.0) -> bool:
    """
    Wait for MCP initialization to complete
    
    Args:
        timeout: Maximum time to wait in seconds
    
    Returns:
        True if initialized, False if timeout
    """
    global _initialization_task
    
    if _mcp_initialized:
        return True
    
    if not _initialization_task:
        return False
    
    try:
        await asyncio.wait_for(_initialization_task, timeout=timeout)
        return _mcp_initialized
    except asyncio.TimeoutError:
        logger.warning(f"MCP initialization timeout after {timeout}s")
        return False

# FastAPI integration helper
def setup_mcp_events(app):
    """
    Setup MCP initialization and shutdown events for FastAPI app
    
    Usage:
        from app.core.mcp_startup import setup_mcp_events
        
        app = FastAPI()
        setup_mcp_events(app)
    """
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize MCP services on startup"""
        # Start in background to not block application startup
        await initialize_mcp_background()
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Shutdown MCP services on application shutdown"""
        await shutdown_mcp_services()
    
    logger.info("MCP startup/shutdown events registered with FastAPI app")