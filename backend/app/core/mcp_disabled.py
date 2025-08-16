"""
MCP Disabled Module - Temporary solution to bypass MCP startup failures
This module provides stub implementations to prevent MCP startup errors
while maintaining API compatibility.
"""
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Global flag to track initialization status
_mcp_initialized = False

async def initialize_mcp_on_startup():
    """
    Stub initialization - MCP servers are managed externally by Claude
    """
    global _mcp_initialized
    
    if _mcp_initialized:
        logger.info("MCP services already marked as initialized (disabled)")
        return
    
    logger.info("MCP startup disabled - servers are managed externally by Claude")
    logger.info("✅ MCP integration bypassed successfully")
    _mcp_initialized = True
    
    return {
        "status": "disabled",
        "message": "MCP servers are managed externally by Claude",
        "started": [],
        "failed": []
    }

async def initialize_mcp_background():
    """
    Stub background initialization
    """
    return await initialize_mcp_on_startup()

async def shutdown_mcp_services():
    """
    Stub shutdown - nothing to shutdown
    """
    global _mcp_initialized
    
    if not _mcp_initialized:
        logger.info("MCP services not initialized, nothing to shutdown")
        return
    
    logger.info("MCP shutdown skipped - servers are managed externally")
    _mcp_initialized = False

def is_mcp_initialized() -> bool:
    """Check if MCP services are initialized"""
    return _mcp_initialized

async def wait_for_mcp_initialization(timeout: float = 60.0) -> bool:
    """
    Stub wait - immediately returns as initialized
    """
    global _mcp_initialized
    _mcp_initialized = True
    return True

def setup_mcp_events(app):
    """
    Setup MCP initialization and shutdown events for FastAPI app
    This version doesn't actually start any servers
    """
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize MCP services on startup"""
        await initialize_mcp_background()
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Shutdown MCP services on application shutdown"""
        await shutdown_mcp_services()
    
    logger.info("MCP startup/shutdown events registered with FastAPI app (disabled mode)")