"""Test the new MCP stdio bridge implementation"""
import asyncio
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, '/opt/sutazaiapp/backend')

from app.mesh.mcp_stdio_bridge import get_mcp_stdio_bridge

async def test_mcp_stdio():
    """Test MCP stdio bridge initialization"""
    logger.info("Testing MCP stdio bridge...")
    
    try:
        bridge = await get_mcp_stdio_bridge()
        logger.info("Bridge instance created")
        
        logger.info("Initializing MCP services...")
        results = await bridge.initialize()
        
        logger.info(f"Results: {results}")
        logger.info(f"Started: {results.get('started', [])}")
        logger.info(f"Failed: {results.get('failed', [])}")
        
        logger.info("Running health checks...")
        health_results = await bridge.health_check_all()
        for service, status in health_results.items():
            logger.info(f"  {service}: {status}")
        
        logger.info("Services running for 5 seconds...")
        await asyncio.sleep(5)
        
        logger.info("Shutting down services...")
        await bridge.shutdown()
        logger.info("Shutdown complete")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = asyncio.run(test_mcp_stdio())
    sys.exit(0 if success else 1)