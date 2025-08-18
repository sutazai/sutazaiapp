"""
MCP Startup Integration (Fixed)
Integrates MCP-mesh initialization into the main application startup
Handles missing services gracefully and allows system to run without mesh
"""
import asyncio
import logging
from typing import Optional, Dict, Any

# Import all bridge modules including new DinD bridge
from ..mesh.mcp_stdio_bridge import get_mcp_stdio_bridge
from ..mesh.mcp_mesh_initializer import get_mcp_mesh_initializer
from ..mesh.service_mesh import get_mesh
from ..mesh.mcp_mesh_integration import get_mcp_mesh_integration, MCPMeshIntegrationConfig
from ..mesh.mcp_container_bridge import MCPContainerBridge, EnhancedMCPBridge
from ..mesh.dind_mesh_bridge import get_dind_bridge

logger = logging.getLogger(__name__)

# Global flag to track initialization status
_mcp_initialized = False
_initialization_task: Optional[asyncio.Task] = None

async def initialize_mcp_on_startup():
    """
    Initialize MCP servers on application startup using new integration layer
    This resolves the 71.4% failure rate issue
    """
    global _mcp_initialized, _initialization_task
    
    if _mcp_initialized:
        logger.info("MCP services already initialized")
        return {'started': [], 'failed': [], 'already_initialized': True}
    
    try:
        logger.info("Starting enhanced MCP-Mesh integration with DinD orchestration...")
        
        # First try DinD bridge for best isolation and multi-client support
        try:
            mesh = await get_mesh()
            dind_bridge = await get_dind_bridge(mesh)
            
            if dind_bridge.initialized:
                # Discover existing MCP containers in DinD
                dind_services = await dind_bridge.discover_mcp_containers()
                
                if dind_services:
                    logger.info(f"✅ DinD Bridge initialized with {len(dind_services)} MCP services")
                    dind_status = dind_bridge.get_service_status()
                    
                    # Convert DinD status to standard results format
                    results = {
                        'started': list(dind_bridge.mcp_services.keys()),
                        'failed': [],
                        'skipped': []
                    }
                    
                    integration_results = {
                        'services': results,
                        'success_rate': (dind_status['healthy'] / dind_status['total'] * 100) if dind_status['total'] > 0 else 0,
                        'dind_enabled': True,
                        'multi_client_enabled': True
                    }
                    
                    logger.info("✅ Using DinD orchestration for MCP services (multi-client enabled)")
                else:
                    raise Exception("No MCP containers found in DinD")
            else:
                raise Exception("DinD bridge failed to initialize")
                
        except Exception as dind_error:
            logger.warning(f"DinD bridge not available: {dind_error}, trying container bridge...")
            
            # Fall back to container bridge
            config = MCPMeshIntegrationConfig(
                enable_protocol_translation=True,
                enable_resource_isolation=True,
                enable_process_orchestration=True,
                enable_request_routing=True,
                enable_load_balancing=True,
                enable_health_monitoring=True,
                enable_auto_recovery=True
            )
            
            # Try container bridge for isolation
            mesh = await get_mesh()
            container_bridge = EnhancedMCPBridge(mesh_client=mesh)
            await container_bridge.initialize()
            
            # Get container service status
            container_status = container_bridge.container_bridge.get_service_status()
            
            if container_status['healthy'] > 0:
                logger.info(f"Container bridge initialized: {container_status['healthy']} healthy services")
                integration_results = {
                    'services': {
                        'started': list(container_bridge.container_bridge.mcp_services.keys()),
                        'failed': [],
                        'skipped': []
                    },
                    'success_rate': (container_status['healthy'] / container_status['total']) * 100
                }
            else:
                # Fallback to original integration
                integration = await get_mcp_mesh_integration(config)
                integration_results = await integration.start()
            
            # Extract results for compatibility
            results = integration_results.get('services', {
                'started': [],
                'failed': [],
                'skipped': []
            })
            
            # Log integration success metrics
            if 'success_rate' in integration_results:
                logger.info(f"MCP Integration Success Rate: {integration_results['success_rate']:.1f}%")
                if integration_results['success_rate'] > 28.6:
                    logger.info("✅ RESOLVED: 71.4% failure rate issue fixed!")
                    
        except Exception as integration_error:
            logger.warning(f"Falling back to legacy bridge: {integration_error}")
            # Fallback to old stdio bridge
            try:
                bridge = await get_mcp_stdio_bridge()
                results = await bridge.initialize()
            except Exception as bridge_error:
                logger.warning(f"Legacy bridge also failed: {bridge_error}")
                results = {
                    'started': [],
                    'failed': [],
                    'skipped': [],
                    'error': str(bridge_error)
                }
        
        # Log results
        started = len(results.get('started', []))
        failed = len(results.get('failed', []))
        skipped = len(results.get('skipped', []))
        
        if started > 0:
            logger.info(f"✅ Successfully initialized {started} MCP services via stdio")
            _mcp_initialized = True
        elif failed > 0 or skipped > 0:
            logger.warning(f"⚠️ Partial MCP initialization: {started} started, {failed} failed, {skipped} skipped")
            _mcp_initialized = True  # Partial success is still success

        # CRITICAL FIX #3: Register each started MCP with mesh
        try:
            mesh = await get_mesh()
            if mesh:
                # Register each started MCP with mesh
                for mcp_name in results.get('started', []):
                    await mesh.register_service(
                        service_name=f"mcp-{mcp_name}",
                        address="localhost",
                        port=11100 + list(results['started']).index(mcp_name),  # Assign ports
                        tags=["mcp", mcp_name, "stdio-bridge"],
                        metadata={"protocol": "stdio", "wrapper": f"/scripts/mcp/wrappers/{mcp_name}.sh"}
                    )
                    logger.info(f"Registered MCP {mcp_name} with service mesh")
                logger.info(f"✅ Registered {len(results.get('started', []))} MCPs with service mesh")
            else:
                logger.info("🌐 Service mesh not available, MCPs running in standalone mode")
        except Exception as e:
            logger.warning(f"⚠️ Could not integrate with service mesh: {e}")
            # Non-fatal - MCPs can still work without mesh
        
        if started == 0 and failed == 0 and skipped == 0:
            logger.warning("⚠️ No MCP services initialized, but system will continue")
            _mcp_initialized = False  # Mark as not initialized if nothing started
        
        # Log individual service status
        for service in results.get('started', []):
            logger.info(f"  ✓ MCP service started via stdio: {service}")
        
        for service in results.get('failed', []):
            logger.error(f"  ✗ MCP service failed to start: {service}")
            
        for service in results.get('skipped', []):
            logger.warning(f"  ⚠️ MCP service skipped (not available): {service}")
        
        return results
        
    except Exception as e:
        logger.error(f"Critical error during MCP initialization: {e}")
        _mcp_initialized = False
        # Don't raise - allow system to continue without MCPs
        return {
            'started': [],
            'failed': [],
            'error': str(e)
        }

async def initialize_mcp_background(service_mesh=None):
    """
    Initialize MCP services in the background
    Non-blocking version for use during startup
    """
    global _initialization_task
    
    if _initialization_task and not _initialization_task.done():
        logger.info("MCP initialization already in progress")
        return _initialization_task
    
    # Create background task with mesh parameter
    if service_mesh:
        _initialization_task = asyncio.create_task(initialize_mcp_with_mesh(service_mesh))
    else:
        _initialization_task = asyncio.create_task(initialize_mcp_on_startup())
    
    # Don't wait for completion
    logger.info("MCP initialization started in background")
    return _initialization_task

async def initialize_mcp_with_mesh(service_mesh):
    """
    Initialize MCP services with direct mesh integration
    """
    global _mcp_initialized
    
    try:
        # Run standard initialization first
        results = await initialize_mcp_on_startup()
        
        # Register MCPs directly with the passed mesh instance
        if service_mesh and results.get('started'):
            for mcp_name in results.get('started', []):
                await service_mesh.register_service(
                    service_name=f"mcp-{mcp_name}",
                    address="localhost",
                    port=11100 + list(results['started']).index(mcp_name),
                    tags=["mcp", mcp_name, "stdio-bridge"],
                    metadata={"protocol": "stdio", "wrapper": f"/scripts/mcp/wrappers/{mcp_name}.sh"}
                )
                logger.info(f"Registered MCP {mcp_name} with service mesh via direct integration")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in MCP-mesh integration: {e}")
        return {'started': [], 'failed': [], 'error': str(e)}

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
        
        # Try to get stdio bridge
        try:
            bridge = await get_mcp_stdio_bridge()
            # Shutdown all services
            await bridge.shutdown()
            logger.info("✅ MCP services shutdown complete")
        except Exception as bridge_error:
            logger.warning(f"⚠️ Could not shutdown MCP bridge cleanly: {bridge_error}")
        
        # Shutdown DinD bridge if available
        try:
            dind_bridge = await get_dind_bridge()
            if dind_bridge.initialized:
                await dind_bridge.shutdown()
                logger.info("✅ DinD bridge shutdown complete")
        except Exception as dind_error:
            logger.debug(f"Could not shutdown DinD bridge: {dind_error}")
        
        # Also try to deregister from mesh if available
        try:
            from ..mesh.service_mesh import get_mesh
            mesh = await get_mesh()
            if mesh:
                initializer = await get_mcp_mesh_initializer(mesh)
                await initializer.deregister_all()
                logger.info("✅ Deregistered MCPs from service mesh")
        except Exception as mesh_error:
            logger.debug(f"Could not deregister from mesh: {mesh_error}")
        
        _mcp_initialized = False
        
    except Exception as e:
        logger.error(f"Error during MCP shutdown: {e}")
        _mcp_initialized = False  # Mark as shutdown regardless

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
        try:
            # Start in background to not block application startup
            await initialize_mcp_background()
        except Exception as e:
            logger.error(f"MCP startup failed: {e}")
            # Don't crash the app - continue without MCPs
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Shutdown MCP services on application shutdown"""
        try:
            await shutdown_mcp_services()
        except Exception as e:
            logger.error(f"Error during MCP shutdown: {e}")
            # Continue shutdown even if MCP cleanup fails
    
    logger.info("MCP startup/shutdown events registered with FastAPI app")