"""
Graceful Shutdown Handler
Provides signal handling and graceful shutdown for all services
"""

import signal
import sys
import asyncio
import logging
from typing import Callable, List, Optional, Awaitable
from datetime import datetime

logger = logging.getLogger(__name__)


class GracefulShutdownHandler:
    """
    Handles graceful shutdown for services
    
    Usage:
        shutdown_handler = GracefulShutdownHandler()
        shutdown_handler.register_cleanup(cleanup_database)
        shutdown_handler.register_cleanup(close_connections)
        shutdown_handler.start()  # Registers signal handlers
    """
    
    def __init__(self, shutdown_timeout: int = 30):
        """
        Initialize shutdown handler
        
        Args:
            shutdown_timeout: Maximum seconds to wait for cleanup (default 30)
        """
        self.shutdown_timeout = shutdown_timeout
        self.cleanup_tasks: List[Callable] = []
        self.is_shutting_down = False
        self.shutdown_event = asyncio.Event()
        
    def register_cleanup(self, cleanup_func: Callable) -> None:
        """
        Register a cleanup function to run on shutdown
        
        Args:
            cleanup_func: Sync or async function to call during shutdown
        """
        self.cleanup_tasks.append(cleanup_func)
        logger.info(f"Registered cleanup task: {cleanup_func.__name__}")
        
    def _signal_handler(self, signum: int, frame) -> None:
        """
        Handle SIGTERM/SIGINT signals
        
        Args:
            signum: Signal number
            frame: Current stack frame
        """
        signal_name = signal.Signals(signum).name
        logger.info(f"Received {signal_name} signal, initiating graceful shutdown...")
        
        if self.is_shutting_down:
            logger.warning("Shutdown already in progress, ignoring signal")
            return
            
        self.is_shutting_down = True
        self.shutdown_event.set()
        
        # For sync contexts, run async shutdown
        if asyncio.get_event_loop().is_running():
            asyncio.create_task(self.shutdown())
        else:
            asyncio.run(self.shutdown())
            
    async def shutdown(self) -> None:
        """
        Execute all registered cleanup tasks with timeout
        """
        if not self.is_shutting_down:
            self.is_shutting_down = True
            
        logger.info(f"Starting graceful shutdown (timeout: {self.shutdown_timeout}s)")
        start_time = datetime.now()
        
        # Run all cleanup tasks
        for cleanup_func in self.cleanup_tasks:
            try:
                logger.info(f"Running cleanup: {cleanup_func.__name__}")
                
                # Handle async and sync cleanup functions
                if asyncio.iscoroutinefunction(cleanup_func):
                    await asyncio.wait_for(
                        cleanup_func(),
                        timeout=self.shutdown_timeout
                    )
                else:
                    cleanup_func()
                    
                logger.info(f"Cleanup completed: {cleanup_func.__name__}")
                
            except asyncio.TimeoutError:
                logger.error(
                    f"Cleanup timeout: {cleanup_func.__name__} "
                    f"exceeded {self.shutdown_timeout}s"
                )
            except Exception as e:
                logger.error(
                    f"Cleanup error in {cleanup_func.__name__}: {e}",
                    exc_info=True
                )
                
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Graceful shutdown completed in {elapsed:.2f}s")
        
    def start(self) -> None:
        """
        Register signal handlers for graceful shutdown
        Call this after registering all cleanup tasks
        """
        # Register handlers for SIGTERM and SIGINT
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        logger.info("Graceful shutdown handlers registered (SIGTERM, SIGINT)")
        
    async def wait_for_shutdown(self) -> None:
        """
        Wait for shutdown signal (useful for main loop)
        """
        await self.shutdown_event.wait()


# Convenience functions for common use cases

async def wait_for_signal(cleanup_func: Optional[Callable] = None) -> None:
    """
    Simple async function to wait for shutdown signal
    
    Usage:
        async def cleanup():
            await close_connections()
            
        await wait_for_signal(cleanup)
    """
    handler = GracefulShutdownHandler()
    
    if cleanup_func:
        handler.register_cleanup(cleanup_func)
        
    handler.start()
    await handler.wait_for_shutdown()
    
    if cleanup_func:
        await handler.shutdown()


def setup_graceful_shutdown(*cleanup_funcs: Callable) -> GracefulShutdownHandler:
    """
    Quick setup for graceful shutdown with cleanup functions
    
    Usage:
        handler = setup_graceful_shutdown(
            close_database,
            stop_background_tasks,
            flush_logs
        )
    
    Returns:
        Configured GracefulShutdownHandler
    """
    handler = GracefulShutdownHandler()
    
    for cleanup_func in cleanup_funcs:
        handler.register_cleanup(cleanup_func)
        
    handler.start()
    return handler


# Example usage patterns:

if __name__ == "__main__":
    # Example 1: Simple async service
    async def example_async_service():
        handler = GracefulShutdownHandler()
        
        async def cleanup_connections():
            logger.info("Closing connections...")
            await asyncio.sleep(1)  # Simulate cleanup
            
        handler.register_cleanup(cleanup_connections)
        handler.start()
        
        # Main service loop
        while not handler.is_shutting_down:
            logger.info("Service running...")
            await asyncio.sleep(5)
            
        await handler.shutdown()
        
    # Example 2: FastAPI service (add to lifespan)
    """
    from app.core.graceful_shutdown import GracefulShutdownHandler
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        shutdown_handler = GracefulShutdownHandler()
        
        # Register cleanup tasks
        shutdown_handler.register_cleanup(service_connections.disconnect_all)
        shutdown_handler.register_cleanup(close_db)
        shutdown_handler.start()
        
        yield
        
        # Cleanup runs automatically via signal handlers
        # Or manually trigger:
        await shutdown_handler.shutdown()
    """
    
    # Example 3: Streamlit app (add to main)
    """
    from app.core.graceful_shutdown import setup_graceful_shutdown
    
    def cleanup_streamlit():
        st.write("Shutting down gracefully...")
        # Close connections, save state, etc.
        
    if __name__ == "__main__":
        handler = setup_graceful_shutdown(cleanup_streamlit)
        st.run()
    """
