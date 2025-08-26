#!/usr/bin/env python3
"""
Purpose: WebSocket cleanup patch for hygiene monitoring backend
Usage: Apply this patch to fix memory leaks in WebSocket handling
Requirements: None - this is a code patch
"""

import logging

# Configure logger for exception handling
logger = logging.getLogger(__name__)


# Add this method to the EnhancedHygieneBackend class:

async def cleanup_websocket_clients(self):
    """Periodic cleanup of disconnected WebSocket clients"""
    cleanup_interval = 30  # seconds
    
    while self.running:
        try:
            # Create a new set with only active clients
            active_clients = set()
            disconnected_count = 0
            
            for ws in list(self.websocket_clients):
                try:
                    # Check if the WebSocket is still connected
                    if hasattr(ws, 'closed') and not ws.closed:
                        # Send a ping to verify the connection is alive
                        await asyncio.wait_for(ws.ping(), timeout=5.0)
                        active_clients.add(ws)
                    else:
                        disconnected_count += 1
                except (Exception, asyncio.TimeoutError):
                    # Client is not responsive or disconnected
                    disconnected_count += 1
                    try:
                        await ws.close()
                    except Exception as e:
                        # Suppressed exception (was bare except)
                        logger.debug(f"Suppressed exception: {e}")
                        pass
            
            # Update the client set
            self.websocket_clients = active_clients
            
            if disconnected_count > 0:
                logger.info(f"Cleaned up {disconnected_count} disconnected WebSocket clients. Active clients: {len(active_clients)}")
            
            # Log memory usage
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            logger.info(f"Current memory usage: {memory_mb:.1f}MB, Active WebSocket clients: {len(active_clients)}")
            
        except Exception as e:
            logger.error(f"Error in WebSocket cleanup: {e}")
        
        await asyncio.sleep(cleanup_interval)

# Add this to the init_app() function after line 886:
# asyncio.create_task(backend.cleanup_websocket_clients())

# Also modify the broadcast_to_websockets method to be more defensive:

async def broadcast_to_websockets_safe(self, message: Dict[str, Any]):
    """Broadcast message to all WebSocket clients with better error handling"""
    if not self.websocket_clients:
        return
    
    # Use a copy to avoid modification during iteration
    clients_snapshot = list(self.websocket_clients)
    closed_clients = set()
    
    try:
        # Safely serialize the message
        safe_message = self._safe_serialize(message)
        message_json = json.dumps(safe_message, ensure_ascii=False)
        
        # Check message size
        if len(message_json) > 100000:  # 100KB limit
            logger.warning("WebSocket message too large, truncating", size=len(message_json))
            simplified_message = {
                'type': message.get('type', 'unknown'),
                'timestamp': datetime.now().isoformat(),
                'error': 'Message truncated due to size limit'
            }
            message_json = json.dumps(simplified_message)
    
    except Exception as e:
        logger.error("Failed to serialize WebSocket message", error=str(e))
        return
    
    # Send to all clients with proper error handling
    send_tasks = []
    for ws in clients_snapshot:
        try:
            if hasattr(ws, 'closed') and ws.closed:
                closed_clients.add(ws)
                continue
                
            # Create a task for each send operation
            task = asyncio.create_task(self._send_with_timeout(ws, message_json))
            send_tasks.append((ws, task))
            
        except Exception as e:
            logger.debug(f"Error preparing send for client: {e}")
            closed_clients.add(ws)
    
    # Wait for all sends to complete with timeout
    if send_tasks:
        try:
            # Give all sends 5 seconds to complete
            await asyncio.wait_for(
                asyncio.gather(*[task for _, task in send_tasks], return_exceptions=True),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            logger.warning("Some WebSocket sends timed out")
        
        # Check results and mark failed clients
        for ws, task in send_tasks:
            try:
                if task.done() and task.exception():
                    closed_clients.add(ws)
            except Exception as e:
                logger.error(f"Unexpected exception: {e}", exc_info=True)
                closed_clients.add(ws)
    
    # Remove closed clients
    if closed_clients:
        self.websocket_clients -= closed_clients
        logger.info(f"Removed {len(closed_clients)} closed WebSocket connections")

async def _send_with_timeout(self, ws, message):
    """Send message to WebSocket with timeout"""
    try:
        await asyncio.wait_for(ws.send_str(message), timeout=2.0)
    except (Exception, asyncio.TimeoutError) as e:
        logger.debug(f"WebSocket send failed: {e}")
