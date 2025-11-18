"""
WebSocket Connection Manager with Limits and Heartbeat
Manages WebSocket connections with per-user limits and health monitoring
"""

import asyncio
import time
import logging
from typing import Dict, Set, Optional
from fastapi import WebSocket, WebSocketDisconnect
from collections import defaultdict

logger = logging.getLogger(__name__)


class WebSocketConnectionManager:
    """
    Enhanced WebSocket manager with:
    - Per-user connection limits
    - Heartbeat/ping-pong monitoring
    - Connection health tracking
    - Automatic cleanup of stale connections
    """
    
    def __init__(
        self,
        max_connections_per_user: int = 5,
        max_total_connections: int = 1000,
        heartbeat_interval: int = 30,  # seconds
        heartbeat_timeout: int = 60,  # seconds
    ):
        # Active connections: session_id -> WebSocket
        self.active_connections: Dict[str, WebSocket] = {}
        
        # User connections: user_id -> set of session_ids
        self.user_connections: Dict[str, Set[str]] = defaultdict(set)
        
        # Last heartbeat timestamps: session_id -> timestamp
        self.last_heartbeat: Dict[str, float] = {}
        
        # Session metadata: session_id -> dict
        self.session_metadata: Dict[str, dict] = {}
        
        # Configuration
        self.max_connections_per_user = max_connections_per_user
        self.max_total_connections = max_total_connections
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_timeout = heartbeat_timeout
        
        # Background task for heartbeat monitoring
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def connect(
        self,
        websocket: WebSocket,
        session_id: str,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Accept WebSocket connection with validation
        
        Args:
            websocket: WebSocket connection
            session_id: Unique session identifier
            user_id: Optional user identifier for per-user limits
            
        Returns:
            True if connection accepted, False if rejected
        """
        # Check total connection limit
        if len(self.active_connections) >= self.max_total_connections:
            logger.warning(
                f"Connection rejected: total limit ({self.max_total_connections}) reached"
            )
            await websocket.close(code=1008, reason="Server connection limit reached")
            return False
        
        # Check per-user connection limit
        if user_id:
            user_conn_count = len(self.user_connections[user_id])
            if user_conn_count >= self.max_connections_per_user:
                logger.warning(
                    f"Connection rejected for user {user_id}: "
                    f"limit ({self.max_connections_per_user}) reached"
                )
                await websocket.close(
                    code=1008,
                    reason=f"User connection limit ({self.max_connections_per_user}) reached"
                )
                return False
        
        # Accept connection
        await websocket.accept()
        
        # Register connection
        self.active_connections[session_id] = websocket
        self.last_heartbeat[session_id] = time.time()
        self.session_metadata[session_id] = {
            "user_id": user_id,
            "connected_at": time.time(),
            "message_count": 0
        }
        
        if user_id:
            self.user_connections[user_id].add(session_id)
        
        logger.info(
            f"WebSocket connected: session={session_id}, user={user_id}, "
            f"total_connections={len(self.active_connections)}"
        )
        
        # Start heartbeat monitoring if not running
        if not self._running:
            self._running = True
            self._heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
        
        return True
    
    async def disconnect(self, session_id: str):
        """Disconnect and cleanup WebSocket session"""
        if session_id not in self.active_connections:
            return
        
        # Get metadata before cleanup
        metadata = self.session_metadata.get(session_id, {})
        user_id = metadata.get("user_id")
        
        # Remove from active connections
        websocket = self.active_connections.pop(session_id, None)
        
        # Cleanup
        self.last_heartbeat.pop(session_id, None)
        self.session_metadata.pop(session_id, None)
        
        if user_id:
            self.user_connections[user_id].discard(session_id)
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]
        
        logger.info(
            f"WebSocket disconnected: session={session_id}, user={user_id}, "
            f"remaining_connections={len(self.active_connections)}"
        )
        
        # Stop heartbeat monitor if no connections
        if not self.active_connections and self._running:
            self._running = False
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
    
    async def send_message(self, session_id: str, message: dict):
        """Send JSON message to specific session"""
        websocket = self.active_connections.get(session_id)
        if websocket:
            try:
                await websocket.send_json(message)
                
                # Update message count
                if session_id in self.session_metadata:
                    self.session_metadata[session_id]["message_count"] += 1
                    
            except Exception as e:
                logger.error(f"Error sending message to {session_id}: {e}")
                await self.disconnect(session_id)
    
    async def broadcast(self, message: dict, exclude: Optional[Set[str]] = None):
        """Broadcast message to all connected sessions"""
        exclude = exclude or set()
        
        disconnected = []
        for session_id, websocket in self.active_connections.items():
            if session_id in exclude:
                continue
            
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to {session_id}: {e}")
                disconnected.append(session_id)
        
        # Cleanup failed connections
        for session_id in disconnected:
            await self.disconnect(session_id)
    
    async def broadcast_to_user(self, user_id: str, message: dict):
        """Broadcast message to all sessions of a specific user"""
        session_ids = self.user_connections.get(user_id, set()).copy()
        
        for session_id in session_ids:
            await self.send_message(session_id, message)
    
    async def handle_heartbeat(self, session_id: str):
        """Update heartbeat timestamp for session"""
        if session_id in self.active_connections:
            self.last_heartbeat[session_id] = time.time()
    
    async def send_ping(self, session_id: str):
        """Send ping to session"""
        await self.send_message(session_id, {
            "type": "ping",
            "timestamp": time.time()
        })
    
    async def _heartbeat_monitor(self):
        """Background task to monitor connection health"""
        logger.info("WebSocket heartbeat monitor started")
        
        try:
            while self._running:
                await asyncio.sleep(self.heartbeat_interval)
                
                current_time = time.time()
                stale_sessions = []
                
                # Check for stale connections
                for session_id, last_beat in self.last_heartbeat.items():
                    time_since_heartbeat = current_time - last_beat
                    
                    if time_since_heartbeat > self.heartbeat_timeout:
                        # Connection is stale
                        stale_sessions.append(session_id)
                        logger.warning(
                            f"Stale connection detected: {session_id} "
                            f"(no heartbeat for {time_since_heartbeat:.0f}s)"
                        )
                    elif time_since_heartbeat > self.heartbeat_interval:
                        # Send ping to keep alive
                        await self.send_ping(session_id)
                
                # Disconnect stale sessions
                for session_id in stale_sessions:
                    await self.disconnect(session_id)
                
        except asyncio.CancelledError:
            logger.info("WebSocket heartbeat monitor stopped")
        except Exception as e:
            logger.error(f"Heartbeat monitor error: {e}", exc_info=True)
    
    def get_stats(self) -> dict:
        """Get connection statistics"""
        return {
            "total_connections": len(self.active_connections),
            "total_users": len(self.user_connections),
            "max_connections_per_user": self.max_connections_per_user,
            "max_total_connections": self.max_total_connections,
            "heartbeat_interval": self.heartbeat_interval,
            "heartbeat_timeout": self.heartbeat_timeout,
            "user_distribution": {
                user_id: len(sessions)
                for user_id, sessions in self.user_connections.items()
            }
        }
    
    async def shutdown(self):
        """Gracefully shutdown all connections"""
        logger.info("Shutting down WebSocket manager...")
        
        self._running = False
        
        # Cancel heartbeat task
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        session_ids = list(self.active_connections.keys())
        for session_id in session_ids:
            websocket = self.active_connections[session_id]
            try:
                await websocket.close(code=1001, reason="Server shutdown")
            except Exception as e:
                logger.error(f"Error closing websocket {session_id}: {e}")
            await self.disconnect(session_id)
        
        logger.info("WebSocket manager shutdown complete")


# Global instance
ws_connection_manager = WebSocketConnectionManager(
    max_connections_per_user=5,
    max_total_connections=1000,
    heartbeat_interval=30,
    heartbeat_timeout=60
)
