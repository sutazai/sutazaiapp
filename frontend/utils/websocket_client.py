"""
WebSocket Client for Real-Time Streaming Communication
Handles connection management, auto-reconnection, and message streaming
"""

import asyncio
import websockets
import json
import threading
import queue
import time
import logging
from typing import Optional, Callable, Dict, Any, List
from datetime import datetime
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConnectionState(Enum):
    """WebSocket connection states"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"

class WebSocketClient:
    """
    Advanced WebSocket client with auto-reconnection and streaming support
    Designed for Streamlit integration with synchronous interface
    """
    
    def __init__(self, base_url: str = "ws://localhost:10200/ws"):
        self.base_url = base_url
        self.websocket = None
        self.state = ConnectionState.DISCONNECTED
        
        # Connection management
        self.max_retries = 5
        self.retry_delay = 2
        self.current_retry = 0
        self.heartbeat_interval = 30
        self.last_heartbeat = None
        
        # Message handling
        self.message_queue = queue.Queue()
        self.response_callbacks: Dict[str, Callable] = {}
        self.stream_callback: Optional[Callable] = None
        
        # Threading
        self.connection_thread: Optional[threading.Thread] = None
        self.heartbeat_thread: Optional[threading.Thread] = None
        self.is_running = False
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        
        # Session management
        self.session_id: Optional[str] = None
        self.message_history: List[Dict[str, Any]] = []
        
        # Metrics
        self.messages_sent = 0
        self.messages_received = 0
        self.connection_start_time = None
        self.total_bytes_sent = 0
        self.total_bytes_received = 0
    
    def connect(self, session_id: Optional[str] = None) -> bool:
        """
        Start WebSocket connection in background thread
        Returns immediately with connection attempt status
        """
        if self.state in [ConnectionState.CONNECTED, ConnectionState.CONNECTING]:
            logger.warning("Already connected or connecting")
            return False
        
        self.session_id = session_id or f"session_{int(time.time())}"
        self.is_running = True
        self.state = ConnectionState.CONNECTING
        
        # Start connection thread
        self.connection_thread = threading.Thread(
            target=self._run_connection_loop,
            daemon=True
        )
        self.connection_thread.start()
        
        # Wait briefly for initial connection
        max_wait = 5
        start = time.time()
        while time.time() - start < max_wait:
            if self.state == ConnectionState.CONNECTED:
                return True
            time.sleep(0.1)
        
        return self.state == ConnectionState.CONNECTED
    
    def disconnect(self):
        """Gracefully close WebSocket connection"""
        self.is_running = False
        self.state = ConnectionState.DISCONNECTED
        
        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self._close_connection(), self.loop)
        
        # Wait for threads to finish
        if self.connection_thread:
            self.connection_thread.join(timeout=3)
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=1)
    
    def send_message(self, message: str, callback: Optional[Callable] = None) -> bool:
        """
        Send message through WebSocket
        Returns True if message was queued successfully
        """
        if self.state != ConnectionState.CONNECTED:
            logger.error("Cannot send message: Not connected")
            return False
        
        message_data = {
            "type": "chat",
            "message": message,
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "message_id": f"msg_{self.messages_sent}"
        }
        
        if callback:
            self.response_callbacks[message_data["message_id"]] = callback
        
        try:
            self.message_queue.put(message_data)
            self.messages_sent += 1
            self.message_history.append({
                "role": "user",
                "content": message,
                "timestamp": datetime.now().isoformat()
            })
            return True
        except Exception as e:
            logger.error(f"Failed to queue message: {e}")
            return False
    
    def send_streaming_request(self, message: str, model: str = "tinyllama:latest",
                               temperature: float = 0.7, stream_callback: Optional[Callable] = None) -> bool:
        """
        Send streaming chat request
        stream_callback will be called for each token received
        """
        if self.state != ConnectionState.CONNECTED:
            logger.error("Cannot send streaming request: Not connected")
            return False
        
        self.stream_callback = stream_callback
        
        message_data = {
            "type": "stream_chat",
            "message": message,
            "model": model,
            "temperature": temperature,
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "stream": True
        }
        
        try:
            self.message_queue.put(message_data)
            self.messages_sent += 1
            return True
        except Exception as e:
            logger.error(f"Failed to queue streaming request: {e}")
            return False
    
    def get_state(self) -> ConnectionState:
        """Get current connection state"""
        return self.state
    
    def is_connected(self) -> bool:
        """Check if WebSocket is connected"""
        return self.state == ConnectionState.CONNECTED
    
    def get_latency(self) -> Optional[float]:
        """Get current latency in milliseconds"""
        if self.last_heartbeat:
            return (time.time() - self.last_heartbeat) * 1000
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        uptime = None
        if self.connection_start_time:
            uptime = int(time.time() - self.connection_start_time)
        
        return {
            "state": self.state.value,
            "session_id": self.session_id,
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "uptime_seconds": uptime,
            "bytes_sent": self.total_bytes_sent,
            "bytes_received": self.total_bytes_received,
            "latency_ms": self.get_latency(),
            "retry_count": self.current_retry
        }
    
    # Internal async methods
    
    def _run_connection_loop(self):
        """Run asyncio event loop in background thread"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        try:
            self.loop.run_until_complete(self._connection_handler())
        except Exception as e:
            logger.error(f"Connection loop error: {e}")
            self.state = ConnectionState.ERROR
        finally:
            self.loop.close()
    
    async def _connection_handler(self):
        """Handle WebSocket connection with auto-reconnection"""
        while self.is_running:
            try:
                logger.info(f"Connecting to {self.base_url}...")
                
                async with websockets.connect(
                    self.base_url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5
                ) as websocket:
                    self.websocket = websocket
                    self.state = ConnectionState.CONNECTED
                    self.connection_start_time = time.time()
                    self.current_retry = 0
                    logger.info("WebSocket connected successfully")
                    
                    # Start heartbeat
                    self._start_heartbeat()
                    
                    # Send initial handshake
                    await self._send_handshake()
                    
                    # Create send and receive tasks
                    send_task = asyncio.create_task(self._send_loop())
                    receive_task = asyncio.create_task(self._receive_loop())
                    
                    # Wait for either task to complete (error or disconnect)
                    done, pending = await asyncio.wait(
                        [send_task, receive_task],
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    # Cancel remaining tasks
                    for task in pending:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                    
            except websockets.exceptions.WebSocketException as e:
                logger.error(f"WebSocket error: {e}")
                self.state = ConnectionState.RECONNECTING
                
            except Exception as e:
                logger.error(f"Connection error: {e}")
                self.state = ConnectionState.ERROR
            
            finally:
                self.websocket = None
                
                # Attempt reconnection
                if self.is_running and self.current_retry < self.max_retries:
                    self.current_retry += 1
                    delay = min(self.retry_delay * (2 ** self.current_retry), 30)
                    logger.info(f"Reconnecting in {delay}s (attempt {self.current_retry}/{self.max_retries})")
                    await asyncio.sleep(delay)
                elif self.current_retry >= self.max_retries:
                    logger.error("Max reconnection attempts reached")
                    self.state = ConnectionState.ERROR
                    break
                else:
                    # is_running is False, clean exit
                    break
    
    async def _send_handshake(self):
        """Send initial handshake message"""
        if self.websocket:
            handshake = {
                "type": "handshake",
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "client": "JARVIS Frontend"
            }
            await self.websocket.send(json.dumps(handshake))
    
    async def _send_loop(self):
        """Send queued messages"""
        while self.is_running and self.websocket:
            try:
                # Non-blocking queue get with timeout
                message = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.message_queue.get(timeout=0.5)
                )
                
                # Send message
                message_str = json.dumps(message)
                await self.websocket.send(message_str)
                self.total_bytes_sent += len(message_str)
                logger.debug(f"Sent message: {message.get('type')}")
                
            except queue.Empty:
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Send error: {e}")
                break
    
    async def _receive_loop(self):
        """Receive and process messages"""
        while self.is_running and self.websocket:
            try:
                message_str = await self.websocket.recv()
                self.total_bytes_received += len(message_str)
                self.messages_received += 1
                
                # Parse message
                try:
                    message = json.loads(message_str)
                    await self._handle_message(message)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse message: {message_str[:100]}")
                
            except websockets.exceptions.ConnectionClosed:
                logger.warning("Connection closed")
                break
            except Exception as e:
                logger.error(f"Receive error: {e}")
                break
    
    async def _handle_message(self, message: Dict[str, Any]):
        """Handle received message based on type"""
        msg_type = message.get("type", "unknown")
        
        if msg_type == "pong":
            # Heartbeat response
            self.last_heartbeat = time.time()
            
        elif msg_type == "chat_response":
            # Full chat response
            message_id = message.get("message_id")
            if message_id and message_id in self.response_callbacks:
                callback = self.response_callbacks.pop(message_id)
                # Run callback in thread pool to avoid blocking
                asyncio.get_event_loop().run_in_executor(None, callback, message)
        
        elif msg_type == "stream_token":
            # Streaming token
            if self.stream_callback:
                token = message.get("token", "")
                asyncio.get_event_loop().run_in_executor(None, self.stream_callback, token)
        
        elif msg_type == "stream_complete":
            # Stream completed
            if self.stream_callback:
                asyncio.get_event_loop().run_in_executor(
                    None,
                    self.stream_callback,
                    {"complete": True, "message": message.get("full_response", "")}
                )
                self.stream_callback = None
        
        elif msg_type == "error":
            # Error message
            error_msg = message.get("error", "Unknown error")
            logger.error(f"Server error: {error_msg}")
        
        else:
            logger.warning(f"Unknown message type: {msg_type}")
    
    async def _close_connection(self):
        """Close WebSocket connection"""
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception as e:
                logger.error(f"Error closing connection: {e}")
    
    def _start_heartbeat(self):
        """Start heartbeat thread"""
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            return
        
        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True
        )
        self.heartbeat_thread.start()
    
    def _heartbeat_loop(self):
        """Send periodic heartbeat messages"""
        while self.is_running and self.state == ConnectionState.CONNECTED:
            try:
                ping_message = {
                    "type": "ping",
                    "timestamp": datetime.now().isoformat()
                }
                self.message_queue.put(ping_message)
                time.sleep(self.heartbeat_interval)
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                break


# Convenience functions for Streamlit integration

def create_websocket_client(base_url: str = "ws://localhost:10200/ws",
                            session_id: Optional[str] = None) -> WebSocketClient:
    """Create and connect WebSocket client"""
    client = WebSocketClient(base_url)
    client.connect(session_id)
    return client


def get_or_create_client(session_state, key: str = "ws_client") -> WebSocketClient:
    """Get existing client from session state or create new one"""
    if key not in session_state or not session_state[key].is_connected():
        session_state[key] = create_websocket_client()
    return session_state[key]
