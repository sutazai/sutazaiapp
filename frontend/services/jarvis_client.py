"""
JARVIS Backend Client Service
Handles all communication with the backend API
"""

import requests
import websocket
import json
import threading
import queue
import time
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
import asyncio
import aiohttp
from enum import Enum

class ConnectionStatus(Enum):
    """Connection status states"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"

@dataclass
class BackendConfig:
    """Backend configuration"""
    base_url: str = "http://localhost:10200"
    ws_url: str = "ws://localhost:10200/ws"
    timeout: int = 10
    retry_attempts: int = 3
    retry_delay: float = 1.0

class WebSocketClient:
    """WebSocket client for real-time communication"""
    
    def __init__(self, url: str, on_message: Optional[Callable] = None):
        self.url = url
        self.ws = None
        self.on_message = on_message
        self.connected = False
        self.message_queue = queue.Queue()
        self.thread = None
    
    def connect(self):
        """Connect to WebSocket server"""
        try:
            self.ws = websocket.WebSocketApp(
                self.url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            
            # Run in separate thread
            self.thread = threading.Thread(target=self.ws.run_forever)
            self.thread.daemon = True
            self.thread.start()
            
            # Wait for connection
            timeout = 5
            start = time.time()
            while not self.connected and time.time() - start < timeout:
                time.sleep(0.1)
            
            return self.connected
        except Exception as e:
            print(f"WebSocket connection error: {e}")
            return False
    
    def _on_open(self, ws):
        """Handle connection open"""
        self.connected = True
        print("WebSocket connected")
    
    def _on_message(self, ws, message):
        """Handle incoming message"""
        try:
            data = json.loads(message)
            self.message_queue.put(data)
            
            if self.on_message:
                self.on_message(data)
        except json.JSONDecodeError:
            # Handle non-JSON messages
            self.message_queue.put({"raw": message})
    
    def _on_error(self, ws, error):
        """Handle connection error"""
        print(f"WebSocket error: {error}")
        self.connected = False
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle connection close"""
        print(f"WebSocket closed: {close_status_code} - {close_msg}")
        self.connected = False
    
    def send(self, data: Dict[str, Any]):
        """Send message through WebSocket"""
        if self.connected and self.ws:
            try:
                self.ws.send(json.dumps(data))
                return True
            except Exception as e:
                print(f"Send error: {e}")
                return False
        return False
    
    def get_message(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """Get message from queue"""
        try:
            return self.message_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def disconnect(self):
        """Disconnect WebSocket"""
        if self.ws:
            self.ws.close()
        self.connected = False

class JARVISClient:
    """Main JARVIS backend client"""
    
    def __init__(self, config: Optional[BackendConfig] = None):
        self.config = config or BackendConfig()
        self.ws_client = None
        self.status = ConnectionStatus.DISCONNECTED
        self._health_cache = {}
        self._cache_ttl = 5  # seconds
    
    # Health Check Methods
    
    def check_health(self) -> bool:
        """Check backend health with caching"""
        cache_key = "backend_health"
        
        # Check cache
        if cache_key in self._health_cache:
            cached_time, cached_value = self._health_cache[cache_key]
            if time.time() - cached_time < self._cache_ttl:
                return cached_value
        
        # Make request
        try:
            response = requests.get(
                f"{self.config.base_url}/health",
                timeout=2
            )
            result = response.status_code == 200
            
            # Update cache
            self._health_cache[cache_key] = (time.time(), result)
            
            return result
        except:
            self._health_cache[cache_key] = (time.time(), False)
            return False
    
    def check_voice_health(self) -> Dict[str, Any]:
        """Check voice service health"""
        try:
            response = requests.get(
                f"{self.config.base_url}/api/v1/voice/health",
                timeout=2
            )
            if response.status_code == 200:
                return response.json()
            return {"status": "unavailable", "error": f"Status code: {response.status_code}"}
        except Exception as e:
            return {"status": "offline", "error": str(e)}
    
    # Chat Methods
    
    def send_message(self, message: str, stream: bool = False) -> Optional[Dict[str, Any]]:
        """Send chat message to backend"""
        endpoint = "/api/v1/chat/"
        
        for attempt in range(self.config.retry_attempts):
            try:
                response = requests.post(
                    f"{self.config.base_url}{endpoint}",
                    json={"message": message, "stream": stream},
                    timeout=self.config.timeout
                )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 500:
                    # Server error, might be temporary
                    if attempt < self.config.retry_attempts - 1:
                        time.sleep(self.config.retry_delay)
                        continue
                    return {"error": "Server error", "status_code": 500}
                else:
                    return {"error": f"Unexpected status: {response.status_code}"}
                    
            except requests.exceptions.Timeout:
                if attempt < self.config.retry_attempts - 1:
                    time.sleep(self.config.retry_delay)
                    continue
                return {"error": "Request timeout"}
            except Exception as e:
                return {"error": str(e)}
        
        return {"error": "Max retries exceeded"}
    
    async def send_message_async(self, message: str) -> Optional[Dict[str, Any]]:
        """Send chat message asynchronously"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.config.base_url}/api/v1/chat/",
                    json={"message": message},
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    return None
            except Exception as e:
                print(f"Async request error: {e}")
                return None
    
    # Voice Methods
    
    def process_voice(self, audio_data: bytes) -> Optional[Dict[str, Any]]:
        """Process voice input"""
        try:
            files = {'audio': ('audio.wav', audio_data, 'audio/wav')}
            response = requests.post(
                f"{self.config.base_url}/api/v1/voice/process",
                files=files,
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                return response.json()
            return {"error": f"Status code: {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    def synthesize_speech(self, text: str) -> Optional[bytes]:
        """Convert text to speech"""
        try:
            response = requests.post(
                f"{self.config.base_url}/api/v1/voice/synthesize",
                json={"text": text},
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                return response.content
            return None
        except Exception as e:
            print(f"TTS error: {e}")
            return None
    
    # WebSocket Methods
    
    def connect_websocket(self, on_message: Optional[Callable] = None) -> bool:
        """Connect to WebSocket for streaming"""
        if self.ws_client and self.ws_client.connected:
            return True
        
        self.ws_client = WebSocketClient(self.config.ws_url, on_message)
        return self.ws_client.connect()
    
    def send_ws_message(self, data: Dict[str, Any]) -> bool:
        """Send message through WebSocket"""
        if not self.ws_client:
            if not self.connect_websocket():
                return False
        
        return self.ws_client.send(data)
    
    def disconnect_websocket(self):
        """Disconnect WebSocket"""
        if self.ws_client:
            self.ws_client.disconnect()
            self.ws_client = None
    
    # Utility Methods
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        try:
            response = requests.get(
                f"{self.config.base_url}/api/v1/system/info",
                timeout=2
            )
            if response.status_code == 200:
                return response.json()
            return {"error": "Unable to fetch system info"}
        except:
            return {
                "model": "tinyllama",
                "backend": self.config.base_url,
                "status": "unknown"
            }
    
    def test_connection(self) -> Dict[str, bool]:
        """Test all connections"""
        results = {
            "backend": self.check_health(),
            "voice": self.check_voice_health().get("status") == "healthy",
            "websocket": False
        }
        
        # Test WebSocket
        if self.connect_websocket():
            results["websocket"] = True
            self.disconnect_websocket()
        
        return results

# Singleton instance
_client_instance = None

def get_jarvis_client() -> JARVISClient:
    """Get singleton JARVIS client instance"""
    global _client_instance
    if _client_instance is None:
        _client_instance = JARVISClient()
    return _client_instance

# Convenience functions

def send_chat_message(message: str) -> Optional[Dict[str, Any]]:
    """Send chat message using singleton client"""
    client = get_jarvis_client()
    return client.send_message(message)

def process_voice_input(audio_data: bytes) -> Optional[Dict[str, Any]]:
    """Process voice input using singleton client"""
    client = get_jarvis_client()
    return client.process_voice(audio_data)

def text_to_speech(text: str) -> Optional[bytes]:
    """Convert text to speech using singleton client"""
    client = get_jarvis_client()
    return client.synthesize_speech(text)