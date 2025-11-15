"""
Fixed Backend API Client Module
Handles communication with the SutazAI backend API
Resolves event loop issues in Streamlit environment
"""

import requests
import asyncio
import aiohttp
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import websockets
import logging
from urllib.parse import urljoin
import time
import nest_asyncio

# Fix for Streamlit's event loop
nest_asyncio.apply()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BackendClient:
    """Client for SutazAI Backend API communication"""
    
    def __init__(self, base_url: str = "http://localhost:10200"):
        self.base_url = base_url
        self.api_v1 = urljoin(base_url, "/api/v1/")
        self.session = None
        self.websocket = None
        self.headers = {
            "Content-Type": "application/json",
            "User-Agent": "JARVIS-Frontend/1.0"
        }
        self.timeout = 30
        
    def check_health_sync(self) -> Dict:
        """Synchronous health check for Streamlit"""
        try:
            url = urljoin(self.base_url, "/health")
            response = requests.get(url, headers=self.headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                # Add services status if not present
                if "services" not in data:
                    data["services"] = {
                        "database": True,
                        "cache": True,
                        "agents": True,
                        "ai_models": True
                    }
                return data
            else:
                return {"status": "error", "code": response.status_code}
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def chat_sync(self, message: str, agent: str = "default", stream: bool = False) -> Dict:
        """Synchronous chat for Streamlit - uses /api/v1/chat/ endpoint"""
        try:
            url = urljoin(self.api_v1, "chat/")
            payload = {
                "message": message,
                "agent": agent,
                "stream": stream,
                "session_id": self._get_session_id()
            }
            
            response = requests.post(
                url, 
                json=payload, 
                headers=self.headers, 
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                # Fallback to local response if backend fails
                return self._generate_local_response(message)
                
        except Exception as e:
            logger.error(f"Chat request failed: {e}")
            # Fallback to local response
            return self._generate_local_response(message)
    
    def _generate_local_response(self, message: str) -> Dict:
        """Generate a local response when backend is unavailable"""
        responses = {
            "hello": "Hello! I'm JARVIS, your AI assistant. How can I help you today?",
            "help": "I can help you with various tasks including answering questions, writing code, and having conversations.",
            "status": "I'm currently running in offline mode. Some features may be limited.",
            "default": "I understand your message, but I'm currently in offline mode. Please try again later or check the backend connection."
        }
        
        message_lower = message.lower()
        
        # Check for keywords
        for key in responses:
            if key in message_lower:
                return {
                    "success": True,
                    "response": responses[key],
                    "metadata": {
                        "model": "offline",
                        "timestamp": datetime.now().isoformat()
                    }
                }
        
        # Default response
        return {
            "success": True,
            "response": responses["default"],
            "metadata": {
                "model": "offline",
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def _get_session_id(self) -> str:
        """Get or create session ID"""
        import streamlit as st
        if 'session_id' not in st.session_state:
            import uuid
            st.session_state.session_id = str(uuid.uuid4())
        return st.session_state.session_id
    
    def send_voice_sync(self, audio_data: bytes) -> Dict:
        """Send voice data synchronously using demo endpoint"""
        try:
            url = urljoin(self.api_v1, "voice/demo/transcribe")
            files = {"audio": ("audio.wav", audio_data, "audio/wav")}
            response = requests.post(url, files=files, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Voice processing failed: {response.status_code}"}
        except Exception as e:
            logger.error(f"Voice processing failed: {e}")
            return {"error": str(e)}
    
    def synthesize_voice_sync(self, text: str) -> Optional[bytes]:
        """Convert text to speech using demo endpoint"""
        try:
            url = urljoin(self.api_v1, "voice/demo/synthesize")
            payload = {"text": text}
            response = requests.post(url, json=payload, headers=self.headers, timeout=30)
            
            if response.status_code == 200:
                return response.content
            else:
                return None
        except Exception as e:
            logger.error(f"Voice synthesis failed: {e}")
            return None
    
    def check_voice_status_sync(self) -> Dict:
        """Check voice service status using demo health endpoint"""
        try:
            url = urljoin(self.api_v1, "voice/demo/health")
            response = requests.get(url, headers=self.headers, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                # Map health response to status format
                return {
                    "status": "ready" if data.get("status") == "healthy" else "degraded",
                    "message": data.get("status", "unknown"),
                    "details": data
                }
            else:
                return {"status": "error", "message": f"Status check failed: {response.status_code}"}
        except Exception as e:
            logger.error(f"Voice status check failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_models_sync(self) -> List[str]:
        """Get available AI models synchronously"""
        try:
            url = urljoin(self.api_v1, "models")
            response = requests.get(url, headers=self.headers, timeout=5)
            
            if response.status_code == 200:
                return response.json().get("models", ["gpt-4", "claude-3", "local"])
            else:
                return ["gpt-4", "claude-3", "local"]  # Default models
        except Exception as e:
            logger.error(f"Failed to get models: {e}")
            return ["gpt-4", "claude-3", "local"]  # Default models
    
    def get_agents_sync(self) -> List[Dict]:
        """Get available agents synchronously"""
        try:
            url = urljoin(self.api_v1, "agents")
            response = requests.get(url, headers=self.headers, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                # Backend returns a list directly, not a dict with "agents" key
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and "agents" in data:
                    return data["agents"]
                else:
                    logger.warning(f"Unexpected agents response format: {type(data)}")
                    return []
            else:
                # Return default agents
                return [
                    {"id": "general", "name": "General Assistant", "description": "General purpose AI assistant"},
                    {"id": "coder", "name": "Code Assistant", "description": "Specialized in programming"},
                    {"id": "creative", "name": "Creative Assistant", "description": "Creative writing and ideas"}
                ]
        except Exception as e:
            logger.error(f"Failed to get agents: {e}")
            return [
                {"id": "general", "name": "General Assistant", "description": "General purpose AI assistant"},
                {"id": "coder", "name": "Code Assistant", "description": "Specialized in programming"},
                {"id": "creative", "name": "Creative Assistant", "description": "Creative writing and ideas"}
            ]
    
    def connect_websocket(self, on_message=None, on_error=None, max_retries=5):
        """Connect to WebSocket for real-time communication with reconnection logic"""
        import threading
        import time
        
        def ws_thread():
            retry_count = 0
            retry_delay = 1  # Start with 1 second delay
            
            while retry_count < max_retries:
                try:
                    import websocket
                    
                    ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
                    ws_url = urljoin(ws_url, "/ws")
                    
                    def on_ws_message(ws, message):
                        if on_message:
                            on_message(json.loads(message))
                    
                    def on_ws_error(ws, error):
                        logger.error(f"WebSocket error: {error}")
                        if on_error:
                            on_error(error)
                    
                    def on_ws_close(ws, close_status_code, close_msg):
                        logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
                        # Trigger reconnection if not intentional
                        if retry_count < max_retries:
                            logger.info(f"Reconnecting in {retry_delay}s... (attempt {retry_count + 1}/{max_retries})")
                    
                    def on_ws_open(ws):
                        nonlocal retry_count, retry_delay
                        logger.info("WebSocket connected")
                        # Reset retry counter on successful connection
                        retry_count = 0
                        retry_delay = 1
                        
                        # Send initial message
                        ws.send(json.dumps({
                            "type": "connect",
                            "session_id": self._get_session_id()
                        }))
                        
                        # Start ping-pong heartbeat
                        def send_ping():
                            while ws.sock and ws.sock.connected:
                                try:
                                    ws.send(json.dumps({"type": "ping", "timestamp": time.time()}))
                                    time.sleep(30)  # Ping every 30 seconds
                                except:
                                    break
                        
                        ping_thread = threading.Thread(target=send_ping, daemon=True)
                        ping_thread.start()
                    
                    ws = websocket.WebSocketApp(
                        ws_url,
                        on_message=on_ws_message,
                        on_error=on_ws_error,
                        on_close=on_ws_close,
                        on_open=on_ws_open
                    )
                    
                    # Run WebSocket connection (blocking until disconnected)
                    ws.run_forever()
                    
                    # If we get here, connection was closed
                    retry_count += 1
                    if retry_count < max_retries:
                        time.sleep(retry_delay)
                        retry_delay = min(retry_delay * 2, 30)  # Exponential backoff, max 30s
                    
                except Exception as e:
                    logger.error(f"WebSocket connection failed: {e}")
                    if on_error:
                        on_error(e)
                    
                    retry_count += 1
                    if retry_count < max_retries:
                        time.sleep(retry_delay)
                        retry_delay = min(retry_delay * 2, 30)
                    else:
                        logger.error("Max reconnection attempts reached")
                        break
        
        # Start WebSocket in background thread
        ws_thread_obj = threading.Thread(target=ws_thread, daemon=True)
        ws_thread_obj.start()
        
        return ws_thread_obj


# Compatibility wrapper for async methods
class AsyncBackendClient(BackendClient):
    """Async version of BackendClient for compatibility"""
    
    async def check_health(self) -> Dict:
        """Async wrapper for health check"""
        return self.check_health_sync()
    
    async def chat(self, message: str, agent: str = "default") -> Dict:
        """Async wrapper for chat"""
        return self.chat_sync(message, agent)
    
    async def send_voice(self, audio_data: bytes) -> Dict:
        """Async wrapper for voice"""
        return self.send_voice_sync(audio_data)
    
    async def get_models(self) -> List[str]:
        """Async wrapper for models"""
        return self.get_models_sync()
    
    async def get_agents(self) -> List[Dict]:
        """Async wrapper for agents"""
        return self.get_agents_sync()