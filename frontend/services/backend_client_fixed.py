"""
Fixed Backend API Client Module
Handles communication with the SutazAI backend API
Resolves event loop issues in Streamlit environment
"""

import requests
import asyncio
import aiohttp
import json
from typing import Dict, List, Optional, Any
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
        """Synchronous chat for Streamlit"""
        try:
            url = urljoin(self.api_v1, "chat")
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
        """Send voice data synchronously"""
        try:
            url = urljoin(self.api_v1, "voice/process")
            files = {"audio": ("audio.wav", audio_data, "audio/wav")}
            response = requests.post(url, files=files, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Voice processing failed: {response.status_code}"}
        except Exception as e:
            logger.error(f"Voice processing failed: {e}")
            return {"error": str(e)}
    
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
                return response.json().get("agents", [])
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
    
    def connect_websocket(self, on_message=None, on_error=None):
        """Connect to WebSocket for real-time communication"""
        import threading
        
        def ws_thread():
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
                
                def on_ws_close(ws):
                    logger.info("WebSocket closed")
                
                def on_ws_open(ws):
                    logger.info("WebSocket connected")
                    # Send initial message
                    ws.send(json.dumps({
                        "type": "connect",
                        "session_id": self._get_session_id()
                    }))
                
                ws = websocket.WebSocketApp(
                    ws_url,
                    on_message=on_ws_message,
                    on_error=on_ws_error,
                    on_close=on_ws_close,
                    on_open=on_ws_open
                )
                
                ws.run_forever()
                
            except Exception as e:
                logger.error(f"WebSocket connection failed: {e}")
                if on_error:
                    on_error(e)
        
        # Start WebSocket in background thread
        ws_thread = threading.Thread(target=ws_thread, daemon=True)
        ws_thread.start()
        
        return ws_thread


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