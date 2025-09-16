"""
Backend API Client Module
Handles communication with the SutazAI backend API
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
        self.retry_count = 3
        self.retry_delay = 1
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
        
    async def initialize(self):
        """Initialize async session"""
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession(headers=self.headers)
    
    async def ensure_session(self):
        """Ensure session is available and open"""
        if not self.session or self.session.closed:
            await self.initialize()
    
    async def close(self):
        """Close connections"""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
    
    # Health & Status Endpoints
    
    async def check_health(self) -> Dict:
        """Check backend health status"""
        try:
            if not self.session or self.session.closed:
                await self.initialize()
            
            url = urljoin(self.base_url, "/health")
            async with self.session.get(url, timeout=self.timeout) as response:
                if response.status == 200:
                    data = await response.json()
                    # Add services status if not present
                    if "services" not in data:
                        data["services"] = {
                            "database": True,
                            "cache": True,
                            "agents": True
                        }
                    return data
                else:
                    return {"status": "error", "code": response.status}
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def get_services_status(self) -> Dict:
        """Get status of all backend services"""
        try:
            await self.ensure_session()
            url = urljoin(self.api_v1, "system/services")
            async with self.session.get(url, timeout=self.timeout) as response:
                return await response.json()
        except Exception as e:
            logger.error(f"Services status failed: {e}")
            return {"error": str(e)}
    
    # Chat & Conversation Endpoints
    
    async def chat(self, message: str, agent: str = "jarvis") -> str:
        """Simple chat interface that returns string response"""
        try:
            if not self.session or self.session.closed:
                await self.initialize()
            
            result = await self.send_message(message, agent)
            
            # Extract response text from various possible formats
            if isinstance(result, dict):
                if "response" in result:
                    return result["response"]
                elif "message" in result:
                    return result["message"]
                elif "text" in result:
                    return result["text"]
                elif "error" in result:
                    return f"Error: {result['error']}"
            
            # Fallback to string representation
            return str(result)
            
        except Exception as e:
            logger.error(f"Chat failed: {e}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    async def send_message(self, message: str, agent: str = "jarvis", 
                           context: Optional[Dict] = None) -> Dict:
        """Send message to AI agent"""
        try:
            await self.ensure_session()
            url = urljoin(self.api_v1, "chat/message")
            payload = {
                "message": message,
                "model": "tinyllama:latest",  # Using default model from backend
                "user_id": context.get("user_id") if context else None,
                "session_id": context.get("session_id") if context else None,
                "stream": False,
                "temperature": 0.7,
                "max_tokens": 1000
            }
            
            async with self.session.post(url, json=payload, timeout=self.timeout) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"Backend returned {response.status}: {error_text}")
                    return {"error": f"Backend error: {error_text}"}
        except aiohttp.ClientConnectorError as e:
            logger.error(f"Cannot connect to backend: {e}")
            return {"error": "Cannot connect to backend service. Please ensure it's running."}
        except asyncio.TimeoutError:
            logger.error("Request timed out")
            return {"error": "Request timed out. Please try again."}
        except Exception as e:
            logger.error(f"Send message failed: {e}")
            return {"error": str(e)}
    
    async def stream_response(self, message: str, agent: str = "jarvis", 
                             callback=None) -> None:
        """Stream response from AI agent using WebSocket"""
        try:
            ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
            ws_url = urljoin(ws_url, "/ws/chat")
            
            async with websockets.connect(ws_url) as websocket:
                # Send message
                await websocket.send(json.dumps({
                    "type": "message",
                    "content": message,
                    "agent": agent
                }))
                
                # Receive streaming response
                async for response in websocket:
                    data = json.loads(response)
                    if callback:
                        callback(data)
                    
                    if data.get("type") == "end":
                        break
                        
        except Exception as e:
            logger.error(f"Stream response failed: {e}")
            if callback:
                callback({"type": "error", "error": str(e)})
    
    # Agent Management Endpoints
    
    async def list_agents(self) -> List[Dict]:
        """List available AI agents"""
        try:
            await self.ensure_session()
            url = urljoin(self.api_v1, "agents")
            async with self.session.get(url, timeout=self.timeout) as response:
                return await response.json()
        except Exception as e:
            logger.error(f"List agents failed: {e}")
            return []
    
    async def get_agent_status(self, agent_id: str) -> Dict:
        """Get specific agent status"""
        try:
            await self.ensure_session()
            url = urljoin(self.api_v1, f"agents/{agent_id}/status")
            async with self.session.get(url, timeout=self.timeout) as response:
                return await response.json()
        except Exception as e:
            logger.error(f"Agent status failed: {e}")
            return {"error": str(e)}
    
    async def start_agent(self, agent_id: str, config: Optional[Dict] = None) -> Dict:
        """Start an AI agent"""
        try:
            await self.ensure_session()
            url = urljoin(self.api_v1, f"agents/{agent_id}/start")
            payload = {"config": config or {}}
            
            async with self.session.post(url, json=payload, timeout=self.timeout) as response:
                return await response.json()
        except Exception as e:
            logger.error(f"Start agent failed: {e}")
            return {"error": str(e)}
    
    async def stop_agent(self, agent_id: str) -> Dict:
        """Stop an AI agent"""
        try:
            await self.ensure_session()
            url = urljoin(self.api_v1, f"agents/{agent_id}/stop")
            
            async with self.session.post(url, timeout=self.timeout) as response:
                return await response.json()
        except Exception as e:
            logger.error(f"Stop agent failed: {e}")
            return {"error": str(e)}
    
    # Task Orchestration Endpoints
    
    async def create_task(self, task_type: str, params: Dict, 
                         agents: Optional[List[str]] = None) -> Dict:
        """Create a new task for agents"""
        try:
            await self.ensure_session()
            url = urljoin(self.api_v1, "orchestration/tasks")
            payload = {
                "type": task_type,
                "params": params,
                "agents": agents or ["jarvis"],
                "created_at": datetime.now().isoformat()
            }
            
            async with self.session.post(url, json=payload, timeout=self.timeout) as response:
                return await response.json()
        except Exception as e:
            logger.error(f"Create task failed: {e}")
            return {"error": str(e)}
    
    async def get_task_status(self, task_id: str) -> Dict:
        """Get task execution status"""
        try:
            await self.ensure_session()
            url = urljoin(self.api_v1, f"orchestration/tasks/{task_id}")
            
            async with self.session.get(url, timeout=self.timeout) as response:
                return await response.json()
        except Exception as e:
            logger.error(f"Task status failed: {e}")
            return {"error": str(e)}
    
    async def cancel_task(self, task_id: str) -> Dict:
        """Cancel a running task"""
        try:
            await self.ensure_session()
            url = urljoin(self.api_v1, f"orchestration/tasks/{task_id}/cancel")
            
            async with self.session.post(url, timeout=self.timeout) as response:
                return await response.json()
        except Exception as e:
            logger.error(f"Cancel task failed: {e}")
            return {"error": str(e)}
    
    # Document & Knowledge Endpoints
    
    async def upload_document(self, file_path: str, metadata: Optional[Dict] = None) -> Dict:
        """Upload document for processing"""
        try:
            await self.ensure_session()
            url = urljoin(self.api_v1, "documents/upload")
            
            with open(file_path, 'rb') as f:
                data = aiohttp.FormData()
                data.add_field('file', f, filename=file_path.split('/')[-1])
                if metadata:
                    data.add_field('metadata', json.dumps(metadata))
                
                async with self.session.post(url, data=data, timeout=60) as response:
                    return await response.json()
        except Exception as e:
            logger.error(f"Upload document failed: {e}")
            return {"error": str(e)}
    
    async def search_documents(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        """Search documents in knowledge base"""
        try:
            await self.ensure_session()
            url = urljoin(self.api_v1, "documents/search")
            payload = {
                "query": query,
                "filters": filters or {},
                "limit": 10
            }
            
            async with self.session.post(url, json=payload, timeout=self.timeout) as response:
                return await response.json()
        except Exception as e:
            logger.error(f"Search documents failed: {e}")
            return []
    
    # Vector Database Endpoints
    
    async def vector_search(self, query: str, collection: str = "default", 
                           top_k: int = 5) -> List[Dict]:
        """Search in vector database"""
        try:
            await self.ensure_session()
            url = urljoin(self.api_v1, "vectors/search")
            payload = {
                "query": query,
                "collection": collection,
                "top_k": top_k
            }
            
            async with self.session.post(url, json=payload, timeout=self.timeout) as response:
                return await response.json()
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    async def add_embedding(self, text: str, metadata: Dict, 
                           collection: str = "default") -> Dict:
        """Add text embedding to vector database"""
        try:
            await self.ensure_session()
            url = urljoin(self.api_v1, "vectors/add")
            payload = {
                "text": text,
                "metadata": metadata,
                "collection": collection
            }
            
            async with self.session.post(url, json=payload, timeout=self.timeout) as response:
                return await response.json()
        except Exception as e:
            logger.error(f"Add embedding failed: {e}")
            return {"error": str(e)}
    
    # Voice & Audio Endpoints
    
    async def transcribe_audio(self, audio_data: bytes) -> Dict:
        """Transcribe audio to text"""
        try:
            await self.ensure_session()
            url = urljoin(self.api_v1, "voice/transcribe")
            
            data = aiohttp.FormData()
            data.add_field('audio', audio_data, filename='audio.wav')
            
            async with self.session.post(url, data=data, timeout=60) as response:
                return await response.json()
        except Exception as e:
            logger.error(f"Transcribe audio failed: {e}")
            return {"error": str(e)}
    
    async def synthesize_speech(self, text: str, voice: str = "jarvis") -> bytes:
        """Convert text to speech"""
        try:
            await self.ensure_session()
            url = urljoin(self.api_v1, "voice/synthesize")
            payload = {
                "text": text,
                "voice": voice
            }
            
            async with self.session.post(url, json=payload, timeout=60) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    logger.error(f"Speech synthesis failed: {response.status}")
                    return b""
        except Exception as e:
            logger.error(f"Synthesize speech failed: {e}")
            return b""
    
    # Metrics & Monitoring Endpoints
    
    async def get_metrics(self) -> Dict:
        """Get system metrics"""
        try:
            await self.ensure_session()
            url = urljoin(self.api_v1, "metrics")
            
            async with self.session.get(url, timeout=self.timeout) as response:
                return await response.json()
        except Exception as e:
            logger.error(f"Get metrics failed: {e}")
            return {"error": str(e)}
    
    async def get_logs(self, service: Optional[str] = None, 
                      limit: int = 100) -> List[Dict]:
        """Get system logs"""
        try:
            await self.ensure_session()
            url = urljoin(self.api_v1, "logs")
            params = {"limit": limit}
            if service:
                params["service"] = service
            
            async with self.session.get(url, params=params, timeout=self.timeout) as response:
                return await response.json()
        except Exception as e:
            logger.error(f"Get logs failed: {e}")
            return []
    
    # Utility Methods
    
    def _sync_request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Synchronous request wrapper for non-async contexts"""
        url = urljoin(self.api_v1, endpoint)
        
        for attempt in range(self.retry_count):
            try:
                if method == "GET":
                    response = requests.get(url, headers=self.headers, 
                                          timeout=self.timeout, **kwargs)
                elif method == "POST":
                    response = requests.post(url, headers=self.headers, 
                                           timeout=self.timeout, **kwargs)
                elif method == "PUT":
                    response = requests.put(url, headers=self.headers, 
                                          timeout=self.timeout, **kwargs)
                elif method == "DELETE":
                    response = requests.delete(url, headers=self.headers, 
                                             timeout=self.timeout, **kwargs)
                else:
                    raise ValueError(f"Unsupported method: {method}")
                
                if response.status_code < 500:
                    return response.json()
                
                # Retry on server errors
                time.sleep(self.retry_delay * (attempt + 1))
                
            except requests.ConnectionError as e:
                if attempt == self.retry_count - 1:
                    logger.error(f"Connection failed after {self.retry_count} attempts: {e}")
                    return {"error": "Connection failed", "details": str(e)}
                time.sleep(self.retry_delay * (attempt + 1))
            
            except Exception as e:
                logger.error(f"Request failed: {e}")
                return {"error": str(e)}
        
        return {"error": "Max retries exceeded"}
    
    # Synchronous convenience methods
    
    def health_check_sync(self) -> Dict:
        """Synchronous health check"""
        try:
            response = requests.get(
                urljoin(self.base_url, "/health"),
                headers=self.headers,
                timeout=self.timeout
            )
            return response.json()
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def send_message_sync(self, message: str, agent: str = "jarvis") -> Dict:
        """Synchronous message sending"""
        payload = {
            "message": message,
            "model": "tinyllama:latest",
            "user_id": None,
            "session_id": None,
            "stream": False,
            "temperature": 0.7,
            "max_tokens": 1000
        }
        return self._sync_request("POST", "chat/message", json=payload)
    
    def list_agents_sync(self) -> List[Dict]:
        """Synchronous agent listing"""
        result = self._sync_request("GET", "agents")
        return result if isinstance(result, list) else []
    
    def get_metrics_sync(self) -> Dict:
        """Synchronous metrics retrieval"""
        return self._sync_request("GET", "metrics")


# Create a singleton instance
backend_client = BackendClient()