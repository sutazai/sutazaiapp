"""
SutazAI Platform Main Application
FastAPI backend with comprehensive service integrations
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Response, Request
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from contextlib import asynccontextmanager
import logging
import sys
import json
import uuid
import httpx
import asyncio
from datetime import datetime
from typing import Dict, List, Any
from app.core.config import settings
from app.core.database import init_db, close_db, Base
from app.services.connections import service_connections
from app.api.v1.router import api_router

# Import Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY, CONTENT_TYPE_LATEST

# Import custom middleware
from app.middleware.metrics import PrometheusMetricsMiddleware
from app.middleware.logging import RequestLoggingMiddleware, configure_structured_logging

# Import models to ensure they're registered
from app.models import User

# Configure structured logging
if settings.DEBUG:
    configure_structured_logging(log_level="DEBUG")
else:
    configure_structured_logging(log_level="INFO")

logger = logging.getLogger(__name__)

# Initialize Prometheus metrics
REQUEST_COUNT = Counter('backend_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('backend_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint'])
ACTIVE_CONNECTIONS = Gauge('backend_active_connections', 'Active database connections')
SERVICE_STATUS = Gauge('backend_service_status', 'Service connection status', ['service'])
CHAT_MESSAGES = Counter('backend_chat_messages_total', 'Total chat messages processed')
WEBSOCKET_CONNECTIONS = Gauge('backend_websocket_connections', 'Active WebSocket connections')


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle
    Initialize connections on startup, cleanup on shutdown
    """
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    
    try:
        # Initialize database
        try:
            await init_db()
            logger.info("Database initialized")
        except Exception as e:
            logger.warning(f"Database init failed or skipped: {e}")
        
        # Connect to all services
        await service_connections.connect_all()
        logger.info("All service connections established")
        
        # Register with Consul
        await register_with_consul()
        
        yield
        
    finally:
        # Cleanup on shutdown
        logger.info("Shutting down application")
        await service_connections.disconnect_all()
        try:
            await close_db()
        except Exception as e:
            logger.warning(f"Database close failed: {e}")
        await deregister_from_consul()
        logger.info("Cleanup completed")


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses"""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        # Content Security Policy
        csp_directives = [
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'",  # Streamlit requires unsafe-eval
            "style-src 'self' 'unsafe-inline'",
            "img-src 'self' data: https:",
            "font-src 'self' data:",
            "connect-src 'self' ws: wss:",
            "frame-ancestors 'none'",
            "base-uri 'self'",
            "form-action 'self'"
        ]
        response.headers["Content-Security-Policy"] = "; ".join(csp_directives)
        
        return response


# Create FastAPI application
app = FastAPI(
    title="SutazAI Backend API",
    description="Comprehensive AI-powered platform with authentication, vector databases, and AI agents",
    version="1.0.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    contact={
        "name": "SutazAI Support",
        "url": "https://github.com/sutazai/sutazaiapp",
        "email": "support@sutazai.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
    servers=[
        {
            "url": "http://localhost:10200",
            "description": "Development server"
        },
        {
            "url": "https://api.sutazai.com",
            "description": "Production server"
        }
    ],
    openapi_tags=[
        {
            "name": "authentication",
            "description": "User authentication and authorization endpoints"
        },
        {
            "name": "health",
            "description": "Health check and monitoring endpoints"
        },
        {
            "name": "metrics",
            "description": "Prometheus metrics for monitoring"
        },
        {
            "name": "chat",
            "description": "AI chat and WebSocket endpoints"
        }
    ]
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add security headers middleware
app.add_middleware(SecurityHeadersMiddleware)

# Add request logging middleware
app.add_middleware(RequestLoggingMiddleware)

# Add Prometheus metrics middleware
app.add_middleware(PrometheusMetricsMiddleware)

# Include API routers
app.include_router(api_router, prefix=settings.API_V1_STR)

# WebSocket connection manager
class WebSocketManager:
    """Manages WebSocket connections for streaming chat"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.session_connections: Dict[str, WebSocket] = {}
        self.chat_sessions: Dict[str, List[Dict[str, Any]]] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept and track a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.session_connections[session_id] = websocket
        logger.info(f"WebSocket connected for session {session_id}")
    
    def disconnect(self, websocket: WebSocket, session_id: str):
        """Remove a WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if session_id in self.session_connections:
            del self.session_connections[session_id]
        logger.info(f"WebSocket disconnected for session {session_id}")
    
    async def send_message(self, websocket: WebSocket, message: dict):
        """Send a JSON message to a specific WebSocket"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}")

# Global WebSocket manager
ws_manager = WebSocketManager()

# Ollama configuration
OLLAMA_HOST = "host.docker.internal"
OLLAMA_PORT = "11434"
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"

async def call_ollama_streaming(prompt: str, model: str = "tinyllama:latest", temperature: float = 0.7):
    """Stream responses from Ollama API"""
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            url = f"{OLLAMA_BASE_URL}/api/generate"
            
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "num_predict": 500
                }
            }
            
            async with client.stream('POST', url, json=payload) as response:
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            yield data
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse JSON: {line}")
                            continue
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield {"error": str(e), "done": True}

async def call_ollama(message: str, model: str = "tinyllama:latest", temperature: float = 0.7) -> str:
    """Call Ollama API for text generation (non-streaming)"""
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            url = f"{OLLAMA_BASE_URL}/api/generate"
            
            payload = {
                "model": model,
                "prompt": message,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": 500
                }
            }
            
            response = await client.post(url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "No response generated")
            else:
                return f"Error: {response.status_code} - {response.text}"
                    
    except Exception as e:
        logger.error(f"Ollama error: {str(e)}")
        return f"Error calling Ollama: {str(e)}"

# WebSocket endpoint for real-time streaming chat
@app.websocket("/ws")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for real-time streaming chat with Ollama"""
    session_id = str(uuid.uuid4())
    await ws_manager.connect(websocket, session_id)
    
    # Initialize session
    if session_id not in ws_manager.chat_sessions:
        ws_manager.chat_sessions[session_id] = []
    
    try:
        # Send welcome message
        await ws_manager.send_message(websocket, {
            "type": "connection",
            "status": "connected",
            "session_id": session_id,
            "message": "WebSocket chat connected successfully"
        })
        
        RECEIVE_TIMEOUT = 20
        while True:
            # Receive message from client with timeout to avoid hangs
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=RECEIVE_TIMEOUT)
            except asyncio.TimeoutError:
                await ws_manager.send_message(websocket, {
                    "type": "error",
                    "message": "Timeout waiting for client message"
                })
                break
            
            # Handle different message types
            message_type = data.get("type", "chat")
            
            if message_type == "ping":
                # Heartbeat response
                await ws_manager.send_message(websocket, {
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                })
                continue
            
            elif message_type == "chat":
                # Process chat message
                user_message = data.get("message", "")
                model = data.get("model", "tinyllama:latest")
                temperature = data.get("temperature", 0.7)
                stream = data.get("stream", True)
                
                if not user_message:
                    await ws_manager.send_message(websocket, {
                        "type": "error",
                        "message": "No message provided"
                    })
                    continue
                
                # Store user message
                ws_manager.chat_sessions[session_id].append({
                    "role": "user",
                    "content": user_message,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                # Send acknowledgment
                await ws_manager.send_message(websocket, {
                    "type": "message_received",
                    "message": user_message,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                if stream:
                    # Stream response from Ollama
                    full_response = ""
                    await ws_manager.send_message(websocket, {
                        "type": "stream_start",
                        "model": model
                    })
                    
                    async for chunk in call_ollama_streaming(user_message, model, temperature):
                        if "error" in chunk:
                            await ws_manager.send_message(websocket, {
                                "type": "error",
                                "message": chunk["error"]
                            })
                            break
                        
                        if "response" in chunk:
                            full_response += chunk["response"]
                            await ws_manager.send_message(websocket, {
                                "type": "stream_chunk",
                                "content": chunk["response"],
                                "done": chunk.get("done", False)
                            })
                        
                        if chunk.get("done", False):
                            # Store assistant response
                            ws_manager.chat_sessions[session_id].append({
                                "role": "assistant",
                                "content": full_response,
                                "timestamp": datetime.utcnow().isoformat(),
                                "model": model
                            })
                            
                            await ws_manager.send_message(websocket, {
                                "type": "stream_end",
                                "full_response": full_response,
                                "timestamp": datetime.utcnow().isoformat()
                            })
                            break
                else:
                    # Non-streaming response
                    response_text = await call_ollama(user_message, model, temperature)
                    
                    # Store assistant response
                    ws_manager.chat_sessions[session_id].append({
                        "role": "assistant",
                        "content": response_text,
                        "timestamp": datetime.utcnow().isoformat(),
                        "model": model
                    })
                    
                    await ws_manager.send_message(websocket, {
                        "type": "response",
                        "content": response_text,
                        "model": model,
                        "timestamp": datetime.utcnow().isoformat()
                    })
            
            elif message_type == "get_history":
                # Send chat history
                await ws_manager.send_message(websocket, {
                    "type": "history",
                    "messages": ws_manager.chat_sessions.get(session_id, []),
                    "count": len(ws_manager.chat_sessions.get(session_id, []))
                })
            
            elif message_type == "clear_history":
                # Clear session history
                ws_manager.chat_sessions[session_id] = []
                await ws_manager.send_message(websocket, {
                    "type": "history_cleared",
                    "session_id": session_id
                })
                
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, session_id)
        logger.info(f"Client {session_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
        try:
            await ws_manager.send_message(websocket, {
                "type": "error",
                "message": str(e)
            })
        except:
            pass
        ws_manager.disconnect(websocket, session_id)


@app.get("/", tags=["health"])
async def root():
    """
    API root endpoint with service information
    
    Returns basic information about the SutazAI API including version, status, and links to documentation.
    
    **Returns:**
    - `200 OK`: Service information
        - `app`: Application name
        - `version`: API version
        - `status`: operational
        - `docs`: Link to interactive API documentation
        - `api`: API base path
    """
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "operational",
        "docs": "/docs",
        "api": settings.API_V1_STR
    }


@app.get("/health", tags=["health"])
async def health_check():
    """
    Basic health check endpoint
    
    Quickly determine if the API is operational. Does not check external service dependencies.
    Use `/health/detailed` for comprehensive service health checks.
    
    **Returns:**
    - `200 OK`: Service is healthy
        - `status`: "healthy"
        - `app`: Application name
    """
    return {"status": "healthy", "app": settings.APP_NAME}


@app.get("/health/detailed", tags=["health"])
async def detailed_health_check():
    """
    Comprehensive health check for all service dependencies
    
    Checks connectivity and health status of all integrated services including:
    - **Redis**: Caching and session management
    - **RabbitMQ**: Message queue
    - **Neo4j**: Graph database
    - **ChromaDB**: Vector database
    - **Qdrant**: Vector database
    - **FAISS**: Vector database
    - **Consul**: Service discovery
    - **Kong**: API gateway
    - **Ollama**: AI model server
    
    **Returns:**
    - `200 OK`: All services healthy or some degraded
        - `status`: "healthy" (all pass) or "degraded" (some failing)
        - `app`: Application name
        - `version`: API version
        - `services`: Object with service names and boolean health status
        - `healthy_count`: Number of healthy services
        - `total_services`: Total number of services checked
    
    - `503 Service Unavailable`: Health check failed
        - `detail`: Error message
    
    **Example Response:**
    ```json
    {
      "status": "healthy",
      "app": "SutazAI Backend API",
      "version": "1.0.0",
      "services": {
        "redis": true,
        "rabbitmq": true,
        "neo4j": true,
        "chromadb": true,
        "qdrant": true,
        "faiss": true,
        "consul": true,
        "kong": true,
        "ollama": true
      },
      "healthy_count": 9,
      "total_services": 9
    }
    ```
    """
    try:
        service_health = await service_connections.health_check()
        all_healthy = all(service_health.values())
        
        # Update service status metrics
        for service, status in service_health.items():
            SERVICE_STATUS.labels(service=service).set(1 if status else 0)
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "app": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "services": service_health,
            "healthy_count": sum(service_health.values()),
            "total_services": len(service_health)
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/metrics", tags=["metrics"])
async def metrics():
    """
    Prometheus-compatible metrics endpoint
    
    Exposes comprehensive application and infrastructure metrics in Prometheus text format.
    Scrape this endpoint with Prometheus/Grafana for monitoring and alerting.
    
    **Metric Categories:**
    
    **HTTP Metrics:**
    - `http_requests_total`: Total HTTP requests by method, endpoint, status code
    - `http_request_duration_seconds`: Request latency histogram (14 buckets)
    - `http_request_size_bytes`: Request body size
    - `http_response_size_bytes`: Response body size
    - `http_requests_in_progress`: Currently processing requests
    
    **Authentication Metrics:**
    - `auth_login_total`: Login attempts by success/failure
    - `auth_token_generation_total`: Token generation by type
    - `auth_account_lockouts_total`: Account lockouts triggered
    - `auth_password_resets_total`: Password reset requests
    
    **Database Metrics:**
    - `db_queries_total`: Database queries by operation
    - `db_query_duration_seconds`: Query execution time
    - `db_connection_pool_size`: Connection pool size
    - `db_connection_pool_active`: Active connections
    - `db_query_errors_total`: Database errors
    
    **Cache Metrics:**
    - `cache_hits_total`: Cache hit count by cache type
    - `cache_misses_total`: Cache miss count
    - `cache_operations_total`: Cache operations (get, set, delete)
    - `cache_operation_duration_seconds`: Cache operation latency
    
    **Message Queue Metrics:**
    - `rabbitmq_messages_published_total`: Messages published by queue
    - `rabbitmq_messages_consumed_total`: Messages consumed
    - `rabbitmq_message_processing_duration_seconds`: Processing time
    
    **Vector Database Metrics:**
    - `vector_db_operations_total`: Vector DB operations by type and database
    - `vector_db_operation_duration_seconds`: Operation latency
    - `vector_db_collection_size`: Collection document count
    
    **External API Metrics:**
    - `external_api_calls_total`: External API calls by service and status
    - `external_api_call_duration_seconds`: API call latency
    
    **Service Health:**
    - `backend_service_status`: Service health by service name (1=healthy, 0=unhealthy)
    
    **Returns:**
    - `200 OK`: Metrics in Prometheus text format
    - Content-Type: `text/plain; version=0.0.4; charset=utf-8`
    
    **Errors:**
    - `500 Internal Server Error`: Failed to generate metrics
    """
    try:
        # Update connection metrics before exposing
        service_health = await service_connections.health_check()
        for service, status in service_health.items():
            SERVICE_STATUS.labels(service=service).set(1 if status else 0)
        
        # Generate metrics
        metrics_output = generate_latest(REGISTRY)
        return Response(content=metrics_output, media_type=CONTENT_TYPE_LATEST)
    except Exception as e:
        logger.error(f"Failed to generate metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def register_with_consul():
    """Register service with Consul"""
    try:
        import socket
        hostname = socket.gethostname()
        
        service_definition = {
            "ID": f"sutazai-backend-{hostname}",
            "Name": "sutazai-backend",
            "Tags": ["api", "backend", "fastapi"],
            "Address": hostname,
            "Port": settings.PORT,
            "Check": {
                "HTTP": f"http://{hostname}:{settings.PORT}/health",
                "Interval": "30s",
                "Timeout": "10s"
            }
        }
        
        response = await service_connections.consul_client.put(
            "/v1/agent/service/register",
            json=service_definition
        )
        
        if response.status_code == 200:
            logger.info("Service registered with Consul")
        else:
            logger.warning(f"Consul registration failed: {response.status_code}")
    except Exception as e:
        logger.warning(f"Could not register with Consul: {e}")


async def deregister_from_consul():
    """Deregister service from Consul"""
    try:
        import socket
        hostname = socket.gethostname()
        service_id = f"sutazai-backend-{hostname}"
        
        response = await service_connections.consul_client.put(
            f"/v1/agent/service/deregister/{service_id}"
        )
        
        if response.status_code == 200:
            logger.info("Service deregistered from Consul")
    except Exception as e:
        logger.warning(f"Could not deregister from Consul: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=settings.WORKERS if not settings.DEBUG else 1
    )
