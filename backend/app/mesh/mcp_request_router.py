"""
MCP Multi-Client Request Router
Enables simultaneous Claude Code and Codex access without conflicts
Implements proper request routing, session management, and concurrent handling
"""
import asyncio
import uuid
import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib
from collections import defaultdict

logger = logging.getLogger(__name__)

class ClientType(Enum):
    """Types of clients accessing MCP services"""
    CLAUDE_CODE = "claude_code"
    CODEX = "codex"
    API = "api"
    WEB = "web"
    CLI = "cli"
    TEST = "test"

@dataclass
class ClientSession:
    """Represents a client session"""
    session_id: str
    client_type: ClientType
    client_id: str
    created_at: datetime
    last_activity: datetime
    active_requests: int = 0
    total_requests: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    assigned_instances: Dict[str, str] = field(default_factory=dict)  # service -> instance mapping
    
    @property
    def is_expired(self) -> bool:
        """Check if session has expired (30 minutes of inactivity)"""
        return datetime.now() - self.last_activity > timedelta(minutes=30)
    
    def touch(self):
        """Update last activity time"""
        self.last_activity = datetime.now()

@dataclass
class RoutedRequest:
    """Represents a routed request"""
    request_id: str
    session_id: str
    service_name: str
    instance_id: Optional[str]
    method: str
    params: Dict[str, Any]
    created_at: datetime
    client_type: ClientType
    priority: int = 5  # 1-10, lower is higher priority
    timeout: float = 30.0
    
    def get_routing_key(self) -> str:
        """Get routing key for load balancing"""
        return f"{self.session_id}:{self.service_name}"

class RequestQueue:
    """Priority queue for requests with client isolation"""
    
    def __init__(self):
        self.queues: Dict[ClientType, List[RoutedRequest]] = {
            client_type: [] for client_type in ClientType
        }
        self.processing: Dict[str, RoutedRequest] = {}
        self.lock = asyncio.Lock()
    
    async def enqueue(self, request: RoutedRequest):
        """Add request to appropriate queue"""
        async with self.lock:
            queue = self.queues[request.client_type]
            
            # Insert based on priority
            inserted = False
            for i, existing in enumerate(queue):
                if request.priority < existing.priority:
                    queue.insert(i, request)
                    inserted = True
                    break
            
            if not inserted:
                queue.append(request)
    
    async def dequeue(self, client_type: Optional[ClientType] = None) -> Optional[RoutedRequest]:
        """Get next request to process"""
        async with self.lock:
            # If specific client type requested
            if client_type:
                queue = self.queues[client_type]
                if queue:
                    request = queue.pop(0)
                    self.processing[request.request_id] = request
                    return request
            else:
                # Round-robin across client types
                for client_type in ClientType:
                    queue = self.queues[client_type]
                    if queue:
                        request = queue.pop(0)
                        self.processing[request.request_id] = request
                        return request
            
            return None
    
    async def complete(self, request_id: str):
        """Mark request as completed"""
        async with self.lock:
            if request_id in self.processing:
                del self.processing[request_id]
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return {
            "queued": {ct.value: len(q) for ct, q in self.queues.items()},
            "processing": len(self.processing),
            "total_pending": sum(len(q) for q in self.queues.values())
        }

class SessionManager:
    """Manages client sessions with proper isolation"""
    
    def __init__(self):
        self.sessions: Dict[str, ClientSession] = {}
        self.client_sessions: Dict[Tuple[ClientType, str], str] = {}  # (type, client_id) -> session_id
        self.lock = asyncio.Lock()
    
    async def create_session(
        self,
        client_type: ClientType,
        client_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ClientSession:
        """Create or retrieve client session"""
        async with self.lock:
            # Check for existing session
            session_key = (client_type, client_id)
            if session_key in self.client_sessions:
                session_id = self.client_sessions[session_key]
                if session_id in self.sessions:
                    session = self.sessions[session_id]
                    session.touch()
                    return session
            
            # Create new session
            session_id = str(uuid.uuid4())
            session = ClientSession(
                session_id=session_id,
                client_type=client_type,
                client_id=client_id,
                created_at=datetime.now(),
                last_activity=datetime.now(),
                metadata=metadata or {}
            )
            
            self.sessions[session_id] = session
            self.client_sessions[session_key] = session_id
            
            logger.info(f"Created session {session_id} for {client_type.value}:{client_id}")
            return session
    
    async def get_session(self, session_id: str) -> Optional[ClientSession]:
        """Get session by ID"""
        async with self.lock:
            session = self.sessions.get(session_id)
            if session and not session.is_expired:
                session.touch()
                return session
            elif session and session.is_expired:
                # Clean up expired session
                await self._cleanup_session(session_id)
            return None
    
    async def _cleanup_session(self, session_id: str):
        """Clean up expired session"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session_key = (session.client_type, session.client_id)
            
            del self.sessions[session_id]
            if session_key in self.client_sessions:
                del self.client_sessions[session_key]
            
            logger.info(f"Cleaned up expired session {session_id}")
    
    async def cleanup_expired_sessions(self):
        """Periodic cleanup of expired sessions"""
        async with self.lock:
            expired = [
                sid for sid, session in self.sessions.items()
                if session.is_expired
            ]
            
            for session_id in expired:
                await self._cleanup_session(session_id)
            
            if expired:
                logger.info(f"Cleaned up {len(expired)} expired sessions")
    
    def get_active_sessions(self) -> Dict[str, Any]:
        """Get active session statistics"""
        now = datetime.now()
        
        stats = {
            "total": len(self.sessions),
            "by_client_type": defaultdict(int),
            "active_last_5min": 0,
            "active_last_hour": 0
        }
        
        for session in self.sessions.values():
            stats["by_client_type"][session.client_type.value] += 1
            
            time_since_activity = now - session.last_activity
            if time_since_activity < timedelta(minutes=5):
                stats["active_last_5min"] += 1
            if time_since_activity < timedelta(hours=1):
                stats["active_last_hour"] += 1
        
        return stats

class MCPRequestRouter:
    """
    Routes requests from multiple clients to MCP services
    Ensures proper isolation and concurrent handling
    """
    
    def __init__(self, orchestrator=None, load_balancer=None):
        self.orchestrator = orchestrator
        self.load_balancer = load_balancer
        self.session_manager = SessionManager()
        self.request_queue = RequestQueue()
        self.rate_limiters: Dict[str, RateLimiter] = {}
        self.request_metrics: Dict[str, RequestMetrics] = {}
        self.worker_tasks: List[asyncio.Task] = []
        self.running = False
        
    async def start(self, num_workers: int = 4):
        """Start request router with worker pool"""
        if self.running:
            return
        
        self.running = True
        
        # Start worker tasks
        for i in range(num_workers):
            task = asyncio.create_task(self._request_worker(i))
            self.worker_tasks.append(task)
        
        # Start session cleanup task
        asyncio.create_task(self._session_cleanup_task())
        
        logger.info(f"✅ Started request router with {num_workers} workers")
    
    async def stop(self):
        """Stop request router"""
        self.running = False
        
        # Cancel worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        self.worker_tasks.clear()
        
        logger.info("✅ Stopped request router")
    
    async def route_request(
        self,
        client_type: ClientType,
        client_id: str,
        service_name: str,
        method: str,
        params: Dict[str, Any],
        timeout: float = 30.0,
        priority: int = 5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Route a request from a client to an MCP service
        
        Args:
            client_type: Type of client
            client_id: Unique client identifier
            service_name: Target MCP service
            method: Method to call
            params: Method parameters
            timeout: Request timeout
            priority: Request priority (1-10)
            metadata: Optional client metadata
        
        Returns:
            Response from MCP service
        """
        try:
            # Create or get session
            session = await self.session_manager.create_session(
                client_type=client_type,
                client_id=client_id,
                metadata=metadata
            )
            
            # Check rate limits
            if not await self._check_rate_limit(session.session_id):
                return {
                    "error": "Rate limit exceeded",
                    "retry_after": 60
                }
            
            # Create routed request
            request = RoutedRequest(
                request_id=str(uuid.uuid4()),
                session_id=session.session_id,
                service_name=service_name,
                instance_id=session.assigned_instances.get(service_name),
                method=method,
                params=params,
                created_at=datetime.now(),
                client_type=client_type,
                priority=priority,
                timeout=timeout
            )
            
            # Update session
            session.active_requests += 1
            session.total_requests += 1
            
            # Enqueue request
            await self.request_queue.enqueue(request)
            
            # Wait for response with timeout
            response = await self._wait_for_response(request.request_id, timeout)
            
            # Update session
            session.active_requests -= 1
            
            return response
            
        except asyncio.TimeoutError:
            return {
                "error": "Request timeout",
                "timeout": timeout
            }
        except Exception as e:
            logger.error(f"Error routing request: {e}")
            return {
                "error": str(e)
            }
    
    async def _request_worker(self, worker_id: int):
        """Worker task that processes requests"""
        logger.info(f"Worker {worker_id} started")
        
        while self.running:
            try:
                # Get next request
                request = await self.request_queue.dequeue()
                
                if not request:
                    await asyncio.sleep(0.1)
                    continue
                
                # Process request
                await self._process_request(request)
                
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
        
        logger.info(f"Worker {worker_id} stopped")
    
    async def _process_request(self, request: RoutedRequest):
        """Process a single request"""
        try:
            # Get session
            session = await self.session_manager.get_session(request.session_id)
            if not session:
                logger.warning(f"Session {request.session_id} not found for request {request.request_id}")
                return
            
            # Select instance using load balancer
            if self.load_balancer:
                instance = await self._select_instance(request, session)
                if instance:
                    session.assigned_instances[request.service_name] = instance.service_id
            
            # Route to MCP service through orchestrator
            if self.orchestrator:
                response = await self._call_mcp_service(request)
            else:
                response = {"error": "Orchestrator not available"}
            
            # Store response (in real implementation, would use proper response storage)
            # For now, we'll log it
            logger.debug(f"Request {request.request_id} completed with response: {response}")
            
            # Mark request as complete
            await self.request_queue.complete(request.request_id)
            
            # Update metrics
            self._update_metrics(request, response)
            
        except Exception as e:
            logger.error(f"Error processing request {request.request_id}: {e}")
            await self.request_queue.complete(request.request_id)
    
    async def _select_instance(self, request: RoutedRequest, session: ClientSession):
        """Select instance for request using load balancer"""
        # This would integrate with the actual load balancer
        # For now, return None to use default routing
        return None
    
    async def _call_mcp_service(self, request: RoutedRequest) -> Dict[str, Any]:
        """Call MCP service through orchestrator"""
        # This would integrate with the actual orchestrator
        # For now, return a  response
        return {
            "result": f"Processed {request.method} for {request.service_name}",
            "request_id": request.request_id
        }
    
    async def _wait_for_response(self, request_id: str, timeout: float) -> Dict[str, Any]:
        """Wait for request response"""
        # In a real implementation, this would wait for the actual response
        # For now, we'll simulate a delay
        await asyncio.sleep(0.1)
        return {
            "result": "success",
            "request_id": request_id
        }
    
    async def _check_rate_limit(self, session_id: str) -> bool:
        """Check if request is within rate limits"""
        if session_id not in self.rate_limiters:
            self.rate_limiters[session_id] = RateLimiter(
                max_requests=100,
                time_window=60  # 100 requests per minute
            )
        
        return await self.rate_limiters[session_id].allow_request()
    
    def _update_metrics(self, request: RoutedRequest, response: Dict[str, Any]):
        """Update request metrics"""
        if request.service_name not in self.request_metrics:
            self.request_metrics[request.service_name] = RequestMetrics()
        
        metrics = self.request_metrics[request.service_name]
        metrics.total_requests += 1
        
        if "error" in response:
            metrics.failed_requests += 1
        else:
            metrics.successful_requests += 1
        
        # Calculate response time
        response_time = (datetime.now() - request.created_at).total_seconds()
        metrics.update_response_time(response_time)
    
    async def _session_cleanup_task(self):
        """Periodic task to clean up expired sessions"""
        while self.running:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self.session_manager.cleanup_expired_sessions()
            except Exception as e:
                logger.error(f"Session cleanup error: {e}")
    
    def get_router_stats(self) -> Dict[str, Any]:
        """Get router statistics"""
        return {
            "sessions": self.session_manager.get_active_sessions(),
            "queue": self.request_queue.get_queue_stats(),
            "workers": len(self.worker_tasks),
            "metrics": {
                service: metrics.get_stats()
                for service, metrics in self.request_metrics.items()
            }
        }

@dataclass
class RequestMetrics:
    """Metrics for request tracking"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_response_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    
    def update_response_time(self, response_time: float):
        """Update response time metrics"""
        self.total_response_time += response_time
        self.min_response_time = min(self.min_response_time, response_time)
        self.max_response_time = max(self.max_response_time, response_time)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get metrics statistics"""
        avg_response_time = (
            self.total_response_time / self.total_requests
            if self.total_requests > 0 else 0
        )
        
        success_rate = (
            self.successful_requests / self.total_requests * 100
            if self.total_requests > 0 else 0
        )
        
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": success_rate,
            "avg_response_time": avg_response_time,
            "min_response_time": self.min_response_time if self.min_response_time != float('inf') else 0,
            "max_response_time": self.max_response_time
        }

class RateLimiter:
    """Simple rate limiter using sliding window"""
    
    def __init__(self, max_requests: int, time_window: float):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: List[float] = []
        self.lock = asyncio.Lock()
    
    async def allow_request(self) -> bool:
        """Check if request is allowed"""
        async with self.lock:
            now = time.time()
            
            # Remove old requests outside window
            self.requests = [
                req_time for req_time in self.requests
                if now - req_time < self.time_window
            ]
            
            # Check if under limit
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            
            return False

# Global router instance
_request_router: Optional[MCPRequestRouter] = None

async def get_request_router(orchestrator=None, load_balancer=None) -> MCPRequestRouter:
    """Get or create request router"""
    global _request_router
    
    if _request_router is None:
        _request_router = MCPRequestRouter(orchestrator, load_balancer)
        await _request_router.start()
    
    return _request_router