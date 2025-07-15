"""
SutazAI AGI System API Layer
Enterprise-grade REST API for the Integrated AGI/ASI System

This module provides a comprehensive REST API for interacting with the SutazAI
AGI system, including task submission, system monitoring, and administrative functions.
"""

import asyncio
import json
import time
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Depends, Security, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import jwt
from passlib.context import CryptContext
import secrets
import hashlib

# Import AGI system components
from ..core.agi_system import (
    IntegratedAGISystem,
    AGITask,
    TaskPriority,
    AGISystemState,
    get_agi_system,
    create_agi_task
)
from ..core.exceptions import SutazaiException
from ..core.security import SecurityManager

# Initialize logging
logger = logging.getLogger(__name__)

# Security components
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# API Models
class TaskRequest(BaseModel):
    name: str = Field(..., description="Task name")
    priority: str = Field(default="medium", description="Task priority: low, medium, high, critical, emergency")
    data: Dict[str, Any] = Field(default_factory=dict, description="Task data")
    
    @validator('priority')
    def validate_priority(cls, v):
        valid_priorities = ["low", "medium", "high", "critical", "emergency"]
        if v.lower() not in valid_priorities:
            raise ValueError(f"Priority must be one of: {valid_priorities}")
        return v.lower()

class TaskResponse(BaseModel):
    task_id: str
    name: str
    priority: str
    status: str
    created_at: datetime
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class SystemStatusResponse(BaseModel):
    state: str
    metrics: Dict[str, Any]
    neural_network: Dict[str, Any]
    components: Dict[str, Any]
    timestamp: datetime

class CodeGenerationRequest(BaseModel):
    description: str = Field(..., description="Code description")
    language: str = Field(default="python", description="Programming language")
    requirements: List[str] = Field(default_factory=list, description="Additional requirements")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")

class KnowledgeQueryRequest(BaseModel):
    query: str = Field(..., description="Knowledge query")
    max_results: int = Field(default=10, description="Maximum results")
    include_metadata: bool = Field(default=True, description="Include metadata")

class NeuralProcessingRequest(BaseModel):
    input_data: List[float] = Field(..., description="Neural network input")
    processing_mode: str = Field(default="standard", description="Processing mode")
    return_internal_state: bool = Field(default=False, description="Return internal neural state")

class SecurityRequest(BaseModel):
    operation: str = Field(..., description="Security operation")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Operation parameters")

class AuthTokenResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int

class SystemHealthResponse(BaseModel):
    health_status: str
    uptime: float
    performance_metrics: Dict[str, Any]
    alerts: List[Dict[str, Any]]
    recommendations: List[str]

# AGI API Application
class AGIAPISystem:
    """Enterprise-grade API system for SutazAI AGI"""
    
    def __init__(self):
        self.app = FastAPI(
            title="SutazAI AGI System API",
            description="Enterprise-grade REST API for the Integrated AGI/ASI System",
            version="1.0.0",
            docs_url="/api/docs",
            redoc_url="/api/redoc"
        )
        
        # Initialize components
        self.agi_system = get_agi_system()
        self.security_manager = SecurityManager()
        self.authorized_user = "chrissuta01@gmail.com"
        
        # JWT configuration
        self.jwt_secret = secrets.token_urlsafe(32)
        self.jwt_algorithm = "HS256"
        self.jwt_expiration = timedelta(hours=24)
        
        # Task storage for tracking
        self.active_tasks: Dict[str, AGITask] = {}
        
        # Setup middleware and routes
        self._setup_middleware()
        self._setup_routes()
        
        logger.info("AGI API System initialized")
    
    def _setup_middleware(self):
        """Setup FastAPI middleware"""
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Custom security middleware
        @self.app.middleware("http")
        async def security_middleware(request, call_next):
            # Log request
            logger.info(f"API Request: {request.method} {request.url}")
            
            # Process request
            response = await call_next(request)
            
            # Log response
            logger.info(f"API Response: {response.status_code}")
            
            return response
    
    def _setup_routes(self):
        """Setup API routes"""
        
        # Health check endpoint
        @self.app.get("/health", response_model=SystemHealthResponse)
        async def health_check():
            """System health check endpoint"""
            try:
                status = self.agi_system.get_system_status()
                
                return SystemHealthResponse(
                    health_status=status["metrics"]["system_health"],
                    uptime=time.time(),
                    performance_metrics=status["metrics"],
                    alerts=[],
                    recommendations=[]
                )
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                raise HTTPException(status_code=500, detail="Health check failed")
        
        # Authentication endpoint
        @self.app.post("/auth/token", response_model=AuthTokenResponse)
        async def authenticate(email: str, password: str):
            """Authenticate user and return JWT token"""
            try:
                # Verify user credentials
                if email != self.authorized_user:
                    raise HTTPException(status_code=401, detail="Unauthorized")
                
                # Generate JWT token
                token_data = {
                    "sub": email,
                    "exp": datetime.utcnow() + self.jwt_expiration,
                    "iat": datetime.utcnow()
                }
                
                access_token = jwt.encode(token_data, self.jwt_secret, algorithm=self.jwt_algorithm)
                
                return AuthTokenResponse(
                    access_token=access_token,
                    token_type="bearer",
                    expires_in=int(self.jwt_expiration.total_seconds())
                )
                
            except Exception as e:
                logger.error(f"Authentication failed: {e}")
                raise HTTPException(status_code=401, detail="Authentication failed")
        
        # System status endpoint
        @self.app.get("/api/v1/system/status", response_model=SystemStatusResponse)
        async def get_system_status(current_user: str = Depends(self.get_current_user)):
            """Get current system status"""
            try:
                status = self.agi_system.get_system_status()
                
                return SystemStatusResponse(
                    state=status["state"],
                    metrics=status["metrics"],
                    neural_network=status["neural_network"],
                    components=status["components"],
                    timestamp=datetime.now()
                )
                
            except Exception as e:
                logger.error(f"Failed to get system status: {e}")
                raise HTTPException(status_code=500, detail="Failed to get system status")
        
        # Task submission endpoint
        @self.app.post("/api/v1/tasks", response_model=TaskResponse)
        async def submit_task(
            task_request: TaskRequest,
            background_tasks: BackgroundTasks,
            current_user: str = Depends(self.get_current_user)
        ):
            """Submit a task to the AGI system"""
            try:
                # Convert priority string to enum
                priority_map = {
                    "low": TaskPriority.LOW,
                    "medium": TaskPriority.MEDIUM,
                    "high": TaskPriority.HIGH,
                    "critical": TaskPriority.CRITICAL,
                    "emergency": TaskPriority.EMERGENCY
                }
                
                priority = priority_map[task_request.priority]
                
                # Create AGI task
                task = create_agi_task(
                    name=task_request.name,
                    priority=priority,
                    data=task_request.data
                )
                
                # Submit to AGI system
                task_id = self.agi_system.submit_task(task)
                
                # Store task for tracking
                self.active_tasks[task_id] = task
                
                # Schedule task cleanup
                background_tasks.add_task(self._cleanup_task, task_id, 3600)  # 1 hour
                
                return TaskResponse(
                    task_id=task.id,
                    name=task.name,
                    priority=task_request.priority,
                    status=task.status,
                    created_at=task.created_at,
                    result=task.result,
                    error=task.error
                )
                
            except Exception as e:
                logger.error(f"Task submission failed: {e}")
                raise HTTPException(status_code=500, detail="Task submission failed")
        
        # Task status endpoint
        @self.app.get("/api/v1/tasks/{task_id}", response_model=TaskResponse)
        async def get_task_status(
            task_id: str,
            current_user: str = Depends(self.get_current_user)
        ):
            """Get task status"""
            try:
                if task_id not in self.active_tasks:
                    raise HTTPException(status_code=404, detail="Task not found")
                
                task = self.active_tasks[task_id]
                
                return TaskResponse(
                    task_id=task.id,
                    name=task.name,
                    priority=task.priority.name.lower(),
                    status=task.status,
                    created_at=task.created_at,
                    result=task.result,
                    error=task.error
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to get task status: {e}")
                raise HTTPException(status_code=500, detail="Failed to get task status")
        
        # Code generation endpoint
        @self.app.post("/api/v1/code/generate")
        async def generate_code(
            request: CodeGenerationRequest,
            current_user: str = Depends(self.get_current_user)
        ):
            """Generate code using AGI system"""
            try:
                # Create code generation task
                task_data = {
                    "code_request": {
                        "description": request.description,
                        "language": request.language,
                        "requirements": request.requirements,
                        "context": request.context
                    }
                }
                
                task = create_agi_task(
                    name="code_generation",
                    priority=TaskPriority.HIGH,
                    data=task_data
                )
                
                # Submit and wait for completion
                task_id = self.agi_system.submit_task(task)
                
                # Wait for task completion (with timeout)
                result = await self._wait_for_task_completion(task, timeout=60)
                
                return {
                    "task_id": task_id,
                    "generated_code": result.get("generated_code", ""),
                    "quality_score": result.get("quality_score", 0.0),
                    "timestamp": result.get("timestamp", datetime.now().isoformat())
                }
                
            except Exception as e:
                logger.error(f"Code generation failed: {e}")
                raise HTTPException(status_code=500, detail="Code generation failed")
        
        # Knowledge query endpoint
        @self.app.post("/api/v1/knowledge/query")
        async def query_knowledge(
            request: KnowledgeQueryRequest,
            current_user: str = Depends(self.get_current_user)
        ):
            """Query knowledge graph"""
            try:
                # Create knowledge query task
                task_data = {
                    "query": request.query,
                    "max_results": request.max_results,
                    "include_metadata": request.include_metadata
                }
                
                task = create_agi_task(
                    name="knowledge_query",
                    priority=TaskPriority.MEDIUM,
                    data=task_data
                )
                
                # Submit and wait for completion
                task_id = self.agi_system.submit_task(task)
                result = await self._wait_for_task_completion(task, timeout=30)
                
                return {
                    "task_id": task_id,
                    "query": result.get("query", ""),
                    "results": result.get("results", []),
                    "timestamp": result.get("timestamp", datetime.now().isoformat())
                }
                
            except Exception as e:
                logger.error(f"Knowledge query failed: {e}")
                raise HTTPException(status_code=500, detail="Knowledge query failed")
        
        # Neural processing endpoint
        @self.app.post("/api/v1/neural/process")
        async def process_neural_input(
            request: NeuralProcessingRequest,
            current_user: str = Depends(self.get_current_user)
        ):
            """Process input through neural network"""
            try:
                # Create neural processing task
                task_data = {
                    "input": request.input_data,
                    "processing_mode": request.processing_mode,
                    "return_internal_state": request.return_internal_state
                }
                
                task = create_agi_task(
                    name="neural_processing",
                    priority=TaskPriority.HIGH,
                    data=task_data
                )
                
                # Submit and wait for completion
                task_id = self.agi_system.submit_task(task)
                result = await self._wait_for_task_completion(task, timeout=10)
                
                return {
                    "task_id": task_id,
                    "input": result.get("input", []),
                    "output": result.get("output", []),
                    "neural_activity": result.get("neural_activity", 0.0),
                    "timestamp": result.get("timestamp", datetime.now().isoformat())
                }
                
            except Exception as e:
                logger.error(f"Neural processing failed: {e}")
                raise HTTPException(status_code=500, detail="Neural processing failed")
        
        # Security endpoint
        @self.app.post("/api/v1/security/process")
        async def process_security_request(
            request: SecurityRequest,
            current_user: str = Depends(self.get_current_user)
        ):
            """Process security-related request"""
            try:
                # Create security task
                task_data = {
                    "security_request": {
                        "operation": request.operation,
                        "parameters": request.parameters
                    }
                }
                
                task = create_agi_task(
                    name="security_check",
                    priority=TaskPriority.CRITICAL,
                    data=task_data
                )
                
                # Submit and wait for completion
                task_id = self.agi_system.submit_task(task)
                result = await self._wait_for_task_completion(task, timeout=30)
                
                return {
                    "task_id": task_id,
                    "security_result": result.get("security_result", {}),
                    "timestamp": result.get("timestamp", datetime.now().isoformat())
                }
                
            except Exception as e:
                logger.error(f"Security processing failed: {e}")
                raise HTTPException(status_code=500, detail="Security processing failed")
        
        # Emergency shutdown endpoint
        @self.app.post("/api/v1/system/emergency-shutdown")
        async def emergency_shutdown(current_user: str = Depends(self.get_current_user)):
            """Emergency shutdown - authorized user only"""
            try:
                # Verify user authorization
                if current_user != self.authorized_user:
                    raise HTTPException(status_code=403, detail="Unauthorized for emergency shutdown")
                
                # Perform emergency shutdown
                success = self.agi_system.emergency_shutdown(current_user)
                
                if success:
                    return {"status": "success", "message": "Emergency shutdown initiated"}
                else:
                    raise HTTPException(status_code=500, detail="Emergency shutdown failed")
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Emergency shutdown failed: {e}")
                raise HTTPException(status_code=500, detail="Emergency shutdown failed")
        
        # System metrics endpoint
        @self.app.get("/api/v1/metrics")
        async def get_metrics(current_user: str = Depends(self.get_current_user)):
            """Get detailed system metrics"""
            try:
                status = self.agi_system.get_system_status()
                
                return {
                    "system_metrics": status["metrics"],
                    "neural_metrics": status["neural_network"],
                    "component_status": status["components"],
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Failed to get metrics: {e}")
                raise HTTPException(status_code=500, detail="Failed to get metrics")
    
    async def get_current_user(self, credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
        """Get current authenticated user"""
        try:
            # Decode JWT token
            payload = jwt.decode(
                credentials.credentials,
                self.jwt_secret,
                algorithms=[self.jwt_algorithm]
            )
            
            username = payload.get("sub")
            if username is None:
                raise HTTPException(status_code=401, detail="Invalid token")
            
            return username
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    async def _wait_for_task_completion(self, task: AGITask, timeout: int = 30) -> Dict[str, Any]:
        """Wait for task completion with timeout"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if task.status == "completed":
                return task.result or {}
            elif task.status == "failed":
                raise SutazaiException(f"Task failed: {task.error}")
            
            await asyncio.sleep(0.1)
        
        raise SutazaiException(f"Task timeout after {timeout} seconds")
    
    async def _cleanup_task(self, task_id: str, delay: int):
        """Clean up task after delay"""
        await asyncio.sleep(delay)
        if task_id in self.active_tasks:
            del self.active_tasks[task_id]
            logger.info(f"Cleaned up task: {task_id}")

# Global API instance
_api_instance = None

def get_api_app() -> FastAPI:
    """Get the FastAPI application instance"""
    global _api_instance
    if _api_instance is None:
        _api_instance = AGIAPISystem()
    return _api_instance.app

def create_api_server() -> AGIAPISystem:
    """Create a new API server instance"""
    return AGIAPISystem()

if __name__ == "__main__":
    import uvicorn
    
    # Create API server
    api_system = create_api_server()
    
    # Run server
    uvicorn.run(
        api_system.app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )