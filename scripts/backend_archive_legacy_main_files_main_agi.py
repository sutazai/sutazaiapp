"""
SutazAI AGI/ASI System - Main Backend
Enterprise-grade AGI implementation with complete local operation
"""

import os
import sys
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import psutil
import torch
import subprocess

# Add the app directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import AGI components
from agi_brain import AGIBrain
from agent_orchestrator import AgentOrchestrator
from knowledge_manager import KnowledgeManager
from self_improvement import SelfImprovementSystem
from reasoning_engine import ReasoningEngine, ReasoningType

# Import core utilities
from core.config import settings
from core.database import get_db, init_db
from utils.logger import setup_logger

# Setup logging
logger = setup_logger("sutazai_agi", "logs/agi_system.log")

# Global instances
agi_brain: Optional[AGIBrain] = None
agent_orchestrator: Optional[AgentOrchestrator] = None
knowledge_manager: Optional[KnowledgeManager] = None
self_improvement: Optional[SelfImprovementSystem] = None
reasoning_engine: Optional[ReasoningEngine] = None

# Request/Response Models
class ThinkRequest(BaseModel):
    query: str = Field(..., description="Query to process through AGI brain")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")
    trace_enabled: bool = Field(default=True, description="Enable cognitive trace")

class ThinkResponse(BaseModel):
    response: str
    cognitive_trace: Optional[List[Dict[str, Any]]] = None
    confidence: float
    timestamp: datetime
    processing_time_ms: float

class ReasonRequest(BaseModel):
    problem: str = Field(..., description="Problem to solve")
    reasoning_type: Optional[ReasoningType] = Field(default=None, description="Specific reasoning type")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")

class ReasonResponse(BaseModel):
    solution: str
    reasoning_type: str
    certainty: float
    steps: List[Dict[str, Any]]
    timestamp: datetime

class LearnRequest(BaseModel):
    content: str = Field(..., description="Knowledge to learn")
    type: str = Field(default="general", description="Knowledge type")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")

class ExecuteRequest(BaseModel):
    description: str = Field(..., description="Task description")
    type: str = Field(..., description="Task type")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Task parameters")
    multi_agent: bool = Field(default=False, description="Use multiple agents")

class ImproveRequest(BaseModel):
    target: str = Field(default="system", description="Improvement target")
    focus: Optional[str] = Field(default=None, description="Specific area to focus on")
    max_improvements: int = Field(default=5, description="Maximum improvements to generate")

class HealthResponse(BaseModel):
    status: str
    service: str = "sutazai-agi"
    version: str = "1.0.0"
    gpu_available: bool
    agents_healthy: int
    agents_total: int
    memory_usage_mb: float
    cpu_percent: float
    uptime_seconds: float

# GPU detection
def check_gpu_availability() -> bool:
    """Check if GPU is available for computation"""
    try:
        if torch.cuda.is_available():
            return True
    except:
        pass
    
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global agi_brain, agent_orchestrator, knowledge_manager, self_improvement, reasoning_engine
    
    logger.info("Starting SutazAI AGI/ASI System...")
    start_time = datetime.now()
    
    try:
        # Initialize database
        await init_db()
        logger.info("Database initialized")
        
        # Initialize core components
        agi_brain = AGIBrain()
        await agi_brain.initialize()
        logger.info("AGI Brain initialized")
        
        agent_orchestrator = AgentOrchestrator()
        await agent_orchestrator.initialize()
        logger.info("Agent Orchestrator initialized with {} agents".format(
            len(agent_orchestrator.agents)
        ))
        
        knowledge_manager = KnowledgeManager()
        await knowledge_manager.initialize()
        logger.info("Knowledge Manager initialized")
        
        self_improvement = SelfImprovementSystem()
        await self_improvement.initialize()
        logger.info("Self-Improvement System initialized")
        
        reasoning_engine = ReasoningEngine()
        logger.info("Reasoning Engine initialized")
        
        # Start background tasks
        asyncio.create_task(agent_orchestrator._health_monitor())
        asyncio.create_task(agent_orchestrator._task_processor())
        asyncio.create_task(self_improvement._improvement_cycle())
        
        logger.info(f"System startup completed in {(datetime.now() - start_time).total_seconds():.2f}s")
        
        yield
        
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise
    finally:
        # Cleanup
        logger.info("Shutting down SutazAI AGI/ASI System...")
        # Add cleanup code here

# Create FastAPI app
app = FastAPI(
    title="SutazAI AGI/ASI System",
    description="Enterprise-grade Autonomous General Intelligence System",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "type": type(exc).__name__,
            "message": str(exc)
        }
    )

# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check endpoint"""
    process = psutil.Process()
    
    # Count healthy agents
    agents_healthy = 0
    for agent in agent_orchestrator.agents.values():
        if await agent_orchestrator._check_agent_health(agent):
            agents_healthy += 1
    
    return HealthResponse(
        status="healthy",
        gpu_available=check_gpu_availability(),
        agents_healthy=agents_healthy,
        agents_total=len(agent_orchestrator.agents),
        memory_usage_mb=process.memory_info().rss / 1024 / 1024,
        cpu_percent=process.cpu_percent(),
        uptime_seconds=(datetime.now() - process.create_time()).total_seconds()
    )

@app.post("/think", response_model=ThinkResponse)
async def think(request: ThinkRequest):
    """Process query through AGI brain"""
    start_time = datetime.now()
    
    try:
        response = await agi_brain.process_query(
            request.query,
            context=request.context
        )
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return ThinkResponse(
            response=response["response"],
            cognitive_trace=response.get("cognitive_trace") if request.trace_enabled else None,
            confidence=response.get("confidence", 0.8),
            timestamp=datetime.now(),
            processing_time_ms=processing_time
        )
    except Exception as e:
        logger.error(f"Think endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reason", response_model=ReasonResponse)
async def reason(request: ReasonRequest):
    """Apply reasoning to solve problems"""
    try:
        result = await reasoning_engine.solve(
            request.problem,
            reasoning_type=request.reasoning_type,
            context=request.context
        )
        
        return ReasonResponse(
            solution=result["solution"],
            reasoning_type=result["reasoning_type"],
            certainty=result["certainty"],
            steps=result["steps"],
            timestamp=datetime.now()
        )
    except Exception as e:
        logger.error(f"Reason endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/learn")
async def learn(request: LearnRequest):
    """Add new knowledge to the system"""
    try:
        result = await knowledge_manager.add_knowledge(
            request.content,
            knowledge_type=request.type,
            metadata=request.metadata
        )
        
        return {
            "status": "success",
            "knowledge_id": result["id"],
            "relationships_created": result["relationships"],
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Learn endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/execute")
async def execute_task(request: ExecuteRequest, background_tasks: BackgroundTasks):
    """Execute task using AI agents"""
    try:
        # Execute task
        result = await agent_orchestrator.execute_task(
            request.description,
            task_type=request.type,
            parameters=request.parameters or {},
            multi_agent=request.multi_agent
        )
        
        # Learn from execution
        if result.get("status") == "completed":
            background_tasks.add_task(
                knowledge_manager.add_knowledge,
                f"Task execution: {request.description}",
                knowledge_type="execution",
                metadata={"result": result}
            )
        
        return result
    except Exception as e:
        logger.error(f"Execute endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents")
async def list_agents():
    """List all available AI agents"""
    return await agent_orchestrator.list_agents()

@app.post("/improve")
async def improve_system(request: ImproveRequest):
    """Trigger self-improvement process"""
    try:
        if not settings.ENABLE_SELF_IMPROVEMENT:
            raise HTTPException(
                status_code=403, 
                detail="Self-improvement is disabled in production"
            )
        
        result = await self_improvement.improve_system(
            target=request.target,
            focus=request.focus,
            max_improvements=request.max_improvements
        )
        
        return {
            "status": "success",
            "improvements": result["improvements"],
            "files_modified": result["files_modified"],
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Improve endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/knowledge/search")
async def search_knowledge(query: str, limit: int = 10):
    """Search knowledge base"""
    try:
        results = await knowledge_manager.query_knowledge(
            query,
            search_type="semantic",
            limit=limit
        )
        return {
            "query": query,
            "results": results,
            "count": len(results),
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Knowledge search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/cognitive-stream")
async def cognitive_stream(websocket: WebSocket):
    """WebSocket for real-time cognitive trace streaming"""
    await websocket.accept()
    
    try:
        while True:
            # Receive query
            data = await websocket.receive_json()
            query = data.get("query")
            
            if not query:
                await websocket.send_json({"error": "No query provided"})
                continue
            
            # Process with streaming
            async for trace_item in agi_brain.process_query_stream(query):
                await websocket.send_json({
                    "type": "cognitive_trace",
                    "data": trace_item,
                    "timestamp": datetime.now().isoformat()
                })
            
            # Send final response
            await websocket.send_json({
                "type": "complete",
                "timestamp": datetime.now().isoformat()
            })
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.close()

@app.get("/metrics")
async def get_metrics():
    """Prometheus-compatible metrics endpoint"""
    metrics = []
    
    # System metrics
    process = psutil.Process()
    metrics.append(f'sutazai_memory_usage_bytes {process.memory_info().rss}')
    metrics.append(f'sutazai_cpu_percent {process.cpu_percent()}')
    metrics.append(f'sutazai_gpu_available {1 if check_gpu_availability() else 0}')
    
    # Agent metrics
    agents_healthy = 0
    for agent in agent_orchestrator.agents.values():
        if await agent_orchestrator._check_agent_health(agent):
            agents_healthy += 1
    
    metrics.append(f'sutazai_agents_total {len(agent_orchestrator.agents)}')
    metrics.append(f'sutazai_agents_healthy {agents_healthy}')
    
    # Knowledge metrics
    knowledge_count = await knowledge_manager.get_knowledge_count()
    metrics.append(f'sutazai_knowledge_items_total {knowledge_count}')
    
    return "\n".join(metrics)

@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "name": "SutazAI AGI/ASI System",
        "version": "1.0.0",
        "status": "operational",
        "description": "Enterprise-grade Autonomous General Intelligence",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "think": "/think",
            "reason": "/reason",
            "learn": "/learn",
            "execute": "/execute",
            "agents": "/agents",
            "improve": "/improve",
            "metrics": "/metrics"
        },
        "gpu_enabled": check_gpu_availability(),
        "timestamp": datetime.now()
    }

if __name__ == "__main__":
    # Run with optimal settings
    uvicorn.run(
        "main_agi:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        workers=1 if settings.DEBUG else 4,
        log_level="debug" if settings.DEBUG else "info"
    ) 