#!/usr/bin/env python3
"""
SutazAI Brain - Main Entry Point
100% Local AGI/ASI System
"""

import asyncio
import logging
import os
import sys
from typing import Dict, Any
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import yaml

# Add brain modules to path
sys.path.insert(0, str(Path(__file__).parent))

from core.orchestrator import BrainOrchestrator
from core.brain_state import BrainConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BrainRequest(BaseModel):
    """Request model for Brain API"""
    input: str
    context: Dict[str, Any] = {}
    expected_output: Dict[str, Any] = None
    stream: bool = False


class BrainResponse(BaseModel):
    """Response model for Brain API"""
    request_id: str
    output: Any
    confidence: float
    execution_time: float
    agents_used: list
    improvements_suggested: int


class BrainSystem:
    """Main Brain system controller"""
    
    def __init__(self, config_path: str = "config/brain_config.yaml"):
        """Initialize the Brain system"""
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize orchestrator
        self.orchestrator = BrainOrchestrator(self.config)
        
        # System state
        self.is_running = True
        self.active_requests = {}
        
        logger.info("🧠 Brain system initialized")
        
    def _load_config(self, config_path: str) -> BrainConfig:
        """Load Brain configuration"""
        default_config = {
            # Hardware
            'max_memory_gb': 48.0,
            'gpu_memory_gb': 4.0,
            'cpu_cores': os.cpu_count() or 8,
            
            # Models
            'default_embedding_model': 'nomic-embed-text',
            'default_reasoning_model': 'deepseek-r1:8b',
            'default_coding_model': 'codellama:7b',
            'evaluation_model': 'deepseek-r1:8b',
            'comparison_model': 'qwen2.5:7b',
            'code_model': 'codellama:7b',
            'analysis_model': 'deepseek-r1:8b',
            
            # Thresholds
            'min_quality_score': 0.85,
            'improvement_threshold': 0.85,
            'memory_retention_days': 30,
            
            # Parallelism
            'max_concurrent_agents': 5,
            'max_model_instances': 3,
            
            # Self-improvement
            'auto_improve': True,
            'pr_batch_size': 50,
            'require_human_approval': True,
            
            # Service hosts
            'ollama_host': 'http://sutazai-ollama:11434',
            'redis_host': 'sutazai-redis',
            'redis_port': 6379,
            'qdrant_host': 'sutazai-qdrant',
            'qdrant_port': 6333,
            'chroma_host': 'sutazai-chromadb',
            'chroma_port': 8000,
            'postgres_host': 'sutazai-postgresql',
            'postgres_port': 5432,
            'postgres_db': 'sutazai_brain',
            'postgres_user': 'sutazai',
            'postgres_password': os.getenv('POSTGRES_PASSWORD', 'sutazai_password'),
            
            # Brain repo
            'brain_repo_path': '/workspace/brain'
        }
        
        # Load user config if exists
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
        
        return default_config
    
    async def process_request(self, request: BrainRequest) -> BrainResponse:
        """Process a single request through the Brain"""
        try:
            # Process through orchestrator
            result = await self.orchestrator.process(request.input)
            
            # Create response
            response = BrainResponse(
                request_id=result['request_id'],
                output=result['output'],
                confidence=result['confidence'],
                execution_time=result['execution_time'],
                agents_used=result['agents_used'],
                improvements_suggested=result['improvements_suggested']
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def stream_process(self, request: BrainRequest):
        """Stream processing results as they become available"""
        # This would implement streaming for real-time updates
        # For now, return a simple generator
        result = await self.process_request(request)
        yield f"data: {result.json()}\n\n"
    
    def get_status(self) -> Dict[str, Any]:
        """Get Brain system status"""
        return {
            'status': 'running' if self.is_running else 'stopped',
            'active_requests': len(self.active_requests),
            'config': {
                'max_memory_gb': self.config['max_memory_gb'],
                'gpu_memory_gb': self.config['gpu_memory_gb'],
                'max_concurrent_agents': self.config['max_concurrent_agents'],
                'auto_improve': self.config['auto_improve']
            }
        }
    
    async def shutdown(self):
        """Gracefully shutdown the Brain"""
        logger.info("🛑 Shutting down Brain system...")
        self.is_running = False
        # Cleanup would happen here
        logger.info("✅ Brain system shut down complete")


# Initialize FastAPI app
app = FastAPI(
    title="SutazAI Brain API",
    description="100% Local AGI/ASI System",
    version="1.0.0"
)

# Initialize Brain system
brain = None


@app.on_event("startup")
async def startup_event():
    """Initialize Brain on startup"""
    global brain
    brain = BrainSystem()
    logger.info("🚀 Brain API started")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if brain:
        await brain.shutdown()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "SutazAI Brain",
        "version": "1.0.0",
        "status": brain.get_status() if brain else "not initialized"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not brain or not brain.is_running:
        raise HTTPException(status_code=503, detail="Brain not running")
    
    return {
        "status": "healthy",
        "brain_status": brain.get_status()
    }


@app.post("/process", response_model=BrainResponse)
async def process_request(request: BrainRequest):
    """Process a request through the Brain"""
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    
    return await brain.process_request(request)


@app.post("/stream")
async def stream_process(request: BrainRequest):
    """Stream processing results"""
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    
    if not request.stream:
        # Regular processing
        result = await brain.process_request(request)
        return result
    
    # Streaming response
    return StreamingResponse(
        brain.stream_process(request),
        media_type="text/event-stream"
    )


@app.get("/status")
async def get_status():
    """Get detailed Brain status"""
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    
    return brain.get_status()


@app.get("/agents")
async def list_agents():
    """List available agents"""
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    
    # Get agent list from router
    return {
        "agents": list(brain.orchestrator.router.agent_registry.keys()),
        "total": len(brain.orchestrator.router.agent_registry)
    }


@app.get("/memory/stats")
async def memory_stats():
    """Get memory system statistics"""
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    
    stats = await brain.orchestrator.memory.get_memory_stats()
    return stats


@app.get("/performance/agents")
async def agent_performance():
    """Get agent performance statistics"""
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    
    stats = brain.orchestrator.evaluator.get_agent_performance_stats()
    return stats


def main():
    """Main entry point"""
    # Configure logging
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    
    # Start the API server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8888,
        log_level="info",
        access_log=False
    )


if __name__ == "__main__":
    main()