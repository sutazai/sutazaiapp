#!/usr/bin/env python3
"""
Brain API Service for SutazAI
Provides HTTP API interface to the brain system
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn

from initialize_brain_minimal import MinimalBrain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Models
class BrainRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None
    priority: float = 1.0
    require_learning: bool = False

class BrainResponse(BaseModel):
    response: str
    confidence: float
    processing_time: float
    intelligence_level: float
    memories_used: int
    request_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class MemoryEntry(BaseModel):
    content: str
    importance: float = 0.5
    memory_type: str = "general"
    metadata: Optional[Dict[str, Any]] = None

class BrainStatus(BaseModel):
    initialized: bool
    intelligence_level: float
    total_requests: int
    success_rate: float
    memory_entries: int
    learning_cycles: int
    components: List[str]
    uptime_seconds: float

# Global brain instance
brain_instance = None
start_time = datetime.now()

# FastAPI app
app = FastAPI(
    title="SutazAI Brain API",
    description="AGI/ASI Brain System with Continuous Learning",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Initialize brain on startup"""
    global brain_instance
    logger.info("🚀 Starting SutazAI Brain API...")
    
    try:
        brain_instance = MinimalBrain()
        await brain_instance.initialize()
        logger.info("✅ Brain API ready")
    except Exception as e:
        logger.error(f"❌ Failed to initialize brain: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "SutazAI AGI/ASI Brain System",
        "status": "active" if brain_instance and brain_instance.is_initialized else "initializing",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not brain_instance or not brain_instance.is_initialized:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    
    return {
        "status": "healthy",
        "intelligence_level": brain_instance.intelligence_level,
        "uptime_seconds": (datetime.now() - start_time).total_seconds(),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/status", response_model=BrainStatus)
async def get_status():
    """Get detailed brain status"""
    if not brain_instance or not brain_instance.is_initialized:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    
    status = brain_instance.get_status()
    uptime = (datetime.now() - start_time).total_seconds()
    
    return BrainStatus(
        initialized=status['initialized'],
        intelligence_level=status['intelligence_level'],
        total_requests=status['performance_metrics']['total_requests'],
        success_rate=(
            status['performance_metrics']['successful_requests'] / 
            max(1, status['performance_metrics']['total_requests'])
        ),
        memory_entries=status['performance_metrics']['memory_entries'],
        learning_cycles=status['performance_metrics']['learning_cycles'],
        components=list(status['components'].keys()),
        uptime_seconds=uptime
    )

@app.post("/process", response_model=BrainResponse)
async def process_request(request: BrainRequest, background_tasks: BackgroundTasks):
    """Process a request through the brain"""
    if not brain_instance or not brain_instance.is_initialized:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    
    try:
        # Process the request
        result = await brain_instance.process_request(request.query)
        
        # Schedule background learning if requested
        if request.require_learning:
            background_tasks.add_task(trigger_learning_cycle, request.query, result)
        
        return BrainResponse(
            response=result['response'],
            confidence=result['confidence'],
            processing_time=result['processing_time'],
            intelligence_level=result['intelligence_level'],
            memories_used=result['memories_used'],
            request_id=f"req_{int(datetime.now().timestamp() * 1000)}",
            metadata={
                'context_provided': request.context is not None,
                'priority': request.priority,
                'learning_triggered': request.require_learning
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/store")
async def store_memory(entry: MemoryEntry):
    """Store a memory in the brain"""
    if not brain_instance or not brain_instance.is_initialized:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    
    try:
        memory = {
            'id': f"api_memory_{int(datetime.now().timestamp() * 1000000)}",
            'content': entry.content,
            'importance': entry.importance,
            'memory_type': entry.memory_type,
            'timestamp': datetime.now().isoformat(),
            'metadata': entry.metadata or {}
        }
        
        await brain_instance.store_memory(memory)
        
        return {
            'success': True,
            'memory_id': memory['id'],
            'message': 'Memory stored successfully'
        }
        
    except Exception as e:
        logger.error(f"Error storing memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/search")
async def search_memories(query: str, top_k: int = 5):
    """Search memories"""
    if not brain_instance or not brain_instance.is_initialized:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    
    try:
        memories = await brain_instance.search_memories(query, top_k=top_k)
        
        return {
            'query': query,
            'memories': memories,
            'count': len(memories)
        }
        
    except Exception as e:
        logger.error(f"Error searching memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/intelligence/level")
async def get_intelligence_level():
    """Get current intelligence level"""
    if not brain_instance or not brain_instance.is_initialized:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    
    return {
        'intelligence_level': brain_instance.intelligence_level,
        'scale': 'continuous (0.0 to 1.0)',
        'description': get_intelligence_description(brain_instance.intelligence_level),
        'timestamp': datetime.now().isoformat()
    }

@app.get("/learning/stats")
async def get_learning_stats():
    """Get learning statistics"""
    if not brain_instance or not brain_instance.is_initialized:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    
    status = brain_instance.get_status()
    
    return {
        'learning_cycles': status['performance_metrics']['learning_cycles'],
        'experiences_processed': len(brain_instance.experiences),
        'continuous_learning_enabled': brain_instance.learning_system['continuous_learning'],
        'meta_learning_enabled': brain_instance.learning_system['meta_learning_enabled'],
        'learning_rate': brain_instance.learning_system['learning_rate'],
        'memory_stats': status['memory_stats']
    }

@app.post("/learning/trigger")
async def trigger_learning():
    """Manually trigger a learning cycle"""
    if not brain_instance or not brain_instance.is_initialized:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    
    try:
        # Trigger immediate learning cycle
        if brain_instance.experiences:
            await brain_instance.process_experience_batch()
            
        return {
            'success': True,
            'message': 'Learning cycle triggered',
            'experiences_processed': min(10, len(brain_instance.experiences)),
            'new_learning_cycles': brain_instance.performance_metrics['learning_cycles']
        }
        
    except Exception as e:
        logger.error(f"Error triggering learning: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/components/status")
async def get_components_status():
    """Get status of all brain components"""
    if not brain_instance or not brain_instance.is_initialized:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    
    return {
        'neural_components': brain_instance.neural_components,
        'learning_system': brain_instance.learning_system,
        'memory_layers': {
            'short_term': len(brain_instance.memory_system['short_term']),
            'long_term': len(brain_instance.memory_system['long_term']),
            'working': len(brain_instance.memory_system['working']),
            'episodic': len(brain_instance.memory_system['episodic'])
        },
        'redis_connected': brain_instance.redis_client is not None
    }

@app.get("/metrics")
async def get_metrics():
    """Get performance metrics for monitoring"""
    if not brain_instance or not brain_instance.is_initialized:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    
    metrics = brain_instance.performance_metrics.copy()
    metrics['uptime_seconds'] = (datetime.now() - start_time).total_seconds()
    metrics['intelligence_level'] = brain_instance.intelligence_level
    metrics['timestamp'] = datetime.now().isoformat()
    
    return metrics

@app.websocket("/ws/brain")
async def websocket_brain_stream(websocket):
    """WebSocket endpoint for real-time brain updates"""
    await websocket.accept()
    
    try:
        while True:
            if brain_instance and brain_instance.is_initialized:
                status = {
                    'type': 'status_update',
                    'intelligence_level': brain_instance.intelligence_level,
                    'total_requests': brain_instance.performance_metrics['total_requests'],
                    'memory_entries': brain_instance.performance_metrics['memory_entries'],
                    'learning_cycles': brain_instance.performance_metrics['learning_cycles'],
                    'timestamp': datetime.now().isoformat()
                }
                
                await websocket.send_json(status)
            
            await asyncio.sleep(5)  # Update every 5 seconds
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()

# Helper functions
async def trigger_learning_cycle(query: str, result: Dict[str, Any]):
    """Background task to trigger learning from a request"""
    try:
        if brain_instance and brain_instance.is_initialized:
            # Create a learning experience
            experience = {
                'request': query,
                'response': result['response'],
                'quality_score': result['confidence'],
                'processing_time': result['processing_time'],
                'success': result['confidence'] > 0.5,
                'timestamp': datetime.now().isoformat(),
                'description': f"API request learning: {query[:50]}...",
                'performance_score': result['confidence']
            }
            
            brain_instance.experiences.append(experience)
            
            # Trigger immediate learning if we have enough experiences
            if len(brain_instance.experiences) >= 5:
                await brain_instance.process_experience_batch()
                
    except Exception as e:
        logger.error(f"Error in background learning: {e}")

def get_intelligence_description(level: float) -> str:
    """Get human-readable description of intelligence level"""
    if level < 0.3:
        return "Basic - Learning fundamental patterns"
    elif level < 0.5:
        return "Developing - Building knowledge base"
    elif level < 0.7:
        return "Competent - Demonstrating understanding"
    elif level < 0.9:
        return "Advanced - Showing sophisticated reasoning"
    else:
        return "Expert - Approaching human-level intelligence"

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "brain_api:app",
        host="0.0.0.0",
        port=8888,
        reload=False,
        log_level="info"
    )