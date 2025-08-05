"""
Cognitive Architecture API Endpoints
===================================

FastAPI endpoints for interacting with the unified cognitive system.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from datetime import datetime
import logging

from .cognitive_integration import (
    get_integration_manager,
    initialize_cognitive_integration
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/cognitive", tags=["cognitive"])


# Request/Response models
class TaskRequest(BaseModel):
    """Request model for cognitive task processing"""
    type: str = Field(..., description="Task type")
    goal: Optional[str] = Field(None, description="Task goal")
    priority: float = Field(0.5, ge=0.0, le=1.0, description="Task priority")
    reasoning_type: str = Field("deductive", description="Type of reasoning to apply")
    max_agents: int = Field(5, ge=1, le=20, description="Maximum agents to use")
    time_limit: int = Field(3600, ge=60, description="Time limit in seconds")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")


class MemoryQuery(BaseModel):
    """Query model for memory retrieval"""
    memory_type: str = Field("episodic", description="Type of memory to query")
    cues: Dict[str, Any] = Field(..., description="Memory retrieval cues")
    time_range: Optional[List[str]] = Field(None, description="Time range for retrieval")
    limit: int = Field(10, ge=1, le=100, description="Maximum results")


class AttentionRequest(BaseModel):
    """Request model for attention allocation"""
    task_id: str = Field(..., description="Task ID")
    agents: List[str] = Field(..., description="Agents requiring attention")
    priority: float = Field(0.5, ge=0.0, le=1.0, description="Priority level")
    mode: str = Field("focused", description="Attention mode")


class LearningFeedback(BaseModel):
    """Feedback model for learning system"""
    task_id: str = Field(..., description="Task ID")
    success: bool = Field(..., description="Whether task was successful")
    feedback: Optional[str] = Field(None, description="Additional feedback")
    performance_metrics: Optional[Dict[str, float]] = Field(None, description="Performance metrics")


# Dependency to get integration manager
async def get_manager():
    """Get the cognitive integration manager"""
    manager = get_integration_manager()
    if not manager:
        raise HTTPException(status_code=503, detail="Cognitive system not initialized")
    return manager


# Endpoints
@router.post("/initialize")
async def initialize_cognitive_system():
    """Initialize the cognitive architecture system"""
    try:
        success = await initialize_cognitive_integration()
        
        if success:
            return {
                "status": "initialized",
                "message": "Cognitive architecture initialized successfully",
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            return {
                "status": "partial",
                "message": "Cognitive architecture initialized with some components unavailable",
                "timestamp": datetime.utcnow().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Failed to initialize cognitive system: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process")
async def process_task(request: TaskRequest, manager=Depends(get_manager)):
    """Process a task through the cognitive system"""
    try:
        task = {
            "type": request.type,
            "goal": request.goal,
            "priority": request.priority,
            "reasoning_type": request.reasoning_type,
            "max_agents": request.max_agents,
            "time_limit": request.time_limit
        }
        
        result = await manager.process_with_cognitive_system(task, request.context)
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/state")
async def get_cognitive_state(manager=Depends(get_manager)):
    """Get current state of the cognitive system"""
    try:
        state = manager.get_cognitive_state()
        return state
        
    except Exception as e:
        logger.error(f"Error getting cognitive state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/memory/store")
async def store_memory(
    content: Dict[str, Any],
    memory_type: str = "episodic",
    importance: float = 0.5,
    context: Optional[Dict[str, Any]] = None,
    manager=Depends(get_manager)
):
    """Store information in cognitive memory"""
    try:
        if not manager.cognitive_system:
            raise HTTPException(status_code=503, detail="Cognitive system not available")
        
        # Store in episodic memory
        if memory_type == "episodic":
            episode_id = manager.cognitive_system.episodic_memory.store_episode(
                content,
                context or {},
                [],  # No specific agents
                importance
            )
            
            return {
                "status": "stored",
                "memory_id": episode_id,
                "memory_type": memory_type,
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported memory type: {memory_type}")
            
    except Exception as e:
        logger.error(f"Error storing memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/memory/recall")
async def recall_memory(query: MemoryQuery, manager=Depends(get_manager)):
    """Recall memories based on cues"""
    try:
        if not manager.cognitive_system:
            raise HTTPException(status_code=503, detail="Cognitive system not available")
        
        # Parse time range if provided
        time_range = None
        if query.time_range and len(query.time_range) == 2:
            try:
                start_time = datetime.fromisoformat(query.time_range[0])
                end_time = datetime.fromisoformat(query.time_range[1])
                time_range = (start_time, end_time)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid time range format")
        
        # Recall from episodic memory
        if query.memory_type == "episodic":
            memories = manager.cognitive_system.episodic_memory.recall(
                query.cues,
                time_range
            )
            
            # Convert to serializable format
            result = []
            for memory in memories[:query.limit]:
                result.append({
                    "id": memory.id,
                    "content": memory.content,
                    "timestamp": memory.timestamp.isoformat(),
                    "importance": memory.importance,
                    "access_count": memory.access_count,
                    "context": memory.context
                })
            
            return {
                "memories": result,
                "count": len(result),
                "query": query.cues
            }
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported memory type: {query.memory_type}")
            
    except Exception as e:
        logger.error(f"Error recalling memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/attention/allocate")
async def allocate_attention(request: AttentionRequest, manager=Depends(get_manager)):
    """Allocate attention resources to a task"""
    try:
        if not manager.cognitive_system:
            raise HTTPException(status_code=503, detail="Cognitive system not available")
        
        # Map string mode to enum
        from ..cognitive_architecture.unified_cognitive_system import AttentionMode
        mode_map = {
            "focused": AttentionMode.FOCUSED,
            "divided": AttentionMode.DIVIDED,
            "selective": AttentionMode.SELECTIVE,
            "sustained": AttentionMode.SUSTAINED,
            "executive": AttentionMode.EXECUTIVE
        }
        
        attention_mode = mode_map.get(request.mode, AttentionMode.FOCUSED)
        
        # Allocate attention
        focus = manager.cognitive_system.attention.allocate_attention(
            request.task_id,
            request.agents,
            request.priority,
            attention_mode
        )
        
        if focus:
            return {
                "status": "allocated",
                "task_id": focus.task_id,
                "allocated_resources": focus.allocated_resources,
                "mode": focus.mode.value,
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            return {
                "status": "failed",
                "reason": "Insufficient attention resources",
                "current_focus": len(manager.cognitive_system.attention.current_focus),
                "available_resources": manager.cognitive_system.attention.resource_pool
            }
            
    except Exception as e:
        logger.error(f"Error allocating attention: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/attention/release/{task_id}")
async def release_attention(task_id: str, manager=Depends(get_manager)):
    """Release attention from a task"""
    try:
        if not manager.cognitive_system:
            raise HTTPException(status_code=503, detail="Cognitive system not available")
        
        manager.cognitive_system.attention.release_attention(task_id)
        
        return {
            "status": "released",
            "task_id": task_id,
            "available_resources": manager.cognitive_system.attention.resource_pool,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error releasing attention: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/attention/distribution")
async def get_attention_distribution(manager=Depends(get_manager)):
    """Get current attention distribution across agents"""
    try:
        if not manager.cognitive_system:
            raise HTTPException(status_code=503, detail="Cognitive system not available")
        
        distribution = manager.cognitive_system.attention.get_attention_distribution()
        
        return {
            "distribution": distribution,
            "current_focus_count": len(manager.cognitive_system.attention.current_focus),
            "resource_utilization": 1.0 - manager.cognitive_system.attention.resource_pool,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting attention distribution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/learning/feedback")
async def provide_learning_feedback(feedback: LearningFeedback, manager=Depends(get_manager)):
    """Provide feedback to the learning system"""
    try:
        if not manager.cognitive_system:
            raise HTTPException(status_code=503, detail="Cognitive system not available")
        
        # Create experience record
        experience = {
            "task_id": feedback.task_id,
            "success": feedback.success,
            "feedback": feedback.feedback,
            "performance_metrics": feedback.performance_metrics or {},
            "timestamp": datetime.utcnow()
        }
        
        # Learn from experience
        await manager.cognitive_system.learning_system.learn_from_experience(experience)
        
        return {
            "status": "processed",
            "task_id": feedback.task_id,
            "learning_applied": True,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error processing learning feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/learning/skills")
async def get_skill_levels(manager=Depends(get_manager)):
    """Get current skill levels across different task types"""
    try:
        if not manager.cognitive_system:
            raise HTTPException(status_code=503, detail="Cognitive system not available")
        
        skills = dict(manager.cognitive_system.learning_system.skill_levels)
        
        return {
            "skill_levels": skills,
            "learning_rate": manager.cognitive_system.learning_system.learning_rate,
            "exploration_rate": manager.cognitive_system.learning_system.exploration_rate,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting skill levels: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reflect")
async def trigger_reflection(manager=Depends(get_manager)):
    """Trigger metacognitive reflection"""
    try:
        reflection = await manager.perform_reflection()
        
        return reflection
        
    except Exception as e:
        logger.error(f"Error performing reflection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_cognitive_metrics(manager=Depends(get_manager)):
    """Get cognitive system performance metrics"""
    try:
        if not manager.cognitive_system:
            raise HTTPException(status_code=503, detail="Cognitive system not available")
        
        metrics = manager.cognitive_system.metrics
        
        # Add calculated metrics
        if metrics["total_tasks_processed"] > 0:
            metrics["success_rate"] = metrics["successful_tasks"] / metrics["total_tasks_processed"]
        else:
            metrics["success_rate"] = 0.0
        
        return {
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reasoning/chains")
async def get_active_reasoning_chains(manager=Depends(get_manager)):
    """Get information about active reasoning chains"""
    try:
        if not manager.cognitive_system:
            raise HTTPException(status_code=503, detail="Cognitive system not available")
        
        chains = []
        for chain_id, chain in manager.cognitive_system.active_reasoning_chains.items():
            chains.append({
                "id": chain.id,
                "reasoning_type": chain.reasoning_type.value,
                "steps": len(chain.steps),
                "agents_involved": list(chain.agents_involved),
                "start_time": chain.start_time.isoformat(),
                "confidence": chain.confidence
            })
        
        return {
            "active_chains": chains,
            "count": len(chains),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting reasoning chains: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check(manager=Depends(get_manager)):
    """Check health of cognitive system"""
    try:
        integration_status = manager.integration_status
        
        # Determine overall health
        all_healthy = all(integration_status.values())
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "components": integration_status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception:
        return {
            "status": "unhealthy",
            "components": {},
            "timestamp": datetime.utcnow().isoformat()
        }