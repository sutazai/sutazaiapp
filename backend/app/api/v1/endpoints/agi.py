"""
AGI API Endpoints for SutazAI
Provides access to advanced AGI capabilities including reasoning and self-improvement
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

try:
    from app.ai_agents.reasoning.agi_orchestrator import AGIOrchestrator
    from app.ai_agents.reasoning import AdvancedReasoningEngine, SelfImprovementEngine
    from app.ai_agents.agent_manager import AgentManager
    from app.ai_agents.orchestrator.workflow_engine import WorkflowEngine
except ImportError:
    # Fallback imports if modules don't exist
    AGIOrchestrator = None
    AdvancedReasoningEngine = None
    SelfImprovementEngine = None
    AgentManager = None
    WorkflowEngine = None

logger = logging.getLogger(__name__)

router = APIRouter()

# Global AGI orchestrator instance (would be initialized in main app)
agi_orchestrator: Optional[AGIOrchestrator] = None

class AGITaskRequest(BaseModel):
    description: str
    task_type: str = "general"
    domain: str = "general"
    require_reasoning: bool = True
    enable_learning: bool = True
    max_thinking_time: int = 300
    min_agents: int = 3

class ReasoningRequest(BaseModel):
    problem: str
    domain: str = "general"
    min_agents: int = 3
    require_consensus: bool = True

class AGIResponse(BaseModel):
    task_id: str
    success: bool
    result: Any
    confidence: float
    approach: str
    reasoning_chain_id: Optional[str] = None
    execution_time: float
    timestamp: str

@router.post("/process", response_model=AGIResponse)
async def process_agi_task(request: AGITaskRequest, background_tasks: BackgroundTasks):
    """
    Process a complex task using AGI capabilities
    
    This endpoint demonstrates the full power of the SutazAI AGI system:
    - Multi-agent reasoning for complex problems
    - Self-improvement and learning
    - Dynamic agent coordination
    - Autonomous capability enhancement
    """
    if not agi_orchestrator:
        raise HTTPException(status_code=503, detail="AGI system not initialized")
    
    try:
        start_time = datetime.now()
        
        task = {
            "id": f"agi_task_{int(start_time.timestamp())}",
            "description": request.description,
            "type": request.task_type,
            "domain": request.domain
        }
        
        logger.info(f"Processing AGI task: {request.description[:100]}...")
        
        result = await agi_orchestrator.process_complex_task(
            task=task,
            require_reasoning=request.require_reasoning,
            enable_learning=request.enable_learning
        )
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return AGIResponse(
            task_id=result.get("task_id", task["id"]),
            success=result.get("success", False),
            result=result.get("result"),
            confidence=result.get("confidence", 0.0),
            approach=result.get("approach", "unknown"),
            reasoning_chain_id=result.get("reasoning_chain_id"),
            execution_time=execution_time,
            timestamp=result.get("timestamp", datetime.now().isoformat())
        )
        
    except Exception as e:
        logger.error(f"AGI task processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"AGI processing failed: {str(e)}")

@router.post("/reason")
async def advanced_reasoning(request: ReasoningRequest):
    """
    Use advanced multi-agent reasoning on a specific problem
    
    This endpoint showcases the reasoning engine that implements
    approaches similar to GPT-o3's multi-step verification process
    """
    if not agi_orchestrator:
        raise HTTPException(status_code=503, detail="AGI system not initialized")
    
    try:
        reasoning_chain = await agi_orchestrator.reasoning_engine.reason_about_problem(
            problem=request.problem,
            domain=request.domain,
            min_agents=request.min_agents,
            require_consensus=request.require_consensus
        )
        
        # Get detailed explanation
        explanation = await agi_orchestrator.reasoning_engine.get_reasoning_explanation(
            reasoning_chain.problem_id
        )
        
        return {
            "reasoning_chain_id": reasoning_chain.problem_id,
            "problem": reasoning_chain.problem_statement,
            "final_answer": reasoning_chain.final_answer,
            "confidence": reasoning_chain.confidence_score,
            "verification_count": reasoning_chain.verification_count,
            "detailed_explanation": explanation,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Advanced reasoning failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reasoning failed: {str(e)}")

@router.get("/status")
async def get_agi_status():
    """
    Get comprehensive AGI system status and capabilities
    """
    if not agi_orchestrator:
        raise HTTPException(status_code=503, detail="AGI system not initialized")
    
    try:
        status = await agi_orchestrator.get_agi_status()
        return status
        
    except Exception as e:
        logger.error(f"Failed to get AGI status: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@router.post("/demonstrate")
async def demonstrate_agi():
    """
    Demonstrate AGI capabilities with a complex showcase task
    
    This endpoint runs a comprehensive demonstration of the system's
    AGI capabilities including reasoning, learning, and coordination
    """
    if not agi_orchestrator:
        raise HTTPException(status_code=503, detail="AGI system not initialized")
    
    try:
        demonstration = await agi_orchestrator.demonstrate_agi_capabilities()
        return demonstration
        
    except Exception as e:
        logger.error(f"AGI demonstration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Demonstration failed: {str(e)}")

@router.get("/learning/report")
async def get_learning_report():
    """
    Get self-improvement and learning report
    """
    if not agi_orchestrator:
        raise HTTPException(status_code=503, detail="AGI system not initialized")
    
    try:
        report = await agi_orchestrator.self_improvement.get_improvement_report()
        return report
        
    except Exception as e:
        logger.error(f"Failed to get learning report: {e}")
        raise HTTPException(status_code=500, detail=f"Learning report failed: {str(e)}")

@router.get("/reasoning/chains")
async def get_active_reasoning_chains():
    """
    Get information about active reasoning chains
    """
    if not agi_orchestrator:
        raise HTTPException(status_code=503, detail="AGI system not initialized")
    
    try:
        chains = []
        for chain_id, chain in agi_orchestrator.reasoning_engine.active_chains.items():
            chains.append({
                "chain_id": chain_id,
                "problem": chain.problem_statement,
                "steps_count": len(chain.steps),
                "confidence": chain.confidence_score,
                "final_answer": chain.final_answer
            })
            
        return {
            "active_chains": chains,
            "total_chains": len(chains),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get reasoning chains: {e}")
        raise HTTPException(status_code=500, detail=f"Chains retrieval failed: {str(e)}")

@router.get("/reasoning/explanation/{chain_id}")
async def get_reasoning_explanation(chain_id: str):
    """
    Get detailed explanation of a specific reasoning chain
    """
    if not agi_orchestrator:
        raise HTTPException(status_code=503, detail="AGI system not initialized")
    
    try:
        explanation = await agi_orchestrator.reasoning_engine.get_reasoning_explanation(chain_id)
        
        if "error" in explanation:
            raise HTTPException(status_code=404, detail=explanation["error"])
            
        return explanation
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get reasoning explanation: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation retrieval failed: {str(e)}")

@router.post("/initialize")
async def initialize_agi_system():
    """
    Initialize the AGI orchestrator system
    
    This should be called during application startup
    """
    global agi_orchestrator
    
    try:
        # Create required components with minimal dependencies for testing
        from ai_agents.health_check import HealthCheck
        from ai_agents.protocols.agent_communication import AgentCommunication
        from ai_agents.memory.agent_memory import MemoryManager
        from ai_agents.memory.shared_memory import SharedMemoryManager
        from ai_agents.interaction.human_interaction import InteractionManager
        
        # Initialize components
        health_check = HealthCheck()
        agent_communication = AgentCommunication()
        memory_manager = MemoryManager()
        shared_memory_manager = SharedMemoryManager()
        interaction_manager = InteractionManager()
        workflow_engine = WorkflowEngine()
        
        # Create agent manager with all required dependencies
        agent_manager = AgentManager(
            agent_communication=agent_communication,
            interaction_manager=interaction_manager,
            workflow_engine=workflow_engine,
            memory_manager=memory_manager,
            shared_memory_manager=shared_memory_manager,
            health_check=health_check
        )
            
        agi_orchestrator = AGIOrchestrator(
            agent_manager=agent_manager,
            workflow_engine=workflow_engine
        )
        
        logger.info("AGI system initialized successfully")
        
        return {
            "status": "initialized",
            "message": "AGI orchestrator is ready",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"AGI initialization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")

@router.get("/capabilities")
async def get_agi_capabilities():
    """
    Get detailed information about AGI capabilities
    """
    if not agi_orchestrator:
        raise HTTPException(status_code=503, detail="AGI system not initialized")
    
    capabilities = {
        "reasoning_capabilities": [
            "Multi-agent collaboration",
            "Step-by-step analysis", 
            "Hypothesis generation",
            "Cross-verification",
            "Confidence scoring",
            "Domain specialization"
        ],
        "self_improvement_features": [
            "Performance tracking",
            "Pattern recognition", 
            "Strategy optimization",
            "Capability metrics",
            "Continuous learning",
            "Auto-enhancement"
        ],
        "coordination_features": [
            "Dynamic task routing",
            "Complexity analysis",
            "Optimal agent selection",
            "Parallel execution",
            "Resource optimization",
            "Performance monitoring"
        ],
        "supported_domains": [
            "code",
            "math", 
            "science",
            "analysis",
            "general"
        ],
        "current_status": {
            "max_reasoning_time": 300,
            "active_chains": len(agi_orchestrator.reasoning_engine.active_chains),
            "learning_events": len(agi_orchestrator.self_improvement.learning_events),
            "capability_scores": agi_orchestrator.capability_scores
        }
    }
    
    return capabilities

# Health check endpoint
@router.get("/health")
async def agi_health_check():
    """AGI system health check"""
    
    if not agi_orchestrator:
        return {
            "status": "not_initialized",
            "message": "AGI system not initialized",
            "timestamp": datetime.now().isoformat()
        }
    
    try:
        # Basic health checks
        health_status = {
            "status": "healthy",
            "reasoning_engine": "operational",
            "self_improvement": "operational", 
            "agent_manager": "operational",
            "workflow_engine": "operational",
            "active_reasoning_chains": len(agi_orchestrator.reasoning_engine.active_chains),
            "learning_events_count": len(agi_orchestrator.self_improvement.learning_events),
            "performance_history_size": len(agi_orchestrator.performance_history),
            "timestamp": datetime.now().isoformat()
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"AGI health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        } 