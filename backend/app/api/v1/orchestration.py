"""
SutazAI Multi-Agent Orchestration API
Comprehensive API endpoints for agent orchestration, task routing,
workflow management, and distributed coordination.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from pydantic import BaseModel, Field
import uuid

# Import orchestration components
try:
    from app.orchestration.agent_orchestrator import SutazAIAgentOrchestrator, Task, TaskPriority, RegisteredAgent
    from app.orchestration.message_bus import MessageBus, Message, MessageType
    from app.orchestration.task_router import IntelligentTaskRouter, TaskRequest, LoadBalancingAlgorithm
    from app.orchestration.workflow_engine import WorkflowEngine, Workflow, WorkflowStatus
    from app.orchestration.agent_discovery import AgentDiscoveryService, AgentInfo
    ORCHESTRATION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Orchestration components not available: {e}")
    ORCHESTRATION_AVAILABLE = False

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize orchestration components
orchestrator: Optional[SutazAIAgentOrchestrator] = None
message_bus: Optional[MessageBus] = None
task_router: Optional[IntelligentTaskRouter] = None
workflow_engine: Optional[WorkflowEngine] = None
agent_discovery: Optional[AgentDiscoveryService] = None

# Request/Response Models
class TaskSubmissionRequest(BaseModel):
    type: str = Field(..., description="Task type")
    description: str = Field(..., description="Task description")
    input_data: Any = Field(default=None, description="Task input data")
    priority: str = Field(default="normal", description="Task priority (low, normal, high, critical, emergency)")
    capabilities_required: List[str] = Field(default=[], description="Required agent capabilities")
    resource_requirements: Dict[str, Any] = Field(default={}, description="Resource requirements")
    deadline: Optional[str] = Field(default=None, description="Task deadline (ISO format)")
    dependencies: List[str] = Field(default=[], description="Task dependencies")

class WorkflowRequest(BaseModel):
    name: str = Field(..., description="Workflow name")
    description: str = Field(default="", description="Workflow description")
    nodes: List[Dict[str, Any]] = Field(..., description="Workflow nodes")
    metadata: Dict[str, Any] = Field(default={}, description="Workflow metadata")
    parallel_limit: int = Field(default=5, description="Maximum parallel execution limit")
    timeout: Optional[int] = Field(default=None, description="Workflow timeout in seconds")

class AgentRegistrationRequest(BaseModel):
    name: str = Field(..., description="Agent name")
    type: str = Field(..., description="Agent type")
    endpoint: str = Field(..., description="Agent endpoint URL")
    capabilities: List[str] = Field(..., description="Agent capabilities")
    health_endpoint: str = Field(..., description="Agent health check endpoint")
    metadata: Dict[str, Any] = Field(default={}, description="Agent metadata")

class MessageRequest(BaseModel):
    type: str = Field(..., description="Message type")
    recipient_id: str = Field(..., description="Recipient agent ID")
    content: Dict[str, Any] = Field(..., description="Message content")
    priority: int = Field(default=1, description="Message priority")
    requires_response: bool = Field(default=False, description="Whether response is required")

class LoadBalancingConfigRequest(BaseModel):
    algorithm: str = Field(..., description="Load balancing algorithm")
    parameters: Dict[str, Any] = Field(default={}, description="Algorithm parameters")

# Initialize orchestration components on startup
@router.on_event("startup")
async def initialize_orchestration():
    """Initialize orchestration components"""
    global orchestrator, message_bus, task_router, workflow_engine, agent_discovery
    
    if not ORCHESTRATION_AVAILABLE:
        logger.warning("Orchestration components not available")
        return
    
    try:
        # Initialize components
        orchestrator = SutazAIAgentOrchestrator()
        message_bus = MessageBus()
        task_router = IntelligentTaskRouter()
        workflow_engine = WorkflowEngine()
        agent_discovery = AgentDiscoveryService()
        
        # Initialize all components
        await orchestrator.initialize()
        await message_bus.initialize()
        await task_router.initialize()
        await workflow_engine.initialize()
        await agent_discovery.initialize()
        
        logger.info("Orchestration system initialized successfully")
        
    except Exception as e:
        logger.error(f"Orchestration initialization failed: {e}")

# Agent Management Endpoints

@router.get("/agents")
async def list_agents():
    """List all discovered and registered agents"""
    if not agent_discovery:
        raise HTTPException(status_code=503, detail="Agent discovery service not available")
    
    try:
        agents = await agent_discovery.get_discovered_agents()
        
        return {
            "agents": [
                {
                    "id": agent.id,
                    "name": agent.name,
                    "type": agent.type,
                    "endpoint": agent.endpoint,
                    "capabilities": agent.capabilities,
                    "status": agent.status,
                    "last_seen": agent.last_seen.isoformat() if agent.last_seen else None,
                    "discovery_method": agent.discovery_method,
                    "metadata": agent.metadata or {}
                }
                for agent in agents
            ],
            "total_count": len(agents),
            "healthy_count": len([a for a in agents if a.status == "healthy"]),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to list agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/agents/healthy")
async def list_healthy_agents():
    """List only healthy agents"""
    if not agent_discovery:
        raise HTTPException(status_code=503, detail="Agent discovery service not available")
    
    try:
        agents = await agent_discovery.get_healthy_agents()
        
        return {
            "agents": [
                {
                    "id": agent.id,
                    "name": agent.name,
                    "type": agent.type,
                    "capabilities": agent.capabilities,
                    "endpoint": agent.endpoint
                }
                for agent in agents
            ],
            "count": len(agents)
        }
        
    except Exception as e:
        logger.error(f"Failed to list healthy agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/agents/capability/{capability}")
async def get_agents_by_capability(capability: str):
    """Get agents with a specific capability"""
    if not agent_discovery:
        raise HTTPException(status_code=503, detail="Agent discovery service not available")
    
    try:
        agents = await agent_discovery.get_agents_by_capability(capability)
        
        return {
            "capability": capability,
            "agents": [
                {
                    "id": agent.id,
                    "name": agent.name,
                    "type": agent.type,
                    "endpoint": agent.endpoint
                }
                for agent in agents
            ],
            "count": len(agents)
        }
        
    except Exception as e:
        logger.error(f"Failed to get agents by capability: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/agents/register")
async def register_agent(request: AgentRegistrationRequest):
    """Manually register an agent"""
    if not agent_discovery:
        raise HTTPException(status_code=503, detail="Agent discovery service not available")
    
    try:
        agent_info = AgentInfo(
            id=f"manual_{request.name}_{uuid.uuid4().hex[:8]}",
            name=request.name,
            type=request.type,
            endpoint=request.endpoint,
            capabilities=request.capabilities,
            health_endpoint=request.health_endpoint,
            metadata=request.metadata
        )
        
        success = await agent_discovery.manually_register_agent(agent_info)
        
        if success:
            return {
                "message": "Agent registered successfully",
                "agent_id": agent_info.id,
                "status": "registered"
            }
        else:
            raise HTTPException(status_code=500, detail="Agent registration failed")
            
    except Exception as e:
        logger.error(f"Agent registration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/agents/{agent_id}")
async def remove_agent(agent_id: str):
    """Remove an agent from the system"""
    if not agent_discovery:
        raise HTTPException(status_code=503, detail="Agent discovery service not available")
    
    try:
        success = await agent_discovery.remove_agent(agent_id)
        
        if success:
            return {
                "message": f"Agent {agent_id} removed successfully",
                "status": "removed"
            }
        else:
            raise HTTPException(status_code=404, detail="Agent not found")
            
    except Exception as e:
        logger.error(f"Agent removal failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/agents/discover")
async def trigger_agent_discovery():
    """Manually trigger agent discovery"""
    if not agent_discovery:
        raise HTTPException(status_code=503, detail="Agent discovery service not available")
    
    try:
        await agent_discovery.trigger_discovery()
        
        return {
            "message": "Agent discovery triggered successfully",
            "status": "discovery_started",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Agent discovery trigger failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Task Management Endpoints

@router.post("/tasks/submit")
async def submit_task(request: TaskSubmissionRequest):
    """Submit a task for execution"""
    if not task_router:
        raise HTTPException(status_code=503, detail="Task router not available")
    
    try:
        # Parse priority
        priority_map = {
            "low": TaskPriority.LOW,
            "normal": TaskPriority.NORMAL,
            "high": TaskPriority.HIGH,
            "critical": TaskPriority.CRITICAL,
            "emergency": TaskPriority.EMERGENCY
        }
        priority = priority_map.get(request.priority.lower(), TaskPriority.NORMAL)
        
        # Parse deadline
        deadline = None
        if request.deadline:
            deadline = datetime.fromisoformat(request.deadline.replace('Z', '+00:00'))
        
        # Create task request
        task_request = TaskRequest(
            id=str(uuid.uuid4()),
            type=request.type,
            description=request.description,
            input_data=request.input_data,
            priority=priority,
            requester_id="api_user",
            capabilities_required=request.capabilities_required,
            resource_requirements=request.resource_requirements,
            deadline=deadline,
            dependencies=request.dependencies
        )
        
        # Submit task
        success = await task_router.submit_task(task_request)
        
        if success:
            return {
                "message": "Task submitted successfully",
                "task_id": task_request.id,
                "status": "queued",
                "priority": priority.name,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Task submission failed")
            
    except Exception as e:
        logger.error(f"Task submission failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tasks/queue/status")
async def get_queue_status():
    """Get current task queue status"""
    if not task_router:
        raise HTTPException(status_code=503, detail="Task router not available")
    
    try:
        status = await task_router.get_queue_status()
        return status
        
    except Exception as e:
        logger.error(f"Failed to get queue status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tasks/routing/history")
async def get_routing_history(limit: int = Query(100, ge=1, le=1000)):
    """Get recent task routing history"""
    if not task_router:
        raise HTTPException(status_code=503, detail="Task router not available")
    
    try:
        history = await task_router.get_routing_history(limit)
        return {
            "routing_history": history,
            "count": len(history)
        }
        
    except Exception as e:
        logger.error(f"Failed to get routing history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Load Balancing Configuration

@router.get("/load-balancing/algorithms")
async def list_load_balancing_algorithms():
    """List available load balancing algorithms"""
    return {
        "algorithms": [
            {
                "name": algo.value,
                "description": f"Load balancing using {algo.value.replace('_', ' ')}"
            }
            for algo in LoadBalancingAlgorithm
        ]
    }

@router.post("/load-balancing/configure")
async def configure_load_balancing(request: LoadBalancingConfigRequest):
    """Configure load balancing algorithm"""
    if not task_router:
        raise HTTPException(status_code=503, detail="Task router not available")
    
    try:
        # Validate algorithm
        try:
            algorithm = LoadBalancingAlgorithm(request.algorithm)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid algorithm: {request.algorithm}")
        
        await task_router.set_load_balancing_algorithm(algorithm)
        
        return {
            "message": "Load balancing algorithm configured successfully",
            "algorithm": algorithm.value,
            "parameters": request.parameters,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Load balancing configuration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Workflow Management Endpoints

@router.post("/workflows/create")
async def create_workflow(request: WorkflowRequest):
    """Create and start a new workflow"""
    if not workflow_engine:
        raise HTTPException(status_code=503, detail="Workflow engine not available")
    
    try:
        workflow_definition = {
            "name": request.name,
            "description": request.description,
            "nodes": request.nodes,
            "metadata": request.metadata,
            "parallel_limit": request.parallel_limit,
            "timeout": request.timeout,
            "created_by": "api_user"
        }
        
        workflow_id = await workflow_engine.create_workflow(workflow_definition)
        execution_id = await workflow_engine.execute_workflow(workflow_id)
        
        return {
            "message": "Workflow created and started successfully",
            "workflow_id": workflow_id,
            "execution_id": execution_id,
            "status": "running",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Workflow creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/workflows/{workflow_id}/status")
async def get_workflow_status(workflow_id: str):
    """Get workflow status"""
    if not workflow_engine:
        raise HTTPException(status_code=503, detail="Workflow engine not available")
    
    try:
        status = await workflow_engine.get_workflow_status(workflow_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/workflows/{workflow_id}/cancel")
async def cancel_workflow(workflow_id: str):
    """Cancel a running workflow"""
    if not workflow_engine:
        raise HTTPException(status_code=503, detail="Workflow engine not available")
    
    try:
        success = await workflow_engine.cancel_workflow(workflow_id)
        
        if success:
            return {
                "message": f"Workflow {workflow_id} cancelled successfully",
                "status": "cancelled",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=400, detail="Cannot cancel workflow")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Workflow cancellation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/workflows/{workflow_id}/pause")
async def pause_workflow(workflow_id: str):
    """Pause a running workflow"""
    if not workflow_engine:
        raise HTTPException(status_code=503, detail="Workflow engine not available")
    
    try:
        success = await workflow_engine.pause_workflow(workflow_id)
        
        if success:
            return {
                "message": f"Workflow {workflow_id} paused successfully",
                "status": "paused",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=400, detail="Cannot pause workflow")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Workflow pause failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/workflows/{workflow_id}/resume")
async def resume_workflow(workflow_id: str):
    """Resume a paused workflow"""
    if not workflow_engine:
        raise HTTPException(status_code=503, detail="Workflow engine not available")
    
    try:
        success = await workflow_engine.resume_workflow(workflow_id)
        
        if success:
            return {
                "message": f"Workflow {workflow_id} resumed successfully",
                "status": "running",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=400, detail="Cannot resume workflow")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Workflow resume failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Message Bus Endpoints

@router.post("/messages/send")
async def send_message(request: MessageRequest):
    """Send a message to an agent"""
    if not message_bus:
        raise HTTPException(status_code=503, detail="Message bus not available")
    
    try:
        message = Message(
            id=str(uuid.uuid4()),
            type=MessageType(request.type),
            sender_id="api_user",
            recipient_id=request.recipient_id,
            content=request.content,
            timestamp=datetime.now(),
            priority=request.priority,
            requires_response=request.requires_response
        )
        
        success = await message_bus.send_message(message)
        
        if success:
            return {
                "message": "Message sent successfully",
                "message_id": message.id,
                "status": "sent",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Message sending failed")
            
    except Exception as e:
        logger.error(f"Message sending failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/messages/broadcast")
async def broadcast_message(content: Dict[str, Any]):
    """Broadcast a message to all agents"""
    if not message_bus:
        raise HTTPException(status_code=503, detail="Message bus not available")
    
    try:
        message_id = await message_bus.broadcast_system_notification(
            notification=content.get("message", "System notification"),
            data=content.get("data", {})
        )
        
        return {
            "message": "Broadcast sent successfully",
            "message_id": message_id,
            "status": "broadcast",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Broadcast failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# System Status and Metrics

@router.get("/system/status")
async def get_orchestration_status():
    """Get comprehensive orchestration system status"""
    if not ORCHESTRATION_AVAILABLE:
        return {
            "status": "unavailable",
            "message": "Orchestration system not available",
            "components": {
                "orchestrator": False,
                "message_bus": False,
                "task_router": False,
                "workflow_engine": False,
                "agent_discovery": False
            }
        }
    
    try:
        status = {
            "status": "running",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "orchestrator": orchestrator is not None,
                "message_bus": message_bus is not None,
                "task_router": task_router is not None,
                "workflow_engine": workflow_engine is not None,
                "agent_discovery": agent_discovery is not None
            }
        }
        
        # Get detailed metrics if components are available
        if orchestrator:
            status["orchestrator_metrics"] = await orchestrator.get_system_metrics()
        
        if task_router:
            status["task_router_metrics"] = await task_router.get_queue_status()
        
        if workflow_engine:
            status["workflow_metrics"] = await workflow_engine.get_metrics()
        
        if agent_discovery:
            status["discovery_metrics"] = await agent_discovery.get_metrics()
        
        if message_bus:
            status["message_bus_metrics"] = await message_bus.get_metrics()
        
        return status
        
    except Exception as e:
        logger.error(f"Failed to get orchestration status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system/metrics")
async def get_orchestration_metrics():
    """Get detailed orchestration metrics"""
    if not ORCHESTRATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Orchestration system not available")
    
    try:
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "system_health": "healthy"
        }
        
        # Collect metrics from all components
        if orchestrator:
            metrics["orchestrator"] = await orchestrator.get_system_metrics()
        
        if task_router:
            metrics["task_router"] = await task_router.get_queue_status()
        
        if workflow_engine:
            metrics["workflow_engine"] = await workflow_engine.get_metrics()
        
        if agent_discovery:
            metrics["agent_discovery"] = await agent_discovery.get_metrics()
        
        if message_bus:
            metrics["message_bus"] = await message_bus.get_metrics()
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get orchestration metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Multi-Agent Coordination Endpoints

@router.post("/coordination/consensus")
async def request_consensus(
    topic: str,
    data: Dict[str, Any],
    agents: List[str] = None,
    threshold: float = 0.7
):
    """Request consensus from multiple agents"""
    if not message_bus or not agent_discovery:
        raise HTTPException(status_code=503, detail="Coordination services not available")
    
    try:
        # Get agents for consensus if not specified
        if not agents:
            healthy_agents = await agent_discovery.get_healthy_agents()
            agents = [agent.id for agent in healthy_agents[:5]]  # Limit to 5 agents
        
        # Send consensus request
        consensus_id = str(uuid.uuid4())
        
        for agent_id in agents:
            await message_bus.request_coordination(
                sender_id="orchestrator",
                coordination_type="consensus",
                data={
                    "consensus_id": consensus_id,
                    "topic": topic,
                    "data": data,
                    "threshold": threshold
                }
            )
        
        return {
            "message": "Consensus request sent",
            "consensus_id": consensus_id,
            "agents_contacted": agents,
            "threshold": threshold,
            "status": "pending",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Consensus request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@router.get("/health")
async def health_check():
    """Orchestration system health check"""
    health_status = {
        "status": "healthy" if ORCHESTRATION_AVAILABLE else "degraded",
        "timestamp": datetime.now().isoformat(),
        "components": {}
    }
    
    if ORCHESTRATION_AVAILABLE:
        # Check component health
        health_status["components"] = {
            "orchestrator": orchestrator is not None and hasattr(orchestrator, 'running') and orchestrator.running,
            "message_bus": message_bus is not None and hasattr(message_bus, 'running') and message_bus.running,
            "task_router": task_router is not None,
            "workflow_engine": workflow_engine is not None,
            "agent_discovery": agent_discovery is not None
        }
        
        # Overall health based on critical components
        critical_components = ["orchestrator", "message_bus", "task_router"]
        health_status["status"] = "healthy" if all(
            health_status["components"].get(comp, False) for comp in critical_components
        ) else "degraded"
    
    return health_status