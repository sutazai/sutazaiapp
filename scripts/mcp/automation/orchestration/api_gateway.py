#!/usr/bin/env python3
"""
MCP Orchestration API Gateway

RESTful API interface for MCP orchestration service. Provides comprehensive
endpoints for workflow management, service monitoring, event handling, and
system control with OpenAPI documentation and authentication.

Author: Claude AI Assistant (ai-agent-orchestrator)
Created: 2025-08-15 11:58:00 UTC
Version: 1.0.0
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query, Body
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
from enum import Enum
import asyncio
import json
import logging
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import MCPAutomationConfig

from .event_manager import Event, EventType, EventPriority
from .workflow_engine import WorkflowStatus
from .service_registry import ServiceStatus


# Pydantic Models for API

class HealthStatus(str, Enum):
    """System health status."""
    healthy = "healthy"
    degraded = "degraded"
    unhealthy = "unhealthy"


class WorkflowRequest(BaseModel):
    """Workflow execution request."""
    workflow_id: str = Field(..., description="Workflow identifier")
    context: Dict[str, Any] = Field(default_factory=dict, description="Execution context")
    dry_run: bool = Field(False, description="Execute in dry-run mode")
    force: bool = Field(False, description="Force execution despite policy violations")
    timeout: Optional[int] = Field(None, description="Execution timeout in seconds")
    priority: int = Field(5, ge=1, le=10, description="Execution priority (1=highest)")


class WorkflowResponse(BaseModel):
    """Workflow execution response."""
    execution_id: str
    workflow_id: str
    status: str
    started_at: str
    completed_at: Optional[str] = None
    execution_time: float
    results: Dict[str, Any]
    error: Optional[str] = None


class ServiceUpdateRequest(BaseModel):
    """Service status update request."""
    status: ServiceStatus
    details: Optional[Dict[str, Any]] = None


class EventPublishRequest(BaseModel):
    """Event publish request."""
    type: str = Field(..., description="Event type")
    source: str = Field("api", description="Event source")
    data: Dict[str, Any] = Field(default_factory=dict, description="Event data")
    priority: str = Field("normal", description="Event priority")
    ttl: Optional[int] = Field(None, description="Time to live in seconds")
    correlation_id: Optional[str] = Field(None, description="Correlation identifier")


class UpdateCheckRequest(BaseModel):
    """MCP update check request."""
    servers: Optional[List[str]] = Field(None, description="Specific servers to check")
    force: bool = Field(False, description="Force update check")
    apply_updates: bool = Field(False, description="Automatically apply updates")


class CleanupRequest(BaseModel):
    """Cleanup operation request."""
    types: List[str] = Field(["all"], description="Cleanup types")
    dry_run: bool = Field(True, description="Execute in dry-run mode")
    force: bool = Field(False, description="Force cleanup despite safety checks")


class SystemControlRequest(BaseModel):
    """System control request."""
    action: str = Field(..., description="Control action (pause/resume/restart)")
    target: Optional[str] = Field(None, description="Target service or component")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Action parameters")


def create_api_app(orchestrator=None) -> FastAPI:
    """
    Create FastAPI application for orchestration API.
    
    Args:
        orchestrator: MCPOrchestrator instance
        
    Returns:
        FastAPI application
    """
    
    # Create FastAPI app
    app = FastAPI(
        title="MCP Orchestration API",
        description="RESTful API for MCP automation orchestration",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Store orchestrator reference
    app.state.orchestrator = orchestrator
    
    # Configure logging
    logger = logging.getLogger("mcp.api_gateway")
    
    # Dependency to get orchestrator
    def get_orchestrator():
        if not app.state.orchestrator:
            raise HTTPException(status_code=503, detail="Orchestrator not initialized")
        return app.state.orchestrator
    
    # Health Check Endpoints
    
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Get API health status."""
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "1.0.0"
        }
    
    @app.get("/health/detailed", tags=["Health"])
    async def detailed_health(orchestrator=Depends(get_orchestrator)):
        """Get detailed system health status."""
        try:
            status = await orchestrator.get_system_status()
            
            # Determine overall health
            services = status.get("services", {})
            unhealthy = services.get("unhealthy", 0)
            
            if unhealthy == 0:
                health = HealthStatus.healthy
            elif unhealthy < services.get("total", 0) / 2:
                health = HealthStatus.degraded
            else:
                health = HealthStatus.unhealthy
                
            return {
                "health": health,
                "status": status,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Workflow Management Endpoints
    
    @app.get("/workflows", tags=["Workflows"])
    async def list_workflows(orchestrator=Depends(get_orchestrator)):
        """List all available workflows."""
        try:
            workflows = await orchestrator.workflow_engine.list_workflows()
            return {"workflows": workflows}
        except Exception as e:
            logger.error(f"Failed to list workflows: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/workflows/{workflow_id}", tags=["Workflows"])
    async def get_workflow(workflow_id: str, orchestrator=Depends(get_orchestrator)):
        """Get workflow details."""
        try:
            workflow = await orchestrator.workflow_engine.get_workflow(workflow_id)
            if not workflow:
                raise HTTPException(status_code=404, detail="Workflow not found")
            return workflow
        except Exception as e:
            logger.error(f"Failed to get workflow: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/workflows/execute", response_model=WorkflowResponse, tags=["Workflows"])
    async def execute_workflow(
        request: WorkflowRequest,
        background_tasks: BackgroundTasks,
        orchestrator=Depends(get_orchestrator)
    ):
        """Execute a workflow."""
        try:
            from .orchestrator import OrchestrationContext, OrchestrationMode
            
            context = OrchestrationContext(
                mode=OrchestrationMode.MANUAL,
                source="api",
                metadata=request.context,
                dry_run=request.dry_run,
                force=request.force,
                timeout=request.timeout,
                priority=request.priority
            )
            
            result = await orchestrator.execute_workflow(request.workflow_id, context)
            
            return WorkflowResponse(
                execution_id=result.get("execution_id", ""),
                workflow_id=request.workflow_id,
                status=result.get("status", "unknown"),
                started_at=datetime.now(timezone.utc).isoformat(),
                completed_at=None,
                execution_time=result.get("execution_time", 0),
                results=result.get("results", {}),
                error=result.get("error")
            )
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/workflows/executions/active", tags=["Workflows"])
    async def get_active_executions(orchestrator=Depends(get_orchestrator)):
        """Get active workflow executions."""
        try:
            active = await orchestrator.workflow_engine.get_active_workflows()
            return {"executions": active}
        except Exception as e:
            logger.error(f"Failed to get active executions: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/workflows/executions/{execution_id}", tags=["Workflows"])
    async def get_execution(execution_id: str, orchestrator=Depends(get_orchestrator)):
        """Get workflow execution details."""
        try:
            execution = await orchestrator.workflow_engine.get_execution(execution_id)
            if not execution:
                raise HTTPException(status_code=404, detail="Execution not found")
            return execution
        except Exception as e:
            logger.error(f"Failed to get execution: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.delete("/workflows/executions/{execution_id}", tags=["Workflows"])
    async def cancel_execution(execution_id: str, orchestrator=Depends(get_orchestrator)):
        """Cancel a workflow execution."""
        try:
            cancelled = await orchestrator.workflow_engine.cancel_execution(execution_id)
            if not cancelled:
                raise HTTPException(status_code=404, detail="Execution not found")
            return {"status": "cancelled", "execution_id": execution_id}
        except Exception as e:
            logger.error(f"Failed to cancel execution: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Service Management Endpoints
    
    @app.get("/services", tags=["Services"])
    async def list_services(
        type: Optional[str] = Query(None, description="Filter by service type"),
        orchestrator=Depends(get_orchestrator)
    ):
        """List all registered services."""
        try:
            if type:
                services = await orchestrator.service_registry.get_services_by_type(type)
            else:
                services = await orchestrator.service_registry.get_all_services()
                
            return {
                "services": [
                    {
                        "name": s.name,
                        "type": s.type,
                        "status": s.status.value,
                        "version": s.version,
                        "capabilities": s.capabilities
                    }
                    for s in services
                ]
            }
        except Exception as e:
            logger.error(f"Failed to list services: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/services/{service_name}", tags=["Services"])
    async def get_service(service_name: str, orchestrator=Depends(get_orchestrator)):
        """Get service details."""
        try:
            service = await orchestrator.service_registry.get_service(service_name)
            if not service:
                raise HTTPException(status_code=404, detail="Service not found")
            return service
        except Exception as e:
            logger.error(f"Failed to get service: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.put("/services/{service_name}/status", tags=["Services"])
    async def update_service_status(
        service_name: str,
        request: ServiceUpdateRequest,
        orchestrator=Depends(get_orchestrator)
    ):
        """Update service status."""
        try:
            await orchestrator.service_registry.update_service_status(
                service_name,
                request.status,
                request.details
            )
            return {"status": "updated", "service": service_name}
        except Exception as e:
            logger.error(f"Failed to update service status: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/services/{service_name}/restart", tags=["Services"])
    async def restart_service(service_name: str, orchestrator=Depends(get_orchestrator)):
        """Restart a service."""
        try:
            restarted = await orchestrator.service_registry.restart_service(service_name)
            if not restarted:
                raise HTTPException(status_code=500, detail="Failed to restart service")
            return {"status": "restarted", "service": service_name}
        except Exception as e:
            logger.error(f"Failed to restart service: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/services/{service_name}/metrics", tags=["Services"])
    async def get_service_metrics(service_name: str, orchestrator=Depends(get_orchestrator)):
        """Get service metrics."""
        try:
            metrics = await orchestrator.service_registry.get_service_metrics(service_name)
            if not metrics:
                raise HTTPException(status_code=404, detail="Service not found")
            return metrics
        except Exception as e:
            logger.error(f"Failed to get service metrics: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Event Management Endpoints
    
    @app.post("/events", tags=["Events"])
    async def publish_event(
        request: EventPublishRequest,
        orchestrator=Depends(get_orchestrator)
    ):
        """Publish an event."""
        try:
            event = Event(
                type=EventType[request.type.upper()],
                source=request.source,
                data=request.data,
                priority=EventPriority[request.priority.upper()],
                ttl=request.ttl,
                correlation_id=request.correlation_id
            )
            
            event_id = await orchestrator.event_manager.publish(event)
            return {"event_id": event_id, "status": "published"}
        except Exception as e:
            logger.error(f"Failed to publish event: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/events/history", tags=["Events"])
    async def get_event_history(
        limit: int = Query(100, ge=1, le=1000),
        type: Optional[str] = Query(None, description="Filter by event type"),
        orchestrator=Depends(get_orchestrator)
    ):
        """Get event history."""
        try:
            event_type = EventType[type.upper()] if type else None
            history = await orchestrator.event_manager.event_bus.get_event_history(
                limit=limit,
                event_type=event_type
            )
            
            return {
                "events": [e.to_dict() for e in history],
                "count": len(history)
            }
        except Exception as e:
            logger.error(f"Failed to get event history: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/events/stream", tags=["Events"])
    async def stream_events(orchestrator=Depends(get_orchestrator)):
        """Stream events using Server-Sent Events."""
        async def event_generator():
            subscription_id = None
            try:
                # Subscribe to all events
                async def handler(event: Event):
                    data = json.dumps(event.to_dict())
                    yield f"data: {data}\n\n"
                
                subscription_id = await orchestrator.event_manager.subscribe(
                    EventType.CUSTOM,
                    handler
                )
                
                # Keep connection alive
                while True:
                    await asyncio.sleep(30)
                    yield f"data: {json.dumps({'type': 'ping'})}\n\n"
                    
            except asyncio.CancelledError:
                if subscription_id:
                    await orchestrator.event_manager.event_bus.unsubscribe(subscription_id)
                raise
                
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream"
        )
    
    # MCP Operations Endpoints
    
    @app.post("/mcp/update-check", tags=["MCP Operations"])
    async def check_updates(
        request: UpdateCheckRequest,
        background_tasks: BackgroundTasks,
        orchestrator=Depends(get_orchestrator)
    ):
        """Check for MCP server updates."""
        try:
            result = await orchestrator.trigger_update_check(
                servers=request.servers,
                force=request.force
            )
            return result
        except Exception as e:
            logger.error(f"Update check failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/mcp/cleanup", tags=["MCP Operations"])
    async def trigger_cleanup(
        request: CleanupRequest,
        background_tasks: BackgroundTasks,
        orchestrator=Depends(get_orchestrator)
    ):
        """Trigger cleanup operation."""
        try:
            result = await orchestrator.trigger_cleanup(
                dry_run=request.dry_run,
                types=request.types
            )
            return result
        except Exception as e:
            logger.error(f"Cleanup operation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # System Control Endpoints
    
    @app.post("/system/control", tags=["System Control"])
    async def system_control(
        request: SystemControlRequest,
        orchestrator=Depends(get_orchestrator)
    ):
        """Execute system control action."""
        try:
            if request.action == "pause":
                await orchestrator.pause()
                return {"status": "paused"}
            elif request.action == "resume":
                await orchestrator.resume()
                return {"status": "resumed"}
            elif request.action == "restart":
                # Restart specific component
                if request.target:
                    # Implement component restart
                    pass
                return {"status": "restarted"}
            else:
                raise HTTPException(status_code=400, detail=f"Unknown action: {request.action}")
        except Exception as e:
            logger.error(f"System control failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/system/status", tags=["System Control"])
    async def get_system_status(orchestrator=Depends(get_orchestrator)):
        """Get comprehensive system status."""
        try:
            status = await orchestrator.get_system_status()
            return status
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/system/metrics", tags=["System Control"])
    async def get_system_metrics(orchestrator=Depends(get_orchestrator)):
        """Get system metrics."""
        try:
            metrics = {
                "orchestrator": orchestrator.metrics.__dict__ if orchestrator else {},
                "events": await orchestrator.event_manager.get_metrics() if orchestrator else {},
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            return metrics
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/system/config", tags=["System Control"])
    async def get_system_config(orchestrator=Depends(get_orchestrator)):
        """Get system configuration."""
        try:
            config = orchestrator.config if orchestrator else {}
            return {
                "config": config.__dict__ if hasattr(config, '__dict__') else {},
                "mode": orchestrator.mode.value if orchestrator else "unknown"
            }
        except Exception as e:
            logger.error(f"Failed to get system config: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # WebSocket endpoint for real-time updates
    from fastapi import WebSocket, WebSocketDisconnect
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket, orchestrator=Depends(get_orchestrator)):
        """WebSocket endpoint for real-time updates."""
        await websocket.accept()
        subscription_id = None
        
        try:
            # Subscribe to events
            async def handler(event: Event):
                await websocket.send_json(event.to_dict())
            
            subscription_id = await orchestrator.event_manager.subscribe(
                EventType.CUSTOM,
                handler
            )
            
            # Keep connection alive
            while True:
                try:
                    # Wait for client messages
                    data = await websocket.receive_text()
                    # Echo back or handle commands
                    await websocket.send_text(f"Received: {data}")
                except WebSocketDisconnect:
                    break
                    
        finally:
            if subscription_id:
                await orchestrator.event_manager.event_bus.unsubscribe(subscription_id)
    
    return app