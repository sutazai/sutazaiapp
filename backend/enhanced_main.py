#!/usr/bin/env python3
"""
Enhanced SutazAI FastAPI Backend Server
Comprehensive AGI/ASI system with full integration of all components
"""

import os
import sys
import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import asyncio
import logging

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

# System imports
from system_manager import (
    SystemManager, 
    get_system_manager, 
    initialize_system_manager,
    startup_system_manager,
    shutdown_system_manager
)

# Import SutazAI Core System
from sutazai_core import SutazAICore, SystemConfig, initialize_system, get_system_instance

# Import monitoring tools
from utils.logging_setup import get_api_logger
from monitoring.monitoring import setup_monitoring

# Import routers
from routers.auth_router import router as auth_router
from routers.code_router import router as code_router
from routers.document_router import router as document_router
from routers.model_router import router as model_router
from routers.agent_interaction import router as agent_interaction_router
from routers.agent_analytics import router as agent_analytics_router
from routers.diagrams import router as diagrams_router

# Security imports
from security.secure_config import get_allowed_origins
from security.rate_limiter import RateLimitMiddleware

# Web Learning imports
from web_learning import WebScraper, ContentProcessor, KnowledgeExtractor, LearningPipeline, WebAutomation

# Agent orchestration imports
from agent_orchestration import AgentOrchestrator, create_agent_orchestrator

# Neural engine imports
from neural_engine import NeuralProcessor, create_neural_processor

# Import schemas
from schemas import (
    ChatRequest, DocumentAnalysisRequest,
    CodeGenerationRequest, CodeExecutionRequest, 
    ServiceControlRequest, ModelControlRequest, LogRequest
)

# Configure logging
logger = get_api_logger()

# Initialize FastAPI app
app = FastAPI(
    title="SutazAI Enhanced API",
    description="Enhanced API backend for the SutazAI AGI/ASI system with improved architecture",
    version="1.1.0",
    docs_url="/docs" if os.getenv("ENABLE_DOCS", "true").lower() == "true" else None,
    redoc_url="/redoc" if os.getenv("ENABLE_DOCS", "true").lower() == "true" else None
)

# Set up monitoring
setup_monitoring(app)

# Add security middleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])  # Configure appropriately for production
app.add_middleware(RateLimitMiddleware, default_limit="100/minute")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)

# Directory paths
UPLOAD_DIR = Path("data/uploads")
DOCUMENT_DIR = Path("data/documents")
CONFIG_DIR = Path("config")

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DOCUMENT_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Global SutazAI Core instance
sutazai_core: Optional[SutazAICore] = None

# Enhanced dependency injection
async def get_sutazai_core() -> SutazAICore:
    """Dependency injection for SutazAI Core"""
    global sutazai_core
    if not sutazai_core:
        raise HTTPException(status_code=503, detail="SutazAI Core not initialized")
    
    status = sutazai_core.get_system_status()
    if status.get("status") != "running":
        raise HTTPException(status_code=503, detail="SutazAI Core not running")
    
    return sutazai_core

async def get_system_manager_dependency() -> SystemManager:
    """Dependency injection for system manager"""
    system_manager = get_system_manager()
    if not system_manager:
        raise HTTPException(status_code=503, detail="System manager not initialized")
    if not system_manager.is_running():
        raise HTTPException(status_code=503, detail="System not running")
    return system_manager

async def get_healthy_system_manager() -> SystemManager:
    """Dependency injection for healthy system manager"""
    system_manager = await get_system_manager_dependency()
    if not system_manager.is_healthy():
        raise HTTPException(status_code=503, detail="System not healthy")
    return system_manager

# Startup event
@app.on_event("startup")
async def startup_event() -> None:
    """Initialize system on startup"""
    global sutazai_core
    logger.info("--- Starting Enhanced SutazAI Backend ---")
    
    try:
        # Initialize SutazAI Core System
        logger.info("Initializing SutazAI Core system...")
        config = SystemConfig(
            system_name="SutazAI Enhanced",
            version="1.1.0",
            environment=os.getenv("SUTAZAI_ENVIRONMENT", "production"),
            debug_mode=os.getenv("DEBUG", "false").lower() == "true",
            max_workers=int(os.getenv("MAX_WORKERS", "4")),
            max_memory_mb=int(os.getenv("MAX_MEMORY_MB", "8192")),
            enable_gpu=os.getenv("ENABLE_GPU", "true").lower() == "true",
            enable_neural_processing=os.getenv("ENABLE_NEURAL_PROCESSING", "true").lower() == "true",
            enable_agent_orchestration=os.getenv("ENABLE_AGENT_ORCHESTRATION", "true").lower() == "true",
            enable_knowledge_management=os.getenv("ENABLE_KNOWLEDGE_MANAGEMENT", "true").lower() == "true",
            enable_web_learning=os.getenv("ENABLE_WEB_LEARNING", "true").lower() == "true",
            enable_monitoring=os.getenv("ENABLE_MONITORING", "true").lower() == "true"
        )
        
        sutazai_core = initialize_system(config)
        
        # Start the system
        logger.info("Starting SutazAI core system...")
        success = await sutazai_core.start()
        
        if success:
            logger.info("✅ SutazAI system started successfully")
            
            # Initialize legacy system manager for backward compatibility
            logger.info("Initializing system manager for legacy compatibility...")
            system_manager = initialize_system_manager()
            legacy_success = await startup_system_manager()
            
            if legacy_success:
                logger.info("✅ Legacy system manager started successfully")
            else:
                logger.warning("⚠️ Legacy system manager failed to start")
            
            # Verify system health
            system_status = sutazai_core.get_system_status()
            if system_status.get("status") == "running":
                logger.info("✅ System health check passed")
            else:
                logger.warning("⚠️ System health check failed")
        else:
            logger.error("❌ Failed to start SutazAI system")
            raise RuntimeError("System startup failed")
            
    except Exception as e:
        logger.error(f"❌ CRITICAL: System startup failed: {e}", exc_info=True)
        raise RuntimeError(f"System startup failed: {e}")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Cleanup on shutdown"""
    global sutazai_core
    logger.info("--- Shutting down Enhanced SutazAI Backend ---")
    
    try:
        # Shutdown SutazAI Core system
        if sutazai_core:
            logger.info("Stopping SutazAI Core system...")
            core_success = await sutazai_core.stop()
            if core_success:
                logger.info("✅ SutazAI Core system shutdown completed")
            else:
                logger.warning("⚠️ SutazAI Core system shutdown had issues")
        
        # Shutdown legacy system manager
        logger.info("Stopping legacy system manager...")
        success = await shutdown_system_manager()
        if success:
            logger.info("✅ Legacy system manager shutdown completed")
        else:
            logger.warning("⚠️ Legacy system manager shutdown had issues")
            
    except Exception as e:
        logger.error(f"❌ System shutdown error: {e}", exc_info=True)

# Enhanced health check endpoint
@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Enhanced health check with system status"""
    try:
        global sutazai_core
        
        # Check SutazAI Core status
        core_status = {"status": "not_initialized"}
        if sutazai_core:
            core_status = sutazai_core.get_system_status()
        
        # Check legacy system manager
        legacy_status = "not_initialized"
        system_manager = get_system_manager()
        if system_manager:
            if system_manager.is_running():
                legacy_status = "healthy" if system_manager.is_healthy() else "unhealthy"
            else:
                legacy_status = "down"
        
        # Determine overall health
        overall_status = "healthy"
        if core_status.get("status") != "running":
            overall_status = "down"
        elif legacy_status in ["down", "unhealthy"]:
            overall_status = "degraded"
        
        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "sutazai_core": core_status,
            "legacy_system": legacy_status,
            "version": "1.1.0"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "error",
            "message": f"Health check failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

# System status endpoint
@app.get("/system/status")
async def get_system_status(
    system_manager: SystemManager = Depends(get_system_manager_dependency)
) -> Dict[str, Any]:
    """Get detailed system status"""
    try:
        return system_manager.get_system_status()
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=f"System status error: {str(e)}")

# System metrics endpoint
@app.get("/system/metrics")
async def get_system_metrics(
    limit: int = 100,
    system_manager: SystemManager = Depends(get_system_manager_dependency)
) -> Dict[str, Any]:
    """Get system metrics"""
    try:
        return system_manager.get_system_metrics(limit)
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        raise HTTPException(status_code=500, detail=f"System metrics error: {str(e)}")

# Enhanced chat endpoint
@app.post("/chat")
async def chat(
    request: ChatRequest,
    system_manager: SystemManager = Depends(get_healthy_system_manager)
) -> Dict[str, Any]:
    """Enhanced chat endpoint with system integration"""
    try:
        # Get ethical verifier
        ethical_verifier = system_manager.get_component("ethical_verifier")
        if not ethical_verifier:
            raise HTTPException(status_code=503, detail="Ethical verifier not available")
        
        # Validate chat content
        for message in request.messages:
            if message.role == "user":
                verification = ethical_verifier.verify_content(message.content)
                if not verification["allowed"]:
                    raise HTTPException(status_code=403, detail=verification["message"])
        
        # Get agent framework
        agent_framework = system_manager.get_component("agent_framework")
        if not agent_framework:
            raise HTTPException(status_code=503, detail="Agent framework not available")
        
        # Process chat request
        agent_name = request.agent
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        parameters = request.parameters or {}
        
        # Create agent instance
        logger.info(f"Creating agent instance for: {agent_name}")
        instance_id = await agent_framework.create_agent(agent_identifier=agent_name)
        
        if not instance_id:
            raise HTTPException(status_code=500, detail="Failed to create agent instance")
        
        try:
            # Execute chat task
            task_input = {
                "messages": messages,
                "parameters": parameters
            }
            task_details = {"task_name": "chat", "input": task_input}
            
            logger.info(f"Executing chat task for instance {instance_id}")
            response = await agent_framework.execute_task(instance_id=instance_id, task=task_details)
            
            # Check response
            if response.get("status") == "error":
                raise HTTPException(status_code=500, detail=f"Agent task failed: {response.get('error', 'Unknown error')}")
            
            chat_response = response.get("result", {}).get("response", "No response generated")
            usage_info = response.get("result", {}).get("usage", {})
            
            return {
                "status": "success",
                "response": chat_response,
                "usage": usage_info,
                "agent": agent_name,
                "instance_id": instance_id
            }
            
        finally:
            # Always terminate the agent instance
            await agent_framework.terminate_agent(instance_id)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing error: {str(e)}")

# Enhanced model management endpoints
@app.get("/models")
async def list_models(
    system_manager: SystemManager = Depends(get_system_manager_dependency)
) -> Dict[str, Any]:
    """List available models"""
    try:
        models = await system_manager.list_models()
        return {"status": "success", "models": models}
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")

@app.post("/models/{model_name}/load")
async def load_model(
    model_name: str,
    system_manager: SystemManager = Depends(get_system_manager_dependency)
) -> Dict[str, Any]:
    """Load a specific model"""
    try:
        result = await system_manager.load_model(model_name)
        return {"status": "success", "model": model_name, "result": result}
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

@app.post("/models/{model_name}/unload")
async def unload_model(
    model_name: str,
    system_manager: SystemManager = Depends(get_system_manager_dependency)
) -> Dict[str, Any]:
    """Unload a specific model"""
    try:
        result = await system_manager.unload_model(model_name)
        return {"status": "success", "model": model_name, "result": result}
    except Exception as e:
        logger.error(f"Error unloading model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error unloading model: {str(e)}")

# Enhanced agent endpoints
@app.get("/agents")
async def list_agents(
    system_manager: SystemManager = Depends(get_system_manager_dependency)
) -> Dict[str, Any]:
    """List available agents"""
    try:
        agent_framework = system_manager.get_component("agent_framework")
        if not agent_framework:
            raise HTTPException(status_code=503, detail="Agent framework not available")
        
        agents = [config.to_dict() for config in agent_framework.agent_configs.values()]
        return {"status": "success", "agents": agents}
    except Exception as e:
        logger.error(f"Error listing agents: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing agents: {str(e)}")

@app.post("/agents/{agent_type}/create")
async def create_agent(
    agent_type: str,
    system_manager: SystemManager = Depends(get_system_manager_dependency)
) -> Dict[str, Any]:
    """Create a new agent instance"""
    try:
        instance_id = await system_manager.create_agent(agent_type)
        return {"status": "success", "agent_type": agent_type, "instance_id": instance_id}
    except Exception as e:
        logger.error(f"Error creating agent {agent_type}: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating agent: {str(e)}")

# Component status endpoints
@app.get("/components")
async def list_components(
    system_manager: SystemManager = Depends(get_system_manager_dependency)
) -> Dict[str, Any]:
    """List all system components"""
    try:
        components = system_manager.get_component_status()
        return {"status": "success", "components": components}
    except Exception as e:
        logger.error(f"Error listing components: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing components: {str(e)}")

@app.get("/components/{component_name}")
async def get_component_status(
    component_name: str,
    system_manager: SystemManager = Depends(get_system_manager_dependency)
) -> Dict[str, Any]:
    """Get specific component status"""
    try:
        component = system_manager.get_component_status(component_name)
        if "error" in component:
            raise HTTPException(status_code=404, detail=component["error"])
        return {"status": "success", "component": component}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting component status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting component status: {str(e)}")

# Configuration endpoints
@app.get("/system/config")
async def get_system_config(
    system_manager: SystemManager = Depends(get_system_manager_dependency)
) -> Dict[str, Any]:
    """Get system configuration"""
    try:
        config = system_manager.get_config()
        return {"status": "success", "config": config}
    except Exception as e:
        logger.error(f"Error getting config: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting config: {str(e)}")

@app.post("/system/config")
async def update_system_config(
    config_updates: Dict[str, Any],
    system_manager: SystemManager = Depends(get_system_manager_dependency)
) -> Dict[str, Any]:
    """Update system configuration"""
    try:
        success = system_manager.update_config(config_updates)
        if success:
            return {"status": "success", "message": "Configuration updated"}
        else:
            raise HTTPException(status_code=500, detail="Configuration update failed")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating config: {str(e)}")

# System command endpoint
@app.post("/system/command")
async def execute_system_command(
    command: str,
    parameters: Optional[Dict[str, Any]] = None,
    system_manager: SystemManager = Depends(get_system_manager_dependency)
) -> Dict[str, Any]:
    """Execute system command"""
    try:
        result = await system_manager.execute_command(command, **(parameters or {}))
        return {"status": "success", "command": command, "result": result}
    except Exception as e:
        logger.error(f"Error executing command {command}: {e}")
        raise HTTPException(status_code=500, detail=f"Command execution error: {str(e)}")

# Enhanced SutazAI Core endpoints
@app.post("/sutazai/command")
async def execute_sutazai_command(
    command: str,
    parameters: Optional[Dict[str, Any]] = None,
    sutazai_core: SutazAICore = Depends(get_sutazai_core)
) -> Dict[str, Any]:
    """Execute SutazAI Core command"""
    try:
        result = await sutazai_core.execute_command(command, **(parameters or {}))
        return {"status": "success", "command": command, "result": result}
    except Exception as e:
        logger.error(f"Error executing SutazAI command {command}: {e}")
        raise HTTPException(status_code=500, detail=f"SutazAI command execution error: {str(e)}")

@app.get("/sutazai/status")
async def get_sutazai_status(
    sutazai_core: SutazAICore = Depends(get_sutazai_core)
) -> Dict[str, Any]:
    """Get comprehensive SutazAI system status"""
    try:
        return sutazai_core.get_system_status()
    except Exception as e:
        logger.error(f"Error getting SutazAI status: {e}")
        raise HTTPException(status_code=500, detail=f"SutazAI status error: {str(e)}")

@app.get("/sutazai/components")
async def get_sutazai_components(
    sutazai_core: SutazAICore = Depends(get_sutazai_core)
) -> Dict[str, Any]:
    """Get SutazAI component information"""
    try:
        components = sutazai_core.get_component_info()
        return {
            "status": "success",
            "components": {name: {
                "name": info.name,
                "version": info.version,
                "status": info.status.value,
                "config": info.config,
                "metrics": info.metrics,
                "last_updated": info.last_updated.isoformat()
            } for name, info in components.items()}
        }
    except Exception as e:
        logger.error(f"Error getting SutazAI components: {e}")
        raise HTTPException(status_code=500, detail=f"SutazAI components error: {str(e)}")

@app.get("/sutazai/metrics")
async def get_sutazai_metrics(
    limit: int = 100,
    sutazai_core: SutazAICore = Depends(get_sutazai_core)
) -> Dict[str, Any]:
    """Get SutazAI system metrics"""
    try:
        metrics = sutazai_core.get_metrics_history(limit)
        return {"status": "success", "metrics": metrics}
    except Exception as e:
        logger.error(f"Error getting SutazAI metrics: {e}")
        raise HTTPException(status_code=500, detail=f"SutazAI metrics error: {str(e)}")

@app.post("/sutazai/restart")
async def restart_sutazai_system(
    sutazai_core: SutazAICore = Depends(get_sutazai_core)
) -> Dict[str, Any]:
    """Restart SutazAI system"""
    try:
        success = await sutazai_core.restart()
        return {
            "status": "success" if success else "error",
            "message": "System restart completed" if success else "System restart failed"
        }
    except Exception as e:
        logger.error(f"Error restarting SutazAI system: {e}")
        raise HTTPException(status_code=500, detail=f"SutazAI restart error: {str(e)}")

# Enhanced neural processing endpoints
@app.post("/neural/process")
async def neural_process(
    input_data: Dict[str, Any],
    sutazai_core: SutazAICore = Depends(get_sutazai_core)
) -> Dict[str, Any]:
    """Process data through neural engine"""
    try:
        result = await sutazai_core.execute_command("neural.process", input_data=input_data)
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Error in neural processing: {e}")
        raise HTTPException(status_code=500, detail=f"Neural processing error: {str(e)}")

@app.post("/neural/train")
async def neural_train(
    training_data: Dict[str, Any],
    sutazai_core: SutazAICore = Depends(get_sutazai_core)
) -> Dict[str, Any]:
    """Train neural network"""
    try:
        result = await sutazai_core.execute_command("neural.train", training_data=training_data)
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Error in neural training: {e}")
        raise HTTPException(status_code=500, detail=f"Neural training error: {str(e)}")

@app.get("/neural/status")
async def get_neural_status(
    sutazai_core: SutazAICore = Depends(get_sutazai_core)
) -> Dict[str, Any]:
    """Get neural processing status"""
    try:
        result = await sutazai_core.execute_command("neural.status")
        return {"status": "success", "neural_status": result}
    except Exception as e:
        logger.error(f"Error getting neural status: {e}")
        raise HTTPException(status_code=500, detail=f"Neural status error: {str(e)}")

# Enhanced agent management endpoints
@app.post("/agents/create_enhanced")
async def create_enhanced_agent(
    agent_type: str,
    config: Optional[Dict[str, Any]] = None,
    sutazai_core: SutazAICore = Depends(get_sutazai_core)
) -> Dict[str, Any]:
    """Create enhanced agent through SutazAI Core"""
    try:
        result = await sutazai_core.execute_command("agent.create", agent_type=agent_type, config=config)
        return {"status": "success", "agent": result}
    except Exception as e:
        logger.error(f"Error creating enhanced agent: {e}")
        raise HTTPException(status_code=500, detail=f"Enhanced agent creation error: {str(e)}")

@app.post("/agents/{instance_id}/execute_enhanced")
async def execute_enhanced_agent_task(
    instance_id: str,
    task: Dict[str, Any],
    sutazai_core: SutazAICore = Depends(get_sutazai_core)
) -> Dict[str, Any]:
    """Execute task with enhanced agent"""
    try:
        result = await sutazai_core.execute_command("agent.execute", instance_id=instance_id, task=task)
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Error executing enhanced agent task: {e}")
        raise HTTPException(status_code=500, detail=f"Enhanced agent execution error: {str(e)}")

# Knowledge Graph endpoints
@app.post("/knowledge/graph/add_node")
async def add_knowledge_node(
    node_data: Dict[str, Any],
    sutazai_core: SutazAICore = Depends(get_sutazai_core)
) -> Dict[str, Any]:
    """Add node to knowledge graph"""
    try:
        from knowledge_graph import GraphNode
        
        node = GraphNode(
            id=node_data.get("id", str(uuid.uuid4())),
            label=node_data["label"],
            node_type=node_data["type"],
            properties=node_data.get("properties", {})
        )
        
        kg_engine = sutazai_core.get_component("knowledge_graph")
        if not kg_engine:
            raise HTTPException(status_code=503, detail="Knowledge graph not available")
        
        success = await kg_engine.add_node(node)
        return {"status": "success" if success else "failed", "node_id": node.id}
    except Exception as e:
        logger.error(f"Error adding knowledge node: {e}")
        raise HTTPException(status_code=500, detail=f"Knowledge graph error: {str(e)}")

@app.get("/knowledge/graph/search")
async def search_knowledge_graph(
    query: str,
    limit: int = 10,
    sutazai_core: SutazAICore = Depends(get_sutazai_core)
) -> Dict[str, Any]:
    """Search knowledge graph by semantic similarity"""
    try:
        kg_engine = sutazai_core.get_component("knowledge_graph")
        if not kg_engine:
            raise HTTPException(status_code=503, detail="Knowledge graph not available")
        
        results = kg_engine.search_by_embedding(query, limit)
        
        formatted_results = [
            {
                "node_id": node.id,
                "label": node.label,
                "type": node.node_type,
                "properties": node.properties,
                "similarity": float(similarity)
            }
            for node, similarity in results
        ]
        
        return {"status": "success", "results": formatted_results}
    except Exception as e:
        logger.error(f"Error searching knowledge graph: {e}")
        raise HTTPException(status_code=500, detail=f"Knowledge search error: {str(e)}")

# Self-Evolution endpoints
@app.post("/evolution/evolve_code")
async def evolve_code(
    code_data: Dict[str, Any],
    sutazai_core: SutazAICore = Depends(get_sutazai_core)
) -> Dict[str, Any]:
    """Evolve code using self-evolution engine"""
    try:
        from self_evolution import SelfEvolutionEngine, EvolutionMetrics
        
        evolution_engine = sutazai_core.get_component("self_evolution")
        if not evolution_engine:
            raise HTTPException(status_code=503, detail="Self-evolution engine not available")
        
        original_code = code_data["code"]
        target_metrics = EvolutionMetrics(
            performance_score=code_data.get("target_performance", 0.8),
            efficiency_score=code_data.get("target_efficiency", 0.8),
            accuracy_score=code_data.get("target_accuracy", 0.8)
        )
        
        result = await evolution_engine.evolve_code(
            original_code, 
            target_metrics,
            max_iterations=code_data.get("max_iterations", 20)
        )
        
        if result:
            return {
                "status": "success",
                "evolved_code": result.code,
                "metrics": result.metrics.__dict__,
                "generation": result.generation
            }
        else:
            return {"status": "failed", "message": "Evolution did not converge"}
            
    except Exception as e:
        logger.error(f"Error in code evolution: {e}")
        raise HTTPException(status_code=500, detail=f"Code evolution error: {str(e)}")

@app.get("/evolution/statistics")
async def get_evolution_statistics(
    sutazai_core: SutazAICore = Depends(get_sutazai_core)
) -> Dict[str, Any]:
    """Get evolution engine statistics"""
    try:
        evolution_engine = sutazai_core.get_component("self_evolution")
        if not evolution_engine:
            raise HTTPException(status_code=503, detail="Self-evolution engine not available")
        
        stats = evolution_engine.get_evolution_statistics()
        return {"status": "success", "statistics": stats}
    except Exception as e:
        logger.error(f"Error getting evolution statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Evolution statistics error: {str(e)}")

# External AI Service Integration endpoints
@app.post("/ai/langflow/execute")
async def execute_langflow(
    flow_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Execute Langflow workflow"""
    try:
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://langflow:7860/api/v1/run",
                json=flow_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {"status": "success", "result": result}
                else:
                    error_text = await response.text()
                    raise HTTPException(status_code=response.status, detail=f"Langflow error: {error_text}")
    except Exception as e:
        logger.error(f"Error executing Langflow: {e}")
        raise HTTPException(status_code=500, detail=f"Langflow execution error: {str(e)}")

@app.post("/ai/pytorch/generate")
async def pytorch_generate(
    data: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate using PyTorch service"""
    try:
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://pytorch:8085/generate",
                json=data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {"status": "success", "result": result}
                else:
                    error_text = await response.text()
                    raise HTTPException(status_code=response.status, detail=f"PyTorch error: {error_text}")
    except Exception as e:
        logger.error(f"Error with PyTorch generation: {e}")
        raise HTTPException(status_code=500, detail=f"PyTorch generation error: {str(e)}")

@app.post("/ai/tensorflow/train")
async def tensorflow_train(
    training_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Train model using TensorFlow service"""
    try:
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://tensorflow:8086/train",
                json=training_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {"status": "success", "result": result}
                else:
                    error_text = await response.text()
                    raise HTTPException(status_code=response.status, detail=f"TensorFlow error: {error_text}")
    except Exception as e:
        logger.error(f"Error with TensorFlow training: {e}")
        raise HTTPException(status_code=500, detail=f"TensorFlow training error: {str(e)}")

@app.post("/ai/jax/compute")
async def jax_compute(
    computation_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Perform computation using JAX service"""
    try:
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://jax:8087/compute",
                json=computation_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {"status": "success", "result": result}
                else:
                    error_text = await response.text()
                    raise HTTPException(status_code=response.status, detail=f"JAX error: {error_text}")
    except Exception as e:
        logger.error(f"Error with JAX computation: {e}")
        raise HTTPException(status_code=500, detail=f"JAX computation error: {str(e)}")

@app.get("/ai/services/status")
async def get_ai_services_status() -> Dict[str, Any]:
    """Get status of all AI services"""
    try:
        import aiohttp
        
        services = {
            "langflow": "http://langflow:7860/health",
            "dify": "http://dify:5001/health", 
            "autogen": "http://autogen:8080/health",
            "pytorch": "http://pytorch:8085/health",
            "tensorflow": "http://tensorflow:8086/health",
            "jax": "http://jax:8087/health"
        }
        
        status_results = {}
        
        async with aiohttp.ClientSession() as session:
            for service_name, health_url in services.items():
                try:
                    async with session.get(health_url, timeout=5) as response:
                        if response.status == 200:
                            health_data = await response.json()
                            status_results[service_name] = {
                                "status": "healthy",
                                "details": health_data
                            }
                        else:
                            status_results[service_name] = {
                                "status": "unhealthy",
                                "details": {"error": f"HTTP {response.status}"}
                            }
                except Exception as e:
                    status_results[service_name] = {
                        "status": "unreachable",
                        "details": {"error": str(e)}
                    }
        
        return {"status": "success", "services": status_results}
        
    except Exception as e:
        logger.error(f"Error getting AI services status: {e}")
        raise HTTPException(status_code=500, detail=f"Services status error: {str(e)}")

# Include existing routers
app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(code_router, prefix="/code", tags=["code"])
app.include_router(document_router, prefix="/documents", tags=["documents"])
app.include_router(model_router, prefix="/models", tags=["models"])
app.include_router(agent_interaction_router, prefix="/agents/interaction", tags=["agent-interaction"])
app.include_router(agent_analytics_router, prefix="/agents/analytics", tags=["agent-analytics"])
app.include_router(diagrams_router, prefix="/diagrams", tags=["diagrams"])

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    logger.error(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.detail, "timestamp": datetime.now().isoformat()}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "Internal server error", "timestamp": datetime.now().isoformat()}
    )

# Run server
if __name__ == "__main__":
    import uvicorn
    
    # Get configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    logger.info(f"Starting Enhanced SutazAI backend server on {host}:{port}")
    logger.info(f"Debug mode: {debug}")
    logger.info(f"Log level: {log_level}")
    
    uvicorn.run(
        "enhanced_main:app",
        host=host,
        port=port,
        reload=debug,
        log_level=log_level,
        access_log=True
    )