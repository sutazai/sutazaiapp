#!/usr/bin/env python3
"""
SutazAI FastAPI Backend Server

This module provides the API backend for the SutazAI AGI/ASI system,
handling requests for chat, document analysis, code generation, and system monitoring.
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
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Custom components
from backend.ai_agents.model_manager import ModelManager
from backend.ai_agents.agent_framework import AgentFramework
from backend.ai_agents.ethical_verifier import EthicalVerifier
from backend.sandbox.code_sandbox import CodeSandbox

# Import monitoring tools
from backend.utils.logging_setup import get_api_logger
from backend.monitoring.monitoring import setup_monitoring

# Import routers from the backend/routers directory
from backend.routers.auth_router import router as auth_router
from backend.routers.code_router import router as code_router
from backend.routers.document_router import router as document_router
from backend.routers.model_router import router as model_router
from backend.routers.agent_interaction import router as agent_interaction_router # Renamed for clarity
from backend.routers.agent_analytics import router as agent_analytics_router # Renamed for clarity
from backend.routers.diagrams import router as diagrams_router # Added based on discovered files
# Note: Routers like config, tasks, health, monitoring, logs, workflows, memory, protocols mentioned
# in the deleted api.py were NOT found in backend/routers/. If they exist elsewhere or are needed,
# they must be imported and included separately.

# Configure logging
logger = get_api_logger()

# Get settings instance
from backend.config.settings import get_settings
settings = get_settings()

# Initialize FastAPI app
app = FastAPI(
    title="SutazAI API",
    description="API backend for the SutazAI AGI/ASI system",
    version="1.0.0",
)

# Set up monitoring
setup_monitoring(app)

# SECURITY FIX: Secure CORS configuration
from backend.security.secure_config import get_allowed_origins
from backend.security.rate_limiter import RateLimitMiddleware

# Add rate limiting middleware first
app.add_middleware(RateLimitMiddleware, default_limit="100/minute")

# Add secure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),  # Environment-specific origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # Restrict methods
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],  # Restrict headers
)

# Global variables for component instances
model_manager = None
agent_framework = None
ethical_verifier = None
code_sandbox = None

# Directory paths
UPLOAD_DIR = Path("data/uploads")
DOCUMENT_DIR = Path("data/documents")
CONFIG_DIR = Path("config")

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DOCUMENT_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

# Add project root to Python path if running as a script
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import necessary models and schemas

# Add SQLModel Session import for type hinting

# Import schemas from the new central location
from backend.schemas import (
    ChatRequest, DocumentAnalysisRequest,
    CodeGenerationRequest, CodeExecutionRequest, ServiceControlRequest, ModelControlRequest, LogRequest
)

# Import settings for log directory path

# Startup event
@app.on_event("startup")
async def startup_event() -> None:
    """Initialize components on startup"""
    global model_manager, agent_framework, ethical_verifier, code_sandbox

    logger.info("--- Starting API Backend Initialization ---")

    # Initialize core and agent components (replace with actual initializations)
    # These initializations might depend on settings, database connections, etc.
    logger.info("Initializing core components...")
    try:
        # Example Initializations (Replace with actual logic)
        # Store initialized instances on app.state
        app.state.app_state = AppState() # Needs actual init
        app.state.llm_service = LLMService() # Needs actual init
        app.state.tool_registry = ToolRegistry() # Needs actual init
        app.state.vector_store = VectorStore() # Needs actual init
        app.state.ethical_verifier = EthicalVerifier() # Use app.state
        app.state.code_sandbox = CodeSandbox() # Use app.state
        app.state.model_manager = ModelManager() # Use app.state

        logger.info("Core components initialized.")

        logger.info("Initializing AI agent components...")
        # Assuming AgentFramework initialization depends on ModelManager
        app.state.agent_framework = AgentFramework(app.state.model_manager)

        # Initialize other agent components (these need actual init logic)
        app.state.agent_communication = AgentCommunication()
        app.state.interaction_manager = InteractionManager()
        app.state.performance_metrics = PerformanceMetrics()
        app.state.workflow_engine = WorkflowEngine()
        app.state.memory_manager = MemoryManager()
        app.state.shared_memory_manager = SharedMemoryManager()
        app.state.health_check = HealthCheck()

        # Initialize AgentManager (depends on many other components)
        app.state.agent_manager = AgentManager(
             agent_communication=app.state.agent_communication,
             interaction_manager=app.state.interaction_manager,
             workflow_engine=app.state.workflow_engine,
             memory_manager=app.state.memory_manager,
             shared_memory_manager=app.state.shared_memory_manager,
             health_check=app.state.health_check,
             # Add other necessary dependencies for AgentManager init
        )
        logger.info("AI agent components initialized.")

        # --- Call the consolidated dependency initializer --- 
        # Pass instances from app.state to set up global references for injectors
        initialize_dependencies(
            llm_service=app.state.llm_service,
            tool_registry=app.state.tool_registry,
            vector_store=app.state.vector_store,
            app_state=app.state.app_state,
            agent_communication=app.state.agent_communication,
            interaction_manager=app.state.interaction_manager,
            performance_metrics=app.state.performance_metrics,
            workflow_engine=app.state.workflow_engine,
            memory_manager=app.state.memory_manager,
            shared_memory_manager=app.state.shared_memory_manager,
            health_check=app.state.health_check,
            agent_manager=app.state.agent_manager,
            # Pass other initialized components ONLY IF initialize_dependencies needs them
            # ethical_verifier=app.state.ethical_verifier, # Example if needed
            # code_sandbox=app.state.code_sandbox,     # Example if needed
        )
        logger.info("Global dependencies set via initialize_dependencies.")

    except Exception as e:
        logger.error(f"CRITICAL: Failed to initialize components: {e}", exc_info=True)
        raise RuntimeError(f"Component initialization failed: {e}")

    logger.info("All components initialized successfully")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Clean up on shutdown"""
    logger.info("Shutting down API backend...")
    # Retrieve components from app.state for cleanup
    
    # Model Manager Cleanup
    model_manager = getattr(app.state, 'model_manager', None)
    if model_manager:
        try:
            # Check if cleanup is async or sync
            if asyncio.iscoroutinefunction(model_manager.cleanup):
                await model_manager.cleanup() 
            else:
                 model_manager.cleanup()
            logger.info("Model manager cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up model manager: {str(e)}", exc_info=True)

    # Agent Framework Cleanup
    agent_framework = getattr(app.state, 'agent_framework', None)
    if agent_framework:
        try:
            if asyncio.iscoroutinefunction(agent_framework.cleanup):
                await agent_framework.cleanup()
            else:
                 agent_framework.cleanup()
            logger.info("Agent framework cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up agent framework: {str(e)}", exc_info=True)
            
    # Code Sandbox Cleanup
    code_sandbox = getattr(app.state, 'code_sandbox', None)
    if code_sandbox:
        try:
            if asyncio.iscoroutinefunction(code_sandbox.cleanup):
                await code_sandbox.cleanup()
            else:
                 code_sandbox.cleanup()
            logger.info("Code sandbox cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up code sandbox: {str(e)}", exc_info=True)

    # Add cleanup for other components stored in app.state as needed...
    # Example: LLM Service
    llm_service = getattr(app.state, 'llm_service', None)
    if llm_service and hasattr(llm_service, 'cleanup'):
        try:
            logger.info("Cleaning up LLM Service...")
            if asyncio.iscoroutinefunction(llm_service.cleanup):
                await llm_service.cleanup()
            else:
                llm_service.cleanup()
            logger.info("LLM Service cleaned up.")
        except Exception as e:
            logger.error(f"Error cleaning up LLM Service: {e}", exc_info=True)
            
    # Example: Vector Store (if it needs cleanup)
    vector_store = getattr(app.state, 'vector_store', None)
    if vector_store and hasattr(vector_store, 'close'): # Assuming a 'close' method
         try:
             logger.info("Closing Vector Store connection...")
             if asyncio.iscoroutinefunction(vector_store.close):
                  await vector_store.close()
             else:
                  vector_store.close()
             logger.info("Vector Store connection closed.")
         except Exception as e:
             logger.error(f"Error closing Vector Store: {e}", exc_info=True)

    logger.info("API backend shutdown complete")


# Helper functions
def check_components() -> None:
    """Check if all components are initialized"""
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    if not agent_framework:
        raise HTTPException(status_code=503, detail="Agent framework not initialized")
    if not ethical_verifier:
        raise HTTPException(status_code=503, detail="Ethical verifier not initialized")
    if not code_sandbox:
        raise HTTPException(status_code=503, detail="Code sandbox not initialized")


# Health check endpoint
@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Check if the API is running"""
    status = "ok"
    components_status = {
        "model_manager": model_manager is not None,
        "agent_framework": agent_framework is not None,
        "ethical_verifier": ethical_verifier is not None,
        "code_sandbox": code_sandbox is not None,
    }

    if not all(components_status.values()):
        status = "degraded"

    return {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "components": components_status,
    }


# Chat endpoints
@app.post("/chat")
async def chat(request: ChatRequest) -> Union[Dict[str, Any], JSONResponse]:
    """Process a chat request"""
    check_components()
    assert ethical_verifier is not None
    assert agent_framework is not None

    try:
        # Validate chat content with ethical verifier
        for message in request.messages:
            if message.role == "user":
                verification = ethical_verifier.verify_content(message.content)
                if not verification["allowed"]:
                    return JSONResponse(
                        status_code=403,
                        content={"status": "error", "message": verification["message"]},
                    )

        # Process chat with agent framework
        agent_name = request.agent # This is the agent configuration ID/name
        messages = [
            {"role": msg.role, "content": msg.content} for msg in request.messages
        ]
        parameters = request.parameters or {}

        # --- Instance Management (Simplified: Create new instance per request) ---
        logger.info(f"Creating agent instance for: {agent_name}")
        # Create agent instance using the agent name (config identifier)
        instance_id = await agent_framework.create_agent(agent_identifier=agent_name)

        if not instance_id:
            logger.error(f"Failed to create agent instance for {agent_name}")
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": "Failed to create agent instance"},
            )
        logger.info(f"Created agent instance {instance_id} for {agent_name}")
        # --- End Instance Management ---

        # Prepare the task input for execute_task
        # The task dict structure depends on what the specific agent execution logic expects
        # We assume here it wants 'messages' and 'parameters' directly
        task_input = {
            "messages": messages,
            "parameters": parameters
        }
        # The task definition itself might need a name or type, let's assume 'chat' task
        task_details = {"task_name": "chat", "input": task_input}

        # Call execute_task with the instance_id and the task dictionary
        logger.info(f"Executing task for instance {instance_id}...")
        response = await agent_framework.execute_task(instance_id=instance_id, task=task_details)
        logger.info(f"Task execution response for {instance_id}: {response}")

        # Terminate the agent instance immediately after use (Inefficient, but avoids resource leaks for now)
        # In a real system, instance lifecycle needs better management.
        await agent_framework.terminate_agent(instance_id)
        logger.info(f"Terminated agent instance {instance_id}")

        # --- Response Handling ---
        # Check if the task execution itself returned an error
        if response.get("status") == "error":
             logger.error(f"Task execution failed for instance {instance_id}: {response.get('error')}")
             # You might want to return a more specific error to the user here
             return JSONResponse(
                 status_code=500,
                 content={"status": "error", "message": f"Agent task execution failed: {response.get('error', 'Unknown error')}"},
             )

        # Assuming successful execution returns results within the response dict
        # Adjust based on the actual structure returned by execute_task for a 'chat' task
        chat_response_content = response.get("result", {}).get("response", "Error: Agent did not provide a valid response.")
        usage_info = response.get("result", {}).get("usage", {})

        return {
            "status": "success",
            "response": chat_response_content,
            "usage": usage_info,
            "agent": agent_name, # Return the original requested agent name
            "instance_id": instance_id # Optionally return the instance ID for debugging
        }

    except Exception as e:
        logger.error(f"Error processing chat: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Chat processing error: {str(e)}"},
        )


@app.get("/agents/list")
async def list_agents() -> Union[Dict[str, Any], JSONResponse]:
    """List available agents and their details"""
    check_components()
    assert agent_framework is not None

    try:
        # Corrected: Removed await for synchronous call
        # agents = agent_framework.list_agents() # OLD LINE
        # Get agent config dicts and convert to list of dicts for JSON response
        agent_list = [config.to_dict() for config in agent_framework.agent_configs.values()]
        return {"status": "success", "agents": agent_list} # Use the formatted list

    except Exception as e:
        logger.error(f"Error listing agents: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Error listing agents: {str(e)}"},
        )


# Document processing endpoints
@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)) -> Union[Dict[str, Any], JSONResponse]:
    """Upload a document for processing"""
    check_components()

    try:
        # --- Handle potential None filename --- #
        filename = file.filename
        if filename is None:
            filename = f"upload_{uuid.uuid4()}.dat" # Generate a fallback filename
            logger.warning(f"Uploaded file has no filename, using generated: {filename}")
        # --------------------------------------- #

        # Generate unique ID for document
        document_id = str(uuid.uuid4())

        # Create directory for document
        document_dir = DOCUMENT_DIR / document_id
        document_dir.mkdir(exist_ok=True)

        # Save the file
        file_path = document_dir / filename # Use potentially updated filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Save metadata
        metadata = {
            "document_id": document_id,
            "filename": filename, # Use potentially updated filename
            "content_type": file.content_type,
            "size": len(content),
            "upload_time": datetime.now().isoformat(),
        }

        with open(document_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return {
            "status": "success",
            "document_id": document_id,
            "filename": filename, # Use potentially updated filename
        }

    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Document upload error: {str(e)}"},
        )


@app.post("/documents/analyze")
async def analyze_document(request: DocumentAnalysisRequest) -> Union[Dict[str, Any], JSONResponse]:
    """Analyze a previously uploaded document"""
    check_components()
    assert agent_framework is not None

    try:
        # Check if document exists
        document_dir = DOCUMENT_DIR / request.document_id
        if not document_dir.exists():
            return JSONResponse(
                status_code=404,
                content={
                    "status": "error",
                    "message": f"Document {request.document_id} not found",
                },
            )

        # Load metadata
        with open(document_dir / "metadata.json", "r") as f:
            metadata = json.load(f)

        # Find the document file
        document_files = list(document_dir.glob("*"))
        document_files = [f for f in document_files if f.name != "metadata.json"]

        if not document_files:
            return JSONResponse(
                status_code=404,
                content={"status": "error", "message": "Document file not found"},
            )

        document_path = document_files[0]

        # Process the document
        analysis_type = request.analysis_type.lower()

        # Create parameters for analysis
        params = {
            "document_path": str(document_path),
            "document_type": metadata.get("content_type", ""),
            "analysis_type": analysis_type,
        }

        # Add parameters based on analysis type
        if analysis_type == "information extraction" and request.extraction_fields:
            params["extraction_fields"] = request.extraction_fields

        if analysis_type == "question answering" and request.question:
            params["question"] = request.question

        # Call agent framework for document analysis
        analysis_results = agent_framework.analyze_document(
            document_id=request.document_id,
            analysis_type=request.analysis_type,
            extraction_fields=request.extraction_fields,
            question=request.question,
        )
        return analysis_results # type: ignore[no-any-return]

    except FileNotFoundError as e:
        logger.error(f"Document not found for analysis: {request.document_id} - {e}")
        return JSONResponse(
            status_code=404,
            content={
                "status": "error",
                "message": f"Document not found: {request.document_id}",
            },
        )
    except Exception as gen_e: # Use different variable name
        logger.error(f"Error analyzing document: {str(gen_e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Document analysis error: {str(gen_e)}", # Use different variable name
            },
        )


# Code generation and execution endpoints
@app.post("/code/generate")
async def generate_code(request: CodeGenerationRequest) -> Union[Dict[str, Any], JSONResponse]:
    """Generate code based on requirements"""
    check_components()
    assert ethical_verifier is not None
    assert agent_framework is not None
    instance_id = None

    try:
        # Verify prompt with ethical verifier
        if not ethical_verifier.verify_content(request.requirements):
            return JSONResponse(
                status_code=400, content={"status": "error", "message": "Request violates ethical guidelines"}
            )

        # Create a temporary agent instance for this task
        # Choose agent config based on request (e.g., language, mode)
        # Simplified: use a default coder agent config ID
        coder_config_id = "coder_agent"
        params = request.parameters or {}
        instance_id = await agent_framework.create_agent(coder_config_id, params)

        if not instance_id:
             raise HTTPException(status_code=500, detail="Failed to create code generation agent instance.")

        task = {
            "id": f"codegen-{uuid.uuid4().hex[:6]}",
            "instruction": request.requirements,
            "language": request.language,
            "mode": request.mode,
            "existing_code": request.existing_code,
            "generate_tests": request.generate_tests,
        }
        result = await agent_framework.execute_task(instance_id, task)
        return result # type: ignore[no-any-return]

    except Exception as gen_e: # Use different variable name
        logger.error(f"Error generating code: {str(gen_e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Code generation error: {str(gen_e)}", # Use different variable name
            },
        )
    finally:
        # Ensure agent instance is terminated
        if instance_id:
            await agent_framework.terminate_agent(instance_id)


@app.post("/code/execute")
async def execute_code(request: CodeExecutionRequest) -> Union[Dict[str, Any], JSONResponse]:
    """Execute code in a sandboxed environment"""
    check_components()
    assert ethical_verifier is not None
    assert code_sandbox is not None

    try:
        # Verify the code with ethical verifier
        verification = ethical_verifier.verify_content(request.code)
        if not verification["allowed"]:
            return JSONResponse(
                status_code=403,
                content={"status": "error", "message": verification["message"]},
            )

        # Execute the code in sandbox
        result = code_sandbox.execute_code(request.code)

        # Add status field
        if result["status"] == "success" or result["status"] == "error":
            result["status"] = (
                "success"  # API success, even if code execution had errors
            )
        else:
            result["status"] = "error"

        return result

    except Exception as e:
        logger.error(f"Error executing code: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Code execution error: {str(e)}"},
        )


# System monitoring endpoints
@app.get("/system/status")
async def system_status() -> Union[Dict[str, Any], JSONResponse]:
    """Get overall system status and resource usage"""
    check_components()
    assert model_manager is not None

    try:
        # Get basic system info
        import psutil

        # CPU info
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()

        # Memory info
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        # Disk info
        disk = psutil.disk_usage("/")
        disk_percent = disk.percent

        # Get uptime
        boot_time = psutil.boot_time()
        uptime_seconds = time.time() - boot_time
        uptime_hours = uptime_seconds / 3600

        # Get running services
        services = []
        for proc in psutil.process_iter(
            ["pid", "name", "username", "status", "create_time", "memory_info"]
        ):
            try:
                pinfo = proc.as_dict()
                # Only include relevant processes (customize this based on your system)
                if pinfo["username"] in ["root", "ubuntu", "nobody", "www-data"]:
                    proc_uptime = time.time() - pinfo["create_time"]
                    services.append(
                        {
                            "name": pinfo["name"],
                            "pid": pinfo["pid"],
                            "status": pinfo["status"],
                            "uptime": int(proc_uptime),
                            "memory_usage": pinfo["memory_info"].rss
                            / (1024 * 1024),  # MB
                            "restarts": 0,  # Would need separate tracking for this
                        }
                    )
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

        # Get available models from model manager
        models = model_manager.list_models()

        # Get GPU info if available
        gpu_info = []
        try:
            import GPUtil

            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_info.append(
                    {
                        "id": gpu.id,
                        "name": gpu.name,
                        "load": gpu.load * 100,
                        "memory_total": gpu.memoryTotal,
                        "memory_used": gpu.memoryUsed,
                        "temperature": gpu.temperature,
                    }
                )
        except (ImportError, Exception) as e:
            logger.warning(f"Could not get GPU info: {str(e)}")

        # Create resource history (would need a time-series DB in a real system)
        # Here we're simulating with random data
        resource_history = {
            "cpu": {
                "timestamps": [time.time() - (i * 60) for i in range(60, 0, -1)],
                "values": [
                    max(0, min(100, cpu_percent + ((i % 10) - 5))) for i in range(60)
                ],
            },
            "memory": {
                "timestamps": [time.time() - (i * 60) for i in range(60, 0, -1)],
                "values": [
                    max(0, min(100, memory_percent + ((i % 10) - 5))) for i in range(60)
                ],
            },
        }

        # Determine system status
        status = "healthy"
        if cpu_percent > 80 or memory_percent > 80 or disk_percent > 90:
            status = "degraded"

        # Format response
        return {
            "status": status,
            "version": "1.0.0",
            "cpu_usage": cpu_percent,
            "cpu_count": cpu_count,
            "memory_usage": memory_percent,
            "memory_total_gb": memory.total / (1024**3),
            "memory_available_gb": memory.available / (1024**3),
            "disk_usage": disk_percent,
            "disk_total_gb": disk.total / (1024**3),
            "disk_free_gb": disk.free / (1024**3),
            "uptime_seconds": int(uptime_seconds),
            "uptime_hours": uptime_hours,
            "services": services,
            "models": models,
            "gpu_info": gpu_info,
            "resource_history": resource_history,
        }

    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error getting system status: {str(e)}",
            },
        )


@app.post("/system/service_control")
async def service_control(request: ServiceControlRequest) -> Union[Dict[str, Any], JSONResponse]:
    """Control system services (start, stop, restart)"""
    check_components()

    try:
        # This is a simplified implementation
        # In a real system, you would use systemd, supervisor, or a similar tool

        service = request.service
        action = request.action.lower()

        # Validate action
        if action not in ["start", "stop", "restart"]:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": f"Invalid action: {action}"},
            )

        # Mock implementation - in a real system you would call the appropriate control system
        # Here we're just returning success
        return {
            "status": "success",
            "message": f"{action.capitalize()} action requested for {service}",
            "service": service,
            "action": action,
        }

    except Exception as e:
        logger.error(f"Error controlling service: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Service control error: {str(e)}"},
        )


@app.post("/system/model_control")
async def model_control(request: ModelControlRequest) -> Union[Dict[str, Any], JSONResponse]:
    """Load or unload models dynamically"""
    check_components()
    assert model_manager is not None

    try:
        model = request.model
        action = request.action.lower()

        # Validate action
        if action not in ["load", "unload"]:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": f"Invalid action: {action}"},
            )

        # Perform action
        if action == "load":
            result = await model_manager.load_model(model)
        else:
            result = await model_manager.unload_model(model)

        return {
            "status": "success",
            "message": f"{action.capitalize()} action completed for {model}",
            "model": model,
            "action": action,
            "result": result,
        }

    except Exception as e:
        logger.error(f"Error controlling model: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Model control error: {str(e)}"},
        )


LOG_FILE_MAP = {
    "backend": "/opt/sutazaiapp/logs/backend.log",
    "agent_framework": "/opt/sutazaiapp/logs/agent_framework.log",
    "webui": "/opt/sutazaiapp/logs/streamlit_ui.log", # Assuming this is the correct web UI log
    "model_manager": "/opt/sutazaiapp/logs/model_manager.log", # Assuming a separate log
    "code_sandbox": "/opt/sutazaiapp/logs/code_sandbox.log", # Assuming a separate log
    "ethical_verifier": "/opt/sutazaiapp/logs/ethical_verifier.log", # Assuming a separate log
    "system_optimizer": "/opt/sutazaiapp/logs/system_optimizer.log",
    # Add other potential service log files here
}

# Function to read last N lines efficiently (handles potential errors)
def read_last_n_lines(file_path: Path, n: int) -> List[str]:
    """Reads the last N lines of a file efficiently."""
    try:
        with open(file_path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            buffer_size = 1024
            buffer = b''
            while f.tell() > 0 and buffer.count(b'\n') < n + 1:
                seek_pos = min(f.tell(), buffer_size)
                f.seek(-seek_pos, os.SEEK_CUR)
                buffer = f.read(seek_pos) + buffer
                f.seek(-seek_pos, os.SEEK_CUR)
            lines = buffer.decode('utf-8', errors='ignore').splitlines()
            return lines[-n:]
    except FileNotFoundError:
        return [f"Error: File not found at {file_path}"]
    except Exception as e:
        return [f"Error reading file {file_path}: {e}"]


# Helper function to robustly parse timestamps for sorting
def _parse_log_timestamp(line: str) -> datetime:
    try:
        ts_str = line.split(" - ")[0]
        for fmt in ("%Y-%m-%d %H:%M:%S,%f", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(ts_str, fmt)
            except ValueError:
                continue
        # If no format matched, return datetime.min
        return datetime.min
    except Exception:
        # On any other error, return min
        return datetime.min

@app.post("/system/logs")
async def get_logs(request: LogRequest) -> Union[Dict[str, Any], JSONResponse]:
    """Retrieve system or service logs"""
    check_components()
    # No component usage here

    try:
        # Determine log file path based on service
        log_file_path: Optional[str] = None
        log_dir = Path(settings.LOG_DIR)

        if request.service.lower() == "all":
            # Read from multiple known log files if 'all' is requested
            target_files = [str(log_dir / f) for f in ["backend.log", "agent_framework.log", "webui.log"]]
        elif request.service.lower() == "backend":
            target_files = [str(log_dir / "backend.log")]
        elif request.service.lower() == "agent_framework":
            target_files = [str(log_dir / "agent_framework.log")]
        elif request.service.lower() == "webui":
             target_files = [str(log_dir / "webui.log")]
        # Add cases for specific agent logs if needed, e.g.:
        # elif request.service.startswith("agent_"):
        #     target_files = [str(log_dir / f"{request.service}.log")]
        else:
             # Assume service is a specific log file name if not recognized
             target_files = [str(log_dir / f"{request.service}.log")]
             logger.warning(f"Requested logs for potentially unknown service/file: {request.service}")

        lines_to_fetch = request.lines

        # Read last N lines from target files
        all_log_lines = []
        for filepath in target_files:
            path_obj = Path(filepath)
            if path_obj.exists():
                 try:
                     # Read slightly more lines initially to allow for filtering
                     raw_lines = read_last_n_lines(path_obj, lines_to_fetch + 50)
                     all_log_lines.extend(raw_lines)
                 except Exception as read_err:
                      logger.error(f"Error reading log file {filepath}: {read_err}")
                      all_log_lines.append(f"ERROR: Could not read log file {filepath}: {read_err}")
            else:
                 logger.warning(f"Log file specified but not found: {filepath}")
                 all_log_lines.append(f"INFO: Log file not found at {filepath}")

        # Sort lines by timestamp using the helper function
        all_log_lines.sort(key=_parse_log_timestamp)

        # Apply filtering by level (case-insensitive)
        filtered_lines = []
        if request.level and request.level.upper() != "ALL":
            min_level_val = logging.getLevelName(request.level.upper())
            if isinstance(min_level_val, int): # Check if level name was valid
                for line in all_log_lines:
                    try:
                        # Attempt to extract level (e.g., after timestamp and ' - ')
                        level_str = line.split(" - ")[2].split(" - ")[0].strip() # Fragile parsing
                        if logging.getLevelName(level_str) >= min_level_val:
                            filtered_lines.append(line.strip())
                    except IndexError:
                        filtered_lines.append(line.strip()) # Include line if parsing fails
                    except Exception:
                        filtered_lines.append(line.strip()) # Include line on other errors
            else:
                 logger.warning(f"Invalid log level specified: {request.level}")
                 filtered_lines = [line.strip() for line in all_log_lines]
        else:
            filtered_lines = [line.strip() for line in all_log_lines]

        # Return the last N requested lines from the potentially filtered and sorted list
        final_lines = filtered_lines[-lines_to_fetch:]

        return {"status": "success", "logs": final_lines}

    except Exception as gen_e:
        logger.error(f"Error retrieving logs: {str(gen_e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Log retrieval error: {str(gen_e)}"},
        )


# Include API routers
app.include_router(auth_router, prefix="/auth", tags=["auth"]) # Standard prefix for auth
app.include_router(code_router, prefix="/code", tags=["code"])
app.include_router(document_router, prefix="/documents", tags=["documents"])
app.include_router(model_router, prefix="/models", tags=["models"])
app.include_router(agent_interaction_router, prefix="/agents/interaction", tags=["agent-interaction"])
app.include_router(agent_analytics_router, prefix="/agents/analytics", tags=["agent-analytics"])
app.include_router(diagrams_router, prefix="/diagrams", tags=["diagrams"])

# Add uvicorn entry point for running directly
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting SutazAI backend server directly via Uvicorn on host={settings.SERVER_HOST} port={settings.SERVER_PORT}")
    uvicorn.run(
        "main:app", 
        host=settings.SERVER_HOST, 
        port=settings.SERVER_PORT, 
        reload=settings.DEBUG_MODE, # Use reload only in debug mode
        log_level=settings.LOG_LEVEL.lower()
    )
