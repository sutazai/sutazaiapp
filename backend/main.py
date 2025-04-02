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
from typing import Dict, List, Any, Optional
from pathlib import Path
import inspect

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Custom components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai_agents.model_manager import ModelManager
from ai_agents.agent_framework import AgentFramework
from ai_agents.ethical_verifier import EthicalVerifier
from app.sandbox.code_sandbox import CodeSandbox

# Import monitoring tools
from utils.logging_setup import get_api_logger
from utils.monitoring import setup_monitoring

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

# Initialize FastAPI app
app = FastAPI(
    title="SutazAI API",
    description="API backend for the SutazAI AGI/ASI system",
    version="1.0.0",
)

# Set up monitoring
setup_monitoring(app)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


# API Models
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    agent: str
    messages: List[ChatMessage]
    parameters: Optional[Dict[str, Any]] = None


class DocumentAnalysisRequest(BaseModel):
    document_id: str
    analysis_type: str
    extraction_fields: Optional[List[str]] = None
    question: Optional[str] = None


class CodeGenerationRequest(BaseModel):
    requirements: str
    language: str
    mode: str
    existing_code: Optional[str] = None
    generate_tests: Optional[bool] = False
    parameters: Optional[Dict[str, Any]] = None


class CodeExecutionRequest(BaseModel):
    code: str
    language: str = "python"
    timeout: Optional[int] = 30


class ServiceControlRequest(BaseModel):
    service: str
    action: str


class ModelControlRequest(BaseModel):
    model: str
    action: str


class LogRequest(BaseModel):
    service: str = "All"
    level: str = "ALL"
    lines: int = 50


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global model_manager, agent_framework, ethical_verifier, code_sandbox

    try:
        logger.info("Initializing API backend components...")
        logger.info(f"Importing AgentFramework from: {inspect.getfile(AgentFramework)}")

        # Initialize components
        ethical_verifier = EthicalVerifier()
        logger.info("Ethical verifier initialized")

        model_manager = ModelManager()
        logger.info("Model manager initialized")

        agent_framework = AgentFramework(model_manager)
        logger.info("Agent framework initialized")

        code_sandbox = CodeSandbox()
        logger.info("Code sandbox initialized")

        logger.info("All components initialized successfully")

    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        # Continue startup even with errors - components will be checked before use


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    logger.info("Shutting down API backend...")

    # Clean up components
    if model_manager:
        try:
            await model_manager.cleanup()
            logger.info("Model manager cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up model manager: {str(e)}")

    if agent_framework:
        try:
            await agent_framework.cleanup()
            logger.info("Agent framework cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up agent framework: {str(e)}")

    if code_sandbox:
        try:
            code_sandbox.cleanup()
            logger.info("Code sandbox cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up code sandbox: {str(e)}")

    logger.info("API backend shutdown complete")


# Helper functions
def check_components():
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
async def health_check():
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
async def chat(request: ChatRequest):
    """Process a chat request"""
    check_components()

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
async def list_agents():
    """List available agents"""
    check_components()

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
async def upload_document(file: UploadFile = File(...)):
    """Upload a document for processing"""
    check_components()

    try:
        # Generate unique ID for document
        document_id = str(uuid.uuid4())

        # Create directory for document
        document_dir = DOCUMENT_DIR / document_id
        document_dir.mkdir(exist_ok=True)

        # Save the file
        file_path = document_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Save metadata
        metadata = {
            "document_id": document_id,
            "filename": file.filename,
            "content_type": file.content_type,
            "size": len(content),
            "upload_time": datetime.now().isoformat(),
        }

        with open(document_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return {
            "status": "success",
            "document_id": document_id,
            "filename": file.filename,
        }

    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Document upload error: {str(e)}"},
        )


@app.post("/documents/analyze")
async def analyze_document(request: DocumentAnalysisRequest):
    """Analyze an uploaded document"""
    check_components()

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
        result = await agent_framework.process_document(params)

        # Add success status to result
        result["status"] = "success"

        return result

    except Exception as e:
        logger.error(f"Error analyzing document: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Document analysis error: {str(e)}",
            },
        )


# Code generation and execution endpoints
@app.post("/code/generate")
async def generate_code(request: CodeGenerationRequest):
    """Generate code based on requirements"""
    check_components()

    try:
        # Verify the requirements with ethical verifier
        verification = ethical_verifier.verify_content(request.requirements)
        if not verification["allowed"]:
            return JSONResponse(
                status_code=403,
                content={"status": "error", "message": verification["message"]},
            )

        # If in edit mode, verify existing code
        if request.mode == "Edit Existing" and request.existing_code:
            verification = ethical_verifier.verify_content(request.existing_code)
            if not verification["allowed"]:
                return JSONResponse(
                    status_code=403,
                    content={"status": "error", "message": verification["message"]},
                )

        # Prepare parameters for code generation
        params = {
            "requirements": request.requirements,
            "language": request.language,
            "mode": request.mode,
            "parameters": request.parameters or {},
        }

        # Add optional parameters
        if request.existing_code:
            params["existing_code"] = request.existing_code

        if request.generate_tests:
            params["generate_tests"] = True

        # Generate code using agent framework
        result = await agent_framework.generate_code(params)

        # Add success status
        result["status"] = "success"

        return result

    except Exception as e:
        logger.error(f"Error generating code: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Code generation error: {str(e)}"},
        )


@app.post("/code/execute")
async def execute_code(request: CodeExecutionRequest):
    """Execute code in a sandbox environment"""
    check_components()

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
async def system_status():
    """Get system status information"""
    check_components()

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
async def service_control(request: ServiceControlRequest):
    """Control system services"""
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
async def model_control(request: ModelControlRequest):
    """Control model loading/unloading"""
    check_components()

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


@app.post("/system/logs")
async def get_logs(request: LogRequest):
    """Get system logs"""
    check_components()

    try:
        # In a real system, you would query log files or a logging service
        # Here we're creating mock logs for demonstration

        service = request.service
        level = request.level.upper()
        lines = request.lines

        # Mock logs
        log_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
        services = [
            "backend",
            "model_manager",
            "agent_framework",
            "ethical_verifier",
            "code_sandbox",
        ]

        # Filter by level
        if level != "ALL":
            level_index = log_levels.index(level)
            filtered_levels = log_levels[level_index:]
        else:
            filtered_levels = log_levels

        # Filter by service
        if service != "All":
            filtered_services = [service]
        else:
            filtered_services = services

        # Generate mock logs
        logs = []
        for i in range(lines):
            service_name = filtered_services[i % len(filtered_services)]
            level_name = filtered_levels[i % len(filtered_levels)]
            timestamp = (datetime.now().timestamp() - (i * 60)) * 1000
            formatted_time = datetime.fromtimestamp(timestamp / 1000).strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            log_entry = f"{formatted_time} - {service_name} - {level_name} - Mock log entry {i + 1}"
            logs.append(log_entry)

        return {
            "status": "success",
            "logs": logs,
            "service": service,
            "level": level,
            "lines": len(logs),
        }

    except Exception as e:
        logger.error(f"Error getting logs: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Error getting logs: {str(e)}"},
        )


# Include API routers
app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
app.include_router(model_router, prefix="/models", tags=["Models"])
app.include_router(document_router, prefix="/documents", tags=["Documents"])
app.include_router(code_router, prefix="/code", tags=["Code"])
app.include_router(agent_interaction_router, prefix="/agents", tags=["Agent Interaction"])
app.include_router(agent_analytics_router, prefix="/analytics", tags=["Agent Analytics"])
app.include_router(diagrams_router, prefix="/diagrams", tags=["Diagrams"])
# Add other routers as needed, checking they exist and are imported correctly first

# Main entry point
if __name__ == "__main__":
    import uvicorn

    # Get port from environment or use default
    port = int(os.environ.get("PORT", 8000))

    # Start server
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True, log_level="info")
