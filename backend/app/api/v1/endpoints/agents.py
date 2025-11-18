"""Agent management endpoints with JARVIS integration"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Annotated, Optional
from datetime import datetime
from app.api.dependencies.auth import get_current_active_user, get_current_user_optional
from app.models.user import User
from app.services.jarvis_orchestrator import JARVISOrchestrator, ModelProvider
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize JARVIS orchestrator for agent information
jarvis_config = {
    "enable_local_models": True,
    "enable_web_search": True,
    "max_context_length": 10000
}
jarvis = JARVISOrchestrator(jarvis_config)

@router.get("/")
async def list_agents() -> List[Dict[str, Any]]:
    """List all available AI agents and models"""
    # Actual deployed agents on SutazAI platform (verified 2025-11-19)
    agents = [
        {
            "id": "letta",
            "name": "Letta (MemGPT)",
            "status": "active",
            "type": "memory",
            "port": 11401,
            "description": "Long-term memory AI agent with persistent context",
            "capabilities": ["memory", "chat", "task-automation"],
            "endpoint": "http://localhost:11401"
        },
        {
            "id": "crewai",
            "name": "CrewAI",
            "status": "active",
            "type": "orchestrator",
            "port": 11403,
            "description": "Multi-agent collaboration framework for complex tasks",
            "capabilities": ["orchestration", "collaboration", "task-delegation"],
            "endpoint": "http://localhost:11403"
        },
        {
            "id": "aider",
            "name": "Aider",
            "status": "active",
            "type": "code-assistant",
            "port": 11404,
            "description": "AI pair programming assistant",
            "capabilities": ["code", "refactoring", "debugging"],
            "endpoint": "http://localhost:11404"
        },
        {
            "id": "langchain",
            "name": "LangChain",
            "status": "active",
            "type": "framework",
            "port": 11405,
            "description": "LLM application development framework",
            "capabilities": ["chat", "chains", "tools", "agents"],
            "endpoint": "http://localhost:11405"
        },
        {
            "id": "finrobot",
            "name": "FinRobot",
            "status": "active",
            "type": "specialist",
            "port": 11410,
            "description": "Financial analysis and insights agent",
            "capabilities": ["financial-analysis", "market-data", "reports"],
            "endpoint": "http://localhost:11410"
        },
        {
            "id": "shellgpt",
            "name": "ShellGPT",
            "status": "active",
            "type": "cli-assistant",
            "port": 11413,
            "description": "Command-line interface assistant",
            "capabilities": ["cli", "automation", "scripting"],
            "endpoint": "http://localhost:11413"
        },
        {
            "id": "documind",
            "name": "Documind",
            "status": "active",
            "type": "document-processor",
            "port": 11414,
            "description": "Document processing and analysis agent",
            "capabilities": ["document-processing", "ocr", "extraction"],
            "endpoint": "http://localhost:11414"
        },
        {
            "id": "gpt-engineer",
            "name": "GPT-Engineer",
            "status": "active",
            "type": "code-generator",
            "port": 11416,
            "description": "Automated code generation from requirements",
            "capabilities": ["code-generation", "project-scaffolding"],
            "endpoint": "http://localhost:11416"
        },
        {
            "id": "tinyllama",
            "name": "TinyLlama",
            "status": "active",
            "type": "llm",
            "provider": "ollama",
            "port": 11434,
            "description": "Lightweight local language model (608MB)",
            "capabilities": ["chat", "text-generation"],
            "max_tokens": 2048,
            "local": True,
            "endpoint": "http://localhost:11434"
        }
    ]
    return agents

@router.get("/models")
async def list_models() -> Dict[str, Any]:
    """List all available models with their capabilities"""
    models = {}
    
    # Get models from JARVIS registry
    for model_name, model_info in jarvis.model_registry.items():
        models[model_name] = {
            "name": model_name,
            "provider": model_info["provider"].value,
            "capabilities": [cap.value for cap in model_info["capabilities"]],
            "max_tokens": model_info["max_tokens"],
            "cost_per_1k": model_info["cost_per_1k"],
            "latency": model_info["latency"],
            "quality": model_info["quality"],
            "local": model_info.get("local", False)
        }
    
    return {
        "models": models,
        "count": len(models),
        "providers": list(set(m["provider"] for m in models.values()))
    }

@router.get("/model/{model_id}")
async def get_model_details(model_id: str) -> Dict[str, Any]:
    """Get detailed information about a specific model"""
    if model_id not in jarvis.model_registry:
        raise HTTPException(
            status_code=404,
            detail=f"Model {model_id} not found"
        )
    
    model_info = jarvis.model_registry[model_id]
    return {
        "id": model_id,
        "name": model_id,
        "provider": model_info["provider"].value,
        "capabilities": [cap.value for cap in model_info["capabilities"]],
        "max_tokens": model_info["max_tokens"],
        "cost_per_1k": model_info["cost_per_1k"],
        "latency": model_info["latency"],
        "quality": model_info["quality"],
        "local": model_info.get("local", False),
        "status": "available" if model_info.get("local") else "api_required"
    }

@router.get("/metrics")
async def get_agent_metrics() -> Dict[str, Any]:
    """Get metrics for all active agents"""
    return {
        "jarvis": jarvis.metrics,
        "active_models": len([m for m in jarvis.model_registry if jarvis.model_registry[m].get("local", False)]),
        "total_models": len(jarvis.model_registry),
        "timestamp": str(datetime.utcnow())
    }

@router.post("/create")
async def create_agent(
    agent_type: str, 
    name: str,
    current_user: Annotated[User, Depends(get_current_active_user)]
) -> Dict[str, Any]:
    """Create a new agent instance (requires authentication)"""
    return {
        "id": "new_agent_id",
        "name": name,
        "type": agent_type,
        "status": "initializing",
        "created_by": current_user.username
    }