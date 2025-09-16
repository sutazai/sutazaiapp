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
    agents = [
        {
            "id": "jarvis-core",
            "name": "JARVIS Orchestrator",
            "status": "active",
            "type": "orchestrator",
            "description": "Microsoft JARVIS-inspired multi-model orchestrator",
            "capabilities": ["chat", "code", "vision", "audio", "analysis"],
            "models": list(jarvis.model_registry.keys())
        },
        {
            "id": "gpt-4",
            "name": "GPT-4",
            "status": "active",
            "type": "llm",
            "provider": "openai",
            "capabilities": ["chat", "code", "analysis"],
            "max_tokens": 8192
        },
        {
            "id": "claude-3",
            "name": "Claude 3 Opus",
            "status": "active",
            "type": "llm",
            "provider": "anthropic",
            "capabilities": ["chat", "code", "creative"],
            "max_tokens": 200000
        },
        {
            "id": "gemini-pro",
            "name": "Gemini Pro",
            "status": "active",
            "type": "llm",
            "provider": "google",
            "capabilities": ["chat", "vision", "analysis"],
            "max_tokens": 32768
        },
        {
            "id": "llama-3",
            "name": "Llama 3 70B",
            "status": "available",
            "type": "llm",
            "provider": "ollama",
            "capabilities": ["chat", "code"],
            "max_tokens": 8192,
            "local": True
        },
        {
            "id": "mistral-7b",
            "name": "Mistral 7B",
            "status": "available",
            "type": "llm",
            "provider": "ollama",
            "capabilities": ["chat"],
            "max_tokens": 4096,
            "local": True
        },
        {
            "id": "codestral",
            "name": "Codestral",
            "status": "available",
            "type": "code",
            "provider": "huggingface",
            "capabilities": ["code"],
            "max_tokens": 32768
        },
        {
            "id": "whisper",
            "name": "Whisper ASR",
            "status": "available",
            "type": "audio",
            "provider": "openai",
            "capabilities": ["transcription"],
            "models": ["tiny", "base", "small", "medium", "large"]
        },
        {
            "id": "letta",
            "name": "Letta (MemGPT)",
            "status": "pending",
            "type": "memory",
            "description": "Long-term memory agent"
        },
        {
            "id": "autogpt",
            "name": "AutoGPT",
            "status": "pending",
            "type": "autonomous",
            "description": "Autonomous task execution agent"
        },
        {
            "id": "crewai",
            "name": "CrewAI",
            "status": "pending",
            "type": "collaborative",
            "description": "Multi-agent collaboration framework"
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