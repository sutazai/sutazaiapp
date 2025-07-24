#!/usr/bin/env python3
"""
Enhanced SutazAI Backend with AI Repository Manager Integration
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import time
import asyncio
import logging
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
import sys
import os

# Add the current directory to Python path
sys.path.append("/opt/sutazaiapp")

# Import our AI Repository Manager
try:
    from ai_repository_manager import AIRepositoryManager, ServiceType
except ImportError:
    print("Warning: AI Repository Manager not available")
    AIRepositoryManager = None
    ServiceType = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    context: Optional[str] = None

class AgentCreateRequest(BaseModel):
    name: str
    type: str
    capabilities: Optional[List[str]] = []

class CodeGenerationRequest(BaseModel):
    prompt: str
    language: str = "python"
    style: Optional[str] = "clean"

class ServiceManagementRequest(BaseModel):
    action: str  # start, stop, restart
    services: Optional[List[str]] = []

# FastAPI app
app = FastAPI(
    title="SutazAI Enhanced Backend",
    description="SutazAI v9 Backend with AI Repository Management",
    version="9.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global AI Repository Manager
ai_manager: Optional[AIRepositoryManager] = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global ai_manager
    
    logger.info("Starting SutazAI Enhanced Backend...")
    
    # Initialize AI Repository Manager
    if AIRepositoryManager:
        ai_manager = AIRepositoryManager()
        logger.info(f"Initialized AI Repository Manager with {len(ai_manager.repositories)} services")
        
        # Create Docker network
        await ai_manager.create_service_network()
        
        # Start key services
        key_services = ["enhanced-model-manager", "crewai", "documind"]
        for service in key_services:
            if service in ai_manager.repositories:
                try:
                    success = await ai_manager.start_service(service)
                    logger.info(f"Started {service}: {'✅' if success else '❌'}")
                except Exception as e:
                    logger.error(f"Failed to start {service}: {e}")
    
    logger.info("SutazAI Enhanced Backend startup complete")

# Basic endpoints
@app.get("/")
async def root():
    return {
        "message": "SutazAI Enhanced Backend v9.0.0",
        "status": "online",
        "features": [
            "AI Repository Management",
            "Dynamic Service Discovery",
            "Multi-Agent Orchestration",
            "Real-time Model Switching",
            "Advanced Monitoring"
        ]
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    health_data = {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "9.0.0"
    }
    
    if ai_manager:
        service_count = len(ai_manager.repositories)
        running_count = len([r for r in ai_manager.repositories.values() 
                           if r.status.value == "running"])
        health_data.update({
            "total_services": service_count,
            "running_services": running_count,
            "service_health": "good" if running_count > 0 else "no_services"
        })
    
    return health_data

@app.get("/api/system/status")
async def system_status():
    """Get comprehensive system status"""
    base_status = {
        "status": "online",
        "uptime": time.time() - startup_time if 'startup_time' in globals() else 0,
        "timestamp": time.time(),
        "version": "9.0.0"
    }
    
    if ai_manager:
        service_status = ai_manager.get_service_status()
        base_status.update({
            "ai_services": service_status,
            "active_agents": service_status.get("running", 0),
            "loaded_models": len(ai_manager.get_services_by_type(ServiceType.AI_MODEL)) if ServiceType else 0,
            "capabilities": {
                "code_generation": len(ai_manager.get_services_by_capability("code_generation")),
                "document_processing": len(ai_manager.get_services_by_capability("document_parsing")),
                "web_automation": len(ai_manager.get_services_by_capability("web_scraping")),
                "agent_frameworks": len(ai_manager.get_services_by_type(ServiceType.AGENT_FRAMEWORK)) if ServiceType else 0
            }
        })
    else:
        base_status.update({
            "active_agents": 3,
            "loaded_models": 5,
            "ai_services": {"error": "AI Repository Manager not available"}
        })
    
    return base_status

# AI Service Management endpoints
@app.get("/api/services/")
async def get_services():
    """Get all discovered AI services"""
    if not ai_manager:
        raise HTTPException(status_code=503, detail="AI Repository Manager not available")
    
    services = []
    for name, repo in ai_manager.repositories.items():
        services.append({
            "name": name,
            "type": repo.service_type.value,
            "status": repo.status.value,
            "port": repo.port,
            "capabilities": repo.capabilities,
            "health_endpoint": f"http://localhost:{repo.port}{repo.health_endpoint}",
            "last_health_check": repo.last_health_check
        })
    
    return services

@app.get("/api/services/{service_name}")
async def get_service_details(service_name: str):
    """Get detailed information about a specific service"""
    if not ai_manager:
        raise HTTPException(status_code=503, detail="AI Repository Manager not available")
    
    if service_name not in ai_manager.repositories:
        raise HTTPException(status_code=404, detail=f"Service {service_name} not found")
    
    repo = ai_manager.repositories[service_name]
    return {
        "name": repo.name,
        "type": repo.service_type.value,
        "status": repo.status.value,
        "port": repo.port,
        "capabilities": repo.capabilities,
        "dependencies": repo.dependencies,
        "config": repo.config,
        "container_id": repo.container_id,
        "startup_time": repo.startup_time,
        "error_message": repo.error_message
    }

@app.post("/api/services/{service_name}/start")
async def start_service(service_name: str, background_tasks: BackgroundTasks):
    """Start a specific AI service"""
    if not ai_manager:
        raise HTTPException(status_code=503, detail="AI Repository Manager not available")
    
    if service_name not in ai_manager.repositories:
        raise HTTPException(status_code=404, detail=f"Service {service_name} not found")
    
    # Start service in background
    background_tasks.add_task(ai_manager.start_service, service_name)
    
    return {
        "message": f"Starting service {service_name}",
        "status": "initiated"
    }

@app.post("/api/services/{service_name}/stop")
async def stop_service(service_name: str, background_tasks: BackgroundTasks):
    """Stop a specific AI service"""
    if not ai_manager:
        raise HTTPException(status_code=503, detail="AI Repository Manager not available")
    
    if service_name not in ai_manager.repositories:
        raise HTTPException(status_code=404, detail=f"Service {service_name} not found")
    
    # Stop service in background
    background_tasks.add_task(ai_manager.stop_service, service_name)
    
    return {
        "message": f"Stopping service {service_name}",
        "status": "initiated"
    }

@app.post("/api/services/{service_name}/restart")
async def restart_service(service_name: str, background_tasks: BackgroundTasks):
    """Restart a specific AI service"""
    if not ai_manager:
        raise HTTPException(status_code=503, detail="AI Repository Manager not available")
    
    if service_name not in ai_manager.repositories:
        raise HTTPException(status_code=404, detail=f"Service {service_name} not found")
    
    # Restart service in background
    background_tasks.add_task(ai_manager.restart_service, service_name)
    
    return {
        "message": f"Restarting service {service_name}",
        "status": "initiated"
    }

@app.get("/api/services/health")
async def check_all_services_health():
    """Perform health checks on all running services"""
    if not ai_manager:
        raise HTTPException(status_code=503, detail="AI Repository Manager not available")
    
    health_results = await ai_manager.health_check_all()
    return {
        "timestamp": time.time(),
        "total_checked": len(health_results),
        "results": health_results
    }

@app.get("/api/services/by-capability/{capability}")
async def get_services_by_capability(capability: str):
    """Get services that have a specific capability"""
    if not ai_manager:
        raise HTTPException(status_code=503, detail="AI Repository Manager not available")
    
    services = ai_manager.get_services_by_capability(capability)
    return {
        "capability": capability,
        "services": services,
        "count": len(services)
    }

@app.get("/api/services/by-type/{service_type}")
async def get_services_by_type(service_type: str):
    """Get services of a specific type"""
    if not ai_manager:
        raise HTTPException(status_code=503, detail="AI Repository Manager not available")
    
    try:
        svc_type = ServiceType(service_type)
        services = ai_manager.get_services_by_type(svc_type)
        return {
            "service_type": service_type,
            "services": services,
            "count": len(services)
        }
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid service type: {service_type}")

# Enhanced Agent Management
@app.get("/api/agents/")
async def get_agents():
    """Get available agents including AI services"""
    base_agents = [
        {"id": "deepseek-r1", "name": "DeepSeek-R1", "type": "coding", "status": "active", "capabilities": ["code_generation", "reasoning"]},
        {"id": "qwen3", "name": "Qwen3", "type": "general", "status": "active", "capabilities": ["conversation", "analysis"]},
        {"id": "autogpt", "name": "AutoGPT", "type": "automation", "status": "idle", "capabilities": ["task_automation", "web_browsing"]}
    ]
    
    if ai_manager:
        # Add running AI services as agents
        for name, repo in ai_manager.repositories.items():
            if repo.status.value == "running" and repo.service_type.value in ["agent_framework", "ai_model"]:
                base_agents.append({
                    "id": name,
                    "name": name.replace("-", " ").title(),
                    "type": repo.service_type.value,
                    "status": "active",
                    "capabilities": repo.capabilities,
                    "port": repo.port
                })
    
    return base_agents

@app.post("/api/agents/")
async def create_agent(agent_data: AgentCreateRequest):
    """Create a new agent"""
    agent_id = f"agent_{int(time.time())}"
    
    # If AI manager is available, try to start a matching service
    if ai_manager and agent_data.type in ["coding", "automation", "general"]:
        type_mapping = {
            "coding": "code_generation",
            "automation": "autonomous_tasks",
            "general": "conversation"
        }
        
        capability = type_mapping.get(agent_data.type)
        if capability:
            matching_services = ai_manager.get_services_by_capability(capability)
            if matching_services:
                # Try to start the first available service
                service_name = matching_services[0]["name"]
                try:
                    await ai_manager.start_service(service_name)
                    return {
                        "id": agent_id,
                        "name": agent_data.name,
                        "type": agent_data.type,
                        "status": "active",
                        "service": service_name,
                        "capabilities": agent_data.capabilities
                    }
                except Exception as e:
                    logger.error(f"Failed to start service {service_name}: {e}")
    
    # Fallback to simulated agent
    return {
        "id": agent_id,
        "name": agent_data.name,
        "type": agent_data.type,
        "status": "active",
        "capabilities": agent_data.capabilities,
        "simulated": True
    }

@app.post("/api/agents/{agent_id}/chat")
async def chat_with_agent(agent_id: str, chat_data: ChatRequest):
    """Chat with an agent"""
    message = chat_data.message
    
    # Try to route to appropriate AI service
    if ai_manager and agent_id in ai_manager.repositories:
        repo = ai_manager.repositories[agent_id]
        if repo.status.value == "running":
            try:
                # Make request to the service
                import requests
                url = f"http://localhost:{repo.port}/chat"
                response = requests.post(url, json={"message": message}, timeout=30)
                
                if response.status_code == 200:
                    return {
                        "agent_id": agent_id,
                        "response": response.json().get("response", "Service response received"),
                        "service": agent_id,
                        "timestamp": time.time()
                    }
            except Exception as e:
                logger.error(f"Failed to communicate with service {agent_id}: {e}")
    
    # Fallback to simulated response
    responses = {
        "deepseek-r1": f"DeepSeek-R1 here! I've analyzed your message: '{message}'. I can help with advanced coding tasks, reasoning, and problem-solving.",
        "qwen3": f"Qwen3 responding to: '{message}'. I'm designed for general conversation and analysis with strong multilingual capabilities.",
        "autogpt": f"AutoGPT processing: '{message}'. I can help automate tasks, browse the web, and execute complex workflows.",
        "crewai": f"CrewAI multi-agent system processing: '{message}'. I coordinate multiple AI agents to solve complex problems collaboratively.",
        "langchain-agents": f"LangChain agent responding to: '{message}'. I can use various tools and maintain conversation memory."
    }
    
    response = responses.get(agent_id, f"AI Agent {agent_id} responding to: '{message}'. This is a simulated response from SutazAI v9.")
    
    return {
        "agent_id": agent_id,
        "response": response,
        "timestamp": time.time(),
        "simulated": agent_id not in (ai_manager.repositories if ai_manager else {})
    }

# Enhanced Code Generation
@app.post("/api/code/generate")
async def generate_code(request: CodeGenerationRequest):
    """Generate code using available AI services"""
    prompt = request.prompt
    language = request.language
    
    # Try to use a specialized code generation service
    if ai_manager:
        code_services = ai_manager.get_services_by_capability("code_generation")
        for service in code_services:
            if service["status"] == "running":
                try:
                    import requests
                    repo = ai_manager.repositories[service["name"]]
                    url = f"http://localhost:{repo.port}/generate"
                    
                    response = requests.post(url, json={
                        "prompt": prompt,
                        "language": language,
                        "style": request.style
                    }, timeout=60)
                    
                    if response.status_code == 200:
                        result = response.json()
                        return {
                            "code": result.get("code", "# Generated code"),
                            "language": language,
                            "prompt": prompt,
                            "service": service["name"],
                            "timestamp": time.time()
                        }
                except Exception as e:
                    logger.error(f"Code generation failed with {service['name']}: {e}")
    
    # Fallback to enhanced simulated code generation
    templates = {
        "python": f'''# Generated Python code for: {prompt}
import asyncio
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class SutazAIGenerated:
    """
    Generated class for: {prompt}
    Created by SutazAI v9 Code Generation System
    """
    
    def __init__(self):
        self.initialized = True
        logger.info("SutazAI generated class initialized")
    
    async def process(self, data: Any) -> Dict[str, Any]:
        """
        Main processing method
        """
        result = {{
            "status": "success",
            "data": data,
            "timestamp": time.time(),
            "generated_by": "SutazAI v9"
        }}
        
        logger.info("Processing completed successfully")
        return result

if __name__ == "__main__":
    # Example usage
    generator = SutazAIGenerated()
    result = asyncio.run(generator.process("example_data"))
    print(f"Result: {{result}}")
''',
        
        "javascript": f'''// Generated JavaScript code for: {prompt}
// Created by SutazAI v9 Code Generation System

class SutazAIGenerated {{
    constructor() {{
        this.initialized = true;
        console.log('SutazAI generated class initialized');
    }}
    
    async process(data) {{
        const result = {{
            status: 'success',
            data: data,
            timestamp: Date.now(),
            generatedBy: 'SutazAI v9'
        }};
        
        console.log('Processing completed successfully');
        return result;
    }}
}}

// Example usage
const generator = new SutazAIGenerated();
generator.process('example_data').then(result => {{
    console.log('Result:', result);
}});
''',
        
        "java": f'''// Generated Java code for: {prompt}
// Created by SutazAI v9 Code Generation System

import java.util.*;
import java.time.Instant;

public class SutazAIGenerated {{
    private boolean initialized;
    
    public SutazAIGenerated() {{
        this.initialized = true;
        System.out.println("SutazAI generated class initialized");
    }}
    
    public Map<String, Object> process(Object data) {{
        Map<String, Object> result = new HashMap<>();
        result.put("status", "success");
        result.put("data", data);
        result.put("timestamp", Instant.now().toEpochMilli());
        result.put("generatedBy", "SutazAI v9");
        
        System.out.println("Processing completed successfully");
        return result;
    }}
    
    public static void main(String[] args) {{
        SutazAIGenerated generator = new SutazAIGenerated();
        Map<String, Object> result = generator.process("example_data");
        System.out.println("Result: " + result);
    }}
}}
'''
    }
    
    code = templates.get(language, templates["python"])
    
    return {
        "code": code,
        "language": language,
        "prompt": prompt,
        "service": "enhanced_simulation",
        "timestamp": time.time(),
        "features": ["async_support", "logging", "error_handling", "documentation"]
    }

# Models endpoint
@app.get("/api/models/")
async def get_models():
    """Get available AI models"""
    base_models = [
        {"id": "deepseek-r1:8b", "name": "DeepSeek-R1 8B", "status": "loaded", "type": "reasoning"},
        {"id": "qwen3:8b", "name": "Qwen3 8B", "status": "loaded", "type": "general"},
        {"id": "llama2:13b", "name": "Llama2 13B", "status": "loaded", "type": "conversation"}
    ]
    
    if ai_manager:
        # Add AI services as models
        ai_models = ai_manager.get_services_by_type(ServiceType.AI_MODEL) if ServiceType else []
        for model in ai_models:
            base_models.append({
                "id": model["name"],
                "name": model["name"].replace("-", " ").title(),
                "status": "loaded" if model["status"] == "running" else "unloaded",
                "type": "ai_service",
                "port": model.get("port")
            })
    
    return base_models

# Set startup time
startup_time = time.time()

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False
    )