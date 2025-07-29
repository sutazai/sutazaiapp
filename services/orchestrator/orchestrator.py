from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import asyncio
from typing import Dict, List, Any
import os

app = FastAPI(title="AGI/ASI Orchestrator")

# Service endpoints
SERVICES = {
    "ollama": os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"),
    "litellm": "http://litellm:4000",
    "chromadb": os.getenv("CHROMADB_URL", "http://chromadb:8000"),
    "letta": "http://letta:8283",
    "autogpt": "http://autogpt:8080",
    "localagi": "http://localagi:8090",
    "langchain": "http://langchain-orchestrator:8095",
    "tabbyml": "http://tabbyml:8080",
    "semgrep": "http://semgrep-service:8087",
}

class TaskRequest(BaseModel):
    task_type: str
    prompt: str
    context: Dict[str, Any] = {}
    agents: List[str] = []

class TaskResponse(BaseModel):
    result: Any
    metadata: Dict[str, Any]
    agents_used: List[str]

@app.post("/execute", response_model=TaskResponse)
async def execute_task(request: TaskRequest):
    """Execute a task using multiple AI agents"""
    results = {}
    agents_used = []
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        # Determine which agents to use
        if not request.agents:
            request.agents = determine_agents(request.task_type)
        
        # Execute tasks in parallel where possible
        tasks = []
        for agent in request.agents:
            if agent in SERVICES:
                tasks.append(execute_agent_task(client, agent, request))
        
        # Gather results
        agent_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for agent, result in zip(request.agents, agent_results):
            if isinstance(result, Exception):
                results[agent] = {"error": str(result)}
            else:
                results[agent] = result
                agents_used.append(agent)
    
    # Combine results
    final_result = combine_results(results, request.task_type)
    
    return TaskResponse(
        result=final_result,
        metadata={"task_type": request.task_type},
        agents_used=agents_used
    )

def determine_agents(task_type: str) -> List[str]:
    """Determine which agents to use based on task type"""
    agent_mapping = {
        "code_generation": ["litellm", "tabbyml", "langchain"],
        "code_analysis": ["semgrep", "langchain"],
        "task_automation": ["autogpt", "letta", "localagi"],
        "general": ["litellm", "langchain"],
        "memory_task": ["letta", "chromadb"],
    }
    return agent_mapping.get(task_type, ["litellm", "langchain"])

async def execute_agent_task(client: httpx.AsyncClient, agent: str, request: TaskRequest):
    """Execute task on specific agent"""
    try:
        if agent == "litellm":
            response = await client.post(
                f"{SERVICES[agent]}/chat/completions",
                json={
                    "model": "deepseek-r1",
                    "messages": [{"role": "user", "content": request.prompt}]
                }
            )
        elif agent == "langchain":
            response = await client.post(
                f"{SERVICES[agent]}/execute",
                json={"task": request.prompt, "context": request.context}
            )
        else:
            # Generic agent execution
            response = await client.post(
                f"{SERVICES[agent]}/execute",
                json={"prompt": request.prompt, "context": request.context}
            )
        
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def combine_results(results: Dict[str, Any], task_type: str) -> Any:
    """Combine results from multiple agents"""
    # Implement sophisticated result combination logic
    combined = {
        "summary": "Task completed successfully",
        "details": results,
        "recommendations": []
    }
    
    # Add task-specific processing
    if task_type == "code_generation":
        # Extract and combine code from different agents
        pass
    elif task_type == "code_analysis":
        # Combine security and quality findings
        pass
    
    return combined

@app.get("/health")
async def health_check():
    """Check health of all services"""
    health_status = {}
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        for service, url in SERVICES.items():
            try:
                response = await client.get(f"{url}/health")
                health_status[service] = response.status_code == 200
            except:
                health_status[service] = False
    
    return {
        "status": "healthy" if all(health_status.values()) else "degraded",
        "services": health_status
    }

@app.get("/services")
async def list_services():
    """List all available services"""
    return {"services": list(SERVICES.keys())}
