import asyncio
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import psutil
import requests
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.multi_agent_orchestrator import MultiAgentOrchestrator

app = FastAPI(
    title="SutazAI Unified Backend",
    version="9.0.0",
    description="Integrated AI backend with multi-agent orchestration."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://frontend:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = MultiAgentOrchestrator()

class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = "llama3"

class ChatResponse(BaseModel):
    response: str
    model: str
    timestamp: str

class AgentTaskRequest(BaseModel):
    description: str
    type: str
    priority: int = 5
    metadata: Optional[Dict] = None

class AgentTaskResponse(BaseModel):
    task_id: str
    status: str

@app.on_event("startup")
async def startup_event():
    print("[INFO] Starting SutazAI Unified Backend...")
    asyncio.create_task(orchestrator.start_orchestration())
    print("[INFO] Multi-agent orchestrator is running.")

@app.on_event("shutdown")
async def shutdown_event():
    print("[INFO] Stopping multi-agent orchestrator...")
    await orchestrator.stop_orchestration()
    print("[INFO] Orchestrator stopped.")

@app.get("/health", summary="Health Check")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/metrics", summary="Prometheus Metrics")
async def get_metrics():
    try:
        cpu_usage = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        agent_count = len(orchestrator.agent_manager.get_all_agents())
        task_count = len(orchestrator.task_manager.get_all_tasks())
        pending_task_count = len(orchestrator.task_manager.get_pending_tasks())
        idle_agent_count = len(orchestrator.agent_manager.get_idle_agents())

        metrics = f"""# HELP sutazai_cpu_usage_percent CPU usage percentage
# TYPE sutazai_cpu_usage_percent gauge
sutazai_cpu_usage_percent {cpu_usage}

# HELP sutazai_memory_usage_bytes Memory usage in bytes
# TYPE sutazai_memory_usage_bytes gauge
sutazai_memory_usage_bytes {memory.used}

# HELP sutazai_memory_total_bytes Total memory in bytes
# TYPE sutazai_memory_total_bytes gauge
sutazai_memory_total_bytes {memory.total}

# HELP sutazai_disk_usage_bytes Disk usage in bytes
# TYPE sutazai_disk_usage_bytes gauge
sutazai_disk_usage_bytes {disk.used}

# HELP sutazai_disk_total_bytes Total disk space in bytes
# TYPE sutazai_disk_total_bytes gauge
sutazai_disk_total_bytes {disk.total}

# HELP sutazai_agents_total Total number of agents
# TYPE sutazai_agents_total gauge
sutazai_agents_total {agent_count}

# HELP sutazai_agents_idle Number of idle agents
# TYPE sutazai_agents_idle gauge
sutazai_agents_idle {idle_agent_count}

# HELP sutazai_tasks_total Total number of tasks
# TYPE sutazai_tasks_total gauge
sutazai_tasks_total {task_count}

# HELP sutazai_tasks_pending Number of pending tasks
# TYPE sutazai_tasks_pending gauge
sutazai_tasks_pending {pending_task_count}

# HELP sutazai_uptime_seconds System uptime in seconds
# TYPE sutazai_uptime_seconds counter
sutazai_uptime_seconds {time.time() - psutil.boot_time()}
"""
        return Response(content=metrics, media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to collect metrics: {str(e)}")

@app.get("/api/models", summary="Get Available Models")
async def get_models():
    try:
        response = requests.get(f"{orchestrator.services['ollama']}/api/tags", timeout=5)
        response.raise_for_status()
        return response.json()
    except (requests.exceptions.RequestException, HTTPException) as e:
        print(f"[ERROR] Could not fetch models from Ollama: {e}")
        raise HTTPException(status_code=503, detail="Ollama service is unavailable.")

@app.post("/api/chat", response_model=ChatResponse, summary="Chat with an AI Model")
async def chat(request: ChatRequest):
    task_id = await orchestrator.submit_task(
        description=request.message,
        task_type="conversation",
        metadata={"model": request.model}
    )

    while True:
        task = orchestrator.task_manager.get_task(task_id)
        if task.status in ["completed", "error"]:
            break
        await asyncio.sleep(0.1)

    if task.status == "error":
        raise HTTPException(status_code=500, detail=task.result.get("error"))

    return ChatResponse(
        response=task.result.get("response", ""),
        model=request.model,
        timestamp=datetime.now().isoformat(),
    )

@app.get("/api/system/status", summary="Get System Status")
async def get_system_status():
    return orchestrator.get_system_metrics()

@app.get("/api/agents", summary="Get Agent Status")
async def get_agent_status():
    return [agent.__dict__ for agent in orchestrator.agent_manager.get_all_agents()]

@app.post("/api/tasks", response_model=AgentTaskResponse, summary="Submit a Task")
async def submit_task(request: AgentTaskRequest):
    task_id = await orchestrator.submit_task(
        description=request.description,
        task_type=request.type,
        priority=request.priority,
        metadata=request.metadata
    )
    return AgentTaskResponse(task_id=task_id, status="submitted")

@app.get("/api/tasks/{task_id}", summary="Get Task Status")
async def get_task_status(task_id: str):
    task = orchestrator.task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    return task.__dict__

if __name__ == "__main__":
    print("Starting SutazAI Unified Backend v9.0")
    uvicorn.run(app, host="0.0.0.0", port=8000)
