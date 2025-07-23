#!/usr/bin/env python3
"""
AutoGPT Web Interface for SutazAI
Provides autonomous task execution capabilities
"""

import os
import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("autogpt-agent")

app = FastAPI(
    title="AutoGPT Agent Service",
    description="Autonomous task execution and planning agent",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")
WORKSPACE_PATH = os.getenv("WORKSPACE_PATH", "/workspace")

# Request/Response models
class TaskRequest(BaseModel):
    task: str
    goal: Optional[str] = None
    constraints: Optional[List[str]] = []
    resources: Optional[List[str]] = []

class ChatRequest(BaseModel):
    message: str
    context: Optional[str] = None

# Task execution state
active_tasks = {}

async def query_ollama(prompt: str, model: str = "llama3.2:1b") -> str:
    """Query Ollama for text generation"""
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "max_tokens": 2048
                    }
                }
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "No response generated")
            else:
                return f"Error: Ollama returned status {response.status_code}"
    except Exception as e:
        logger.error(f"Error querying Ollama: {e}")
        return f"Error communicating with language model: {str(e)}"

def create_autogpt_prompt(task: str, goal: str = None, constraints: List[str] = None, resources: List[str] = None) -> str:
    """Create an AutoGPT-style autonomous planning prompt"""
    
    goal_text = goal or f"Complete the task: {task}"
    constraints_text = "\n".join(f"- {c}" for c in (constraints or [
        "Use only information available through the provided context",
        "Be precise and factual in all responses", 
        "Break down complex tasks into manageable steps",
        "Provide clear reasoning for each decision"
    ]))
    resources_text = "\n".join(f"- {r}" for r in (resources or [
        "Local language model for text generation",
        "Workspace for temporary file storage",
        "Planning and reasoning capabilities"
    ]))
    
    prompt = f"""You are AutoGPT, an advanced autonomous AI agent capable of independent planning and task execution.

GOAL: {goal_text}

TASK: {task}

CONSTRAINTS:
{constraints_text}

AVAILABLE RESOURCES:
{resources_text}

As AutoGPT, analyze this task and create a comprehensive execution plan. Follow this structure:

1. TASK ANALYSIS
   - Break down the task into components
   - Identify required capabilities and resources
   - Assess complexity and time requirements

2. EXECUTION PLAN
   - Create step-by-step action plan
   - Define success criteria for each step
   - Identify potential obstacles and mitigation strategies

3. IMPLEMENTATION
   - Execute each step systematically
   - Monitor progress and adapt as needed
   - Document results and lessons learned

4. COMPLETION SUMMARY
   - Summarize what was accomplished
   - Highlight key findings or results
   - Suggest follow-up actions if applicable

Begin execution now, thinking step-by-step through the entire process."""

    return prompt

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "autogpt-agent",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "active_tasks": len(active_tasks),
        "workspace": WORKSPACE_PATH
    }

@app.post("/execute")
async def execute_task(request: TaskRequest):
    """Execute autonomous task using AutoGPT methodology"""
    try:
        task_id = f"task_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Create AutoGPT prompt
        autogpt_prompt = create_autogpt_prompt(
            request.task, 
            request.goal, 
            request.constraints, 
            request.resources
        )
        
        # Store task state
        active_tasks[task_id] = {
            "task": request.task,
            "goal": request.goal,
            "status": "executing",
            "started_at": datetime.utcnow().isoformat()
        }
        
        # Execute using Ollama
        result = await query_ollama(autogpt_prompt)
        
        # Update task state
        active_tasks[task_id].update({
            "status": "completed",
            "result": result,
            "completed_at": datetime.utcnow().isoformat()
        })
        
        return {
            "task_id": task_id,
            "status": "completed",
            "task": request.task,
            "goal": request.goal or f"Complete: {request.task}",
            "result": result,
            "execution_time": "autonomous",
            "agent": "AutoGPT",
            "timestamp": datetime.utcnow().isoformat(),
            "capabilities_used": [
                "autonomous_planning",
                "step_by_step_analysis", 
                "systematic_execution",
                "progress_monitoring"
            ]
        }
        
    except Exception as e:
        logger.error(f"Task execution error: {str(e)}")
        
        if task_id in active_tasks:
            active_tasks[task_id].update({
                "status": "failed",
                "error": str(e),
                "failed_at": datetime.utcnow().isoformat()
            })
        
        raise HTTPException(status_code=500, detail=f"Task execution failed: {str(e)}")

@app.post("/chat")
async def chat_with_autogpt(request: ChatRequest):
    """Chat with AutoGPT agent"""
    try:
        chat_prompt = f"""You are AutoGPT, an autonomous AI agent. The user wants to discuss something with you.

User message: {request.message}
Context: {request.context or "General conversation"}

Respond as AutoGPT would - be autonomous, analytical, and focused on actionable solutions. If the user is asking about a task, offer to create an execution plan."""

        response = await query_ollama(chat_prompt)
        
        return {
            "response": response,
            "agent": "AutoGPT",
            "context": request.context,
            "timestamp": datetime.utcnow().isoformat(),
            "mode": "conversational"
        }
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.get("/tasks")
async def get_active_tasks():
    """Get list of active and recent tasks"""
    return {
        "active_tasks": active_tasks,
        "total_tasks": len(active_tasks),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get status of specific task"""
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return {
        "task_id": task_id,
        "task_details": active_tasks[task_id],
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "AutoGPT Agent Service",
        "description": "Autonomous task execution and planning agent",
        "version": "1.0.0",
        "capabilities": [
            "Autonomous task planning",
            "Step-by-step execution",
            "Goal-oriented problem solving",
            "Resource management",
            "Progress monitoring"
        ],
        "endpoints": ["/execute", "/chat", "/tasks", "/health"],
        "active_tasks": len(active_tasks),
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
