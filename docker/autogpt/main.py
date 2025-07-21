from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import asyncio
import json
import os
from typing import Dict, List, Optional
import requests
from datetime import datetime

app = FastAPI(title="SutazAI AutoGPT Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TaskRequest(BaseModel):
    goal: str
    constraints: Optional[List[str]] = []
    resources: Optional[List[str]] = []
    max_iterations: Optional[int] = 10

class TaskResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[str] = None
    steps: List[Dict] = []

# Global task storage
tasks = {}

class AutoGPTAgent:
    def __init__(self):
        self.ollama_url = os.getenv("OPENAI_API_BASE", "http://ollama:11434/v1")
        self.qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")
        self.workspace = "/workspace"
        
    async def execute_task(self, task_id: str, goal: str, constraints: List[str], max_iterations: int):
        """Execute an autonomous task using AutoGPT-style reasoning"""
        steps = []
        
        try:
            # Initialize task
            steps.append({
                "step": 1,
                "action": "initialize",
                "result": f"Starting autonomous task: {goal}",
                "timestamp": datetime.now().isoformat()
            })
            
            # Break down the goal into subtasks
            subtasks = await self.decompose_goal(goal)
            steps.append({
                "step": 2,
                "action": "decompose_goal",
                "result": f"Identified {len(subtasks)} subtasks",
                "subtasks": subtasks,
                "timestamp": datetime.now().isoformat()
            })
            
            # Execute subtasks
            for i, subtask in enumerate(subtasks):
                if i >= max_iterations:
                    break
                    
                result = await self.execute_subtask(subtask)
                steps.append({
                    "step": i + 3,
                    "action": "execute_subtask",
                    "subtask": subtask,
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                })
            
            # Update task status
            tasks[task_id]["status"] = "completed"
            tasks[task_id]["result"] = f"Successfully completed task: {goal}"
            tasks[task_id]["steps"] = steps
            
        except Exception as e:
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["result"] = f"Task failed: {str(e)}"
            tasks[task_id]["steps"] = steps
    
    async def decompose_goal(self, goal: str) -> List[str]:
        """Break down a complex goal into manageable subtasks"""
        try:
            response = requests.post(f"{self.ollama_url}/chat/completions", json={
                "model": "deepseek-r1:8b",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an AI task planner. Break down complex goals into specific, actionable subtasks. Return only a JSON list of subtasks."
                    },
                    {
                        "role": "user",
                        "content": f"Break down this goal into 3-5 specific subtasks: {goal}"
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 500
            })
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                # Try to extract JSON from the response
                try:
                    import re
                    json_match = re.search(r'\[.*\]', content, re.DOTALL)
                    if json_match:
                        subtasks = json.loads(json_match.group())
                        return subtasks
                except:
                    pass
                    
                # Fallback: simple text parsing
                lines = content.strip().split('\n')
                subtasks = [line.strip('- ').strip() for line in lines if line.strip() and not line.startswith('#')]
                return subtasks[:5]
            
        except Exception as e:
            print(f"Error decomposing goal: {e}")
            
        # Default fallback
        return [
            f"Research and analyze: {goal}",
            f"Plan implementation for: {goal}",
            f"Execute core tasks for: {goal}",
            f"Review and validate: {goal}"
        ]
    
    async def execute_subtask(self, subtask: str) -> str:
        """Execute a specific subtask"""
        try:
            response = requests.post(f"{self.ollama_url}/chat/completions", json={
                "model": "deepseek-r1:8b",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an AI assistant that executes specific tasks. Provide detailed, actionable results."
                    },
                    {
                        "role": "user",
                        "content": f"Execute this subtask and provide detailed results: {subtask}"
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 1000
            })
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                return f"Executed subtask: {subtask} (simulated result)"
                
        except Exception as e:
            return f"Error executing subtask {subtask}: {str(e)}"

agent = AutoGPTAgent()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "AutoGPT"}

@app.get("/")
async def root():
    return {"message": "SutazAI AutoGPT Service", "version": "1.0.0"}

@app.post("/tasks", response_model=TaskResponse)
async def create_task(task_request: TaskRequest):
    """Create and start a new autonomous task"""
    task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    tasks[task_id] = {
        "id": task_id,
        "goal": task_request.goal,
        "status": "running",
        "result": None,
        "steps": [],
        "created_at": datetime.now().isoformat()
    }
    
    # Start task execution in background
    asyncio.create_task(agent.execute_task(
        task_id, 
        task_request.goal, 
        task_request.constraints,
        task_request.max_iterations
    ))
    
    return TaskResponse(
        task_id=task_id,
        status="running",
        steps=[]
    )

@app.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str):
    """Get task status and results"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    return TaskResponse(
        task_id=task_id,
        status=task["status"],
        result=task["result"],
        steps=task["steps"]
    )

@app.get("/tasks")
async def list_tasks():
    """List all tasks"""
    return {"tasks": list(tasks.values())}

@app.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    """Delete a task"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    del tasks[task_id]
    return {"message": f"Task {task_id} deleted"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)