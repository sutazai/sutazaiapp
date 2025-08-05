#!/usr/bin/env python3
"""
FastAPI app for agent-orchestrator
"""

import os
import sys
from pathlib import Path
from fastapi import FastAPI
from datetime import datetime
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Create FastAPI app
app = FastAPI(
    title="Agent Orchestrator",
    description="Specialized agent for coordinating and orchestrating tasks",
    version="1.0.0"
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agent": "agent-orchestrator",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "agent": "agent-orchestrator",
        "status": "running",
        "description": "Specialized agent for coordinating and orchestrating tasks"
    }

@app.post("/task")
async def process_task(task: Dict[str, Any]):
    """Process incoming tasks"""
    try:
        task_type = task.get("type", "unknown")
        
        if task_type == "health":
            return {"status": "healthy", "agent": "agent-orchestrator"}
        
        # Process task with Ollama model
        model = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
        result = {
            "message": f"Processed by Agent Orchestrator using model {model}",
            "task": task
        }
        
        return {
            "status": "success",
            "result": result,
            "agent": "agent-orchestrator"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "agent": "agent-orchestrator"
        }

# For direct execution
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
