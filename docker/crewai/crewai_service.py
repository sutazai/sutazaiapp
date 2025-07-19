#\!/usr/bin/env python3
"""
CrewAI Multi-Agent Service for SutazAI
Provides multi-agent collaboration and task orchestration
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SutazAI CrewAI Service", version="1.0.0")

class TaskRequest(BaseModel):
    task: str
    agents: List[str] = ["researcher", "writer", "analyst"]
    task_type: str = "collaboration"
    parameters: Dict[str, Any] = {}

class CrewAIService:
    """CrewAI multi-agent service implementation"""
    
    def __init__(self):
        self.ollama_url = "http://ollama:11434"
        self.service_name = "CrewAI"
        self.available_agents = {
            "researcher": {
                "role": "Research Specialist",
                "goal": "Gather comprehensive information on any topic",
                "backstory": "Expert researcher with deep analytical skills"
            },
            "writer": {
                "role": "Content Writer", 
                "goal": "Create clear, engaging content",
                "backstory": "Skilled writer who can adapt to any writing style"
            },
            "analyst": {
                "role": "Data Analyst",
                "goal": "Analyze data and provide insights",
                "backstory": "Experienced analyst with strong problem-solving skills"
            },
            "coder": {
                "role": "Software Developer",
                "goal": "Write high-quality code and solutions",
                "backstory": "Senior developer with expertise in multiple languages"
            },
            "manager": {
                "role": "Project Manager",
                "goal": "Coordinate tasks and ensure successful completion",
                "backstory": "Experienced manager with strong leadership skills"
            }
        }
        
    async def execute_crew_task(self, task: str, agent_names: List[str] = None, **kwargs) -> Dict[str, Any]:
        """Execute a task using multi-agent simulation"""
        try:
            logger.info(f"CrewAI executing task: {task}")
            
            if not agent_names:
                agent_names = ["researcher", "writer", "analyst"]
            
            # Multi-agent simulation using local LLM
            agents_context = ", ".join([
                f"{name} ({self.available_agents.get(name, {}).get('role', 'Agent')})"
                for name in agent_names
            ])
            
            prompt = f"""
As a multi-agent system with specialized agents: {agents_context}, work together to address this task:

{task}

Each agent should contribute their expertise:
- Researcher: Gather and analyze information
- Writer: Structure and communicate findings clearly  
- Analyst: Provide data insights and recommendations
- Coder: Provide technical solutions if needed
- Manager: Coordinate and summarize

Provide a comprehensive collaborative response.
"""
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": "llama3.2:1b",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "result": result.get("response", "Task completed using multi-agent approach"),
                    "agents_used": agent_names,
                    "task_type": "multi_agent_collaboration",
                    "service": "CrewAI"
                }
            else:
                return {
                    "success": False,
                    "error": f"LLM request failed with status {response.status_code}"
                }
                
        except Exception as e:
            logger.error(f"CrewAI task execution failed: {e}")
            return {
                "success": False,
                "error": f"Task execution failed: {str(e)}"
            }

# Initialize service
crewai_service = CrewAIService()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "CrewAI",
        "available_agents": list(crewai_service.available_agents.keys()),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/execute")
async def execute_task(request: TaskRequest):
    """Execute a multi-agent task"""
    try:
        result = await crewai_service.execute_crew_task(
            request.task,
            request.agents,
            **request.parameters
        )
        
        return {
            "success": result.get("success", True),
            "result": result,
            "service": "CrewAI",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Task execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents")
async def list_agents():
    """List available agents"""
    return {
        "available_agents": list(crewai_service.available_agents.keys()),
        "agent_details": crewai_service.available_agents,
        "service": "CrewAI"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "CrewAI Multi-Agent System",
        "status": "online",
        "version": "1.0.0",
        "description": "Multi-agent collaboration and task orchestration for SutazAI"
    }
EOF < /dev/null
