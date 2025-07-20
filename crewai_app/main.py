# CrewAI Service Application
# -------------------------

import asyncio
import time
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from crewai import Agent, Task, Crew

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CrewAI Service", version="1.0.0")

class TaskRequest(BaseModel):
    task_type: str
    description: str
    expected_output: str

class CrewAIService:
    def __init__(self):
        self.researchers = {}
        self.writers = {}
        self.setup_agents()
        
    def setup_agents(self):
        """Initialize default agents"""
        self.researcher = Agent(
            role='Senior Research Analyst',
            goal='Uncover cutting-edge developments in AI and data science',
            backstory="""You work at a leading tech think tank.
            Your expertise lies in identifying emerging trends.
            You have a knack for dissecting complex data and presenting actionable insights.""",
            verbose=True,
            allow_delegation=False
        )

        self.writer = Agent(
            role='Tech Content Strategist',
            goal='Craft compelling content on tech advancements',
            backstory="""You are a renowned Content Strategist, known for your insightful and engaging articles.
            You transform complex concepts into compelling narratives.""",
            verbose=True,
            allow_delegation=True
        )
        
        logger.info("CrewAI agents initialized successfully")
    
    async def execute_task(self, task_request: TaskRequest) -> Dict[str, Any]:
        """Execute a task using CrewAI agents"""
        try:
            # Create task
            task = Task(
                description=task_request.description,
                expected_output=task_request.expected_output,
                agent=self.researcher if task_request.task_type == "research" else self.writer
            )
            
            # Create crew
            crew = Crew(
                agents=[self.researcher, self.writer],
                tasks=[task],
                verbose=2
            )
            
            # Execute task
            result = crew.kickoff()
            
            return {
                "status": "success",
                "result": str(result),
                "task_type": task_request.task_type
            }
            
        except Exception as e:
            logger.error(f"CrewAI task execution failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "task_type": task_request.task_type
            }

# Initialize service
crewai_service = CrewAIService()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "crewai"}

@app.post("/execute")
async def execute_task(task_request: TaskRequest):
    """Execute a CrewAI task"""
    result = await crewai_service.execute_task(task_request)
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["error"])
    return result

@app.get("/agents")
async def list_agents():
    """List available agents"""
    return {
        "agents": [
            {"name": "researcher", "role": "Senior Research Analyst"},
            {"name": "writer", "role": "Tech Content Strategist"}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting CrewAI service...")
    uvicorn.run(app, host="0.0.0.0", port=8080)
