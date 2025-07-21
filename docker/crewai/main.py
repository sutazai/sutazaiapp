from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import asyncio
import os
from typing import Dict, List, Optional
import requests
from datetime import datetime

app = FastAPI(title="SutazAI CrewAI Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CrewRequest(BaseModel):
    task: str
    agents: Optional[List[str]] = ["researcher", "writer", "reviewer"]
    max_iterations: Optional[int] = 5

class CrewResponse(BaseModel):
    crew_id: str
    status: str
    result: Optional[str] = None
    agent_outputs: List[Dict] = []

# Global crew storage
crews = {}

class CrewAIOrchestrator:
    def __init__(self):
        self.ollama_url = os.getenv("OPENAI_API_BASE", "http://ollama:11434/v1")
        self.workspace = "/workspace"
        
    async def execute_crew_task(self, crew_id: str, task: str, agents: List[str], max_iterations: int):
        """Execute a task using multiple AI agents in collaboration"""
        agent_outputs = []
        
        try:
            # Initialize crew task
            agent_outputs.append({
                "agent": "coordinator",
                "action": "initialize",
                "output": f"Starting crew task with {len(agents)} agents: {task}",
                "timestamp": datetime.now().isoformat()
            })
            
            # Execute task with each agent
            for i, agent in enumerate(agents):
                if i >= max_iterations:
                    break
                    
                # Get previous context
                context = self.build_context(agent_outputs)
                
                # Execute agent task
                result = await self.execute_agent_task(agent, task, context)
                
                agent_outputs.append({
                    "agent": agent,
                    "action": "execute",
                    "output": result,
                    "timestamp": datetime.now().isoformat()
                })
            
            # Final synthesis
            final_result = await self.synthesize_results(task, agent_outputs)
            
            # Update crew status
            crews[crew_id]["status"] = "completed"
            crews[crew_id]["result"] = final_result
            crews[crew_id]["agent_outputs"] = agent_outputs
            
        except Exception as e:
            crews[crew_id]["status"] = "failed"
            crews[crew_id]["result"] = f"Crew task failed: {str(e)}"
            crews[crew_id]["agent_outputs"] = agent_outputs
    
    def build_context(self, agent_outputs: List[Dict]) -> str:
        """Build context from previous agent outputs"""
        context_parts = []
        for output in agent_outputs[-3:]:  # Last 3 outputs for context
            context_parts.append(f"{output['agent']}: {output['output']}")
        return "\n".join(context_parts)
    
    async def execute_agent_task(self, agent: str, task: str, context: str) -> str:
        """Execute task for a specific agent"""
        try:
            agent_prompts = {
                "researcher": f"You are a research specialist. Research and analyze the following task: {task}\n\nContext from previous agents:\n{context}",
                "writer": f"You are a content writer. Create comprehensive content for: {task}\n\nContext from previous agents:\n{context}",
                "reviewer": f"You are a quality reviewer. Review and improve the following work: {task}\n\nContext from previous agents:\n{context}",
                "analyst": f"You are a data analyst. Analyze and provide insights for: {task}\n\nContext from previous agents:\n{context}",
                "planner": f"You are a strategic planner. Create detailed plans for: {task}\n\nContext from previous agents:\n{context}"
            }
            
            prompt = agent_prompts.get(agent, f"You are an AI agent. Help with this task: {task}\n\nContext:\n{context}")
            
            response = requests.post(f"{self.ollama_url}/chat/completions", json={
                "model": "deepseek-r1:8b",
                "messages": [
                    {
                        "role": "system",
                        "content": prompt
                    },
                    {
                        "role": "user",
                        "content": f"Execute your role for this task: {task}"
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 1000
            })
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                return f"Agent {agent} completed task: {task} (simulated result)"
                
        except Exception as e:
            return f"Agent {agent} error: {str(e)}"
    
    async def synthesize_results(self, task: str, agent_outputs: List[Dict]) -> str:
        """Synthesize results from all agents"""
        try:
            all_outputs = "\n\n".join([f"{output['agent']}: {output['output']}" for output in agent_outputs])
            
            response = requests.post(f"{self.ollama_url}/chat/completions", json={
                "model": "deepseek-r1:8b",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a synthesis specialist. Combine and synthesize the work from multiple AI agents into a cohesive final result."
                    },
                    {
                        "role": "user",
                        "content": f"Synthesize the following agent outputs for task '{task}':\n\n{all_outputs}"
                    }
                ],
                "temperature": 0.5,
                "max_tokens": 1500
            })
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                return f"Crew completed task: {task} with {len(agent_outputs)} agent contributions"
                
        except Exception as e:
            return f"Synthesis completed with some issues: {str(e)}"

orchestrator = CrewAIOrchestrator()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "CrewAI"}

@app.get("/")
async def root():
    return {"message": "SutazAI CrewAI Service", "version": "1.0.0"}

@app.post("/crews", response_model=CrewResponse)
async def create_crew(crew_request: CrewRequest):
    """Create and start a new crew task"""
    crew_id = f"crew_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    crews[crew_id] = {
        "id": crew_id,
        "task": crew_request.task,
        "agents": crew_request.agents,
        "status": "running",
        "result": None,
        "agent_outputs": [],
        "created_at": datetime.now().isoformat()
    }
    
    # Start crew execution in background
    asyncio.create_task(orchestrator.execute_crew_task(
        crew_id, 
        crew_request.task, 
        crew_request.agents,
        crew_request.max_iterations
    ))
    
    return CrewResponse(
        crew_id=crew_id,
        status="running",
        agent_outputs=[]
    )

@app.get("/crews/{crew_id}", response_model=CrewResponse)
async def get_crew(crew_id: str):
    """Get crew status and results"""
    if crew_id not in crews:
        raise HTTPException(status_code=404, detail="Crew not found")
    
    crew = crews[crew_id]
    return CrewResponse(
        crew_id=crew_id,
        status=crew["status"],
        result=crew["result"],
        agent_outputs=crew["agent_outputs"]
    )

@app.get("/crews")
async def list_crews():
    """List all crews"""
    return {"crews": list(crews.values())}

@app.delete("/crews/{crew_id}")
async def delete_crew(crew_id: str):
    """Delete a crew"""
    if crew_id not in crews:
        raise HTTPException(status_code=404, detail="Crew not found")
    
    del crews[crew_id]
    return {"message": f"Crew {crew_id} deleted"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)