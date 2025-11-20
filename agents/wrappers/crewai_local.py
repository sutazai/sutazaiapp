#!/usr/bin/env python3
"""
CrewAI Wrapper - Multi-Agent Orchestration
"""

import os
import sys
import json
from typing import Dict, Any, List
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from base_agent_wrapper import BaseAgentWrapper, ChatRequest

class CrewAILocal(BaseAgentWrapper):
    """CrewAI multi-agent orchestration wrapper"""
    
    def __init__(self):
        super().__init__(
            agent_name="CrewAI",
            agent_description="Multi-agent crew orchestration and collaboration",
            port=8000
        )
        self.crews = {}
        self.agents = {}
        self.tasks = []
        self.setup_crewai_routes()
    
    def setup_crewai_routes(self):
        """Setup CrewAI-specific routes"""
        
        @self.app.get("/capabilities")
        async def get_capabilities():
            """Return CrewAI agent capabilities"""
            return {
                "agent": "CrewAI",
                "version": "1.0.0",
                "capabilities": [
                    "multi_agent_orchestration",
                    "task_delegation",
                    "agent_coordination",
                    "workflow_management",
                    "collaborative_problem_solving"
                ],
                "features": [
                    "role_based_agents",
                    "hierarchical_teams",
                    "task_distribution",
                    "result_aggregation"
                ],
                "endpoints": [
                    "/health",
                    "/capabilities",
                    "/chat",
                    "/orchestrate",
                    "/agents/list"
                ]
            }
        
        @self.app.post("/orchestrate")
        async def create_crew(request: Dict[str, Any]):
            """Create a new crew of agents"""
            crew_name = request.get("name")
            agents = request.get("agents", [])
            goal = request.get("goal", "")
            
            crew_id = f"crew_{datetime.now().timestamp()}"
            self.crews[crew_id] = {
                "id": crew_id,
                "name": crew_name,
                "agents": agents,
                "goal": goal,
                "created_at": datetime.now().isoformat(),
                "status": "ready"
            }
            
            return {
                "success": True,
                "crew_id": crew_id,
                "crew": self.crews[crew_id]
            }
        
        @self.app.post("/agent/create")
        async def create_agent(request: Dict[str, Any]):
            """Create a specialized agent"""
            agent_name = request.get("name")
            role = request.get("role")
            goal = request.get("goal")
            backstory = request.get("backstory", "")
            
            agent_id = f"agent_{datetime.now().timestamp()}"
            self.agents[agent_id] = {
                "id": agent_id,
                "name": agent_name,
                "role": role,
                "goal": goal,
                "backstory": backstory,
                "created_at": datetime.now().isoformat()
            }
            
            return {
                "success": True,
                "agent_id": agent_id,
                "agent": self.agents[agent_id]
            }
        
        @self.app.post("/task/assign")
        async def assign_task(request: Dict[str, Any]):
            """Assign a task to a crew"""
            try:
                crew_id = request.get("crew_id")
                task_description = request.get("task")
                
                if crew_id not in self.crews:
                    return {"success": False, "error": "Crew not found"}
                
                crew = self.crews[crew_id]
                
                # Generate task plan using crew agents
                planning_prompt = f"""As CrewAI, create a task plan for crew '{crew['name']}':
                Goal: {crew['goal']}
                Task: {task_description}
                Agents available: {crew['agents']}
                
                Break down the task and assign to specific agents."""
                
                chat_request = ChatRequest(
                    messages=[
                        {"role": "system", "content": "You are CrewAI orchestrating a crew of agents."},
                        {"role": "user", "content": planning_prompt}
                    ]
                )
                
                response = await self.generate_completion(chat_request)
                task_plan = response.choices[0]["message"]["content"]
                
                task_id = f"task_{datetime.now().timestamp()}"
                task_record = {
                    "id": task_id,
                    "crew_id": crew_id,
                    "description": task_description,
                    "plan": task_plan,
                    "status": "planned",
                    "created_at": datetime.now().isoformat()
                }
                
                self.tasks.append(task_record)
                
                return {
                    "success": True,
                    "task_id": task_id,
                    "task_plan": task_plan
                }
                
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        @self.app.post("/crew/execute")
        async def execute_crew(request: Dict[str, Any]):
            """Execute crew on assigned tasks"""
            try:
                crew_id = request.get("crew_id")
                
                if crew_id not in self.crews:
                    return {"success": False, "error": "Crew not found"}
                
                crew = self.crews[crew_id]
                crew_tasks = [t for t in self.tasks if t["crew_id"] == crew_id and t["status"] == "planned"]
                
                if not crew_tasks:
                    return {"success": False, "error": "No tasks assigned to crew"}
                
                results = []
                for task in crew_tasks:
                    # Simulate task execution
                    execution_prompt = f"""Execute this task as crew '{crew['name']}':
                    Task: {task['description']}
                    Plan: {task['plan']}
                    
                    Provide detailed execution results."""
                    
                    chat_request = ChatRequest(
                        messages=[
                            {"role": "system", "content": f"You are the {crew['name']} crew executing tasks."},
                            {"role": "user", "content": execution_prompt}
                        ]
                    )
                    
                    response = await self.generate_completion(chat_request)
                    result = response.choices[0]["message"]["content"]
                    
                    task["status"] = "completed"
                    task["result"] = result
                    results.append({
                        "task_id": task["id"],
                        "result": result
                    })
                
                return {
                    "success": True,
                    "crew_id": crew_id,
                    "results": results
                }
                
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        @self.app.get("/crews")
        async def list_crews():
            """List all crews"""
            return {"crews": self.crews}
        
        @self.app.get("/agents")
        async def list_agents():
            """List all agents"""
            return {"agents": self.agents}
        
        @self.app.get("/tasks")
        async def list_tasks():
            """List all tasks"""
            return {"tasks": self.tasks}

def main():
    agent = CrewAILocal()
    agent.run()

if __name__ == "__main__":
    main()