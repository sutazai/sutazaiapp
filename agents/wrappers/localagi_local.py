#!/usr/bin/env python3
"""
LocalAGI Wrapper - Local AI Orchestration
"""

import os
import sys
import json
from typing import Dict, Any, List
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from base_agent_wrapper import BaseAgentWrapper, ChatRequest

class LocalAGI(BaseAgentWrapper):
    """LocalAGI wrapper for AI orchestration"""
    
    def __init__(self):
        super().__init__(
            agent_name="LocalAGI",
            agent_description="Local AI orchestration and coordination",
            port=8000
        )
        self.agents = {}
        self.workflows = []
        self.setup_localagi_routes()
    
    def setup_localagi_routes(self):
        """Setup LocalAGI-specific routes"""
        
        @self.app.post("/orchestrate")
        async def orchestrate(request: Dict[str, Any]):
            """Orchestrate AI agents"""
            try:
                task = request.get("task", "")
                agents_needed = request.get("agents", [])
                
                orchestration_prompt = f"""As LocalAGI, orchestrate this task: {task}
                Available agents: {agents_needed}
                Create a workflow to coordinate these agents."""
                
                chat_request = ChatRequest(
                    messages=[
                        {"role": "system", "content": "You are LocalAGI, orchestrating AI agents."},
                        {"role": "user", "content": orchestration_prompt}
                    ]
                )
                
                response = await self.generate_completion(chat_request)
                workflow = response.choices[0]["message"]["content"]
                
                workflow_id = f"workflow_{datetime.now().timestamp()}"
                self.workflows.append({
                    "id": workflow_id,
                    "task": task,
                    "agents": agents_needed,
                    "workflow": workflow,
                    "status": "created"
                })
                
                return {
                    "success": True,
                    "workflow_id": workflow_id,
                    "workflow": workflow
                }
                
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        @self.app.post("/agent/register")
        async def register_agent(request: Dict[str, Any]):
            """Register an agent with LocalAGI"""
            agent_id = request.get("id")
            capabilities = request.get("capabilities", [])
            
            self.agents[agent_id] = {
                "id": agent_id,
                "capabilities": capabilities,
                "status": "active",
                "registered_at": datetime.now().isoformat()
            }
            
            return {"success": True, "agent": self.agents[agent_id]}
        
        @self.app.get("/agents")
        async def list_agents():
            """List registered agents"""
            return {"agents": self.agents}
        
        @self.app.post("/coordinate")
        async def coordinate(request: Dict[str, Any]):
            """Coordinate multiple agents for a task"""
            try:
                task = request.get("task")
                required_capabilities = request.get("capabilities", [])
                
                # Find suitable agents
                suitable_agents = [
                    agent for agent in self.agents.values()
                    if any(cap in agent["capabilities"] for cap in required_capabilities)
                ]
                
                coordination_prompt = f"""Coordinate these agents for task: {task}
                Agents available: {[a['id'] for a in suitable_agents]}
                Required capabilities: {required_capabilities}"""
                
                chat_request = ChatRequest(
                    messages=[
                        {"role": "system", "content": "You are LocalAGI coordinating agents."},
                        {"role": "user", "content": coordination_prompt}
                    ]
                )
                
                response = await self.generate_completion(chat_request)
                plan = response.choices[0]["message"]["content"]
                
                return {
                    "success": True,
                    "task": task,
                    "agents_assigned": suitable_agents,
                    "coordination_plan": plan
                }
                
            except Exception as e:
                return {"success": False, "error": str(e)}

def main():
    """Run LocalAGI"""
    agent = LocalAGI()
    agent.run()

if __name__ == "__main__":
    main()