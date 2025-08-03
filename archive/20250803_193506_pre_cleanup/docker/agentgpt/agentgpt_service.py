#!/usr/bin/env python3
"""
AgentGPT Service for SutazAI
Autonomous AI agent that can create and execute tasks
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SutazAI AgentGPT Service", version="1.0.0")

class AgentRequest(BaseModel):
    goal: str
    agent_name: str = "AutoAgent"
    max_iterations: int = 3

class TaskRequest(BaseModel):
    task: str
    context: str = ""

class AgentGPTService:
    """AgentGPT autonomous agent service"""
    
    def __init__(self):
        self.ollama_url = "http://ollama:11434"
        self.service_name = "AgentGPT"
        self.active_agents = {}
        
    async def create_agent(self, goal: str, agent_name: str = "AutoAgent", max_iterations: int = 3) -> Dict[str, Any]:
        """Create and start an autonomous agent"""
        try:
            logger.info(f"Creating agent {agent_name} with goal: {goal}")
            
            agent_id = f"agent_{len(self.active_agents)}_{int(datetime.now().timestamp())}"
            agent = {
                "id": agent_id,
                "name": agent_name,
                "goal": goal,
                "max_iterations": max_iterations,
                "current_iteration": 0,
                "status": "active",
                "created_at": datetime.now().isoformat(),
                "tasks": [],
                "completed_tasks": [],
                "thoughts": []
            }
            
            self.active_agents[agent_id] = agent
            
            # Start agent execution
            result = await self._execute_agent(agent_id)
            
            return {
                "success": True,
                "agent_id": agent_id,
                "agent_name": agent_name,
                "initial_result": result,
                "message": f"Agent {agent_name} created and started"
            }
            
        except Exception as e:
            logger.error(f"Agent creation failed: {e}")
            return {
                "success": False,
                "error": f"Agent creation failed: {str(e)}"
            }
    
    async def _execute_agent(self, agent_id: str) -> Dict[str, Any]:
        """Execute agent autonomously"""
        try:
            agent = self.active_agents[agent_id]
            results = []
            
            while agent["current_iteration"] < agent["max_iterations"] and agent["status"] == "active":
                logger.info(f"Agent {agent['name']} - Iteration {agent['current_iteration'] + 1}")
                
                # Think about next action
                thought = await self._agent_think(agent)
                agent["thoughts"].append(thought)
                
                # Create task based on thought
                task = await self._create_task(agent, thought)
                agent["tasks"].append(task)
                
                # Execute task
                task_result = await self._execute_task(agent, task)
                agent["completed_tasks"].append({
                    "task": task,
                    "result": task_result,
                    "iteration": agent["current_iteration"] + 1
                })
                
                results.append({
                    "iteration": agent["current_iteration"] + 1,
                    "thought": thought,
                    "task": task,
                    "result": task_result
                })
                
                agent["current_iteration"] += 1
                
                # Check if goal is achieved
                if await self._is_goal_achieved(agent):
                    agent["status"] = "completed"
                    break
                    
                # Small delay between iterations
                await asyncio.sleep(1)
            
            if agent["status"] == "active":
                agent["status"] = "max_iterations_reached"
            
            return {
                "agent_id": agent_id,
                "final_status": agent["status"],
                "iterations_completed": agent["current_iteration"],
                "results": results,
                "summary": await self._generate_summary(agent)
            }
            
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            return {
                "error": f"Agent execution failed: {str(e)}"
            }
    
    async def _agent_think(self, agent: Dict[str, Any]) -> str:
        """Agent thinking process"""
        try:
            context = f"Goal: {agent['goal']}\n"
            context += f"Current iteration: {agent['current_iteration'] + 1}/{agent['max_iterations']}\n"
            context += f"Previous thoughts: {agent['thoughts'][-3:] if agent['thoughts'] else 'None'}\n"
            context += f"Completed tasks: {len(agent['completed_tasks'])}\n"
            
            prompt = f"""You are an autonomous AI agent named {agent['name']}.
            
{context}

Think about what you should do next to achieve your goal. Consider:
1. What specific action would move you closer to the goal?
2. What information do you need?
3. What obstacles might you face?

Provide a clear, actionable thought in 1-2 sentences."""

            response = await self._call_llm(prompt)
            return response
            
        except Exception as e:
            return f"Thinking error: {str(e)}"
    
    async def _create_task(self, agent: Dict[str, Any], thought: str) -> str:
        """Create a specific task based on agent's thought"""
        try:
            prompt = f"""Based on this thought: "{thought}"
            
Create a specific, actionable task that will help achieve the goal: "{agent['goal']}"

The task should be:
- Concrete and specific
- Achievable with available tools
- One clear action

Task:"""

            response = await self._call_llm(prompt)
            return response.strip()
            
        except Exception as e:
            return f"Create task for: {thought}"
    
    async def _execute_task(self, agent: Dict[str, Any], task: str) -> str:
        """Execute a specific task"""
        try:
            prompt = f"""Execute this task: "{task}"

This task is part of achieving the goal: "{agent['goal']}"

Simulate executing this task and provide a realistic result. Consider:
- What would actually happen if this task was performed?
- What information would be gathered?
- What next steps might be needed?

Result:"""

            response = await self._call_llm(prompt)
            return response
            
        except Exception as e:
            return f"Task execution simulation: {task}"
    
    async def _is_goal_achieved(self, agent: Dict[str, Any]) -> bool:
        """Check if the agent's goal has been achieved"""
        try:
            if len(agent["completed_tasks"]) < 2:
                return False
                
            summary = "\n".join([f"Task: {t['task']}\nResult: {t['result']}" for t in agent["completed_tasks"]])
            
            prompt = f"""Goal: {agent['goal']}

Completed work:
{summary}

Has this goal been achieved based on the completed tasks? Answer only 'YES' or 'NO' with a brief explanation."""

            response = await self._call_llm(prompt)
            return response.lower().startswith('yes')
            
        except Exception as e:
            return False
    
    async def _generate_summary(self, agent: Dict[str, Any]) -> str:
        """Generate a summary of agent's work"""
        try:
            tasks_summary = "\n".join([f"- {t['task']}: {t['result'][:100]}..." for t in agent["completed_tasks"]])
            
            prompt = f"""Summarize the work done by agent {agent['name']} to achieve the goal: "{agent['goal']}"

Tasks completed:
{tasks_summary}

Status: {agent['status']}
Iterations: {agent['current_iteration']}

Provide a concise summary of what was accomplished."""

            response = await self._call_llm(prompt)
            return response
            
        except Exception as e:
            return f"Agent completed {len(agent['completed_tasks'])} tasks in {agent['current_iteration']} iterations"
    
    async def _call_llm(self, prompt: str) -> str:
        """Call local LLM for agent reasoning"""
        try:
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
                return result.get("response", "Unable to generate response")
            else:
                return "LLM service unavailable"
                
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"Error: {str(e)}"
    
    def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get status of specific agent"""
        if agent_id not in self.active_agents:
            return {"error": "Agent not found"}
        
        agent = self.active_agents[agent_id]
        return {
            "agent_id": agent_id,
            "name": agent["name"],
            "goal": agent["goal"],
            "status": agent["status"],
            "current_iteration": agent["current_iteration"],
            "max_iterations": agent["max_iterations"],
            "tasks_completed": len(agent["completed_tasks"]),
            "last_thought": agent["thoughts"][-1] if agent["thoughts"] else None
        }
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """List all active agents"""
        return [self.get_agent_status(agent_id) for agent_id in self.active_agents.keys()]

# Initialize service
agentgpt_service = AgentGPTService()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "AgentGPT",
        "active_agents": len(agentgpt_service.active_agents),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/create_agent")
async def create_agent(request: AgentRequest):
    """Create and start an autonomous agent"""
    try:
        result = await agentgpt_service.create_agent(
            request.goal,
            request.agent_name,
            request.max_iterations
        )
        
        return {
            "success": result.get("success", True),
            "result": result,
            "service": "AgentGPT",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Create agent failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents/{agent_id}/status")
async def get_agent_status(agent_id: str):
    """Get status of specific agent"""
    result = agentgpt_service.get_agent_status(agent_id)
    return {
        "result": result,
        "service": "AgentGPT"
    }

@app.get("/agents")
async def list_agents():
    """List all agents"""
    return {
        "agents": agentgpt_service.list_agents(),
        "service": "AgentGPT"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "AgentGPT Autonomous Agent System",
        "status": "online",
        "version": "1.0.0",
        "description": "Autonomous AI agents for goal achievement"
    }