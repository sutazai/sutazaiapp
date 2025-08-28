#!/usr/bin/env python3
"""
AutoGPT Local Wrapper - Autonomous Task Execution with Local LLM
"""

import os
import sys
import json
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

# Add base wrapper to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from base_agent_wrapper import BaseAgentWrapper, ChatRequest

logger = logging.getLogger(__name__)

class AutoGPTLocal(BaseAgentWrapper):
    """AutoGPT wrapper for autonomous task execution with local LLM"""
    
    def __init__(self):
        super().__init__(
            agent_name="AutoGPT",
            agent_description="Autonomous task execution and planning agent",
            port=8000
        )
        self.tasks = []
        self.memory = []
        self.setup_autogpt_routes()
    
    def setup_autogpt_routes(self):
        """Setup AutoGPT-specific routes"""
        
        @self.app.post("/task/create")
        async def create_task(request: Dict[str, Any]):
            """Create a new autonomous task"""
            try:
                task_id = f"task_{datetime.now().timestamp()}"
                task = {
                    "id": task_id,
                    "goal": request.get("goal", ""),
                    "steps": [],
                    "status": "pending",
                    "created_at": datetime.now().isoformat()
                }
                
                # Generate task plan using LLM
                plan_prompt = f"""You are AutoGPT, an autonomous AI agent. 
                Create a detailed step-by-step plan to achieve this goal: {task['goal']}
                
                Respond with a JSON array of steps, each with 'action' and 'description' fields."""
                
                chat_request = ChatRequest(
                    messages=[
                        {"role": "system", "content": "You are AutoGPT, an autonomous task execution agent."},
                        {"role": "user", "content": plan_prompt}
                    ]
                )
                
                response = await self.generate_completion(chat_request)
                
                # Parse the response to extract steps
                try:
                    content = response.choices[0]["message"]["content"]
                    # Try to extract JSON from the response
                    import re
                    json_match = re.search(r'\[.*\]', content, re.DOTALL)
                    if json_match:
                        steps = json.loads(json_match.group())
                        task["steps"] = steps
                    else:
                        # Fallback to simple parsing
                        task["steps"] = [{"action": "analyze", "description": content}]
                except:
                    task["steps"] = [{"action": "process", "description": "Process the goal"}]
                
                self.tasks.append(task)
                return {"success": True, "task": task}
                
            except Exception as e:
                logger.error(f"Failed to create task: {e}")
                return {"success": False, "error": str(e)}
        
        @self.app.post("/task/execute")
        async def execute_task(request: Dict[str, Any]):
            """Execute a task step"""
            try:
                task_id = request.get("task_id")
                step_index = request.get("step_index", 0)
                
                # Find the task
                task = next((t for t in self.tasks if t["id"] == task_id), None)
                if not task:
                    return {"success": False, "error": "Task not found"}
                
                if step_index >= len(task["steps"]):
                    return {"success": False, "error": "No more steps"}
                
                step = task["steps"][step_index]
                
                # Execute the step using LLM
                execution_prompt = f"""Execute this step: {step['description']}
                Provide a detailed response about what was done and the result."""
                
                chat_request = ChatRequest(
                    messages=[
                        {"role": "system", "content": "You are AutoGPT executing a task step."},
                        {"role": "user", "content": execution_prompt}
                    ]
                )
                
                response = await self.generate_completion(chat_request)
                result = response.choices[0]["message"]["content"]
                
                # Update step with result
                step["result"] = result
                step["completed"] = True
                
                # Store in memory
                self.memory.append({
                    "task_id": task_id,
                    "step": step_index,
                    "action": step["action"],
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                })
                
                return {
                    "success": True,
                    "step": step,
                    "result": result,
                    "next_step": step_index + 1 < len(task["steps"])
                }
                
            except Exception as e:
                logger.error(f"Failed to execute task: {e}")
                return {"success": False, "error": str(e)}
        
        @self.app.get("/task/list")
        async def list_tasks():
            """List all tasks"""
            return {"tasks": self.tasks}
        
        @self.app.get("/memory")
        async def get_memory():
            """Get agent memory"""
            return {"memory": self.memory[-100:]}  # Return last 100 items
        
        @self.app.post("/think")
        async def think(request: Dict[str, Any]):
            """Let AutoGPT think about a problem"""
            try:
                thought = request.get("thought", "")
                context = request.get("context", [])
                
                # Build context from memory
                memory_context = "\n".join([
                    f"- {m['action']}: {m['result'][:100]}..." 
                    for m in self.memory[-5:]
                ])
                
                thinking_prompt = f"""Context from memory:
                {memory_context}
                
                Current thought: {thought}
                
                Analyze this and provide insights, next steps, or solutions."""
                
                chat_request = ChatRequest(
                    messages=[
                        {"role": "system", "content": "You are AutoGPT with autonomous reasoning capabilities."},
                        {"role": "user", "content": thinking_prompt}
                    ],
                    temperature=0.8
                )
                
                response = await self.generate_completion(chat_request)
                analysis = response.choices[0]["message"]["content"]
                
                return {
                    "success": True,
                    "thought": thought,
                    "analysis": analysis,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Failed to think: {e}")
                return {"success": False, "error": str(e)}

def main():
    """Run AutoGPT Local"""
    agent = AutoGPTLocal()
    agent.run()

if __name__ == "__main__":
    main()