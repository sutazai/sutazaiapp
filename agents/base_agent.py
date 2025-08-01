#!/usr/bin/env python3
"""
Base Agent Class for Native Ollama Integration
100% Local LLM - No External API Dependencies
"""

import os
import json
import httpx
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime

class OllamaLocalAgent:
    """Base agent class using native Ollama API directly"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
        self.model = os.getenv("MODEL_NAME", "tinyllama:latest")
        self.context_window = int(os.getenv("AGENT_CONTEXT_WINDOW", "2048"))
        self.timeout = int(os.getenv("AGENT_TIMEOUT_SECONDS", "30"))
        
    async def generate(self, prompt: str, context: Optional[List[int]] = None) -> Dict[str, Any]:
        """Generate response using native Ollama API"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "context": context,
                        "options": {
                            "temperature": 0.7,
                            "num_predict": self.context_window,
                            "stop": ["Human:", "Assistant:"]
                        }
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return {
                        "response": result.get("response", ""),
                        "context": result.get("context", []),
                        "done": result.get("done", True),
                        "model": self.model
                    }
                else:
                    return {"error": f"Ollama API error: {response.status_code}"}
                    
            except Exception as e:
                return {"error": f"Failed to generate response: {str(e)}"}
    
    async def chat(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Chat using native Ollama chat API"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    f"{self.ollama_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "options": {
                            "temperature": 0.7,
                            "num_predict": self.context_window
                        }
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return {
                        "message": result.get("message", {}),
                        "done": result.get("done", True),
                        "model": self.model
                    }
                else:
                    return {"error": f"Ollama API error: {response.status_code}"}
                    
            except Exception as e:
                return {"error": f"Failed to chat: {str(e)}"}
    
    async def embeddings(self, text: str) -> List[float]:
        """Generate embeddings using Ollama"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    f"{self.ollama_url}/api/embeddings",
                    json={
                        "model": self.model,
                        "prompt": text
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("embedding", [])
                else:
                    return []
                    
            except Exception as e:
                print(f"Failed to generate embeddings: {str(e)}")
                return []
    
    async def check_model_availability(self) -> bool:
        """Check if model is available in Ollama"""
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                response = await client.get(f"{self.ollama_url}/api/tags")
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    return any(m["name"] == self.model for m in models)
                return False
            except:
                return False
    
    async def pull_model_if_needed(self) -> bool:
        """Pull model if not available"""
        if not await self.check_model_availability():
            async with httpx.AsyncClient(timeout=600) as client:
                try:
                    response = await client.post(
                        f"{self.ollama_url}/api/pull",
                        json={"name": self.model}
                    )
                    return response.status_code == 200
                except:
                    return False
        return True
    
    def format_agent_prompt(self, task: str, context: Optional[Dict] = None) -> str:
        """Format prompt with agent personality and context"""
        agent_description = self.load_agent_description()
        
        prompt = f"""You are {self.agent_name}, a specialized AI agent.

{agent_description}

Current Task: {task}

"""
        if context:
            prompt += f"Context:\n{json.dumps(context, indent=2)}\n\n"
        
        prompt += "Please provide a detailed response following your specialized expertise:\n"
        
        return prompt
    
    def load_agent_description(self) -> str:
        """Load agent description from definition file"""
        agent_file = f"/agents/{self.agent_name}.md"
        
        if os.path.exists(agent_file):
            with open(agent_file, 'r') as f:
                content = f.read()
                # Extract system prompt after frontmatter
                parts = content.split('---', 2)
                if len(parts) >= 3:
                    return parts[2].strip()
        
        return f"A specialized agent for {self.agent_name.replace('-', ' ')} tasks."
    
    async def execute_task(self, task: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute a task with the agent's expertise"""
        # Ensure model is available
        if not await self.pull_model_if_needed():
            return {"error": "Failed to load model"}
        
        # Format prompt
        prompt = self.format_agent_prompt(task, context)
        
        # Generate response
        result = await self.generate(prompt)
        
        if "error" not in result:
            return {
                "agent": self.agent_name,
                "task": task,
                "response": result["response"],
                "model": self.model,
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            return result


# Example usage for specific agents
class TaskCoordinatorAgent(OllamaLocalAgent):
    """Task Assignment Coordinator using native Ollama"""
    
    def __init__(self):
        super().__init__("task-assignment-coordinator")
        self.available_agents = self.load_available_agents()
    
    def load_available_agents(self) -> List[str]:
        """Load list of available agents"""
        agents_dir = "/agents"
        if os.path.exists(agents_dir):
            return [f.replace('.md', '') for f in os.listdir(agents_dir) 
                    if f.endswith('.md') and not f.startswith('_')]
        return []
    
    async def route_task(self, task_description: str) -> Dict[str, Any]:
        """Route task to appropriate agent"""
        routing_prompt = f"""As the Task Assignment Coordinator, analyze this task and determine the best agent to handle it.

Available agents:
{json.dumps(self.available_agents, indent=2)}

Task: {task_description}

Respond with a JSON object containing:
- "selected_agent": The name of the best agent for this task
- "reasoning": Why this agent was selected
- "alternative_agents": List of other suitable agents
"""
        
        result = await self.generate(routing_prompt)
        
        if "error" not in result:
            try:
                # Parse JSON from response
                response_text = result["response"]
                # Extract JSON if wrapped in markdown
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    response_text = response_text[json_start:json_end]
                
                routing_decision = json.loads(response_text)
                return routing_decision
            except:
                # Fallback to simple text parsing
                return {
                    "selected_agent": "senior-ai-engineer",
                    "reasoning": "Failed to parse routing decision",
                    "alternative_agents": []
                }
        
        return {"error": "Failed to route task"}


if __name__ == "__main__":
    # Test the base agent
    async def test_agent():
        agent = OllamaLocalAgent("test-agent")
        result = await agent.execute_task("Explain how you work with native Ollama API")
        print(json.dumps(result, indent=2))
    
    asyncio.run(test_agent())