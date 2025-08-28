#!/usr/bin/env python3
"""
AgentZero Wrapper - Autonomous Agent Framework
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from base_agent_wrapper import BaseAgentWrapper, ChatRequest
from typing import Dict, Any
from datetime import datetime

class AgentZero(BaseAgentWrapper):
    """AgentZero autonomous agent wrapper"""
    
    def __init__(self):
        super().__init__(
            agent_name="AgentZero",
            agent_description="Autonomous agent with zero-shot learning capabilities",
            port=8000
        )
        self.knowledge_base = []
        self.setup_agentzero_routes()
    
    def setup_agentzero_routes(self):
        """Setup AgentZero-specific routes"""
        
        @self.app.post("/learn")
        async def learn(request: Dict[str, Any]):
            """Learn from examples"""
            try:
                example = request.get("example")
                category = request.get("category", "general")
                
                self.knowledge_base.append({
                    "example": example,
                    "category": category,
                    "timestamp": datetime.now().isoformat()
                })
                
                learning_prompt = f"Learn from this example: {example}"
                chat_request = ChatRequest(
                    messages=[
                        {"role": "system", "content": "You are AgentZero, learning from examples."},
                        {"role": "user", "content": learning_prompt}
                    ]
                )
                
                response = await self.generate_completion(chat_request)
                understanding = response.choices[0]["message"]["content"]
                
                return {
                    "success": True,
                    "learned": example,
                    "understanding": understanding
                }
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        @self.app.post("/zero-shot")
        async def zero_shot(request: Dict[str, Any]):
            """Zero-shot task execution"""
            try:
                task = request.get("task")
                
                zero_shot_prompt = f"""Execute this task with no prior training: {task}
                Use reasoning and general knowledge to complete it."""
                
                chat_request = ChatRequest(
                    messages=[
                        {"role": "system", "content": "You are AgentZero with zero-shot capabilities."},
                        {"role": "user", "content": zero_shot_prompt}
                    ],
                    temperature=0.9
                )
                
                response = await self.generate_completion(chat_request)
                result = response.choices[0]["message"]["content"]
                
                return {
                    "success": True,
                    "task": task,
                    "result": result
                }
            except Exception as e:
                return {"success": False, "error": str(e)}

def main():
    agent = AgentZero()
    agent.run()

if __name__ == "__main__":
    main()