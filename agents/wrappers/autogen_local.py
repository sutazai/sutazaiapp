#!/usr/bin/env python3
"""
AutoGen Wrapper - Multi-Agent Configuration and Coordination
"""

import os
import sys
import json
from typing import Dict, Any, List
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from base_agent_wrapper import BaseAgentWrapper, ChatRequest

class AutoGenLocal(BaseAgentWrapper):
    """AutoGen multi-agent configuration wrapper"""
    
    def __init__(self):
        super().__init__(
            agent_name="AutoGen",
            agent_description="Multi-agent configuration and conversation framework",
            port=8000
        )
        self.agents_config = {}
        self.conversations = []
        self.setup_autogen_routes()
    
    def setup_autogen_routes(self):
        """Setup AutoGen-specific routes"""
        
        @self.app.post("/agent/configure")
        async def configure_agent(request: Dict[str, Any]):
            """Configure a new agent"""
            agent_name = request.get("name")
            role = request.get("role", "assistant")
            system_message = request.get("system_message", "")
            
            self.agents_config[agent_name] = {
                "name": agent_name,
                "role": role,
                "system_message": system_message,
                "created_at": datetime.now().isoformat()
            }
            
            return {
                "success": True,
                "agent": self.agents_config[agent_name]
            }
        
        @self.app.post("/conversation/multi-agent")
        async def multi_agent_conversation(request: Dict[str, Any]):
            """Run a multi-agent conversation"""
            try:
                agents = request.get("agents", [])
                topic = request.get("topic")
                rounds = request.get("rounds", 3)
                
                conversation = {
                    "topic": topic,
                    "agents": agents,
                    "messages": []
                }
                
                # Simulate multi-agent conversation
                for round_num in range(rounds):
                    for agent_name in agents:
                        agent_config = self.agents_config.get(agent_name, {})
                        
                        # Build prompt based on conversation history
                        conv_history = "\n".join([
                            f"{msg['agent']}: {msg['content']}" 
                            for msg in conversation["messages"][-5:]
                        ])
                        
                        agent_prompt = f"""You are {agent_name} with role: {agent_config.get('role', 'assistant')}
                        Topic: {topic}
                        
                        Conversation so far:
                        {conv_history}
                        
                        Provide your input on this topic."""
                        
                        chat_request = ChatRequest(
                            messages=[
                                {"role": "system", "content": agent_config.get('system_message', '')},
                                {"role": "user", "content": agent_prompt}
                            ]
                        )
                        
                        response = await self.generate_completion(chat_request)
                        content = response.choices[0]["message"]["content"]
                        
                        conversation["messages"].append({
                            "agent": agent_name,
                            "content": content,
                            "round": round_num + 1
                        })
                
                self.conversations.append(conversation)
                
                return {
                    "success": True,
                    "conversation": conversation
                }
                
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        @self.app.get("/agents/list")
        async def list_agents():
            """List configured agents"""
            return {"agents": self.agents_config}
        
        @self.app.post("/workflow/create")
        async def create_workflow(request: Dict[str, Any]):
            """Create an agent workflow"""
            workflow_name = request.get("name")
            steps = request.get("steps", [])
            
            return {
                "success": True,
                "workflow": {
                    "name": workflow_name,
                    "steps": steps,
                    "created_at": datetime.now().isoformat()
                }
            }

def main():
    agent = AutoGenLocal()
    agent.run()

if __name__ == "__main__":
    main()