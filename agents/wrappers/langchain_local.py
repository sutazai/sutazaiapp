#!/usr/bin/env python3
"""
LangChain Wrapper - LLM Framework Integration
"""

import os
import sys
from typing import Dict, Any, List

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from base_agent_wrapper import BaseAgentWrapper, ChatRequest

class LangChainLocal(BaseAgentWrapper):
    """LangChain framework wrapper"""
    
    def __init__(self):
        super().__init__(
            agent_name="LangChain",
            agent_description="LLM application framework with chains and tools",
            port=8000
        )
        self.chains = {}
        self.setup_langchain_routes()
    
    def setup_langchain_routes(self):
        """Setup LangChain routes"""
        
        @self.app.get("/capabilities")
        async def get_capabilities():
            """Return LangChain agent capabilities"""
            return {
                "agent": "LangChain",
                "version": "1.0.0",
                "capabilities": [
                    "chain_execution",
                    "document_qa",
                    "memory_management",
                    "tool_integration",
                    "retrieval_augmented_generation"
                ],
                "chain_types": ["llm", "sequential", "map_reduce", "retrieval_qa"],
                "endpoints": ["/health", "/capabilities", "/chat", "/chain/create", "/chain/execute"]
            }
        
        @self.app.post("/chain/create")
        async def create_chain(request: Dict[str, Any]):
            """Create a processing chain"""
            chain_name = request.get("name")
            chain_type = request.get("type", "simple")
            steps = request.get("steps", [])
            
            self.chains[chain_name] = {
                "name": chain_name,
                "type": chain_type,
                "steps": steps,
                "created_at": datetime.now().isoformat()
            }
            
            return {"success": True, "chain": self.chains[chain_name]}
        
        @self.app.post("/chain/run")
        async def run_chain(request: Dict[str, Any]):
            """Run a processing chain"""
            try:
                chain_name = request.get("chain_name")
                input_data = request.get("input")
                
                if chain_name not in self.chains:
                    # Create default chain
                    self.chains[chain_name] = {
                        "name": chain_name,
                        "type": "default",
                        "steps": ["process"]
                    }
                
                chain = self.chains[chain_name]
                
                chain_prompt = f"""Execute this chain: {chain['name']}
                Steps: {chain['steps']}
                Input: {input_data}
                
                Process the input through each step."""
                
                chat_request = ChatRequest(
                    messages=[
                        {"role": "system", "content": "You are LangChain executing processing chains."},
                        {"role": "user", "content": chain_prompt}
                    ]
                )
                
                response = await self.generate_completion(chat_request)
                result = response.choices[0]["message"]["content"]
                
                return {"success": True, "result": result, "chain": chain_name}
                
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        @self.app.post("/tool/use")
        async def use_tool(request: Dict[str, Any]):
            """Use a LangChain tool"""
            try:
                tool_name = request.get("tool")
                input_data = request.get("input")
                
                tool_prompt = f"""Use the {tool_name} tool on this input:
                {input_data}
                
                Provide the tool output."""
                
                chat_request = ChatRequest(
                    messages=[
                        {"role": "system", "content": "You are LangChain using tools."},
                        {"role": "user", "content": tool_prompt}
                    ]
                )
                
                response = await self.generate_completion(chat_request)
                output = response.choices[0]["message"]["content"]
                
                return {"success": True, "tool": tool_name, "output": output}
                
            except Exception as e:
                return {"success": False, "error": str(e)}

from datetime import datetime

def main():
    agent = LangChainLocal()
    agent.run()

if __name__ == "__main__":
    main()