#!/usr/bin/env python3
"""
Letta (MemGPT) Wrapper - Long-term Memory AI
"""

import os
import sys
import json
from typing import Dict, Any, List
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from base_agent_wrapper import BaseAgentWrapper, ChatRequest

class LettaLocal(BaseAgentWrapper):
    """Letta/MemGPT with persistent memory"""
    
    def __init__(self):
        super().__init__(
            agent_name="Letta",
            agent_description="AI with long-term memory and context management",
            port=8000,
            capabilities=["memory", "conversation", "task-automation"]
        )
        self.memory = {"core": [], "recall": [], "archival": []}
        self.personas = {}
        self.setup_letta_routes()
    
    def setup_letta_routes(self):
        """Setup Letta-specific routes"""
        
        @self.app.get("/capabilities")
        async def get_capabilities():
            """Return Letta agent capabilities"""
            return {
                "agent": "Letta",
                "version": "1.0.0",
                "capabilities": [
                    "long_term_memory",
                    "context_management",
                    "conversation_recall",
                    "memory_search",
                    "personalization"
                ],
                "memory_types": [
                    "short_term", "long_term", "episodic", "semantic"
                ],
                "endpoints": [
                    "/health",
                    "/capabilities",
                    "/chat",
                    "/memory/store",
                    "/memory/recall",
                    "/memory/search"
                ]
            }
        
        @self.app.post("/memory/store")
        async def store_memory(request: Dict[str, Any]):
            """Store information in memory"""
            memory_type = request.get("type", "core")
            content = request.get("content")
            
            memory_item = {
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "type": memory_type
            }
            
            if memory_type in self.memory:
                self.memory[memory_type].append(memory_item)
            
            return {"success": True, "stored": memory_item}
        
        @self.app.post("/memory/recall")
        async def recall_memory(request: Dict[str, Any]):
            """Recall information from memory"""
            query = request.get("query")
            memory_type = request.get("type", "all")
            
            # Search through memory
            relevant_memories = []
            if memory_type == "all":
                for mem_type in self.memory:
                    relevant_memories.extend(self.memory[mem_type][-10:])
            else:
                relevant_memories = self.memory.get(memory_type, [])[-10:]
            
            return {"success": True, "memories": relevant_memories}
        
        @self.app.post("/persona/create")
        async def create_persona(request: Dict[str, Any]):
            """Create a persona with memory"""
            persona_name = request.get("name")
            traits = request.get("traits", [])
            background = request.get("background", "")
            
            self.personas[persona_name] = {
                "name": persona_name,
                "traits": traits,
                "background": background,
                "memories": [],
                "created_at": datetime.now().isoformat()
            }
            
            return {"success": True, "persona": self.personas[persona_name]}
        
        @self.app.post("/chat/with-memory")
        async def chat_with_memory(request: Dict[str, Any]):
            """Chat with memory context"""
            try:
                message = request.get("message")
                persona = request.get("persona", "default")
                
                # Get relevant memories
                memory_context = "\n".join([
                    f"- {m['content']}" 
                    for m in self.memory["core"][-5:]
                ])
                
                memory_prompt = f"""You have these memories:
                {memory_context}
                
                User message: {message}
                
                Respond using your memory context."""
                
                chat_request = ChatRequest(
                    messages=[
                        {"role": "system", "content": "You are Letta with persistent memory."},
                        {"role": "user", "content": memory_prompt}
                    ]
                )
                
                response = await self.generate_completion(chat_request)
                reply = response.choices[0]["message"]["content"]
                
                # Store this interaction in memory
                self.memory["core"].append({
                    "content": f"User: {message} | Assistant: {reply[:100]}",
                    "timestamp": datetime.now().isoformat(),
                    "type": "interaction"
                })
                
                return {"success": True, "reply": reply, "memories_used": len(self.memory["core"])}
                
            except Exception as e:
                return {"success": False, "error": str(e)}

def main():
    agent = LettaLocal()
    agent.run()

if __name__ == "__main__":
    main()