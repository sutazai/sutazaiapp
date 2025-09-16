#!/usr/bin/env python3
"""
BigAGI Wrapper - Advanced Chat Interface
"""

import os
import sys
import json
import asyncio
from typing import Dict, Any, List
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from base_agent_wrapper import BaseAgentWrapper, ChatRequest

class BigAGI(BaseAgentWrapper):
    """BigAGI chat interface wrapper"""
    
    def __init__(self):
        super().__init__(
            agent_name="BigAGI",
            agent_description="Advanced AGI chat interface with personality modes",
            port=8000
        )
        self.conversations = {}
        self.personalities = {
            "assistant": "Helpful AI assistant",
            "creative": "Creative and imaginative",
            "analytical": "Logical and analytical",
            "teacher": "Educational and informative"
        }
        self.setup_bigagi_routes()
    
    def setup_bigagi_routes(self):
        """Setup BigAGI-specific routes"""
        
        @self.app.post("/conversation/start")
        async def start_conversation(request: Dict[str, Any]):
            """Start a new conversation"""
            conv_id = f"conv_{datetime.now().timestamp()}"
            personality = request.get("personality", "assistant")
            
            self.conversations[conv_id] = {
                "id": conv_id,
                "personality": personality,
                "messages": [],
                "created_at": datetime.now().isoformat()
            }
            
            return {
                "success": True,
                "conversation_id": conv_id,
                "personality": personality
            }
        
        @self.app.post("/message")
        async def send_message(request: Dict[str, Any]):
            """Send a message in conversation"""
            try:
                conv_id = request.get("conversation_id")
                message = request.get("message")
                
                if conv_id not in self.conversations:
                    return {"success": False, "error": "Conversation not found"}
                
                conv = self.conversations[conv_id]
                conv["messages"].append({"role": "user", "content": message})
                
                # Get personality context
                personality = conv["personality"]
                system_prompt = f"You are BigAGI with {personality} personality: {self.personalities[personality]}"
                
                # Build conversation history
                messages = [{"role": "system", "content": system_prompt}]
                messages.extend(conv["messages"][-10:])  # Last 10 messages for context
                
                chat_request = ChatRequest(messages=messages)
                response = await self.generate_completion(chat_request)
                
                reply = response.choices[0]["message"]["content"]
                conv["messages"].append({"role": "assistant", "content": reply})
                
                return {
                    "success": True,
                    "reply": reply,
                    "conversation_id": conv_id
                }
                
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        @self.app.get("/conversation/{conv_id}")
        async def get_conversation(conv_id: str):
            """Get conversation history"""
            if conv_id in self.conversations:
                return {"success": True, "conversation": self.conversations[conv_id]}
            return {"success": False, "error": "Conversation not found"}
        
        @self.app.post("/personality/switch")
        async def switch_personality(request: Dict[str, Any]):
            """Switch conversation personality"""
            conv_id = request.get("conversation_id")
            new_personality = request.get("personality")
            
            if conv_id in self.conversations and new_personality in self.personalities:
                self.conversations[conv_id]["personality"] = new_personality
                return {"success": True, "new_personality": new_personality}
            
            return {"success": False, "error": "Invalid conversation or personality"}

def main():
    agent = BigAGI()
    agent.run()

if __name__ == "__main__":
    main()