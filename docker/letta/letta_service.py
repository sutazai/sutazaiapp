"""
Letta (MemGPT) Service for SutazAI
Provides persistent memory AI agent capabilities
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import logging
import os
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="SutazAI Letta Service",
    description="Persistent memory AI agent service",
    version="1.0.0"
)

# Request/Response models
class LettaRequest(BaseModel):
    message: str
    agent_name: Optional[str] = "sutazai_agent"
    session_id: Optional[str] = "default"

class LettaResponse(BaseModel):
    status: str
    response: str
    agent_name: str
    session_id: str
    memory_usage: Optional[Dict] = None

# Letta Manager
class LettaManager:
    def __init__(self):
        self.workspace = "/app/workspace"
        self.agents = {}
        self.config = {
            "model": "deepseek-r1:8b",
            "api_base": "http://ollama:11434/v1",
            "api_key": "local"
        }
        
        # Ensure workspace exists
        os.makedirs(self.workspace, exist_ok=True)
        
        # Initialize Letta configuration
        self.init_letta_config()
        
    def init_letta_config(self):
        """Initialize Letta configuration"""
        try:
            # Set environment variables for Letta
            os.environ["OPENAI_API_BASE"] = self.config["api_base"]
            os.environ["OPENAI_API_KEY"] = self.config["api_key"]
            
            logger.info("Letta configuration initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Letta config: {e}")
    
    def get_or_create_agent(self, agent_name: str) -> Dict[str, Any]:
        """Get existing agent or create a new one"""
        try:
            if agent_name not in self.agents:
                # Create a simple agent simulation
                self.agents[agent_name] = {
                    "name": agent_name,
                    "memory": {
                        "core_memory": "I am an AI assistant with persistent memory.",
                        "conversation_history": []
                    },
                    "created_at": "2024-01-01",
                    "sessions": {}
                }
                logger.info(f"Created new agent: {agent_name}")
            
            return self.agents[agent_name]
        except Exception as e:
            logger.error(f"Failed to get/create agent {agent_name}: {e}")
            raise
    
    def process_message(self, request: LettaRequest) -> Dict[str, Any]:
        """Process message with Letta agent"""
        try:
            # Get or create agent
            agent = self.get_or_create_agent(request.agent_name)
            
            # Ensure session exists
            if request.session_id not in agent["sessions"]:
                agent["sessions"][request.session_id] = {
                    "messages": [],
                    "created_at": "2024-01-01"
                }
            
            session = agent["sessions"][request.session_id]
            
            # Add user message to history
            session["messages"].append({
                "role": "user",
                "content": request.message,
                "timestamp": "2024-01-01"
            })
            
            # Simulate Letta processing with memory context
            memory_context = agent["memory"]["core_memory"]
            recent_messages = session["messages"][-5:]  # Last 5 messages for context
            
            # Simple response generation (in real implementation, this would use Letta)
            response_text = f"As an AI with persistent memory, I remember our previous conversations. "
            response_text += f"You said: '{request.message}'. "
            response_text += f"Based on my memory: {memory_context}"
            
            # Add assistant response to history
            session["messages"].append({
                "role": "assistant",
                "content": response_text,
                "timestamp": "2024-01-01"
            })
            
            # Update core memory occasionally
            if len(session["messages"]) % 10 == 0:
                agent["memory"]["core_memory"] += f" Recent topic: {request.message[:50]}..."
            
            return {
                "status": "success",
                "response": response_text,
                "agent_name": request.agent_name,
                "session_id": request.session_id,
                "memory_usage": {
                    "core_memory_size": len(agent["memory"]["core_memory"]),
                    "conversation_length": len(session["messages"]),
                    "sessions_count": len(agent["sessions"])
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to process message: {e}")
            return {
                "status": "failed",
                "response": f"Error processing message: {str(e)}",
                "agent_name": request.agent_name,
                "session_id": request.session_id,
                "memory_usage": None
            }

# Initialize Letta manager
letta_manager = LettaManager()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Letta",
        "status": "operational",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "letta"}

@app.post("/chat", response_model=LettaResponse)
async def chat_with_agent(request: LettaRequest):
    """Chat with a Letta agent"""
    try:
        result = letta_manager.process_message(request)
        return LettaResponse(**result)
    except Exception as e:
        logger.error(f"Failed to chat with agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents")
async def list_agents():
    """List all agents"""
    try:
        agents_info = []
        for name, agent in letta_manager.agents.items():
            agents_info.append({
                "name": name,
                "sessions_count": len(agent["sessions"]),
                "memory_size": len(agent["memory"]["core_memory"]),
                "created_at": agent["created_at"]
            })
        return {"agents": agents_info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agent/{agent_name}")
async def get_agent_info(agent_name: str):
    """Get detailed information about an agent"""
    try:
        if agent_name not in letta_manager.agents:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        agent = letta_manager.agents[agent_name]
        return {
            "name": agent_name,
            "memory": agent["memory"],
            "sessions": list(agent["sessions"].keys()),
            "created_at": agent["created_at"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agent/{agent_name}/session/{session_id}")
async def get_session_history(agent_name: str, session_id: str):
    """Get conversation history for a session"""
    try:
        if agent_name not in letta_manager.agents:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        agent = letta_manager.agents[agent_name]
        if session_id not in agent["sessions"]:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return agent["sessions"][session_id]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)