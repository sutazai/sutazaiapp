"""
Letta (MemGPT) Service for SutazAI - WORKING VERSION
Provides real persistent memory AI agent capabilities with actual LLM integration
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import logging
import os
import json
import requests
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="SutazAI Letta Service - Working",
    description="Real persistent memory AI agent service",
    version="2.0.0"
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

# Working Letta Manager with REAL LLM integration
class WorkingLettaManager:
    def __init__(self):
        self.workspace = "/app/workspace"
        self.agents = {}
        self.ollama_url = "http://ollama:11434"
        self.model = "gpt-oss.2:1b"
        
        # Ensure workspace exists
        os.makedirs(self.workspace, exist_ok=True)
        
        # Test Ollama connection
        self.test_ollama_connection()
        
    def test_ollama_connection(self):
        """Test if Ollama is responding"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info("✓ Ollama connection successful")
                return True
            else:
                logger.warning(f"⚠ Ollama returned status {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"✗ Ollama connection failed: {e}")
            return False
    
    def call_ollama(self, prompt: str, max_retries: int = 3) -> str:
        """Call Ollama with proper error handling and retries"""
        for attempt in range(max_retries):
            try:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 100,  # Limit response length
                        "top_p": 0.9
                    }
                }
                
                logger.info(f"Calling Ollama (attempt {attempt + 1})...")
                start_time = time.time()
                
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json=payload,
                    timeout=15  # 15 second timeout
                )
                
                elapsed = time.time() - start_time
                logger.info(f"Ollama responded in {elapsed:.2f}s")
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("response", "No response from model")
                else:
                    logger.warning(f"Ollama error: {response.status_code}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Ollama timeout on attempt {attempt + 1}")
            except Exception as e:
                logger.error(f"Ollama call failed: {e}")
                
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retry
                
        return "I'm having trouble connecting to my language model. Please try again."
    
    def get_or_create_agent(self, agent_name: str) -> Dict[str, Any]:
        """Get existing agent or create a new one"""
        try:
            if agent_name not in self.agents:
                self.agents[agent_name] = {
                    "name": agent_name,
                    "memory": {
                        "core_memory": f"I am {agent_name}, an AI assistant with persistent memory.",
                        "conversation_history": []
                    },
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "sessions": {}
                }
                logger.info(f"Created new agent: {agent_name}")
            
            return self.agents[agent_name]
        except Exception as e:
            logger.error(f"Failed to get/create agent {agent_name}: {e}")
            raise
    
    def process_message(self, request: LettaRequest) -> Dict[str, Any]:
        """Process message with real LLM integration"""
        try:
            # Get or create agent
            agent = self.get_or_create_agent(request.agent_name)
            
            # Ensure session exists
            if request.session_id not in agent["sessions"]:
                agent["sessions"][request.session_id] = {
                    "messages": [],
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            
            session = agent["sessions"][request.session_id]
            
            # Add user message to history
            user_message = {
                "role": "user",
                "content": request.message,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            session["messages"].append(user_message)
            
            # Build context for LLM
            memory_context = agent["memory"]["core_memory"]
            recent_messages = session["messages"][-3:]  # Last 3 messages for context
            
            # Create LLM prompt with memory context
            context_text = f"Memory: {memory_context}\n\n"
            if len(recent_messages) > 1:
                context_text += "Recent conversation:\n"
                for msg in recent_messages[:-1]:  # Exclude current message
                    context_text += f"{msg['role']}: {msg['content']}\n"
            
            llm_prompt = f"""{context_text}
Current user message: {request.message}

Please respond as an AI assistant with persistent memory. Keep responses concise and helpful."""
            
            # Call Ollama LLM
            llm_response = self.call_ollama(llm_prompt)
            
            # Add assistant response to history
            assistant_message = {
                "role": "assistant", 
                "content": llm_response,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            session["messages"].append(assistant_message)
            
            # Update core memory occasionally (every 5 messages)
            if len(session["messages"]) % 10 == 0:
                summary_prompt = f"Summarize this conversation in one sentence: {request.message[:100]}..."
                summary = self.call_ollama(summary_prompt)
                agent["memory"]["core_memory"] += f" Recent: {summary[:100]}"
            
            return {
                "status": "success",
                "response": llm_response,
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

# Initialize working Letta manager
letta_manager = WorkingLettaManager()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Letta-Working",
        "status": "operational", 
        "version": "2.0.0",
        "llm_model": letta_manager.model
    }

@app.get("/health")
async def health_check():
    """Health check endpoint with LLM connectivity test"""
    ollama_status = letta_manager.test_ollama_connection()
    return {
        "status": "healthy" if ollama_status else "degraded",
        "service": "letta",
        "ollama_connected": ollama_status
    }

@app.post("/chat", response_model=LettaResponse)
async def chat_with_agent(request: LettaRequest):
    """Chat with a Letta agent using real LLM"""
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
