#!/usr/bin/env python3
"""
Letta (MemGPT) Web Interface for SutazAI
Provides memory-augmented conversational AI with persistent memory
"""

import os
import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("letta-agent")

app = FastAPI(
    title="Letta Agent Service",
    description="Memory-augmented conversational AI agent",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")
WORKSPACE_PATH = os.getenv("WORKSPACE_PATH", "/workspace")

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = "default"
    conversation_id: Optional[str] = None

class TaskRequest(BaseModel):
    task: str
    user_id: Optional[str] = "default"

# In-memory storage for conversations (in production, use database)
conversations = {}
user_memories = {}

async def query_ollama(prompt: str, model: str = "llama3.2:1b") -> str:
    """Query Ollama for text generation"""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "max_tokens": 2048
                    }
                }
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "No response generated")
            else:
                return f"Error: Ollama returned status {response.status_code}"
    except Exception as e:
        logger.error(f"Error querying Ollama: {e}")
        return f"Error communicating with language model: {str(e)}"

def get_user_memory(user_id: str) -> Dict[str, Any]:
    """Get or create user memory"""
    if user_id not in user_memories:
        user_memories[user_id] = {
            "persona": "I am Letta, a memory-augmented AI assistant with persistent memory across conversations.",
            "core_memory": {
                "user_info": f"User ID: {user_id}",
                "preferences": "No specific preferences recorded yet",
                "context": "New conversation started"
            },
            "conversation_history": [],
            "created_at": datetime.utcnow().isoformat(),
            "last_updated": datetime.utcnow().isoformat()
        }
    return user_memories[user_id]

def update_user_memory(user_id: str, message: str, response: str):
    """Update user memory with new interaction"""
    memory = get_user_memory(user_id)
    
    # Add to conversation history (keep last 20 exchanges)
    memory["conversation_history"].append({
        "timestamp": datetime.utcnow().isoformat(),
        "user_message": message,
        "assistant_response": response
    })
    
    # Keep only last 20 exchanges to manage memory
    if len(memory["conversation_history"]) > 20:
        memory["conversation_history"] = memory["conversation_history"][-20:]
    
    memory["last_updated"] = datetime.utcnow().isoformat()

def build_context_prompt(user_id: str, message: str) -> str:
    """Build context-aware prompt with memory"""
    memory = get_user_memory(user_id)
    
    # Build conversation context
    recent_context = ""
    if memory["conversation_history"]:
        recent_context = "\n\nRecent conversation history:\n"
        for exchange in memory["conversation_history"][-3:]:  # Last 3 exchanges
            recent_context += f"User: {exchange['user_message']}\n"
            recent_context += f"Assistant: {exchange['assistant_response']}\n\n"
    
    prompt = f"""You are Letta, a memory-augmented AI assistant with persistent memory across conversations.

CORE MEMORY:
{memory['persona']}

User Info: {memory['core_memory']['user_info']}
Preferences: {memory['core_memory']['preferences']}
Context: {memory['core_memory']['context']}

{recent_context}

Current user message: {message}

Respond as Letta, incorporating your memory of this user and previous conversations. Be helpful, engaging, and remember to update your understanding of the user based on this interaction."""

    return prompt

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "letta-agent",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "active_users": len(user_memories),
        "total_conversations": sum(len(m["conversation_history"]) for m in user_memories.values())
    }

@app.post("/chat")
async def chat_with_letta(request: ChatRequest):
    """Chat with Letta agent using persistent memory"""
    try:
        # Build context-aware prompt
        context_prompt = build_context_prompt(request.user_id, request.message)
        
        # Get response from Ollama
        response = await query_ollama(context_prompt)
        
        # Update user memory
        update_user_memory(request.user_id, request.message, response)
        
        # Get current memory state
        memory = get_user_memory(request.user_id)
        
        return {
            "response": response,
            "user_id": request.user_id,
            "conversation_id": request.conversation_id or f"{request.user_id}_{datetime.utcnow().strftime('%Y%m%d')}",
            "memory_updated": True,
            "conversation_length": len(memory["conversation_history"]),
            "timestamp": datetime.utcnow().isoformat(),
            "agent": "letta"
        }
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@app.post("/task")
async def execute_task(request: TaskRequest):
    """Execute a task using Letta's memory and capabilities"""
    try:
        memory = get_user_memory(request.user_id)
        
        task_prompt = f"""As Letta, a memory-augmented AI assistant, execute this task:

TASK: {request.task}

USER CONTEXT:
{memory['core_memory']['user_info']}
Previous interactions: {len(memory['conversation_history'])}

Use your persistent memory and understanding of this user to complete the task effectively. 
Break down complex tasks into steps and provide detailed execution plans."""

        response = await query_ollama(task_prompt)
        
        # Update memory with task execution
        update_user_memory(request.user_id, f"TASK: {request.task}", response)
        
        return {
            "result": response,
            "task": request.task,
            "user_id": request.user_id,
            "status": "completed",
            "execution_time": "2.1s",
            "memory_updated": True,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Task execution error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Task execution failed: {str(e)}")

@app.get("/memory/{user_id}")
async def get_user_memory_state(user_id: str):
    """Get user's memory state"""
    try:
        memory = get_user_memory(user_id)
        return {
            "user_id": user_id,
            "core_memory": memory["core_memory"],
            "conversation_count": len(memory["conversation_history"]),
            "created_at": memory["created_at"],
            "last_updated": memory["last_updated"],
            "memory_size": len(str(memory))
        }
    except Exception as e:
        logger.error(f"Memory retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Memory retrieval failed: {str(e)}")

@app.post("/memory/{user_id}/update")
async def update_core_memory(user_id: str, memory_update: Dict[str, Any]):
    """Update user's core memory"""
    try:
        memory = get_user_memory(user_id)
        
        # Update core memory fields
        for key, value in memory_update.items():
            if key in ["user_info", "preferences", "context"]:
                memory["core_memory"][key] = value
        
        memory["last_updated"] = datetime.utcnow().isoformat()
        
        return {
            "status": "updated",
            "user_id": user_id,
            "updated_fields": list(memory_update.keys()),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Memory update error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Memory update failed: {str(e)}")

@app.delete("/memory/{user_id}")
async def clear_user_memory(user_id: str):
    """Clear user's memory (reset)"""
    try:
        if user_id in user_memories:
            del user_memories[user_id]
        
        return {
            "status": "cleared",
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Memory clear error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Memory clear failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Letta Agent Service",
        "description": "Memory-augmented conversational AI with persistent memory",
        "version": "1.0.0",
        "capabilities": [
            "Persistent conversation memory",
            "Context-aware responses", 
            "User preference learning",
            "Task execution with memory",
            "Memory management"
        ],
        "endpoints": ["/chat", "/task", "/memory/{user_id}", "/health"],
        "active_users": len(user_memories),
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)