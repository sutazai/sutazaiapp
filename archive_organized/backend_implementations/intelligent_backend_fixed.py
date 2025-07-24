#!/usr/bin/env python3
"""
Fixed Intelligent SutazAI Backend with Proper Ollama Integration
Ensures AI models respond correctly without default fallback messages
"""

import json
import time
import asyncio
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import requests
import logging
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="SutazAI Intelligent Backend API",
    description="Advanced AI system with reliable model integration",
    version="3.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service endpoints
SERVICES = {
    "ollama": "http://localhost:11434",
    "qdrant": "http://localhost:6333",
    "chromadb": "http://localhost:8001",
    "postgres": "postgresql://sutazai:sutazai_password@localhost:5432/sutazai",
    "redis": "redis://localhost:6379"
}

# Ollama client with proper error handling
class OllamaClient:
    def __init__(self, base_url=SERVICES["ollama"]):
        self.base_url = base_url
        self.default_timeout = 120
        
    async def generate(self, prompt: str, model: str = "llama3.2:1b") -> Dict[str, Any]:
        """Generate response from Ollama model with proper error handling"""
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 500
            }
        }
        
        try:
            logger.info(f"Requesting Ollama: model={model}, prompt_length={len(prompt)}")
            start_time = time.time()
            
            response = requests.post(url, json=payload, timeout=self.default_timeout)
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "")
                
                if response_text:
                    elapsed = time.time() - start_time
                    logger.info(f"Ollama success: {elapsed:.2f}s, {len(response_text)} chars")
                    
                    return {
                        "success": True,
                        "response": response_text,
                        "model": model,
                        "duration": elapsed,
                        "tokens": len(response_text.split())
                    }
                else:
                    return {
                        "success": False,
                        "error": "Model returned empty response",
                        "response": None
                    }
            else:
                logger.error(f"Ollama HTTP {response.status_code}")
                return {
                    "success": False,
                    "error": f"Model service returned status {response.status_code}",
                    "response": None
                }
                
        except requests.exceptions.Timeout:
            logger.error(f"Ollama timeout after {self.default_timeout}s")
            return {
                "success": False,
                "error": f"Request timed out after {self.default_timeout}s. Try a smaller model or shorter prompt.",
                "response": None
            }
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Ollama")
            return {
                "success": False,
                "error": "Cannot connect to AI model service. Please check if Ollama is running.",
                "response": None
            }
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "response": None
            }
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                return response.json().get("models", [])
            return []
        except:
            return []

# Initialize Ollama client
ollama_client = OllamaClient()

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = "llama3.2:1b"
    session_id: Optional[str] = None
    temperature: Optional[float] = 0.7

class ChatResponse(BaseModel):
    response: str
    model: str
    timestamp: str
    tokens_used: int
    duration: Optional[float] = None
    error: Optional[str] = None

# Main chat endpoint
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint with Ollama integration"""
    try:
        # Generate response using Ollama
        result = await ollama_client.generate(request.message, request.model)
        
        if result["success"]:
            return ChatResponse(
                response=result["response"],
                model=request.model,
                timestamp=datetime.now().isoformat(),
                tokens_used=result.get("tokens", 0),
                duration=result.get("duration")
            )
        else:
            # Return error information instead of default message
            return ChatResponse(
                response=f"Error: {result['error']}",
                model=request.model,
                timestamp=datetime.now().isoformat(),
                tokens_used=0,
                error=result['error']
            )
            
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Models endpoint
@app.get("/api/models")
async def list_models():
    """List available AI models"""
    try:
        models = await ollama_client.list_models()
        return {"models": models}
    except Exception as e:
        logger.error(f"Models endpoint error: {str(e)}")
        return {"models": [], "error": str(e)}

# Health check
@app.get("/health")
async def health_check():
    """System health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "3.0.0"
    }

# System status
@app.get("/api/status")
async def system_status():
    """Get comprehensive system status"""
    status = {
        "timestamp": datetime.now().isoformat(),
        "services": {},
        "models": []
    }
    
    # Check Ollama
    try:
        models = await ollama_client.list_models()
        status["services"]["ollama"] = "online"
        status["models"] = [m["name"] for m in models]
    except:
        status["services"]["ollama"] = "offline"
    
    # Check other services
    for service, url in SERVICES.items():
        if service != "ollama":
            try:
                if service in ["postgres", "redis"]:
                    status["services"][service] = "configured"
                else:
                    response = requests.get(f"{url}/", timeout=2)
                    status["services"][service] = "online" if response.status_code < 500 else "error"
            except:
                status["services"][service] = "offline"
    
    return status

# WebSocket for real-time chat
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for real-time chat"""
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            message = data.get("message", "")
            model = data.get("model", "llama3.2:1b")
            
            # Generate response
            result = await ollama_client.generate(message, model)
            
            if result["success"]:
                await websocket.send_json({
                    "type": "response",
                    "message": result["response"],
                    "model": model,
                    "timestamp": datetime.now().isoformat()
                })
            else:
                await websocket.send_json({
                    "type": "error",
                    "error": result["error"],
                    "model": model,
                    "timestamp": datetime.now().isoformat()
                })
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.close()

# Test endpoint
@app.post("/api/test")
async def test_model():
    """Test model functionality"""
    test_prompts = [
        ("What is 2+2?", "llama3.2:1b"),
        ("Explain AI in one sentence", "qwen2.5:3b"),
        ("Hello", "deepseek-r1:8b")
    ]
    
    results = []
    for prompt, model in test_prompts:
        models = await ollama_client.list_models()
        model_names = [m["name"] for m in models]
        
        if model in model_names:
            result = await ollama_client.generate(prompt, model)
            results.append({
                "prompt": prompt,
                "model": model,
                "success": result["success"],
                "response": result.get("response", "")[:100] + "..." if result["success"] else result.get("error"),
                "duration": result.get("duration", 0)
            })
        else:
            results.append({
                "prompt": prompt,
                "model": model,
                "success": False,
                "response": "Model not available"
            })
    
    return {"tests": results}

if __name__ == "__main__":
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000)