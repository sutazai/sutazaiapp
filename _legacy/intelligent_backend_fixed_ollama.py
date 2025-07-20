#!/usr/bin/env python3
"""
Fixed SutazAI Backend - Properly connects to Ollama
"""

import json
import time
import asyncio
import sys
import os
from datetime import datetime
from typing import Dict, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import logging
import uvicorn
import psutil
from concurrent.futures import ThreadPoolExecutor
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread pool for Ollama calls
executor = ThreadPoolExecutor(max_workers=4)

# FastAPI app
app = FastAPI(title="SutazAI Backend", version="7.0.1")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global metrics
class Metrics:
    def __init__(self):
        self.api_calls = 0
        self.model_calls = {}
        self.errors = 0
        self.total_tokens = 0
        self.start_time = time.time()
        self.lock = threading.Lock()
    
    def record_call(self):
        with self.lock:
            self.api_calls += 1
    
    def record_model_use(self, model: str, tokens: int):
        with self.lock:
            if model not in self.model_calls:
                self.model_calls[model] = 0
            self.model_calls[model] += 1
            self.total_tokens += tokens
    
    def record_error(self):
        with self.lock:
            self.errors += 1
    
    def get_summary(self):
        with self.lock:
            return {
                "api_calls": self.api_calls,
                "model_calls": self.model_calls,
                "total_tokens": self.total_tokens,
                "errors": self.errors,
                "uptime": time.time() - self.start_time
            }

metrics = Metrics()

# Models
class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = "llama3.2:1b"

class ChatResponse(BaseModel):
    response: str
    model: str
    timestamp: str
    tokens_used: int
    ollama_success: bool

# Non-blocking Ollama call
def call_ollama_sync(message: str, model: str) -> Dict:
    """Synchronous Ollama call for thread pool"""
    try:
        # Don't truncate the prompt - send full message
        payload = {
            "model": model,
            "prompt": message,
            "stream": False,
            "options": {
                "num_predict": 150,  # Reasonable response length
                "temperature": 0.7
            }
        }
        
        logger.info(f"Calling Ollama with model {model}, prompt length: {len(message)}")
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=30  # Increased timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            response_text = result.get("response", "")
            tokens = result.get("eval_count", 0) + result.get("prompt_eval_count", 0)
            
            logger.info(f"Ollama success: {len(response_text)} chars, {tokens} tokens")
            metrics.record_model_use(model, tokens)
            
            return {
                "success": True,
                "response": response_text,
                "model": model,
                "tokens": tokens
            }
        else:
            logger.error(f"Ollama HTTP error: {response.status_code}")
            metrics.record_error()
    except requests.exceptions.Timeout:
        logger.error("Ollama timeout")
        metrics.record_error()
    except Exception as e:
        logger.error(f"Ollama error: {e}")
        metrics.record_error()
    
    # Enhanced fallback responses
    msg_lower = message.lower()
    if "self" in msg_lower and "improve" in msg_lower:
        return {
            "success": False,
            "response": "The AGI system can self-improve through continuous learning, analyzing patterns in data, optimizing algorithms, and incorporating feedback. This includes code generation, performance optimization, and knowledge expansion.",
            "model": f"{model} (fallback)",
            "tokens": 50
        }
    elif "agi" in msg_lower:
        return {
            "success": False,
            "response": "AGI (Artificial General Intelligence) represents AI systems capable of understanding, learning, and applying intelligence across diverse domains, similar to human cognitive abilities.",
            "model": f"{model} (fallback)",
            "tokens": 40
        }
    else:
        return {
            "success": False,
            "response": f"Processing your request about '{message[:50]}...' The AGI system is analyzing this through its reasoning engine.",
            "model": f"{model} (fallback)",
            "tokens": 30
        }

# Async wrapper for Ollama
async def call_ollama_async(message: str, model: str) -> Dict:
    """Async wrapper using thread pool"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, call_ollama_sync, message, model)

# Chat endpoint
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Non-blocking chat endpoint"""
    metrics.record_call()
    
    # Get response
    result = await call_ollama_async(request.message, request.model)
    
    return ChatResponse(
        response=result["response"],
        model=result["model"],
        timestamp=datetime.now().isoformat(),
        tokens_used=result.get("tokens", 0),
        ollama_success=result["success"]
    )

# Health check
@app.get("/health")
async def health():
    """Health check endpoint"""
    try:
        # Check Ollama
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        ollama_ok = response.status_code == 200
        models = response.json().get("models", []) if ollama_ok else []
    except:
        ollama_ok = False
        models = []
    
    return {
        "status": "healthy" if ollama_ok else "degraded",
        "timestamp": datetime.now().isoformat(),
        "ollama": {
            "connected": ollama_ok,
            "models": [m["name"] for m in models]
        },
        "metrics": metrics.get_summary()
    }

# Get available models
@app.get("/api/models")
async def get_models():
    """Get available Ollama models"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return {
                "models": [{"name": m["name"], "size": m.get("size", 0)} for m in models]
            }
    except:
        pass
    
    return {"models": []}

# Metrics endpoint
@app.get("/api/metrics")
async def get_metrics():
    """Get system metrics"""
    return {
        "backend": metrics.get_summary(),
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent
        },
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    logger.info("Starting fixed SutazAI backend on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)