#!/usr/bin/env python3
"""
Awesome Code AI Integration Service for SutazAI
Provides access to the curated collection of AI code tools and models
"""

import os
import json
import logging
import asyncio
import aiohttp
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import time
import uvicorn
from pathlib import Path
from code_ai_manager import CodeAIManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="SutazAI Awesome Code AI Service",
    description="Integration with curated AI code tools and models",
    version="1.0.0"
)

class CodeAnalysisRequest(BaseModel):
    code: str
    language: Optional[str] = None
    analysis_type: List[str] = ["quality", "security", "performance"]

class CodeGenerationRequest(BaseModel):
    prompt: str
    language: str = "python"
    max_tokens: int = 1000
    temperature: float = 0.7

class ToolExecutionRequest(BaseModel):
    tool_name: str
    parameters: Dict[str, Any]
    input_data: Optional[Any] = None

# Initialize Code AI Manager
code_ai_manager = CodeAIManager()

@app.on_event("startup")
async def startup_event():
    """Initialize the service on startup"""
    logger.info("Starting Awesome Code AI Service...")
    await code_ai_manager.initialize()
    logger.info("Service initialized successfully")

@app.get("/")
async def root():
    return {
        "service": "SutazAI Awesome Code AI",
        "status": "running",
        "version": "1.0.0",
        "description": "Curated AI code tools integration"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "available_tools": len(code_ai_manager.get_available_tools())
    }

@app.get("/tools")
async def list_tools():
    """List all available AI code tools"""
    tools = code_ai_manager.get_available_tools()
    return {"tools": tools, "total": len(tools)}

@app.get("/tools/{tool_name}")
async def get_tool_info(tool_name: str):
    """Get detailed information about a specific tool"""
    try:
        tool_info = code_ai_manager.get_tool_info(tool_name)
        return tool_info
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/analyze")
async def analyze_code(request: CodeAnalysisRequest):
    """Analyze code using multiple AI tools"""
    try:
        results = await code_ai_manager.analyze_code(
            request.code,
            request.language,
            request.analysis_type
        )
        return {"analysis_results": results, "timestamp": time.time()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
async def generate_code(request: CodeGenerationRequest):
    """Generate code using AI models"""
    try:
        generated_code = await code_ai_manager.generate_code(
            request.prompt,
            request.language,
            request.max_tokens,
            request.temperature
        )
        return {
            "generated_code": generated_code,
            "language": request.language,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/execute/{tool_name}")
async def execute_tool(tool_name: str, request: ToolExecutionRequest):
    """Execute a specific AI code tool"""
    try:
        result = await code_ai_manager.execute_tool(
            tool_name,
            request.parameters,
            request.input_data
        )
        return {"result": result, "tool": tool_name, "timestamp": time.time()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    """List available AI models for code tasks"""
    models = code_ai_manager.get_available_models()
    return {"models": models}

@app.post("/optimize")
async def optimize_code(code: str, language: str = "python"):
    """Optimize code using AI models"""
    try:
        optimized_code = await code_ai_manager.optimize_code(code, language)
        return {
            "original_code": code,
            "optimized_code": optimized_code,
            "language": language,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/review")
async def review_code(code: str, language: str = "python"):
    """Review code and provide suggestions"""
    try:
        review_results = await code_ai_manager.review_code(code, language)
        return {
            "code": code,
            "review": review_results,
            "language": language,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/refactor")
async def refactor_code(code: str, language: str = "python", style: str = "clean"):
    """Refactor code using AI"""
    try:
        refactored_code = await code_ai_manager.refactor_code(code, language, style)
        return {
            "original_code": code,
            "refactored_code": refactored_code,
            "language": language,
            "style": style,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_service_stats():
    """Get service usage statistics"""
    stats = code_ai_manager.get_stats()
    return {
        "stats": stats,
        "uptime": time.time() - code_ai_manager.start_time,
        "timestamp": time.time()
    }

if __name__ == "__main__":
    uvicorn.run(
        "awesome_code_service:app",
        host="0.0.0.0",
        port=8089,
        log_level="info",
        reload=False
    )