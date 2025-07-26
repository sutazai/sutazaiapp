#!/usr/bin/env python3
"""
Local Model Server

This module provides a local FastAPI server for model inference
without requiring external API calls.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from pathlib import Path

from .offline_model_manager import OfflineModelManager, ModelConfig, ModelFramework

logger = logging.getLogger("LocalModelServer")

# Request/Response models
class TextGenerationRequest(BaseModel):
    prompt: str
    model_id: Optional[str] = None
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False

class TextGenerationResponse(BaseModel):
    text: str
    model_id: str
    inference_time: float
    tokens_generated: int
    success: bool

class ModelLoadRequest(BaseModel):
    model_id: str
    force_reload: bool = False

class LocalModelServer:
    """
    Local FastAPI server for offline model inference
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.app = FastAPI(
            title="SutazAI Local Model Server",
            description="Local model inference server for offline operation",
            version="1.0.0"
        )
        
        # Initialize model manager
        self.model_manager = OfflineModelManager(config.get('model_manager', {}))
        
        # Add CORS middleware for local development
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:*", "http://127.0.0.1:*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self._setup_routes()
        
        logger.info("Local Model Server initialized")
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "server": "local_model_server"
            }
        
        @self.app.get("/models")
        async def list_models():
            """List all available models"""
            try:
                # Discover models if not already done
                if not self.model_manager.model_configs:
                    await self.model_manager.discover_local_models()
                
                status = self.model_manager.get_model_status()
                return status
                
            except Exception as e:
                logger.error(f"Error listing models: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/models/load")
        async def load_model(request: ModelLoadRequest):
            """Load a specific model"""
            try:
                result = await self.model_manager.load_model(
                    request.model_id, 
                    request.force_reload
                )
                
                if not result["success"]:
                    raise HTTPException(status_code=400, detail=result["error"])
                
                return result
                
            except Exception as e:
                logger.error(f"Error loading model {request.model_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/generate", response_model=TextGenerationResponse)
        async def generate_text(request: TextGenerationRequest):
            """Generate text using a loaded model"""
            try:
                # Auto-select model if not specified
                if not request.model_id:
                    # Find first loaded model
                    loaded_models = [
                        mid for mid, state in self.model_manager.model_states.items()
                        if state.value == "ready"
                    ]
                    
                    if not loaded_models:
                        raise HTTPException(
                            status_code=400, 
                            detail="No model specified and no models are loaded"
                        )
                    
                    request.model_id = loaded_models[0]
                
                # Generate text
                result = await self.model_manager.generate_text(
                    model_id=request.model_id,
                    prompt=request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p
                )
                
                if not result["success"]:
                    raise HTTPException(status_code=400, detail=result["error"])
                
                return TextGenerationResponse(
                    text=result["text"],
                    model_id=result["model_id"],
                    inference_time=result["inference_time"],
                    tokens_generated=result["tokens_generated"],
                    success=True
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error generating text: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/system/resources")
        async def get_system_resources():
            """Get current system resource usage"""
            try:
                return self.model_manager.get_resource_usage()
                
            except Exception as e:
                logger.error(f"Error getting system resources: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def run(self, host: str = "127.0.0.1", port: int = 8001):
        """Run the local model server"""
        logger.info(f"Starting Local Model Server on {host}:{port}")
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )


if __name__ == "__main__":
    # Example usage
    config = {
        'model_manager': {
            'models_dir': '/opt/sutazaiapp/data/models',
            'cache_dir': '/opt/sutazaiapp/data/model_cache',
            'max_total_memory_gb': 16
        }
    }
    
    server = LocalModelServer(config)
    server.run(host="127.0.0.1", port=8001)