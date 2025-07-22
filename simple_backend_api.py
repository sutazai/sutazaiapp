#!/usr/bin/env python3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import json
from typing import Dict, List, Any, Optional
import asyncio
import uvicorn
import os

app = FastAPI(title="SutazAI Backend API", version="10.0.0")

class TaskRequest(BaseModel):
    task: str
    model: str = "deepseek-r1:8b"
    parameters: Dict[str, Any] = {}

class ServiceStatus(BaseModel):
    service: str
    status: str
    url: str
    details: Optional[Dict[str, Any]] = None

class SutazAIOrchestrator:
    def __init__(self):
        self.services = {
            'ollama': 'http://localhost:11434',
            'chromadb': 'http://localhost:8001',
            'qdrant': 'http://localhost:6333',
            'enhanced_model_manager': 'http://localhost:8098'
        }
        
    async def check_service_health(self, service: str, url: str) -> ServiceStatus:
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                details = response.json()
                return ServiceStatus(service=service, status="healthy", url=url, details=details)
            else:
                return ServiceStatus(service=service, status="unhealthy", url=url)
        except Exception as e:
            return ServiceStatus(service=service, status="error", url=url, details={"error": str(e)})
    
    async def process_with_ollama(self, request: TaskRequest) -> Dict[str, Any]:
        try:
            ollama_request = {
                "model": request.model,
                "prompt": request.task,
                "stream": False,
                "options": request.parameters
            }
            
            response = requests.post(f"{self.services['ollama']}/api/generate", json=ollama_request, timeout=30)
            result = response.json()
            
            return {
                "service": "ollama",
                "model": request.model,
                "response": result.get('response', ''),
                "done": result.get('done', False),
                "stats": {
                    "total_duration": result.get('total_duration', 0),
                    "load_duration": result.get('load_duration', 0),
                    "prompt_eval_count": result.get('prompt_eval_count', 0),
                    "eval_count": result.get('eval_count', 0)
                }
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ollama processing error: {str(e)}")
            
    async def get_available_models(self) -> List[str]:
        try:
            response = requests.get(f"{self.services['ollama']}/api/tags", timeout=10)
            models = response.json().get('models', [])
            return [model['name'] for model in models]
        except Exception as e:
            return ["Error fetching models: " + str(e)]
            
    async def store_in_vector_db(self, text: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        try:
            # Try ChromaDB first
            chroma_data = {
                "documents": [text],
                "metadatas": [metadata or {}],
                "ids": [f"doc_{len(text)}"]
            }
            
            response = requests.post(f"{self.services['chromadb']}/api/v1/collections/sutazai/add", 
                                   json=chroma_data, timeout=10)
            
            if response.status_code == 200:
                return {"status": "stored", "service": "chromadb", "id": chroma_data["ids"][0]}
            else:
                # Fallback to Qdrant
                qdrant_data = {
                    "points": [{
                        "id": len(text),
                        "vector": [0.1] * 384,  # Dummy vector
                        "payload": {"text": text, **(metadata or {})}
                    }]
                }
                
                qdrant_response = requests.put(f"{self.services['qdrant']}/collections/sutazai/points", 
                                             json=qdrant_data, timeout=10)
                return {"status": "stored", "service": "qdrant", "id": len(text)}
                
        except Exception as e:
            return {"status": "error", "error": str(e)}

orchestrator = SutazAIOrchestrator()

@app.get("/")
def root():
    return {"message": "SutazAI Backend API v10.0.0", "status": "active"}

@app.get("/health")
def health():
    return {"status": "healthy", "version": "10.0.0", "services": list(orchestrator.services.keys())}

@app.get("/services/status", response_model=List[ServiceStatus])
async def get_services_status():
    tasks = []
    for service, url in orchestrator.services.items():
        tasks.append(orchestrator.check_service_health(service, url))
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [result for result in results if isinstance(result, ServiceStatus)]

@app.get("/models")
async def get_models():
    models = await orchestrator.get_available_models()
    return {"available_models": models}

@app.post("/process")
async def process_task(request: TaskRequest):
    try:
        # Process with AI model
        result = await orchestrator.process_with_ollama(request)
        
        # Store in vector database
        storage_result = await orchestrator.store_in_vector_db(
            request.task, 
            {"model": request.model, "timestamp": "now"}
        )
        
        return {
            "task": request.task,
            "result": result,
            "storage": storage_result,
            "timestamp": "now"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_completion(request: TaskRequest):
    """Chat completion endpoint similar to OpenAI API"""
    try:
        result = await orchestrator.process_with_ollama(request)
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": result["response"]
                },
                "finish_reason": "stop"
            }],
            "model": request.model,
            "usage": result["stats"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_system_stats():
    services_status = await get_services_status()
    models = await orchestrator.get_available_models()
    
    return {
        "total_services": len(orchestrator.services),
        "healthy_services": len([s for s in services_status if s.status == "healthy"]),
        "available_models": len(models),
        "models": models,
        "services": services_status
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)