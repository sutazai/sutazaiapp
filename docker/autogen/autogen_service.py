from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import autogen
from autogen import AssistantAgent, UserProxyAgent
import asyncio
import json
import httpx

app = FastAPI(title="AutoGen Multi-Agent Service - Native Ollama")

class TaskRequest(BaseModel):
    task: str
    agents: List[str] = ["assistant", "user_proxy"]
    max_rounds: int = 10
    require_human_input: bool = False
    code_execution: bool = True

class TaskResponse(BaseModel):
    status: str
    result: Any
    chat_history: List[Dict[str, Any]]
    execution_log: List[str]

# Native Ollama client
class OllamaClient:
    def __init__(self, base_url="http://ollama:11434"):
        self.base_url = base_url
        
    async def chat(self, messages, model="gpt-oss:latest"):
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False
                }
            )
            return response.json()

ollama_client = OllamaClient()

def get_llm_config():
    """Get LLM configuration for native Ollama"""
    return {
        "config_list": [{
            "model": "gpt-oss:latest",
            "base_url": "http://ollama:11434",
            "api_type": "ollama"
        }],
        "temperature": 0.7,
        "cache_seed": 42
    }

# Rest of the AutoGen code remains the same...
@app.post("/execute", response_model=TaskResponse)
async def execute_task(request: TaskRequest):
    """Execute a task using AutoGen agents with native Ollama"""
    # Implementation continues...
    return TaskResponse(
        status="completed",
        result="Task executed with native Ollama",
        chat_history=[],
        execution_log=[]
    )
