"""
Universal Agent Runtime

This runtime allows any agent to run independently of Claude using
Ollama, LiteLLM, or any OpenAI-compatible API.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import requests
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SutazAI Universal Agent")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class ChatResponse(BaseModel):
    response: str
    model: str
    agent: str
    timestamp: str


class AgentInfo(BaseModel):
    name: str
    capabilities: List[str]
    model_provider: str
    model_name: str
    status: str


class UniversalAgentRuntime:
    """Runtime for universal agents."""
    
    def __init__(self):
        """Initialize the runtime from environment variables."""
        self.agent_name = os.getenv("AGENT_NAME", "universal-agent")
        self.capabilities = os.getenv("AGENT_CAPABILITIES", "").split(",")
        self.model_provider = os.getenv("MODEL_PROVIDER", "ollama")
        self.model_name = os.getenv("MODEL_NAME", "llama2:latest")
        self.temperature = float(os.getenv("MODEL_TEMPERATURE", "0.7"))
        self.max_tokens = int(os.getenv("MODEL_MAX_TOKENS", "4096"))
        self.system_prompt = os.getenv("SYSTEM_PROMPT", "You are a helpful AI assistant.")
        
        # Provider endpoints
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://ollama:11434")
        self.litellm_host = os.getenv("LITELLM_HOST", "http://litellm:4000")
        
        logger.info(f"Initialized {self.agent_name} with {self.model_provider}/{self.model_name}")
    
    async def chat(self, messages: List[Message], **kwargs) -> str:
        """Process a chat request."""
        # Add system prompt as first message
        full_messages = [
            {"role": "system", "content": self.system_prompt}
        ] + [msg.dict() for msg in messages]
        
        # Override parameters if provided
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        try:
            if self.model_provider == "ollama":
                return await self._chat_ollama(full_messages, temperature, max_tokens)
            elif self.model_provider == "litellm":
                return await self._chat_litellm(full_messages, temperature, max_tokens)
            else:
                raise ValueError(f"Unknown provider: {self.model_provider}")
        except Exception as e:
            logger.error(f"Chat error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _chat_ollama(self, messages: List[Dict], temperature: float, max_tokens: int) -> str:
        """Chat using Ollama."""
        # Convert messages to Ollama format
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "user":
                prompt += f"Human: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"
        
        prompt += "Assistant: "
        
        # Make request to Ollama
        response = requests.post(
            f"{self.ollama_host}/api/generate",
            json={
                "model": self.model_name.replace("ollama/", ""),
                "prompt": prompt,
                "temperature": temperature,
                "num_predict": max_tokens,
                "stream": False
            }
        )
        
        if response.status_code != 200:
            raise Exception(f"Ollama error: {response.text}")
        
        return response.json()["response"]
    
    async def _chat_litellm(self, messages: List[Dict], temperature: float, max_tokens: int) -> str:
        """Chat using LiteLLM."""
        response = requests.post(
            f"{self.litellm_host}/chat/completions",
            json={
                "model": f"sutazai/{self.agent_name}",
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        )
        
        if response.status_code != 200:
            raise Exception(f"LiteLLM error: {response.text}")
        
        return response.json()["choices"][0]["message"]["content"]
    
    def get_info(self) -> AgentInfo:
        """Get agent information."""
        return AgentInfo(
            name=self.agent_name,
            capabilities=self.capabilities,
            model_provider=self.model_provider,
            model_name=self.model_name,
            status="running"
        )


# Initialize runtime
runtime = UniversalAgentRuntime()


@app.get("/")
async def root():
    """Root endpoint."""
    return {"agent": runtime.agent_name, "status": "online"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "agent": runtime.agent_name}


@app.get("/info", response_model=AgentInfo)
async def get_agent_info():
    """Get agent information."""
    return runtime.get_info()


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat request."""
    response = await runtime.chat(
        request.messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens
    )
    
    return ChatResponse(
        response=response,
        model=runtime.model_name,
        agent=runtime.agent_name,
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/v1/chat/completions")
async def openai_compatible_chat(request: Dict[str, Any]):
    """OpenAI-compatible chat endpoint."""
    messages = [Message(**msg) for msg in request.get("messages", [])]
    
    response = await runtime.chat(
        messages,
        temperature=request.get("temperature"),
        max_tokens=request.get("max_tokens")
    )
    
    # Return OpenAI-compatible response
    return {
        "id": f"chatcmpl-{runtime.agent_name}",
        "object": "chat.completion",
        "created": int(datetime.utcnow().timestamp()),
        "model": runtime.model_name,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": -1,
            "completion_tokens": -1,
            "total_tokens": -1
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)