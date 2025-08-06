#!/usr/bin/env python3
"""
Purpose: LangChain service providing chain execution and agent capabilities
Usage: python main.py
Requirements: langchain, fastapi, uvicorn, redis
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import redis
import consul
from langchain.llms import Ollama
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.callbacks import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="LangChain Service", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)
consul_client = consul.Consul(host='consul', port=8500)

# Service configuration
SERVICE_NAME = os.getenv('SERVICE_NAME', 'langchain')
INSTANCE_ID = os.getenv('INSTANCE_ID', '0')
OLLAMA_BASE_URL = 'http://ollama:10104'

# Memory storage
conversations = {}

class ChainRequest(BaseModel):
    """Request model for chain execution"""
    prompt: str
    model: str = "tinyllama"
    temperature: float = 0.7
    max_tokens: int = 256
    conversation_id: Optional[str] = None
    chain_type: str = "simple"  # simple, conversation, custom

class ChainResponse(BaseModel):
    """Response model for chain execution"""
    result: str
    conversation_id: str
    tokens_used: int
    execution_time: float

@app.on_event("startup")
async def startup_event():
    """Register service with Consul on startup"""
    try:
        consul_client.agent.service.register(
            name=SERVICE_NAME,
            service_id=f"{SERVICE_NAME}-{INSTANCE_ID}",
            address=SERVICE_NAME,
            port=8080,
            tags=['ai', 'langchain', 'nlp'],
            check=consul.Check.http(f"http://{SERVICE_NAME}:8080/health", interval="30s")
        )
        logger.info(f"Registered {SERVICE_NAME}-{INSTANCE_ID} with Consul")
    except Exception as e:
        logger.error(f"Failed to register with Consul: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Deregister service from Consul on shutdown"""
    try:
        consul_client.agent.service.deregister(f"{SERVICE_NAME}-{INSTANCE_ID}")
        logger.info(f"Deregistered {SERVICE_NAME}-{INSTANCE_ID} from Consul")
    except Exception as e:
        logger.error(f"Failed to deregister from Consul: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": SERVICE_NAME,
        "instance": INSTANCE_ID,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/execute", response_model=ChainResponse)
async def execute_chain(request: ChainRequest, background_tasks: BackgroundTasks):
    """Execute a LangChain chain"""
    start_time = datetime.utcnow()
    
    try:
        # Update activity timestamp
        redis_client.set(f"service:activity:{SERVICE_NAME}", datetime.utcnow().isoformat(), ex=3600)
        
        # Initialize LLM
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        llm = Ollama(
            model=request.model,
            base_url=OLLAMA_BASE_URL,
            temperature=request.temperature,
            callback_manager=callback_manager
        )
        
        # Get or create conversation memory
        conversation_id = request.conversation_id or f"{SERVICE_NAME}-{datetime.utcnow().timestamp()}"
        
        if request.chain_type == "conversation":
            if conversation_id not in conversations:
                conversations[conversation_id] = ConversationBufferMemory()
            memory = conversations[conversation_id]
            
            # Create conversation chain
            template = """The following is a friendly conversation between a human and an AI. 
            The AI is talkative and provides lots of specific details from its context. 
            If the AI does not know the answer to a question, it truthfully says it does not know.

            Current conversation:
            {history}
            Human: {input}
            AI:"""
            
            prompt = PromptTemplate(
                input_variables=["history", "input"],
                template=template
            )
            
            chain = LLMChain(
                llm=llm,
                prompt=prompt,
                memory=memory,
                verbose=True
            )
            
            result = chain.predict(input=request.prompt)
            
        else:
            # Simple chain without memory
            template = """{input}"""
            prompt = PromptTemplate(
                input_variables=["input"],
                template=template
            )
            
            chain = LLMChain(
                llm=llm,
                prompt=prompt,
                verbose=True
            )
            
            result = chain.predict(input=request.prompt)
        
        # Calculate execution time
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Estimate tokens (rough estimation)
        tokens_used = len(request.prompt.split()) + len(result.split())
        
        # Cache result
        cache_key = f"langchain:result:{conversation_id}:{request.prompt[:50]}"
        redis_client.set(cache_key, json.dumps({
            "result": result,
            "tokens_used": tokens_used,
            "execution_time": execution_time
        }), ex=3600)
        
        # Log metrics
        background_tasks.add_task(log_metrics, {
            "service": SERVICE_NAME,
            "model": request.model,
            "chain_type": request.chain_type,
            "tokens_used": tokens_used,
            "execution_time": execution_time
        })
        
        return ChainResponse(
            result=result,
            conversation_id=conversation_id,
            tokens_used=tokens_used,
            execution_time=execution_time
        )
        
    except Exception as e:
        logger.error(f"Chain execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history"""
    if conversation_id in conversations:
        memory = conversations[conversation_id]
        return {
            "conversation_id": conversation_id,
            "history": memory.buffer
        }
    else:
        raise HTTPException(status_code=404, detail="Conversation not found")

@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation"""
    if conversation_id in conversations:
        del conversations[conversation_id]
        return {"message": "Conversation deleted"}
    else:
        raise HTTPException(status_code=404, detail="Conversation not found")

@app.get("/models")
async def list_models():
    """List available models from Ollama"""
    try:
        import requests
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
        if response.status_code == 200:
            return response.json()
        else:
            return {"models": []}
    except Exception as e:
        logger.error(f"Failed to fetch models: {e}")
        return {"models": []}

async def log_metrics(metrics: Dict[str, Any]):
    """Log metrics to monitoring system"""
    try:
        # Store in Redis for Prometheus exporter
        redis_client.hincrby("metrics:requests", metrics["service"], 1)
        redis_client.hincrby("metrics:tokens", metrics["model"], metrics["tokens_used"])
        redis_client.lpush("metrics:execution_times", metrics["execution_time"])
        redis_client.ltrim("metrics:execution_times", 0, 999)  # Keep last 1000
    except Exception as e:
        logger.error(f"Failed to log metrics: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)