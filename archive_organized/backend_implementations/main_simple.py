from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI(title="SutazAI Backend", version="9.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "SutazAI Backend", "version": "9.0"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "services": {
            "postgres": "connected",
            "redis": "connected",
            "ollama": "connected"
        }
    }

@app.get("/api/system/status")
async def system_status():
    return {
        "status": "operational",
        "agents": 48,
        "models": 4,
        "requests": 1234
    }

@app.get("/api/models")
async def list_models():
    return {
        "models": [
            {"name": "deepseek-r1:8b", "size": "8B"},
            {"name": "qwen2.5:3b", "size": "3B"},
            {"name": "llama3.2:3b", "size": "3B"},
            {"name": "nomic-embed-text", "size": "137M"}
        ]
    }

@app.get("/api/agents")
async def list_agents():
    return {
        "agents": [
            {"name": "AutoGPT", "status": "ready"},
            {"name": "CrewAI", "status": "ready"},
            {"name": "GPT-Engineer", "status": "ready"},
            {"name": "Aider", "status": "ready"}
        ]
    }

@app.post("/api/v1/chat")
async def chat(message: dict):
    return {
        "response": f"Echo: {message.get('content', '')}",
        "model": "local",
        "timestamp": "2025-07-21T20:00:00Z"
    }
