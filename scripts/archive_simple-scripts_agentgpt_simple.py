#!/usr/bin/env python3
"""
Simple AgentGPT Agent Service - Lightweight Version
"""

from fastapi import FastAPI
import uvicorn
import json
from datetime import datetime

app = FastAPI(title="SutazAI AgentGPT Agent", version="1.0")

@app.get("/")
async def root():
    return {"agent": "AgentGPT", "status": "active", "timestamp": datetime.now().isoformat()}

@app.get("/health")
async def health():
    return {"status": "healthy", "agent": "agentgpt", "port": 8104}

@app.post("/chat")
async def chat(data: dict):
    return {
        "agent": "AgentGPT",
        "response": f"AgentGPT Agent processed: {data.get('message', 'No message')}",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8104)