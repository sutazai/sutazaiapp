#!/usr/bin/env python3
"""
Simple PrivateGPT Agent Service - Lightweight Version
"""

from fastapi import FastAPI
import uvicorn
import json
from datetime import datetime

app = FastAPI(title="SutazAI PrivateGPT Agent", version="1.0")

@app.get("/")
async def root():
    return {"agent": "PrivateGPT", "status": "active", "timestamp": datetime.now().isoformat()}

@app.get("/health")
async def health():
    return {"status": "healthy", "agent": "privategpt", "port": 8105}

@app.post("/chat")
async def chat(data: dict):
    return {
        "agent": "PrivateGPT",
        "response": f"PrivateGPT Agent processed: {data.get('message', 'No message')}",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8105)