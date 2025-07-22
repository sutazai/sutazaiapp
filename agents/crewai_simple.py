#!/usr/bin/env python3
"""
Simple CrewAI Agent Service - Lightweight Version
"""

from fastapi import FastAPI
import uvicorn
import json
from datetime import datetime

app = FastAPI(title="SutazAI CrewAI Agent", version="1.0")

@app.get("/")
async def root():
    return {"agent": "CrewAI", "status": "active", "timestamp": datetime.now().isoformat()}

@app.get("/health")
async def health():
    return {"status": "healthy", "agent": "crewai", "port": 8102}

@app.post("/chat")
async def chat(data: dict):
    return {
        "agent": "CrewAI",
        "response": f"CrewAI Agent processed: {data.get('message', 'No message')}",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8102)