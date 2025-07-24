#!/usr/bin/env python3
"""
Sample External Agent for Testing
Simulates an agent running on port 8102 (CrewAI)
"""

import json
import time
from datetime import datetime
from fastapi import FastAPI
import uvicorn

app = FastAPI(title="Sample CrewAI Agent", version="1.0")

@app.get("/api/health")
async def health():
    return {
        "status": "online",
        "name": "CrewAI",
        "timestamp": datetime.now().isoformat(),
        "capabilities": ["reasoning", "multi-agent", "task_planning"]
    }

@app.get("/health")
async def alt_health():
    return await health()

@app.post("/api/generate")
async def generate(request: dict):
    # Simulate processing time
    await asyncio.sleep(1)
    
    return {
        "response": f"CrewAI response: {request.get('prompt', 'Hello')}",
        "agent": "crewai",
        "timestamp": datetime.now().isoformat(),
        "processing_time": 1.0
    }

if __name__ == "__main__":
    print("🤖 Starting Sample CrewAI Agent on port 8102")
    uvicorn.run(app, host="0.0.0.0", port=8102)