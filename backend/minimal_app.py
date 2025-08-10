#!/usr/bin/env python3
"""
Quick minimal automation backend for immediate deployment
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="SutazAI Backend automation",
    description="SutazAI automation Backend Service - Quick Deploy",
    version="1.0.0"
)

# Add CORS middleware - SECURE: No wildcards
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",    # Streamlit dev server
        "http://localhost:10011",   # Frontend Streamlit UI
        "http://localhost:10010",   # Backend API
        "http://127.0.0.1:8501",    # Alt Streamlit dev server
        "http://127.0.0.1:10011",   # Alt frontend
        "http://127.0.0.1:10010",   # Alt backend
        "http://172.31.77.193:8501", # Specific IP if needed
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

@app.get("/")
async def root():
    return {
        "message": "SutazAI Backend automation is running!",
        "status": "healthy",
        "version": "1.0.0",
        "service": "backend"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "backend",
        "uptime": "running"
    }

@app.get("/api/v1/agents")
async def list_agents():
    return {
        "agents": [
            {
                "id": "sutazai-core",
                "name": "SutazAI Core Agent",
                "status": "active",
                "type": "core"
            }
        ],
        "message": "Agent system ready",
        "total": 1
    }

@app.post("/api/v1/agents/{agent_id}/invoke")
async def invoke_agent(agent_id: str, request_data: dict = None):
    return {
        "agent_id": agent_id,
        "status": "invoked",
        "response": "Agent system is initializing, please check back soon.",
        "message": "Request processed"
    }

@app.get("/api/v1/system/status")
async def system_status():
    return {
        "backend": "healthy",
        "database": "connecting",
        "redis": "connecting", 
        "agents": "initializing",
        "overall_status": "operational"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)