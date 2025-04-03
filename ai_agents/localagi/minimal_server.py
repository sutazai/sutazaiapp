#!/usr/bin/env python3
"""
Minimal LocalAGI Server
This is a simplified LocalAGI server that provides basic AGI endpoints.
"""

import sys
import logging
import argparse

# Set up command line arguments
parser = argparse.ArgumentParser(description='Minimal LocalAGI Server')
parser.add_argument('--port', type=int, default=8090, help='Port to run the server on')
parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
args = parser.parse_args()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='/opt/sutazaiapp/logs/localagi.log',
    filemode='a'
)
logger = logging.getLogger("LocalAGI")

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    import uvicorn
except ImportError:
    logger.error("Required packages not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn"])
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    import uvicorn

# Create FastAPI app
app = FastAPI(title="LocalAGI", description="Minimal LocalAGI Server")

@app.get("/health")
async def health_check():
    """Check if the server is running."""
    return {"status": "ok", "service": "LocalAGI"}

@app.post("/agent/execute")
async def execute_agent(request: Request):
    """Execute an agent task."""
    try:
        data = await request.json()
        agent_type = data.get("agent_type", "default")
        task = data.get("task", {})
        
        logger.info(f"Executing agent: {agent_type}")
        
        # Simulate agent execution
        result = {
            "status": "success",
            "agent_type": agent_type,
            "result": f"Processed task with {agent_type} agent",
            "details": task
        }
        
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error executing agent: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.get("/agents/list")
async def list_agents():
    """List available agents."""
    agents = [
        {"id": "general", "name": "General Purpose Agent", "description": "Handles general tasks"},
        {"id": "document", "name": "Document Agent", "description": "Processes documents"},
        {"id": "code", "name": "Code Agent", "description": "Analyzes and generates code"}
    ]
    return {"agents": agents}

if __name__ == "__main__":
    logger.info(f"Starting LocalAGI server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
