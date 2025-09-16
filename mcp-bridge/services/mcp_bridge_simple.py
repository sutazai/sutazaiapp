#!/usr/bin/env python3
"""
Simplified MCP Bridge Server
Minimal version with basic functionality
"""

import json
import logging
from typing import Dict, Any
from datetime import datetime
import asyncio

# Try to import FastAPI, fallback to basic HTTP server if not available
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    print("FastAPI not available, using basic HTTP server")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if HAS_FASTAPI:
    # FastAPI app
    app = FastAPI(
        title="SutazAI MCP Bridge",
        description="Message Control Protocol Bridge for AI Agent Integration",
        version="1.0.0"
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Service Registry (simplified)
    SERVICE_REGISTRY = {
        "postgres": {"url": "postgresql://localhost:5432", "type": "database", "status": "unknown"},
        "redis": {"url": "redis://localhost:6379", "type": "cache", "status": "unknown"},
        "backend": {"url": "http://localhost:8000", "type": "api", "status": "unknown"},
        "mcp-bridge": {"url": "http://localhost:11100", "type": "bridge", "status": "running"},
    }

    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "service": "MCP Bridge",
            "status": "running",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat()
        }

    @app.get("/health")
    async def health():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "service": "mcp-bridge",
            "timestamp": datetime.now().isoformat(),
            "uptime": "running"
        }

    @app.get("/status")
    async def status():
        """Status endpoint"""
        return {
            "service": "MCP Bridge",
            "status": "operational",
            "services": SERVICE_REGISTRY,
            "timestamp": datetime.now().isoformat()
        }

    @app.get("/api/services")
    async def list_services():
        """List all registered services"""
        return {
            "services": SERVICE_REGISTRY,
            "count": len(SERVICE_REGISTRY),
            "timestamp": datetime.now().isoformat()
        }

    @app.get("/api/agents")
    async def list_agents():
        """List all registered agents"""
        return {
            "agents": [],
            "count": 0,
            "message": "No agents currently registered",
            "timestamp": datetime.now().isoformat()
        }

    @app.post("/api/message")
    async def send_message(data: dict):
        """Send a message through the bridge"""
        return {
            "status": "received",
            "message_id": f"msg_{datetime.now().timestamp()}",
            "data": data,
            "timestamp": datetime.now().isoformat()
        }

    def run_server():
        """Run the FastAPI server"""
        logger.info("Starting MCP Bridge Server on port 11100")
        uvicorn.run(app, host="0.0.0.0", port=11100, log_level="info")

else:
    # Fallback to basic HTTP server
    from http.server import HTTPServer, BaseHTTPRequestHandler
    
    class MCPBridgeHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            """Handle GET requests"""
            if self.path == '/health':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {
                    "status": "healthy",
                    "service": "mcp-bridge",
                    "timestamp": datetime.now().isoformat()
                }
                self.wfile.write(json.dumps(response).encode())
            elif self.path == '/':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {
                    "service": "MCP Bridge",
                    "status": "running",
                    "version": "1.0.0",
                    "timestamp": datetime.now().isoformat()
                }
                self.wfile.write(json.dumps(response).encode())
            else:
                self.send_response(404)
                self.end_headers()

    def run_server():
        """Run the basic HTTP server"""
        logger.info("Starting Basic MCP Bridge Server on port 11100")
        server = HTTPServer(('0.0.0.0', 11100), MCPBridgeHandler)
        server.serve_forever()

if __name__ == "__main__":
    try:
        run_server()
    except KeyboardInterrupt:
        logger.info("MCP Bridge Server stopped")
    except Exception as e:
        logger.error(f"Error running server: {e}")
        raise