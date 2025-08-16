#!/usr/bin/env python3
"""
Base MCP HTTP Server
Provides HTTP interface for MCP services with service discovery integration
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Metrics
mcp_requests = Counter('mcp_requests_total', 'Total MCP requests', ['service', 'method'])
mcp_request_duration = Histogram('mcp_request_duration_seconds', 'MCP request duration', ['service', 'method'])
mcp_errors = Counter('mcp_errors_total', 'Total MCP errors', ['service', 'error_type'])

class MCPHTTPServer:
    def __init__(self):
        self.service_name = os.getenv('MCP_SERVICE_NAME', 'mcp-service')
        self.service_port = int(os.getenv('MCP_SERVICE_PORT', '11100'))
        self.consul_agent = os.getenv('CONSUL_AGENT', 'mcp-consul-agent:8500')
        self.mcp_process: Optional[subprocess.Popen] = None
        self.consul_client = None
        self.service_id = f"{self.service_name}-{self.service_port}"
        
    async def start_mcp_process(self):
        """Start the underlying MCP service process"""
        try:
            # Look for wrapper script
            wrapper_script = f"/opt/mcp/wrappers/{self.service_name}.sh"
            if not os.path.exists(wrapper_script):
                logger.warning(f"Wrapper script not found: {wrapper_script}")
                return False
            
            # Start MCP process
            self.mcp_process = subprocess.Popen(
                [wrapper_script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={
                    **os.environ,
                    'MCP_STDIO_MODE': 'true'
                }
            )
            
            logger.info(f"Started MCP process for {self.service_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start MCP process: {e}")
            return False
    
    async def register_with_consul(self):
        """Register service with Consul"""
        try:
            import consul
            self.consul_client = consul.Consul(host=self.consul_agent.split(':')[0], 
                                             port=int(self.consul_agent.split(':')[1]))
            
            # Service registration
            service_data = {
                'name': f"mcp-{self.service_name}",
                'service_id': self.service_id,
                'port': self.service_port,
                'address': 'localhost',
                'tags': ['mcp', self.service_name],
                'check': {
                    'http': f"http://localhost:{self.service_port}/health",
                    'interval': '10s',
                    'timeout': '5s',
                    'deregister_critical_service_after': '1m'
                },
                'meta': {
                    'version': '1.0.0',
                    'mcp_service': 'true'
                }
            }
            
            self.consul_client.agent.service.register(**service_data)
            logger.info(f"Registered {self.service_id} with Consul")
            return True
            
        except Exception as e:
            logger.warning(f"Consul registration failed: {e}")
            return False
    
    async def deregister_from_consul(self):
        """Deregister service from Consul"""
        if self.consul_client:
            try:
                self.consul_client.agent.service.deregister(self.service_id)
                logger.info(f"Deregistered {self.service_id} from Consul")
            except Exception as e:
                logger.error(f"Failed to deregister from Consul: {e}")
    
    async def call_mcp_method(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Call MCP method through STDIO interface"""
        if not self.mcp_process or self.mcp_process.poll() is not None:
            raise HTTPException(status_code=503, detail="MCP process not running")
        
        try:
            # Prepare JSON-RPC request
            request = {
                "jsonrpc": "2.0",
                "id": int(time.time() * 1000),
                "method": method,
                "params": params
            }
            
            # Send request to MCP process
            request_data = json.dumps(request) + '\n'
            self.mcp_process.stdin.write(request_data.encode())
            self.mcp_process.stdin.flush()
            
            # Read response
            response_line = self.mcp_process.stdout.readline()
            if not response_line:
                raise Exception("No response from MCP process")
            
            response = json.loads(response_line.decode().strip())
            
            if 'error' in response:
                raise HTTPException(status_code=400, detail=response['error'])
            
            return response.get('result', {})
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            raise HTTPException(status_code=500, detail="Invalid JSON response from MCP")
        except Exception as e:
            logger.error(f"MCP call failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def shutdown(self):
        """Shutdown the server"""
        await self.deregister_from_consul()
        
        if self.mcp_process:
            self.mcp_process.terminate()
            try:
                self.mcp_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.mcp_process.kill()
                self.mcp_process.wait()

# Global server instance
mcp_server = MCPHTTPServer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info(f"Starting MCP HTTP server for {mcp_server.service_name}")
    await mcp_server.start_mcp_process()
    await mcp_server.register_with_consul()
    yield
    # Shutdown
    logger.info(f"Shutting down MCP HTTP server for {mcp_server.service_name}")
    await mcp_server.shutdown()

# FastAPI application
app = FastAPI(
    title=f"MCP {mcp_server.service_name.title()} Service",
    description=f"HTTP interface for {mcp_server.service_name} MCP service",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check if MCP process is running
        if not mcp_server.mcp_process or mcp_server.mcp_process.poll() is not None:
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "reason": "MCP process not running"}
            )
        
        # Check if Consul is available
        consul_healthy = mcp_server.consul_client is not None
        
        return {
            "status": "healthy",
            "service": mcp_server.service_name,
            "port": mcp_server.service_port,
            "consul_registered": consul_healthy,
            "mcp_process_running": True,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "reason": str(e)}
        )

@app.post("/execute")
async def execute_mcp_method(request: Dict[str, Any], background_tasks: BackgroundTasks):
    """Execute MCP method"""
    start_time = time.time()
    method = request.get('method')
    params = request.get('params', {})
    
    try:
        # Record metrics
        mcp_requests.labels(service=mcp_server.service_name, method=method).inc()
        
        # Call MCP method
        result = await mcp_server.call_mcp_method(method, params)
        
        # Record duration
        duration = time.time() - start_time
        mcp_request_duration.labels(service=mcp_server.service_name, method=method).observe(duration)
        
        return {
            "status": "success",
            "result": result,
            "duration": duration,
            "service": mcp_server.service_name,
            "method": method
        }
        
    except HTTPException:
        raise
    except Exception as e:
        mcp_errors.labels(service=mcp_server.service_name, error_type=type(e).__name__).inc()
        logger.error(f"MCP execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

@app.get("/info")
async def get_service_info():
    """Get service information"""
    return {
        "service_name": mcp_server.service_name,
        "service_id": mcp_server.service_id,
        "port": mcp_server.service_port,
        "consul_agent": mcp_server.consul_agent,
        "process_running": mcp_server.mcp_process is not None and mcp_server.mcp_process.poll() is None,
        "consul_registered": mcp_server.consul_client is not None
    }

if __name__ == "__main__":
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    uvicorn.run(
        "mcp_http_server:app",
        host="0.0.0.0",
        port=mcp_server.service_port,
        log_level=log_level.lower(),
        reload=False
    )