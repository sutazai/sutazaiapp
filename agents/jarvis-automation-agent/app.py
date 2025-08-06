#!/usr/bin/env python3
"""
Jarvis Automation Agent - Perfect Implementation
Handles task automation, system integration, and command execution
Based on Dipeshpal/Jarvis_AI automation features with enterprise integration
"""

import os
import asyncio
import logging
import json
import time
import subprocess
import shutil
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import httpx
import redis
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://sutazai:password@postgres:5432/sutazai")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

# Redis connection
redis_client = redis.from_url(REDIS_URL)

# Pydantic models
class AutomationRequest(BaseModel):
    task_type: str = Field(..., description="Type of automation task")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    target_system: Optional[str] = None
    session_id: str = Field(default_factory=lambda: f"automation_{int(time.time())}")

class AutomationResponse(BaseModel):
    status: str
    result: Dict[str, Any]
    task_type: str
    execution_time: float
    session_id: str
    system_impact: Optional[str] = None

# FastAPI app
app = FastAPI(title="Jarvis Automation Agent", version="1.0.0")

class SystemAutomationEngine:
    """Core automation engine for system tasks"""
    
    def __init__(self):
        self.supported_tasks = [
            "system_info", "file_operations", "network_diagnostics", 
            "service_management", "monitoring", "cleanup", "backup"
        ]
        self.safe_mode = True  # Enable safe execution mode
    
    async def get_system_info(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Gather system information"""
        try:
            info = {
                "cpu_usage": psutil.cpu_percent(interval=1),
                "memory": {
                    "total": psutil.virtual_memory().total,
                    "available": psutil.virtual_memory().available,
                    "percent": psutil.virtual_memory().percent
                },
                "disk": {
                    "total": psutil.disk_usage('/').total,
                    "free": psutil.disk_usage('/').free,
                    "percent": psutil.disk_usage('/').percent
                },
                "processes": len(psutil.pids()),
                "uptime": time.time() - psutil.boot_time(),
                "network_connections": len(psutil.net_connections())
            }
            
            return {"system_info": info, "timestamp": datetime.now().isoformat()}
            
        except Exception as e:
            logger.error(f"System info error: {e}")
            return {"error": str(e)}
    
    async def handle_file_operations(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle safe file operations"""
        try:
            operation = parameters.get("operation", "list")
            path = parameters.get("path", "/tmp")
            
            # Safety check - only allow operations in safe directories
            safe_dirs = ["/tmp", "/opt/sutazaiapp", "/var/log"]
            if not any(path.startswith(safe_dir) for safe_dir in safe_dirs):
                return {"error": "Operation not allowed in this directory for safety"}
            
            if operation == "list":
                if os.path.exists(path):
                    items = os.listdir(path)
                    return {"operation": "list", "path": path, "items": items[:50]}  # Limit output
                else:
                    return {"error": "Path does not exist"}
            
            elif operation == "info":
                if os.path.exists(path):
                    stat = os.stat(path)
                    return {
                        "operation": "info",
                        "path": path,
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "is_file": os.path.isfile(path),
                        "is_dir": os.path.isdir(path)
                    }
                else:
                    return {"error": "Path does not exist"}
            
            elif operation == "create_dir":
                new_dir = parameters.get("new_path")
                if new_dir and any(new_dir.startswith(safe_dir) for safe_dir in safe_dirs):
                    os.makedirs(new_dir, exist_ok=True)
                    return {"operation": "create_dir", "path": new_dir, "status": "created"}
                else:
                    return {"error": "Invalid path for directory creation"}
            
            else:
                return {"error": f"Unsupported file operation: {operation}"}
                
        except Exception as e:
            logger.error(f"File operations error: {e}")
            return {"error": str(e)}
    
    async def network_diagnostics(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform network diagnostics"""
        try:
            target = parameters.get("target", "google.com")
            diagnostic_type = parameters.get("type", "ping")
            
            if diagnostic_type == "ping":
                # Safe ping command
                result = subprocess.run(
                    ["ping", "-c", "3", target],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                return {
                    "diagnostic": "ping",
                    "target": target,
                    "success": result.returncode == 0,
                    "output": result.stdout[:500]  # Limit output
                }
            
            elif diagnostic_type == "connectivity":
                # Test connectivity to known services
                services = {
                    "ollama": f"http://ollama:11434",
                    "backend": f"http://backend:8000",
                    "postgres": "postgres:5432",
                    "redis": "redis:6379"
                }
                
                connectivity_results = {}
                for service, url in services.items():
                    try:
                        if url.startswith("http"):
                            async with httpx.AsyncClient() as client:
                                response = await client.get(url, timeout=5.0)
                                connectivity_results[service] = response.status_code == 200
                        else:
                            # For non-HTTP services, just mark as testable
                            connectivity_results[service] = "connection_test_available"
                    except:
                        connectivity_results[service] = False
                
                return {"diagnostic": "connectivity", "results": connectivity_results}
            
            else:
                return {"error": f"Unsupported diagnostic type: {diagnostic_type}"}
                
        except Exception as e:
            logger.error(f"Network diagnostics error: {e}")
            return {"error": str(e)}
    
    async def service_management(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Manage system services safely"""
        try:
            action = parameters.get("action", "status")
            service_name = parameters.get("service", "docker")
            
            # Only allow status checks for safety
            if action == "status":
                try:
                    # Check if service is running via process name
                    for proc in psutil.process_iter(['pid', 'name', 'status']):
                        if service_name in proc.info['name']:
                            return {
                                "service": service_name,
                                "status": "running",
                                "pid": proc.info['pid']
                            }
                    
                    return {"service": service_name, "status": "not_found"}
                    
                except Exception:
                    return {"service": service_name, "status": "unknown"}
            
            else:
                return {"error": "Only status checks are allowed for safety"}
                
        except Exception as e:
            logger.error(f"Service management error: {e}")
            return {"error": str(e)}
    
    async def system_monitoring(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Provide system monitoring data"""
        try:
            monitor_type = parameters.get("type", "overview")
            
            if monitor_type == "overview":
                return await self.get_system_info(parameters)
            
            elif monitor_type == "processes":
                top_processes = []
                for proc in sorted(psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']), 
                                 key=lambda x: x.info['cpu_percent'], reverse=True)[:10]:
                    top_processes.append(proc.info)
                
                return {"monitor_type": "processes", "top_processes": top_processes}
            
            elif monitor_type == "network":
                net_stats = psutil.net_io_counters()
                return {
                    "monitor_type": "network",
                    "bytes_sent": net_stats.bytes_sent,
                    "bytes_recv": net_stats.bytes_recv,
                    "packets_sent": net_stats.packets_sent,
                    "packets_recv": net_stats.packets_recv
                }
            
            else:
                return {"error": f"Unsupported monitor type: {monitor_type}"}
                
        except Exception as e:
            logger.error(f"System monitoring error: {e}")
            return {"error": str(e)}
    
    async def system_cleanup(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform safe system cleanup"""
        try:
            cleanup_type = parameters.get("type", "temp")
            
            if cleanup_type == "temp":
                # Clean safe temp directories
                temp_dirs = ["/tmp", "/opt/sutazaiapp/temp"]
                cleaned_files = 0
                cleaned_size = 0
                
                for temp_dir in temp_dirs:
                    if os.path.exists(temp_dir):
                        for item in os.listdir(temp_dir):
                            item_path = os.path.join(temp_dir, item)
                            if os.path.isfile(item_path) and item.startswith("temp_"):
                                try:
                                    size = os.path.getsize(item_path)
                                    os.remove(item_path)
                                    cleaned_files += 1
                                    cleaned_size += size
                                except:
                                    pass
                
                return {
                    "cleanup_type": "temp",
                    "files_cleaned": cleaned_files,
                    "size_freed": cleaned_size,
                    "status": "completed"
                }
            
            else:
                return {"error": f"Unsupported cleanup type: {cleanup_type}"}
                
        except Exception as e:
            logger.error(f"System cleanup error: {e}")
            return {"error": str(e)}

# Initialize automation engine
automation_engine = SystemAutomationEngine()

# API Routes
@app.get("/")
async def root():
    return {"agent": "Jarvis Automation Agent", "status": "active", "version": "1.0.0"}

@app.get("/health")
async def health():
    """Health check endpoint"""
    try:
        # Test Redis connection
        redis_client.ping()
        
        # Test system access
        cpu_usage = psutil.cpu_percent()
        
        return {
            "status": "healthy",
            "agent": "jarvis-automation-agent",
            "redis": "connected",
            "system_access": "available",
            "cpu_usage": cpu_usage,
            "supported_tasks": automation_engine.supported_tasks
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))

@app.post("/process", response_model=AutomationResponse)
async def process_automation_request(request: AutomationRequest):
    """Main automation processing endpoint"""
    start_time = time.time()
    
    try:
        if request.task_type not in automation_engine.supported_tasks:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported task type. Supported: {automation_engine.supported_tasks}"
            )
        
        # Route to appropriate handler
        if request.task_type == "system_info":
            result = await automation_engine.get_system_info(request.parameters)
        elif request.task_type == "file_operations":
            result = await automation_engine.handle_file_operations(request.parameters)
        elif request.task_type == "network_diagnostics":
            result = await automation_engine.network_diagnostics(request.parameters)
        elif request.task_type == "service_management":
            result = await automation_engine.service_management(request.parameters)
        elif request.task_type == "monitoring":
            result = await automation_engine.system_monitoring(request.parameters)
        elif request.task_type == "cleanup":
            result = await automation_engine.system_cleanup(request.parameters)
        else:
            result = {"error": f"Handler not implemented for {request.task_type}"}
        
        execution_time = time.time() - start_time
        
        return AutomationResponse(
            status="success" if "error" not in result else "error",
            result=result,
            task_type=request.task_type,
            execution_time=execution_time,
            session_id=request.session_id,
            system_impact="minimal" if "error" not in result else None
        )
        
    except Exception as e:
        logger.error(f"Automation processing error: {e}")
        execution_time = time.time() - start_time
        
        return AutomationResponse(
            status="error",
            result={"error": str(e)},
            task_type=request.task_type,
            execution_time=execution_time,
            session_id=request.session_id
        )

@app.get("/capabilities")
async def get_automation_capabilities():
    """Return automation capabilities"""
    return {
        "supported_tasks": automation_engine.supported_tasks,
        "safe_mode": automation_engine.safe_mode,
        "system_access": ["cpu_monitoring", "memory_monitoring", "disk_monitoring", "process_monitoring"],
        "file_operations": ["list", "info", "create_dir"],
        "network_diagnostics": ["ping", "connectivity"],
        "service_management": ["status_check"],
        "monitoring": ["overview", "processes", "network"],
        "cleanup": ["temp_files"]
    }

@app.post("/quick-system-check")
async def quick_system_check():
    """Quick system health and status check"""
    try:
        system_info = await automation_engine.get_system_info({})
        return {"status": "success", "system_info": system_info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("Starting Jarvis Automation Agent")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        log_level="info"
    )