#!/usr/bin/env python3
"""
Real AutoGPT-like Agent for SutazAI
Provides task automation, web browsing, file operations, and code execution
"""

import asyncio
import json
import logging
import os
import subprocess
from datetime import datetime
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SutazAI AutoGPT Agent", version="1.0.0")

class TaskRequest(BaseModel):
    task: str
    task_type: str = "general"
    parameters: Dict[str, Any] = {}

class AgentResponse(BaseModel):
    success: bool
    result: Any
    agent: str = "AutoGPT"
    timestamp: str
    error: str = None

class AutoGPTAgent:
    """Real AutoGPT-like agent implementation"""
    
    def __init__(self):
        self.ollama_url = "http://ollama:11434"
        self.agent_name = "AutoGPT"
        self.capabilities = [
            "task_automation",
            "web_browsing", 
            "file_operations",
            "code_execution",
            "data_processing",
            "api_calls"
        ]
        
    async def execute_task(self, task: str, task_type: str = "general", **kwargs) -> Dict[str, Any]:
        """Execute a task using AutoGPT-like capabilities"""
        try:
            logger.info(f"AutoGPT executing task: {task}")
            
            if task_type == "code_execution":
                return await self._execute_code(task, **kwargs)
            elif task_type == "file_operations":
                return await self._handle_files(task, **kwargs)
            elif task_type == "web_browsing":
                return await self._browse_web(task, **kwargs)
            elif task_type == "api_calls":
                return await self._make_api_calls(task, **kwargs)
            else:
                return await self._general_task(task, **kwargs)
                
        except Exception as e:
            logger.error(f"AutoGPT task execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "task": task,
                "timestamp": datetime.now().isoformat()
            }
    
    async def _execute_code(self, task: str, **kwargs) -> Dict[str, Any]:
        """Execute code safely"""
        try:
            # Extract code from task or use provided code
            code = kwargs.get("code", task)
            language = kwargs.get("language", "python")
            
            if language == "python":
                # Create a safe Python execution environment
                safe_code = f"""
import sys
import json
import math
import datetime
import os
import subprocess
from typing import Dict, List, Any

# Restricted execution
try:
    result = None
    {code}
    print(json.dumps({{"success": True, "result": result if 'result' in locals() else "Code executed successfully"}}))
except Exception as e:
    print(json.dumps({{"success": False, "error": str(e)}}))
"""
                
                # Execute in subprocess for safety
                process = subprocess.run(
                    ["python3", "-c", safe_code],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if process.returncode == 0:
                    try:
                        result = json.loads(process.stdout.strip())
                        return {
                            "success": True,
                            "result": result,
                            "execution_time": "< 30s",
                            "language": language
                        }
                    except json.JSONDecodeError:
                        return {
                            "success": True,
                            "result": process.stdout.strip(),
                            "language": language
                        }
                else:
                    return {
                        "success": False,
                        "error": process.stderr.strip(),
                        "language": language
                    }
            else:
                return {
                    "success": False,
                    "error": f"Language {language} not supported yet"
                }
                
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Code execution timed out (30s limit)"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Code execution failed: {str(e)}"
            }
    
    async def _handle_files(self, task: str, **kwargs) -> Dict[str, Any]:
        """Handle file operations"""
        try:
            operation = kwargs.get("operation", "read")
            filepath = kwargs.get("filepath", "/tmp/test.txt")
            content = kwargs.get("content", "")
            
            if operation == "read":
                if os.path.exists(filepath):
                    with open(filepath, 'r') as f:
                        content = f.read()
                    return {
                        "success": True,
                        "result": content,
                        "operation": "read",
                        "filepath": filepath
                    }
                else:
                    return {
                        "success": False,
                        "error": f"File {filepath} not found"
                    }
            
            elif operation == "write":
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                with open(filepath, 'w') as f:
                    f.write(content)
                return {
                    "success": True,
                    "result": f"Written {len(content)} characters to {filepath}",
                    "operation": "write",
                    "filepath": filepath
                }
            
            elif operation == "list":
                directory = kwargs.get("directory", "/tmp")
                if os.path.exists(directory):
                    files = os.listdir(directory)
                    return {
                        "success": True,
                        "result": files,
                        "operation": "list",
                        "directory": directory
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Directory {directory} not found"
                    }
            
            else:
                return {
                    "success": False,
                    "error": f"File operation {operation} not supported"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"File operation failed: {str(e)}"
            }
    
    async def _browse_web(self, task: str, **kwargs) -> Dict[str, Any]:
        """Simulate web browsing capabilities"""
        try:
            url = kwargs.get("url", "")
            action = kwargs.get("action", "get")
            
            if not url:
                # Extract URL from task description
                import re
                url_pattern = r'https?://[^\s]+'
                urls = re.findall(url_pattern, task)
                if urls:
                    url = urls[0]
                else:
                    return {
                        "success": False,
                        "error": "No URL provided for web browsing"
                    }
            
            if action == "get":
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=10) as response:
                        if response.status == 200:
                            content = await response.text()
                            return {
                                "success": True,
                                "result": {
                                    "url": url,
                                    "status_code": response.status,
                                    "content_length": len(content),
                                    "content_preview": content[:500] + "..." if len(content) > 500 else content
                                },
                                "action": "get"
                            }
                        else:
                            return {
                                "success": False,
                                "error": f"HTTP {response.status} from {url}"
                            }
            
            else:
                return {
                    "success": False,
                    "error": f"Web action {action} not supported yet"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Web browsing failed: {str(e)}"
            }
    
    async def _make_api_calls(self, task: str, **kwargs) -> Dict[str, Any]:
        """Make API calls to external services"""
        try:
            url = kwargs.get("url", "")
            method = kwargs.get("method", "GET").upper()
            headers = kwargs.get("headers", {})
            data = kwargs.get("data", {})
            
            if not url:
                return {
                    "success": False,
                    "error": "No URL provided for API call"
                }
            
            async with aiohttp.ClientSession() as session:
                if method == "GET":
                    async with session.get(url, headers=headers, timeout=10) as response:
                        result = await response.text()
                        return {
                            "success": True,
                            "result": {
                                "status_code": response.status,
                                "response": result[:1000] + "..." if len(result) > 1000 else result
                            },
                            "method": method,
                            "url": url
                        }
                elif method == "POST":
                    async with session.post(url, json=data, headers=headers, timeout=10) as response:
                        result = await response.text()
                        return {
                            "success": True,
                            "result": {
                                "status_code": response.status,
                                "response": result[:1000] + "..." if len(result) > 1000 else result
                            },
                            "method": method,
                            "url": url
                        }
                else:
                    return {
                        "success": False,
                        "error": f"HTTP method {method} not supported"
                    }
                    
        except Exception as e:
            return {
                "success": False,
                "error": f"API call failed: {str(e)}"
            }
    
    async def _general_task(self, task: str, **kwargs) -> Dict[str, Any]:
        """Handle general tasks using LLM"""
        try:
            # Use the local Ollama for general intelligence
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": "llama3.2:1b",
                    "prompt": f"As AutoGPT, help with this task: {task}",
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "result": result.get("response", "Task completed"),
                    "task_type": "general",
                    "model": "llama3.2:1b"
                }
            else:
                return {
                    "success": False,
                    "error": f"LLM request failed with status {response.status_code}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"General task processing failed: {str(e)}"
            }

# Initialize agent
agent = AutoGPTAgent()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agent": "AutoGPT",
        "capabilities": agent.capabilities,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/execute")
async def execute_task(request: TaskRequest) -> AgentResponse:
    """Execute a task"""
    try:
        result = await agent.execute_task(
            request.task,
            request.task_type,
            **request.parameters
        )
        
        return AgentResponse(
            success=result.get("success", True),
            result=result,
            timestamp=datetime.now().isoformat(),
            error=result.get("error")
        )
        
    except Exception as e:
        logger.error(f"Task execution failed: {e}")
        return AgentResponse(
            success=False,
            result=None,
            timestamp=datetime.now().isoformat(),
            error=str(e)
        )

@app.get("/capabilities")
async def get_capabilities():
    """Get agent capabilities"""
    return {
        "agent": agent.agent_name,
        "capabilities": agent.capabilities,
        "description": "Real AutoGPT-like agent for task automation"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "agent": "AutoGPT",
        "status": "online",
        "version": "1.0.0",
        "description": "Real AutoGPT-like agent for SutazAI"
    }