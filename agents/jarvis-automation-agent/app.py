#!/usr/bin/env python3
"""
Jarvis Automation Agent

This agent provides intelligent task automation capabilities using AI.
It can analyze tasks, generate automation scripts, and execute workflows.
"""

import os
import sys
import json
import asyncio
import logging
import subprocess
import tempfile
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

# Add the agents core path for BaseAgent import
sys.path.insert(0, '/opt/sutazaiapp/agents')
sys.path.insert(0, '/opt/sutazaiapp')

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Import the consolidated BaseAgent
try:
    from agents.core.base_agent import BaseAgent, AgentCapability, TaskResult
except ImportError:
    from core.base_agent import BaseAgent, AgentCapability, TaskResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AutomationRequest(BaseModel):
    """Request model for automation tasks"""
    task_description: str
    task_type: str = "general"
    parameters: Dict[str, Any] = {}
    execute: bool = False  # If False, only generate script


class JarvisAutomationAgent(BaseAgent):
    """
    Intelligent Automation Agent powered by AI
    
    This agent can:
    - Analyze automation tasks
    - Generate shell scripts and Python code
    - Execute automation workflows
    - Monitor and report on task execution
    """
    
    def __init__(self):
        super().__init__(
            agent_id="jarvis-automation-agent",
            name="Jarvis Automation Agent",
            port=int(os.getenv("PORT", "11102")),
            description="AI-powered task automation and workflow management"
        )
        
        # Add capabilities
        self.add_capability(AgentCapability.CODE_GENERATION)
        self.add_capability(AgentCapability.AUTONOMOUS_EXECUTION)
        self.add_capability(AgentCapability.FILE_OPERATIONS)
        self.add_capability(AgentCapability.API_INTEGRATION)
        
        # Task execution history
        self.execution_history = []
        self.max_history = 100
        
        # Safety settings
        self.safe_mode = True
        self.allowed_commands = {
            'ls', 'cat', 'echo', 'date', 'pwd', 'whoami', 'ps', 'df', 'free',
            'curl', 'wget', 'grep', 'find', 'sort', 'uniq', 'wc', 'head', 'tail'
        }
        
        logger.info(f"Initialized {self.agent_name} with AI automation capabilities")
    
    async def on_initialize(self):
        """Initialize agent-specific components"""
        logger.info("Jarvis Automation Agent initialized with AI capabilities")
        
        # Test Ollama connectivity
        test_response = await self.query_ollama("Hello", max_tokens=10)
        if test_response:
            logger.info("AI capabilities verified - Ollama connection successful")
        else:
            logger.warning("AI capabilities limited - Ollama connection failed")
    
    async def on_task_execute(self, task_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute automation task with AI assistance
        """
        try:
            task_type = task_data.get("type", "automation")
            
            if task_type == "automation":
                return await self._handle_automation_task(task_data)
            elif task_type == "script_generation":
                return await self._handle_script_generation(task_data)
            elif task_type == "workflow_execution":
                return await self._handle_workflow_execution(task_data)
            elif task_type == "system_analysis":
                return await self._handle_system_analysis(task_data)
            else:
                # Default AI-powered task processing
                return await self._process_with_ai(task_data)
                
        except Exception as e:
            logger.error(f"Error executing automation task: {e}")
            return {
                "success": False,
                "error": str(e),
                "task_id": task_id
            }
    
    async def _handle_automation_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general automation tasks"""
        description = task_data.get("description", "")
        execute = task_data.get("execute", False)
        
        # Use AI to analyze and generate automation solution
        ai_prompt = f"""
        You are an expert automation engineer. Analyze this task and provide a solution:
        
        Task: {description}
        
        Please provide:
        1. Analysis of the task
        2. Step-by-step approach
        3. Shell commands or Python code needed
        4. Potential risks or considerations
        
        Format your response as JSON with keys: analysis, steps, commands, risks
        """
        
        ai_response = await self.query_ollama(ai_prompt, max_tokens=500)
        
        if not ai_response:
            return {
                "success": False,
                "error": "AI analysis failed - Ollama unavailable",
                "fallback_analysis": self._fallback_task_analysis(description)
            }
        
        try:
            # Try to parse AI response as JSON
            ai_analysis = json.loads(ai_response)
        except json.JSONDecodeError:
            # If not JSON, create structured response
            ai_analysis = {
                "analysis": "AI provided unstructured response",
                "steps": ["Review AI response", "Extract actionable items"],
                "commands": [],
                "risks": ["Manual review required"],
                "raw_response": ai_response
            }
        
        result = {
            "success": True,
            "task_description": description,
            "ai_analysis": ai_analysis,
            "executed": False,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Execute if requested and safe
        if execute and self._is_safe_to_execute(ai_analysis.get("commands", [])):
            execution_result = await self._execute_commands(ai_analysis.get("commands", []))
            result["execution_result"] = execution_result
            result["executed"] = True
        elif execute:
            result["execution_blocked"] = "Commands blocked by safety checks"
        
        # Store in history
        self._add_to_history(result)
        
        return result
    
    async def _process_with_ai(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generic AI-powered task processing"""
        task_description = task_data.get("description", str(task_data))
        
        ai_prompt = f"""
        You are Jarvis, an intelligent automation assistant. 
        
        Task: {task_description}
        
        Please analyze this task and provide:
        1. Your understanding of what's needed
        2. Suggested approach or solution
        3. Any automation opportunities
        4. Next steps or recommendations
        
        Be helpful, concise, and actionable.
        """
        
        ai_response = await self.query_ollama(ai_prompt, max_tokens=400)
        
        result = {
            "success": True,
            "task_description": task_description,
            "ai_response": ai_response or "AI processing unavailable",
            "agent": self.agent_name,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self._add_to_history(result)
        return result
    
    def _fallback_task_analysis(self, description: str) -> Dict[str, Any]:
        """Provide fallback analysis when AI is unavailable"""
        return {
            "analysis": f"Task requires automation for: {description}",
            "steps": ["Define requirements", "Design solution", "Implement", "Test"],
            "commands": [],
            "risks": ["Manual analysis required - AI unavailable"]
        }
    
    def _is_safe_to_execute(self, commands: List[str]) -> bool:
        """Check if commands are safe to execute"""
        if not self.safe_mode:
            return True
        
        for command in commands:
            if isinstance(command, str):
                cmd_parts = command.strip().split()
                if cmd_parts and cmd_parts[0] not in self.allowed_commands:
                    return False
        
        return True
    
    async def _execute_commands(self, commands: List[str]) -> Dict[str, Any]:
        """Execute a list of shell commands safely"""
        results = []
        
        for cmd in commands:
            if not isinstance(cmd, str):
                continue
            
            try:
                result = subprocess.run(
                    cmd.split(),
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False
                )
                
                results.append({
                    "command": cmd,
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "success": result.returncode == 0
                })
                
            except subprocess.TimeoutExpired:
                results.append({
                    "command": cmd,
                    "error": "Command timed out",
                    "success": False
                })
            except Exception as e:
                results.append({
                    "command": cmd,
                    "error": str(e),
                    "success": False
                })
        
        return {
            "commands_executed": len(results),
            "results": results,
            "overall_success": all(r.get("success", False) for r in results)
        }
    
    def _add_to_history(self, result: Dict[str, Any]):
        """Add task result to execution history"""
        self.execution_history.append(result)
        if len(self.execution_history) > self.max_history:
            self.execution_history.pop(0)  # Remove oldest
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get recent execution history"""
        return self.execution_history.copy()


# FastAPI Integration
app = FastAPI(title="Jarvis Automation Agent", version="2.0.0", 
              description="AI-powered task automation and workflow management")

# Global agent instance
agent_instance: Optional[JarvisAutomationAgent] = None


@app.on_event("startup")
async def startup():
    """Initialize the agent"""
    global agent_instance
    agent_instance = JarvisAutomationAgent()
    await agent_instance.initialize()
    logger.info("Jarvis Automation Agent started with AI capabilities")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    global agent_instance
    if agent_instance:
        await agent_instance.shutdown()
        logger.info("Jarvis Automation Agent shut down")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "jarvis-automation-agent",
        "version": "2.0.0",
        "status": "ready",
        "ai_enabled": True,
        "capabilities": [cap.value for cap in agent_instance.capabilities] if agent_instance else []
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    if not agent_instance:
        return {"status": "initializing"}
    
    health_info = await agent_instance.health_check()
    return health_info


@app.post("/automate")
async def automate_task(request: AutomationRequest):
    """Main automation endpoint"""
    if not agent_instance:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    task_data = {
        "type": "automation",
        "description": request.task_description,
        "task_type": request.task_type,
        "parameters": request.parameters,
        "execute": request.execute
    }
    
    result = await agent_instance.on_task_execute("automation_task", task_data)
    return result


@app.post("/process")
async def process_task(task_data: Dict[str, Any]):
    """Generic task processing endpoint"""
    if not agent_instance:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    result = await agent_instance.on_task_execute("generic", task_data)
    return result


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "11102"))
    logger.info(f"Starting Jarvis Automation Agent on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
