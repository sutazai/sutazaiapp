#!/usr/bin/env python3
"""
Jarvis Hardware Resource Optimizer

AI-powered hardware resource monitoring and optimization agent.
Provides intelligent recommendations for system performance improvements.
"""

import os
import sys
import json
import asyncio
import logging
import psutil
import subprocess
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

# Add the agents core path for BaseAgent import
sys.path.insert(0, '/opt/sutazaiapp/agents')
sys.path.insert(0, '/opt/sutazaiapp')

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Import the consolidated BaseAgent from the core module
from agents.core.base_agent import BaseAgent, AgentCapability, TaskResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OptimizationRequest(BaseModel):
    """Request model for optimization tasks"""
    resource_type: str = "all"  # cpu, memory, disk, network, all
    severity: str = "medium"    # low, medium, high
    apply_fixes: bool = False   # If True, apply recommended fixes


class JarvisHardwareOptimizer(BaseAgent):
    """
    AI-powered Hardware Resource Optimizer
    
    This agent can:
    - Monitor system resources with AI analysis
    - Identify performance bottlenecks
    - Generate optimization recommendations
    - Apply fixes automatically (with safety checks)
    """
    
    def __init__(self):
        super().__init__(
            agent_id="jarvis-hardware-optimizer",
            name="Jarvis Hardware Optimizer",
            port=int(os.getenv("PORT", "11104")),
            description="AI-powered hardware resource monitoring and optimization"
        )
        
        # Add capabilities
        self.add_capability(AgentCapability.MONITORING)
        self.add_capability(AgentCapability.AUTONOMOUS_EXECUTION)
        self.add_capability(AgentCapability.REASONING)
        
        # Resource monitoring history
        self.resource_history = []
        self.max_history = 1000
        
        logger.info(f"Initialized {self.agent_name} with AI-powered resource optimization")
    
    async def on_initialize(self):
        """Initialize agent-specific components"""
        logger.info("Jarvis Hardware Optimizer initialized with AI capabilities")
        
        # Test Ollama connectivity
        test_response = await self.query_ollama("System status check", max_tokens=10)
        if test_response:
            logger.info("AI analysis capabilities verified - Ollama connection successful")
        else:
            logger.warning("AI analysis limited - Ollama connection failed")
    
    async def on_task_execute(self, task_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute hardware optimization task with AI assistance
        """
        try:
            task_type = task_data.get("type", "optimization")
            
            if task_type == "optimization":
                return await self._handle_optimization_request(task_data)
            elif task_type == "analysis":
                return await self._handle_resource_analysis(task_data)
            elif task_type == "monitoring":
                return await self._handle_monitoring_request(task_data)
            elif task_type == "recommendations":
                return await self._handle_recommendations_request(task_data)
            else:
                # Default AI-powered task processing
                return await self._process_with_ai(task_data)
                
        except Exception as e:
            logger.error(f"Error executing hardware optimization task: {e}")
            return {
                "success": False,
                "error": str(e),
                "task_id": task_id
            }
    
    async def _handle_optimization_request(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle optimization requests with AI analysis"""
        resource_type = task_data.get("resource_type", "all")
        severity = task_data.get("severity", "medium")
        apply_fixes = task_data.get("apply_fixes", False)
        
        # Collect current system metrics
        system_metrics = await self._collect_system_metrics()
        
        # Analyze with AI
        ai_analysis = await self._analyze_with_ai(system_metrics, resource_type, severity)
        
        result = {
            "success": True,
            "resource_type": resource_type,
            "severity": severity,
            "current_metrics": system_metrics,
            "ai_analysis": ai_analysis,
            "fixes_applied": False,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Apply fixes if requested and safe
        if apply_fixes and ai_analysis.get("safe_to_apply", False):
            fix_results = await self._apply_optimization_fixes(ai_analysis.get("recommendations", []))
            result["fix_results"] = fix_results
            result["fixes_applied"] = True
        elif apply_fixes:
            result["fixes_blocked"] = "Fixes blocked by safety analysis"
        
        self._add_to_history(result)
        return result
    
    async def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            
            # Disk metrics
            disk_usage = psutil.disk_usage('/')
            
            return {
                "cpu": {
                    "usage_percent": cpu_percent,
                    "count": cpu_count
                },
                "memory": {
                    "total_gb": memory.total / (1024**3),
                    "used_gb": memory.used / (1024**3),
                    "available_gb": memory.available / (1024**3),
                    "usage_percent": memory.percent
                },
                "disk": {
                    "total_gb": disk_usage.total / (1024**3),
                    "used_gb": disk_usage.used / (1024**3),
                    "free_gb": disk_usage.free / (1024**3),
                    "usage_percent": (disk_usage.used / disk_usage.total) * 100
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}
    
    async def _analyze_with_ai(self, system_metrics: Dict[str, Any], resource_type: str, severity: str) -> Dict[str, Any]:
        """Analyze system metrics with AI"""
        ai_prompt = f"""
        You are a system performance expert. Analyze these metrics and provide optimization recommendations:
        
        System Metrics: {json.dumps(system_metrics, indent=2)}
        Focus: {resource_type}
        Severity Level: {severity}
        
        Please provide:
        1. Critical issues identification
        2. Performance optimization recommendations
        3. Safety assessment for applying automated fixes
        4. Priority ranking of recommendations
        
        Format as JSON with keys: critical_issues, recommendations, safe_to_apply, priority_ranking
        """
        
        ai_response = await self.query_ollama(ai_prompt, max_tokens=600)
        
        try:
            return json.loads(ai_response) if ai_response else {"error": "AI analysis failed"}
        except json.JSONDecodeError:
            return {"raw_analysis": ai_response or "Analysis failed", "safe_to_apply": False}
    
    async def _handle_resource_analysis(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resource analysis requests"""
        focus = task_data.get("focus", "all")
        
        system_metrics = await self._collect_system_metrics()
        
        # AI analysis of current state
        ai_prompt = f"""
        Analyze these system metrics for performance insights:
        
        Current Metrics: {json.dumps(system_metrics, indent=2)}
        Focus Area: {focus}
        
        Please provide:
        1. Current system health assessment
        2. Resource utilization analysis
        3. Potential bottlenecks or issues
        4. Performance recommendations
        
        Format as JSON with keys: health_status, utilization, bottlenecks, recommendations
        """
        
        ai_analysis = await self.query_ollama(ai_prompt, max_tokens=500)
        
        try:
            analysis_data = json.loads(ai_analysis) if ai_analysis else {}
        except json.JSONDecodeError:
            analysis_data = {"raw_analysis": ai_analysis or "Analysis failed"}
        
        result = {
            "success": True,
            "focus": focus,
            "current_metrics": system_metrics,
            "ai_analysis": analysis_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self._add_to_history(result)
        return result
    
    async def _handle_recommendations_request(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle recommendations requests"""
        system_metrics = await self._collect_system_metrics()
        
        ai_prompt = f"""
        You are a system optimization expert. Based on these metrics, provide actionable recommendations:
        
        System Metrics: {json.dumps(system_metrics, indent=2)}
        
        Provide specific recommendations for:
        1. Performance improvements
        2. Resource optimization
        3. System maintenance
        4. Configuration tuning
        
        Be specific and actionable.
        """
        
        ai_recommendations = await self.query_ollama(ai_prompt, max_tokens=400)
        
        result = {
            "success": True,
            "system_metrics": system_metrics,
            "ai_recommendations": ai_recommendations or "AI recommendations unavailable",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self._add_to_history(result)
        return result
    
    async def _apply_optimization_fixes(self, recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply safe optimization fixes"""
        results = []
        
        for rec in recommendations:
            if not isinstance(rec, dict):
                continue
            
            # Only apply very safe fixes for now
            fix_type = rec.get("type", "unknown")
            if fix_type == "memory_sync":
                try:
                    result = subprocess.run(["sync"], capture_output=True, text=True, timeout=10)
                    results.append({
                        "recommendation": rec,
                        "applied": result.returncode == 0,
                        "output": result.stdout
                    })
                except Exception as e:
                    results.append({
                        "recommendation": rec,
                        "applied": False,
                        "error": str(e)
                    })
            else:
                results.append({
                    "recommendation": rec,
                    "applied": False,
                    "reason": "Not implemented for safety"
                })
        
        return {
            "fixes_attempted": len(recommendations),
            "fixes_applied": len([r for r in results if r.get("applied", False)]),
            "results": results
        }
    
    async def _process_with_ai(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generic AI-powered task processing"""
        task_description = task_data.get("description", str(task_data))
        
        ai_prompt = f"""
        You are Jarvis Hardware Optimizer, an intelligent system performance assistant.
        
        Task: {task_description}
        
        Please analyze this request and provide:
        1. Your understanding of the hardware/performance aspect
        2. Relevant system metrics to check
        3. Optimization suggestions
        4. Next steps for implementation
        
        Be technical, specific, and actionable.
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
    
    def _add_to_history(self, result: Dict[str, Any]):
        """Add result to resource history"""
        self.resource_history.append(result)
        if len(self.resource_history) > self.max_history:
            self.resource_history.pop(0)  # Remove oldest


# FastAPI Integration
app = FastAPI(title="Jarvis Hardware Resource Optimizer", version="2.0.0", 
              description="AI-powered hardware resource monitoring and optimization")

# Global agent instance
agent_instance: Optional[JarvisHardwareOptimizer] = None


@app.on_event("startup")
async def startup():
    """Initialize the agent"""
    global agent_instance
    agent_instance = JarvisHardwareOptimizer()
    await agent_instance.initialize()
    logger.info("Jarvis Hardware Optimizer started with AI capabilities")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    global agent_instance
    if agent_instance:
        await agent_instance.shutdown()
        logger.info("Jarvis Hardware Optimizer shut down")


@app.get("/health")
async def health():
    """Health check endpoint"""
    if not agent_instance:
        return {"status": "initializing"}
    
    # Get current system metrics
    current_metrics = await agent_instance._collect_system_metrics()
    health_info = await agent_instance.health_check()
    
    return {
        **health_info,
        "system_metrics": current_metrics
    }


@app.post("/optimize")
async def optimize_resources(request: OptimizationRequest):
    """Optimize system resources with AI analysis"""
    if not agent_instance:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    task_data = {
        "type": "optimization",
        "resource_type": request.resource_type,
        "severity": request.severity,
        "apply_fixes": request.apply_fixes
    }
    
    result = await agent_instance.on_task_execute("optimization", task_data)
    return result


@app.get("/analyze")
async def analyze_resources(focus: str = Query("all", description="Focus area: cpu, memory, disk, network, all")):
    """Analyze system resources with AI"""
    if not agent_instance:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    task_data = {
        "type": "analysis",
        "focus": focus
    }
    
    result = await agent_instance.on_task_execute("analysis", task_data)
    return result


@app.get("/recommendations")
async def get_recommendations():
    """Get AI-powered optimization recommendations"""
    if not agent_instance:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    task_data = {
        "type": "recommendations"
    }
    
    result = await agent_instance.on_task_execute("recommendations", task_data)
    return result


@app.get("/metrics")
async def get_current_metrics():
    """Get current system metrics"""
    if not agent_instance:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    metrics = await agent_instance._collect_system_metrics()
    return metrics


@app.post("/process")
async def process_task(task_data: Dict[str, Any]):
    """Generic task processing endpoint"""
    if not agent_instance:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    result = await agent_instance.on_task_execute("generic", task_data)
    return result


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "11104"))
    logger.info(f"Starting Jarvis Hardware Optimizer on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)

