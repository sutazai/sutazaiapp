#!/usr/bin/env python3
"""
Enhanced AI Agent Orchestrator with Real Intelligence
Version 2.0 - Full implementation with Ollama integration
"""

import asyncio
import httpx
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import pika
import json
import redis
from datetime import datetime
import logging
import os
import uuid
import uvicorn
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskRequest(BaseModel):
    """Request model for task orchestration"""
    task_type: str = Field(..., description="Type of task to orchestrate")
    payload: Dict[str, Any] = Field(..., description="Task payload data")
    priority: int = Field(default=5, ge=1, le=10, description="Task priority (1-10)")
    timeout: int = Field(default=300, ge=10, le=3600, description="Task timeout in seconds")

class TaskResponse(BaseModel):
    """Response model for task orchestration"""
    task_id: str
    status: str
    plan: Dict[str, Any]
    estimated_time: int
    message: str

class AIAgentOrchestrator:
    """Real AI Agent Orchestrator with actual intelligence"""
    
    def __init__(self):
        self.port = int(os.getenv("PORT", "8589"))
        self.redis_client = None
        self.rabbit_connection = None
        self.rabbit_channel = None
        self.ollama_url = os.getenv("OLLAMA_URL", "http://ollama:10104")
        self.app = None
        self.active_tasks = {}
        
    async def startup(self):
        """Initialize connections on startup"""
        try:
            # Initialize Redis
            self.redis_client = redis.Redis(
                host=os.getenv("REDIS_HOST", "redis"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info("Redis connected successfully")
            
            # Initialize RabbitMQ
            rabbit_host = os.getenv("RABBITMQ_HOST", "rabbitmq")
            rabbit_port = int(os.getenv("RABBITMQ_PORT", "5672"))
            
            # Try to connect to RabbitMQ with retries
            for attempt in range(5):
                try:
                    self.rabbit_connection = pika.BlockingConnection(
                        pika.ConnectionParameters(
                            host=rabbit_host,
                            port=rabbit_port,
                            connection_attempts=3,
                            retry_delay=2
                        )
                    )
                    self.rabbit_channel = self.rabbit_connection.channel()
                    
                    # Declare exchanges
                    self.rabbit_channel.exchange_declare(
                        exchange='ai.tasks',
                        exchange_type='topic',
                        durable=True
                    )
                    self.rabbit_channel.exchange_declare(
                        exchange='ai.events',
                        exchange_type='fanout',
                        durable=True
                    )
                    
                    logger.info("RabbitMQ connected successfully")
                    break
                except Exception as e:
                    logger.warning(f"RabbitMQ connection attempt {attempt + 1} failed: {e}")
                    if attempt == 4:
                        logger.error("Could not connect to RabbitMQ, continuing without it")
                    else:
                        await asyncio.sleep(5)
            
            # Test Ollama connection
            await self.test_ollama_connection()
            
        except Exception as e:
            logger.error(f"Startup error: {e}")
    
    async def shutdown(self):
        """Cleanup connections on shutdown"""
        try:
            if self.rabbit_connection and not self.rabbit_connection.is_closed:
                self.rabbit_connection.close()
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Shutdown error: {e}")
    
    async def test_ollama_connection(self):
        """Test connection to Ollama"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.ollama_url}/api/tags",
                    timeout=5.0
                )
                if response.status_code == 200:
                    models = response.json().get('models', [])
                    logger.info(f"Ollama connected. Available models: {[m['name'] for m in models]}")
                    return True
        except Exception as e:
            logger.warning(f"Ollama connection test failed: {e}")
            return False
    
    async def analyze_task(self, request: TaskRequest) -> Dict[str, Any]:
        """Use Ollama to analyze and plan task execution"""
        
        prompt = f"""You are an AI task orchestrator. Analyze this task and create an execution plan.

Task Type: {request.task_type}
Payload: {json.dumps(request.payload, indent=2)}
Priority: {request.priority}

Create a detailed execution plan. Consider which agents should handle this task and in what order.
Available agent types:
- code-generator: For code generation tasks
- data-processor: For data analysis and processing
- security-scanner: For security analysis
- deployment-agent: For deployment tasks
- monitoring-agent: For system monitoring
- general-agent: For general purpose tasks

Respond with a JSON object containing:
{{
    "agents_required": ["list of agent types needed"],
    "execution_steps": [
        {{"step": 1, "agent": "agent-type", "action": "what to do", "dependencies": []}}
    ],
    "estimated_time": 60,
    "confidence": 0.85,
    "reasoning": "Brief explanation of the plan"
}}"""
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": os.getenv("OLLAMA_MODEL", "tinyllama"),
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "top_p": 0.9
                        }
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    generated_text = result.get('response', '{}')
                    
                    # Try to parse JSON from response
                    try:
                        # Find JSON in the response
                        import re
                        json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
                        if json_match:
                            plan = json.loads(json_match.group())
                            logger.info(f"AI generated plan: {plan}")
                            return plan
                    except json.JSONDecodeError:
                        logger.warning("Could not parse AI response as JSON")
                        
        except Exception as e:
            logger.error(f"Ollama analysis failed: {e}")
        
        # Fallback to rule-based planning
        return self.fallback_analysis(request)
    
    def fallback_analysis(self, request: TaskRequest) -> Dict[str, Any]:
        """Fallback logic when AI analysis fails"""
        
        # Rule-based task routing
        task_mappings = {
            "code_generation": {
                "agents": ["code-generator", "qa-validator"],
                "time": 120
            },
            "data_processing": {
                "agents": ["data-processor"],
                "time": 60
            },
            "system_optimization": {
                "agents": ["hardware-optimizer", "resource-arbitrator"],
                "time": 90
            },
            "security_scan": {
                "agents": ["security-scanner"],
                "time": 180
            },
            "deployment": {
                "agents": ["deployment-agent", "monitoring-agent"],
                "time": 150
            }
        }
        
        mapping = task_mappings.get(request.task_type, {
            "agents": ["general-agent"],
            "time": 60
        })
        
        execution_steps = []
        for i, agent in enumerate(mapping["agents"]):
            execution_steps.append({
                "step": i + 1,
                "agent": agent,
                "action": f"Process {request.task_type}",
                "dependencies": [i] if i > 0 else []
            })
        
        return {
            "agents_required": mapping["agents"],
            "execution_steps": execution_steps,
            "estimated_time": mapping["time"],
            "confidence": 0.7,
            "reasoning": "Using rule-based routing (fallback mode)"
        }
    
    async def coordinate_execution(
        self, 
        task_id: str, 
        plan: Dict[str, Any],
        request: TaskRequest
    ) -> Dict[str, Any]:
        """Coordinate multi-agent task execution"""
        
        execution_log = []
        
        # Store task in Redis
        task_data = {
            "task_id": task_id,
            "type": request.task_type,
            "status": "executing",
            "plan": json.dumps(plan),
            "created_at": datetime.utcnow().isoformat(),
            "priority": request.priority
        }
        
        if self.redis_client:
            self.redis_client.hset(f"task:{task_id}", mapping=task_data)
        
        # Execute each step
        for step in plan.get("execution_steps", []):
            step_result = await self.execute_step(task_id, step, request.payload)
            execution_log.append(step_result)
            
            # Update progress in Redis
            if self.redis_client:
                self.redis_client.hset(
                    f"task:{task_id}",
                    f"step_{step['step']}",
                    json.dumps(step_result)
                )
        
        # Publish completion event
        if self.rabbit_channel:
            try:
                self.rabbit_channel.basic_publish(
                    exchange='ai.events',
                    routing_key='',
                    body=json.dumps({
                        "event": "task_completed",
                        "task_id": task_id,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                )
            except Exception as e:
                logger.error(f"Failed to publish event: {e}")
        
        return {
            "task_id": task_id,
            "status": "completed",
            "execution_log": execution_log
        }
    
    async def execute_step(
        self,
        task_id: str,
        step: Dict[str, Any],
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single step in the plan"""
        
        agent = step.get("agent", "general-agent")
        action = step.get("action", "process")
        
        # Dispatch to agent via RabbitMQ if available
        if self.rabbit_channel:
            try:
                message = {
                    "task_id": task_id,
                    "step": step["step"],
                    "action": action,
                    "payload": payload,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                self.rabbit_channel.basic_publish(
                    exchange='ai.tasks',
                    routing_key=f'task.{agent}.execute',
                    body=json.dumps(message),
                    properties=pika.BasicProperties(
                        delivery_mode=2,  # Persistent
                        priority=5
                    )
                )
                
                return {
                    "step": step["step"],
                    "agent": agent,
                    "status": "dispatched",
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Failed to dispatch to agent: {e}")
        
        # Simulate execution if RabbitMQ not available
        await asyncio.sleep(1)  # Simulate processing
        
        return {
            "step": step["step"],
            "agent": agent,
            "status": "simulated",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get current status of a task"""
        
        if self.redis_client:
            task_data = self.redis_client.hgetall(f"task:{task_id}")
            if task_data:
                # Parse stored JSON fields
                if "plan" in task_data:
                    task_data["plan"] = json.loads(task_data["plan"])
                
                # Get step results
                steps = {}
                for key in task_data.keys():
                    if key.startswith("step_"):
                        steps[key] = json.loads(task_data[key])
                
                return {
                    "task_id": task_id,
                    "status": task_data.get("status", "unknown"),
                    "type": task_data.get("type", "unknown"),
                    "created_at": task_data.get("created_at"),
                    "plan": task_data.get("plan", {}),
                    "steps": steps
                }
        
        # Check in-memory storage
        if task_id in self.active_tasks:
            return self.active_tasks[task_id]
        
        return {"task_id": task_id, "status": "not_found"}
    
    def create_app(self):
        """Create FastAPI application"""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            await self.startup()
            yield
            # Shutdown
            await self.shutdown()
        
        app = FastAPI(
            title="AI Agent Orchestrator",
            version="2.0.0",
            description="Intelligent task orchestration with Ollama integration",
            lifespan=lifespan
        )
        
        @app.post("/orchestrate", response_model=TaskResponse)
        async def orchestrate_task(
            request: TaskRequest,
            background_tasks: BackgroundTasks
        ):
            """Main orchestration endpoint"""
            
            # Generate unique task ID
            task_id = f"task_{uuid.uuid4().hex[:12]}_{int(datetime.utcnow().timestamp())}"
            
            # Analyze task with AI
            plan = await self.analyze_task(request)
            
            # Store task in memory
            self.active_tasks[task_id] = {
                "task_id": task_id,
                "status": "planned",
                "plan": plan,
                "created_at": datetime.utcnow().isoformat()
            }
            
            # Start execution in background
            background_tasks.add_task(
                self.coordinate_execution,
                task_id,
                plan,
                request
            )
            
            return TaskResponse(
                task_id=task_id,
                status="executing",
                plan=plan,
                estimated_time=plan.get("estimated_time", 60),
                message=f"Task orchestration started with {len(plan.get('agents_required', []))} agents"
            )
        
        @app.get("/task/{task_id}")
        async def get_task_status(task_id: str):
            """Get task execution status"""
            status = await self.get_task_status(task_id)
            if status["status"] == "not_found":
                raise HTTPException(status_code=404, detail="Task not found")
            return status
        
        @app.get("/health")
        async def health_check():
            """Health check endpoint"""
            
            health_status = {
                "status": "healthy",
                "service": "ai-agent-orchestrator",
                "version": "2.0.0",
                "timestamp": datetime.utcnow().isoformat(),
                "connections": {
                    "redis": False,
                    "rabbitmq": False,
                    "ollama": False
                }
            }
            
            # Check Redis
            try:
                if self.redis_client:
                    self.redis_client.ping()
                    health_status["connections"]["redis"] = True
            except:
                pass
            
            # Check RabbitMQ
            try:
                if self.rabbit_connection and not self.rabbit_connection.is_closed:
                    health_status["connections"]["rabbitmq"] = True
            except:
                pass
            
            # Check Ollama
            health_status["connections"]["ollama"] = await self.test_ollama_connection()
            
            # Determine overall health
            if not any(health_status["connections"].values()):
                health_status["status"] = "degraded"
            
            return health_status
        
        @app.get("/")
        async def root():
            """Root endpoint"""
            return {
                "service": "AI Agent Orchestrator",
                "version": "2.0.0",
                "endpoints": [
                    "/orchestrate - POST - Submit task for orchestration",
                    "/task/{task_id} - GET - Get task status",
                    "/health - GET - Health check"
                ]
            }
        
        return app

# Main execution
if __name__ == "__main__":
    orchestrator = AIAgentOrchestrator()
    app = orchestrator.create_app()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=orchestrator.port,
        log_level="info"
    )