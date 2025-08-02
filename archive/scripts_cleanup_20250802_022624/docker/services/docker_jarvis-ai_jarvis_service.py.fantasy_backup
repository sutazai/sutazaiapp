#!/usr/bin/env python3
"""
SutazAI JARVIS - Autonomous AI Assistant
Integrates multiple JARVIS implementations for comprehensive AI assistance
"""

import os
import sys
import json
import logging
import asyncio
import threading
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import requests
import speech_recognition as sr
import pyttsx3
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SutazAI JARVIS AI Assistant", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class JARVISRequest(BaseModel):
    command: str
    context: Optional[Dict] = None
    voice_enabled: bool = False
    priority: str = "normal"

class JARVISResponse(BaseModel):
    response: str
    action_taken: Optional[str] = None
    status: str
    timestamp: str
    confidence: float

class TaskRequest(BaseModel):
    task_description: str
    parameters: Optional[Dict] = None
    async_execution: bool = False

class JARVISCore:
    def __init__(self):
        self.config = self.load_config()
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
        self.backend_url = os.getenv("BACKEND_URL", "http://backend-agi:8000")
        
        # Initialize components
        self.speech_engine = None
        self.recognizer = None
        self.microphone = None
        self.active_tasks = {}
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        # Initialize speech components
        self.init_speech_components()
        
        # Task categories
        self.task_handlers = {
            "code": self.handle_code_task,
            "analysis": self.handle_analysis_task,
            "system": self.handle_system_task,
            "conversation": self.handle_conversation_task,
            "automation": self.handle_automation_task,
            "research": self.handle_research_task,
            "monitoring": self.handle_monitoring_task
        }
    
    def load_config(self):
        """Load JARVIS configuration"""
        try:
            with open("jarvis_config.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "voice_enabled": True,
                "auto_execute": True,
                "learning_mode": True,
                "security_level": "high",
                "response_style": "professional"
            }
    
    def init_speech_components(self):
        """Initialize speech recognition and synthesis"""
        try:
            # Text-to-speech
            self.speech_engine = pyttsx3.init()
            self.speech_engine.setProperty('rate', 150)
            self.speech_engine.setProperty('volume', 0.8)
            
            # Speech recognition
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            
            # Calibrate microphone
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
                
            logger.info("Speech components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize speech components: {e}")
            self.speech_engine = None
            self.recognizer = None
    
    def speak(self, text: str):
        """Convert text to speech"""
        if self.speech_engine and self.config.get("voice_enabled", True):
            try:
                self.speech_engine.say(text)
                self.speech_engine.runAndWait()
            except Exception as e:
                logger.error(f"Speech synthesis error: {e}")
    
    def listen(self) -> Optional[str]:
        """Listen for voice commands"""
        if not self.recognizer or not self.microphone:
            return None
        
        try:
            with self.microphone as source:
                logger.info("Listening for command...")
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
            
            command = self.recognizer.recognize_google(audio)
            logger.info(f"Recognized command: {command}")
            return command
        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            return None
        except Exception as e:
            logger.error(f"Speech recognition error: {e}")
            return None
    
    async def query_ollama(self, prompt: str, model: str = "tinyllama") -> str:
        """Query Ollama for AI responses"""
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 2048
                }
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get("response", "No response generated")
            else:
                return f"Error querying Ollama: {response.status_code}"
        except Exception as e:
            logger.error(f"Ollama query error: {e}")
            return f"Failed to get AI response: {e}"
    
    async def handle_code_task(self, task: str, parameters: Dict = None) -> str:
        """Handle code-related tasks"""
        prompt = f"""
        You are JARVIS, an advanced AI coding assistant. 
        Task: {task}
        Parameters: {parameters or {}}
        
        Provide a comprehensive solution including:
        1. Code implementation
        2. Explanation of approach
        3. Testing recommendations
        4. Integration suggestions
        
        Focus on clean, efficient, and maintainable code.
        """
        
        response = await self.query_ollama(prompt, "tinyllama")
        
        # Execute code if auto-execute is enabled
        if self.config.get("auto_execute", False) and parameters and parameters.get("execute", False):
            # Safe code execution logic here
            pass
        
        return response
    
    async def handle_analysis_task(self, task: str, parameters: Dict = None) -> str:
        """Handle analysis and research tasks"""
        prompt = f"""
        You are JARVIS, performing advanced analysis.
        Task: {task}
        Data: {parameters or {}}
        
        Provide:
        1. Detailed analysis
        2. Key insights
        3. Recommendations
        4. Action items
        5. Confidence metrics
        
        Be thorough and precise in your analysis.
        """
        
        return await self.query_ollama(prompt, "qwen2.5:7b")
    
    async def handle_system_task(self, task: str, parameters: Dict = None) -> str:
        """Handle system administration tasks"""
        try:
            # System status check
            if "status" in task.lower():
                return await self.get_system_status()
            
            # Service management
            if "service" in task.lower():
                return await self.manage_services(task, parameters)
            
            # Resource monitoring
            if "monitor" in task.lower():
                return await self.monitor_resources()
            
            # Default system analysis
            prompt = f"""
            You are JARVIS system administrator.
            Task: {task}
            Parameters: {parameters or {}}
            
            Provide system analysis and recommendations for:
            1. Performance optimization
            2. Security enhancements
            3. Resource management
            4. Maintenance tasks
            """
            
            return await self.query_ollama(prompt)
        except Exception as e:
            return f"System task error: {e}"
    
    async def handle_conversation_task(self, task: str, parameters: Dict = None) -> str:
        """Handle conversational interactions"""
        prompt = f"""
        You are JARVIS, an advanced AI assistant with personality.
        User said: {task}
        Context: {parameters or {}}
        
        Respond as JARVIS would - intelligent, helpful, slightly witty, and always professional.
        Provide useful information and actionable suggestions.
        """
        
        return await self.query_ollama(prompt, "llama3.2:1b")
    
    async def handle_automation_task(self, task: str, parameters: Dict = None) -> str:
        """Handle automation and workflow tasks"""
        prompt = f"""
        You are JARVIS automation specialist.
        Task: {task}
        Parameters: {parameters or {}}
        
        Design automation solution including:
        1. Workflow steps
        2. Dependencies
        3. Error handling
        4. Monitoring points
        5. Success metrics
        """
        
        return await self.query_ollama(prompt)
    
    async def handle_research_task(self, task: str, parameters: Dict = None) -> str:
        """Handle research and information gathering"""
        prompt = f"""
        You are JARVIS research assistant.
        Research topic: {task}
        Parameters: {parameters or {}}
        
        Provide comprehensive research including:
        1. Key findings
        2. Sources and references
        3. Analysis and insights
        4. Future considerations
        5. Practical applications
        """
        
        return await self.query_ollama(prompt, "qwen2.5:7b")
    
    async def handle_monitoring_task(self, task: str, parameters: Dict = None) -> str:
        """Handle monitoring and alerting tasks"""
        try:
            # Check SutazAI system health
            health_status = await self.check_sutazai_health()
            
            prompt = f"""
            You are JARVIS monitoring specialist.
            Task: {task}
            Current system status: {health_status}
            Parameters: {parameters or {}}
            
            Provide monitoring analysis and recommendations.
            """
            
            return await self.query_ollama(prompt)
        except Exception as e:
            return f"Monitoring task error: {e}"
    
    async def get_system_status(self) -> str:
        """Get comprehensive system status"""
        try:
            # Check Docker containers
            import subprocess
            docker_ps = subprocess.run(
                ["docker", "ps", "--format", "table {{.Names}}\t{{.Status}}"],
                capture_output=True, text=True
            )
            
            # Check system resources
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            status = {
                "timestamp": datetime.now().isoformat(),
                "docker_containers": docker_ps.stdout if docker_ps.returncode == 0 else "Error getting containers",
                "cpu_usage": f"{cpu_percent}%",
                "memory_usage": f"{memory.percent}%",
                "disk_usage": f"{disk.percent}%",
                "system_load": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else "N/A"
            }
            
            return f"System Status Report:\n{json.dumps(status, indent=2)}"
        except Exception as e:
            return f"Error getting system status: {e}"
    
    async def manage_services(self, task: str, parameters: Dict = None) -> str:
        """Manage Docker services"""
        try:
            # Service management logic here
            # This would integrate with Docker Compose to start/stop/restart services
            return f"Service management task: {task} - Parameters: {parameters}"
        except Exception as e:
            return f"Service management error: {e}"
    
    async def monitor_resources(self) -> str:
        """Monitor system resources"""
        try:
            import psutil
            
            resources = {
                "cpu": {
                    "usage_percent": psutil.cpu_percent(interval=1),
                    "count": psutil.cpu_count(),
                    "load_avg": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
                },
                "memory": {
                    "total": psutil.virtual_memory().total,
                    "available": psutil.virtual_memory().available,
                    "percent": psutil.virtual_memory().percent
                },
                "disk": {
                    "total": psutil.disk_usage('/').total,
                    "used": psutil.disk_usage('/').used,
                    "free": psutil.disk_usage('/').free,
                    "percent": psutil.disk_usage('/').percent
                }
            }
            
            return f"Resource Monitor:\n{json.dumps(resources, indent=2)}"
        except Exception as e:
            return f"Resource monitoring error: {e}"
    
    async def check_sutazai_health(self) -> Dict:
        """Check SutazAI system health"""
        try:
            # Check backend health
            backend_response = requests.get(f"{self.backend_url}/health", timeout=5)
            backend_healthy = backend_response.status_code == 200
            
            # Check Ollama health
            ollama_response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            ollama_healthy = ollama_response.status_code == 200
            
            return {
                "backend": "healthy" if backend_healthy else "unhealthy",
                "ollama": "healthy" if ollama_healthy else "unhealthy",
                "jarvis": "healthy",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def process_command(self, command: str, context: Dict = None, voice_enabled: bool = False) -> JARVISResponse:
        """Process incoming commands"""
        try:
            start_time = time.time()
            
            # Determine task category
            task_category = self.classify_task(command)
            
            # Get appropriate handler
            handler = self.task_handlers.get(task_category, self.handle_conversation_task)
            
            # Process the task
            response_text = await handler(command, context)
            
            # Calculate confidence based on response quality
            confidence = self.calculate_confidence(command, response_text)
            
            # Voice output if enabled
            if voice_enabled and self.speech_engine:
                self.speak(response_text)
            
            return JARVISResponse(
                response=response_text,
                action_taken=f"Processed {task_category} task",
                status="success",
                timestamp=datetime.now().isoformat(),
                confidence=confidence
            )
        
        except Exception as e:
            error_msg = f"Error processing command: {e}"
            logger.error(error_msg)
            
            return JARVISResponse(
                response=error_msg,
                action_taken="error_handling",
                status="error",
                timestamp=datetime.now().isoformat(),
                confidence=0.0
            )
    
    def classify_task(self, command: str) -> str:
        """Classify the type of task based on command"""
        command_lower = command.lower()
        
        if any(word in command_lower for word in ["code", "program", "script", "function", "debug"]):
            return "code"
        elif any(word in command_lower for word in ["analyze", "analysis", "research", "study"]):
            return "analysis"
        elif any(word in command_lower for word in ["system", "status", "service", "docker", "monitor"]):
            return "system"
        elif any(word in command_lower for word in ["automate", "workflow", "schedule", "task"]):
            return "automation"
        elif any(word in command_lower for word in ["research", "find", "search", "investigate"]):
            return "research"
        elif any(word in command_lower for word in ["monitor", "check", "health", "performance"]):
            return "monitoring"
        else:
            return "conversation"
    
    def calculate_confidence(self, command: str, response: str) -> float:
        """Calculate confidence score for the response"""
        try:
            # Simple confidence calculation based on response length and relevance
            if len(response) > 100 and "error" not in response.lower():
                return min(0.95, 0.7 + (len(response) / 1000))
            elif len(response) > 50:
                return 0.6
            else:
                return 0.3
        except:
            return 0.5

# Initialize JARVIS
jarvis = JARVISCore()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    system_health = await jarvis.check_sutazai_health()
    return {
        "status": "healthy",
        "service": "JARVIS AI Assistant",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "speech_enabled": jarvis.speech_engine is not None,
        "system_health": system_health
    }

@app.post("/command", response_model=JARVISResponse)
async def process_command(request: JARVISRequest):
    """Process text commands"""
    return await jarvis.process_command(
        request.command,
        request.context,
        request.voice_enabled
    )

@app.post("/task", response_model=JARVISResponse)
async def execute_task(request: TaskRequest):
    """Execute specific tasks"""
    if request.async_execution:
        # Execute asynchronously
        task_id = f"task_{int(time.time())}"
        jarvis.active_tasks[task_id] = {
            "status": "running",
            "start_time": datetime.now().isoformat()
        }
        
        # Start task in background
        jarvis.executor.submit(
            asyncio.run,
            jarvis.process_command(request.task_description, request.parameters)
        )
        
        return JARVISResponse(
            response=f"Task started asynchronously with ID: {task_id}",
            action_taken="async_task_started",
            status="success",
            timestamp=datetime.now().isoformat(),
            confidence=1.0
        )
    else:
        # Execute synchronously
        return await jarvis.process_command(request.task_description, request.parameters)

@app.get("/voice/listen")
async def voice_listen():
    """Listen for voice commands"""
    command = jarvis.listen()
    if command:
        response = await jarvis.process_command(command, voice_enabled=True)
        return {"command": command, "response": response}
    else:
        return {"command": None, "response": "No command detected"}

@app.post("/voice/speak")
async def voice_speak(text: str):
    """Convert text to speech"""
    jarvis.speak(text)
    return {"status": "spoken", "text": text}

@app.get("/tasks")
async def get_active_tasks():
    """Get active tasks"""
    return {"active_tasks": jarvis.active_tasks}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            command_data = json.loads(data)
            
            response = await jarvis.process_command(
                command_data.get("command", ""),
                command_data.get("context", {}),
                command_data.get("voice_enabled", False)
            )
            
            await websocket.send_text(response.json())
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8120))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info("Starting JARVIS AI Assistant...")
    logger.info(f"Ollama URL: {jarvis.ollama_url}")
    logger.info(f"Backend URL: {jarvis.backend_url}")
    logger.info(f"Speech enabled: {jarvis.speech_engine is not None}")
    
    uvicorn.run(app, host=host, port=port) 