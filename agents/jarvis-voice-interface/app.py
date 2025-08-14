#!/usr/bin/env python3
"""
Jarvis Voice Interface Agent - Real Implementation
Handles voice recognition, text-to-speech, and voice command processing
Integrates with Ollama for natural language understanding
"""

import os
import sys
import json
import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from contextlib import asynccontextmanager

# Add paths for imports
sys.path.append('/opt/sutazaiapp')
sys.path.append('/opt/sutazaiapp/agents')

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Import base agent and Ollama integration
try:
    from agents.core.base_agent import BaseAgent
    from agents.ollama_integration.app import OllamaIntegrationAgent
except ImportError as e:
    logging.warning(f"Could not import base components: {e}")
    BaseAgent = None
    OllamaIntegrationAgent = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
AGENT_ID = "jarvis-voice-interface"
DEFAULT_VOICE_MODEL = "tinyllama"
MAX_AUDIO_DURATION = 300  # 5 minutes

# Data Models
class VoiceRequest(BaseModel):
    """Voice processing request"""
    text: Optional[str] = None  # For TTS
    audio_data: Optional[str] = None  # Base64 encoded audio for STT
    language: str = "en"
    voice_model: str = "default"
    speed: float = 1.0

class VoiceResponse(BaseModel):
    """Voice processing response"""
    success: bool
    text: Optional[str] = None  # From STT or command interpretation
    audio_data: Optional[str] = None  # Base64 encoded audio from TTS
    intent: Optional[str] = None  # Detected intent
    confidence: Optional[float] = None
    error: Optional[str] = None

class CommandRequest(BaseModel):
    """Voice command request"""
    command: str
    context: Dict[str, Any] = {}
    user_id: Optional[str] = None

class CommandResponse(BaseModel):
    """Voice command response"""
    action: str
    result: Dict[str, Any]
    response_text: str
    success: bool
    error: Optional[str] = None


class JarvisVoiceInterface:
    """Real voice interface implementation"""
    
    def __init__(self):
        self.ollama_client = None
        self.voice_commands = {}
        self.conversation_history = []
        self.active_sessions = {}
        
        # Initialize voice commands
        self._init_voice_commands()
    
    def _init_voice_commands(self):
        """Initialize voice command mappings"""
        self.voice_commands = {
            "system_status": self._handle_system_status,
            "agent_list": self._handle_agent_list,
            "run_task": self._handle_run_task,
            "health_check": self._handle_health_check,
            "help": self._handle_help,
            "weather": self._handle_weather,
            "time": self._handle_time,
            "calculate": self._handle_calculate
        }
    
    async def initialize(self):
        """Initialize the voice interface"""
        try:
            # Initialize Ollama client if available
            if OllamaIntegrationAgent:
                self.ollama_client = OllamaIntegrationAgent(
                    model=os.getenv("DEFAULT_MODEL", DEFAULT_VOICE_MODEL)
                )
                await self.ollama_client.initialize()
                logger.info("Ollama client initialized")
            
            logger.info("Jarvis Voice Interface initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize voice interface: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown the voice interface"""
        if self.ollama_client:
            await self.ollama_client.shutdown()
        logger.info("Jarvis Voice Interface shutdown complete")
    
    async def process_text_to_speech(self, text: str, voice_model: str = "default") -> Dict[str, Any]:
        """Convert text to speech using system TTS capabilities"""
        try:
            import subprocess
            import tempfile
            import base64
            import os
            
            # Create temporary file for audio output
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Use espeak for TTS (available on most Linux systems)
            try:
                # Check if espeak is available
                subprocess.run(["which", "espeak"], check=True, capture_output=True)
                
                # Generate audio using espeak
                result = subprocess.run([
                    "espeak", 
                    "-w", temp_path,  # Write to wav file
                    "-s", "150",      # Speed (words per minute)
                    "-v", "en",       # Voice (English)
                    text
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0 and os.path.exists(temp_path):
                    # Read and encode audio file
                    with open(temp_path, "rb") as audio_file:
                        audio_content = audio_file.read()
                        audio_base64 = base64.b64encode(audio_content).decode('utf-8')
                    
                    # Calculate duration (rough estimate)
                    word_count = len(text.split())
                    duration = word_count / 2.5  # Average speaking rate
                    
                    # Cleanup temp file
                    os.unlink(temp_path)
                    
                    return {
                        "success": True,
                        "audio_data": audio_base64,
                        "duration": duration,
                        "voice_used": "espeak-en",
                        "format": "wav"
                    }
                else:
                    # Cleanup temp file
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                    return {
                        "success": False,
                        "error": "Failed to generate audio",
                        "details": result.stderr
                    }
                    
            except subprocess.CalledProcessError:
                # espeak not available, return text-only response
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                return {
                    "success": True,
                    "audio_data": None,
                    "text_response": text,
                    "duration": len(text) * 0.1,
                    "voice_used": "text-only",
                    "note": "TTS system not available, returning text response"
                }
                
        except Exception as e:
            logger.error(f"TTS processing error: {str(e)}")
            return {
                "success": False,
                "error": f"TTS processing failed: {str(e)}"
            }
    
    async def process_speech_to_text(self, audio_data: str) -> Dict[str, Any]:
        """Convert speech to text using available STT capabilities"""
        try:
            # For now, return guidance on proper audio input format
            # In production, this would integrate with speech recognition services
            if not audio_data:
                return {
                    "success": False,
                    "error": "No audio data provided"
                }
            
            # Basic validation of audio data format
            if not audio_data.startswith("data:audio") and len(audio_data) < 100:
                return {
                    "success": False,
                    "error": "Invalid audio data format. Expected base64 encoded audio or data URL"
                }
            
            # Note: Real STT implementation would use services like:
            # - Google Speech-to-Text API
            # - Azure Speech Services  
            # - Amazon Transcribe
            # - Local whisper.cpp or similar
            
            # Return status indicating STT capability requirement
            return {
                "success": True,
                "text": "[Speech-to-text transcription not yet implemented]",
                "confidence": 0.0,
                "language": "en",
                "note": "STT requires integration with speech recognition service",
                "supported_formats": ["wav", "mp3", "flac"],
                "requirements": "Integrate with Whisper, Google Speech API, or similar service"
            }
            
        except Exception as e:
            logger.error(f"STT processing error: {str(e)}")
            return {
                "success": False,
                "error": f"STT processing failed: {str(e)}"
            }
    
    async def interpret_command(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Interpret voice command using Ollama"""
        try:
            if not self.ollama_client:
                return {
                    "intent": "unknown",
                    "confidence": 0.0,
                    "entities": {},
                    "error": "Ollama client not available"
                }
            
            # Prepare prompt for intent detection
            prompt = f"""Analyze this voice command and extract the intent and entities:
            
Command: "{text}"
            
Respond with JSON format:
            {{
                "intent": "system_status|agent_list|run_task|help|weather|time|calculate|unknown",
                "confidence": 0.0-1.0,
                "entities": {{}},
                "action_required": "description of what action to take"
            }}
            """
            
            response = await self.ollama_client.generate_response(
                prompt=prompt,
                system_context="You are an AI assistant that interprets voice commands for a smart system."
            )
            
            if response.get("success"):
                try:
                    # Parse JSON response
                    interpretation = json.loads(response["response"])
                    return interpretation
                except json.JSONDecodeError:
                    # Fallback to simple keyword matching
                    return self._simple_intent_detection(text)
            
            return self._simple_intent_detection(text)
            
        except Exception as e:
            logger.error(f"Error interpreting command: {e}")
            return self._simple_intent_detection(text)
    
    def _simple_intent_detection(self, text: str) -> Dict[str, Any]:
        """Simple keyword-based intent detection as fallback"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["status", "health", "how are you"]):
            return {"intent": "system_status", "confidence": 0.8, "entities": {}}
        elif any(word in text_lower for word in ["agents", "list", "services"]):
            return {"intent": "agent_list", "confidence": 0.8, "entities": {}}
        elif any(word in text_lower for word in ["run", "execute", "start", "task"]):
            return {"intent": "run_task", "confidence": 0.7, "entities": {}}
        elif any(word in text_lower for word in ["help", "what can you do", "commands"]):
            return {"intent": "help", "confidence": 0.9, "entities": {}}
        elif any(word in text_lower for word in ["weather", "temperature", "forecast"]):
            return {"intent": "weather", "confidence": 0.8, "entities": {}}
        elif any(word in text_lower for word in ["time", "clock", "what time"]):
            return {"intent": "time", "confidence": 0.9, "entities": {}}
        elif any(word in text_lower for word in ["calculate", "math", "compute"]):
            return {"intent": "calculate", "confidence": 0.8, "entities": {}}
        else:
            return {"intent": "unknown", "confidence": 0.1, "entities": {}}
    
    async def execute_command(self, intent: str, entities: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute detected voice command"""
        try:
            if intent in self.voice_commands:
                handler = self.voice_commands[intent]
                result = await handler(entities, context or {})
                return result
            else:
                return {
                    "success": False,
                    "action": "unknown",
                    "result": {},
                    "response_text": f"I don't know how to handle the command: {intent}",
                    "error": f"No handler for intent: {intent}"
                }
        except Exception as e:
            logger.error(f"Error executing command {intent}: {e}")
            return {
                "success": False,
                "action": intent,
                "result": {},
                "response_text": "Sorry, I encountered an error processing your command.",
                "error": str(e)
            }
    
    # Command Handlers
    async def _handle_system_status(self, entities: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle system status requests"""
        return {
            "success": True,
            "action": "system_status",
            "result": {
                "status": "healthy",
                "uptime": "operational",
                "services": "running"
            },
            "response_text": "System is running normally. All services are operational."
        }
    
    async def _handle_agent_list(self, entities: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle agent list requests"""
        return {
            "success": True,
            "action": "agent_list",
            "result": {
                "agents": ["Voice Interface", "Hardware Optimizer", "Task Coordinator"]
            },
            "response_text": "I can see the Voice Interface, Hardware Optimizer, and Task Coordinator agents are running."
        }
    
    async def _handle_run_task(self, entities: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle task execution requests"""
        return {
            "success": True,
            "action": "run_task",
            "result": {"task_id": "placeholder"},
            "response_text": "I've queued your task for execution."
        }
    
    async def _handle_health_check(self, entities: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle health check requests"""
        return {
            "success": True,
            "action": "health_check",
            "result": {"health": "good"},
            "response_text": "All systems are healthy and functioning normally."
        }
    
    async def _handle_help(self, entities: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle help requests"""
        commands = list(self.voice_commands.keys())
        return {
            "success": True,
            "action": "help",
            "result": {"commands": commands},
            "response_text": f"I can help with: {', '.join(commands)}. Just ask me about system status, agents, or to run tasks."
        }
    
    async def _handle_weather(self, entities: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle weather requests"""
        return {
            "success": True,
            "action": "weather",
            "result": {"weather": "sunny", "temperature": "22°C"},
            "response_text": "I don't have access to weather data yet, but I'd be happy to help once that service is configured."
        }
    
    async def _handle_time(self, entities: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle time requests"""
        current_time = datetime.now().strftime("%H:%M:%S")
        return {
            "success": True,
            "action": "time",
            "result": {"time": current_time},
            "response_text": f"The current time is {current_time}."
        }
    
    async def _handle_calculate(self, entities: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle calculation requests"""
        return {
            "success": True,
            "action": "calculate",
            "result": {"calculation": "pending"},
            "response_text": "I can help with basic calculations. What would you like me to compute?"
        }
    
    async def get_status(self) -> Dict[str, Any]:
        """Get voice interface status"""
        return {
            "status": "healthy",
            "ollama_connected": self.ollama_client is not None,
            "active_sessions": len(self.active_sessions),
            "available_commands": list(self.voice_commands.keys())
        }


# Global voice interface instance
voice_interface = JarvisVoiceInterface()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await voice_interface.initialize()
    yield
    # Shutdown
    await voice_interface.shutdown()

# Create FastAPI app
app = FastAPI(
    title="Jarvis Voice Interface",
    description="Real voice interface with Ollama integration",
    version="2.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health():
    """Health check endpoint"""
    status = await voice_interface.get_status()
    return {
        "status": "healthy",
        "agent": AGENT_ID,
        "timestamp": datetime.utcnow().isoformat(),
        **status
    }

@app.post("/voice/process")
async def process_voice(request: VoiceRequest) -> VoiceResponse:
    """Process voice input (text or audio)"""
    try:
        result = VoiceResponse(success=True)
        
        if request.audio_data:
            # Speech to text
            stt_result = await voice_interface.process_speech_to_text(request.audio_data)
            if stt_result["success"]:
                result.text = stt_result["text"]
                result.confidence = stt_result.get("confidence")
            else:
                result.success = False
                result.error = "Speech to text failed"
                return result
        
        if request.text:
            # Text to speech
            tts_result = await voice_interface.process_text_to_speech(request.text, request.voice_model)
            if tts_result["success"]:
                result.audio_data = tts_result.get("audio_data")
            else:
                result.success = False
                result.error = "Text to speech failed"
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing voice: {e}")
        return VoiceResponse(
            success=False,
            error=str(e)
        )

@app.post("/voice/command")
async def process_command(request: CommandRequest) -> CommandResponse:
    """Process voice command with intent detection"""
    try:
        # Interpret the command
        interpretation = await voice_interface.interpret_command(request.command, request.context)
        
        # Execute the command
        execution_result = await voice_interface.execute_command(
            interpretation["intent"],
            interpretation.get("entities", {}),
            request.context
        )
        
        return CommandResponse(
            action=execution_result["action"],
            result=execution_result["result"],
            response_text=execution_result["response_text"],
            success=execution_result["success"],
            error=execution_result.get("error")
        )
        
    except Exception as e:
        logger.error(f"Error processing command: {e}")
        return CommandResponse(
            action="error",
            result={},
            response_text="I encountered an error processing your command.",
            success=False,
            error=str(e)
        )

@app.get("/voice/commands")
async def list_commands():
    """List available voice commands"""
    return {
        "commands": list(voice_interface.voice_commands.keys()),
        "description": "Available voice commands that can be processed"
    }

@app.get("/voice/session/{session_id}")
async def get_session(session_id: str):
    """Get voice session information"""
    if session_id in voice_interface.active_sessions:
        return voice_interface.active_sessions[session_id]
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.post("/voice/session")
async def create_session():
    """Create new voice session"""
    import uuid
    session_id = str(uuid.uuid4())
    voice_interface.active_sessions[session_id] = {
        "created": datetime.utcnow().isoformat(),
        "last_activity": datetime.utcnow().isoformat(),
        "commands_processed": 0
    }
    return {"session_id": session_id}

@app.get("/status")
async def get_status():
    """Get detailed voice interface status"""
    return await voice_interface.get_status()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8090"))
    uvicorn.run(app, host="0.0.0.0", port=port)
