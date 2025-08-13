#!/usr/bin/env python3
"""
Jarvis Orchestrator - Core coordination logic
"""

import asyncio
import json
import logging
import time
import yaml
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from pathlib import Path

from .task_planner import TaskPlanner
try:
    from .voice_interface import VoiceInterface
except ImportError:
    from .voice_interface_  import VoiceInterface
from .plugin_manager import PluginManager
from .agent_coordinator import AgentCoordinator
from .memory_manager import MemoryManager

logger = logging.getLogger(__name__)

class JarvisOrchestrator:
    """Main orchestrator for Jarvis AI system"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = {}
        self.task_planner = None
        self.voice_interface = None
        self.plugin_manager = None
        self.agent_coordinator = None
        self.memory_manager = None
        self.active_sessions: Dict[str, Any] = {}
        self.command_history: List[Dict[str, Any]] = []
        self.initialized = False
        
    async def initialize(self):
        """Initialize all components"""
        try:
            # Load configuration
            self._load_config()
            
            # Initialize components
            self.memory_manager = MemoryManager(self.config.get('memory', {}))
            await self.memory_manager.initialize()
            
            self.agent_coordinator = AgentCoordinator(self.config.get('agents', {}))
            await self.agent_coordinator.initialize()
            
            self.task_planner = TaskPlanner(self.config.get('planner', {}))
            await self.task_planner.initialize()
            
            self.plugin_manager = PluginManager(self.config.get('plugins', {}))
            await self.plugin_manager.initialize()
            
            self.voice_interface = VoiceInterface(self.config.get('voice', {}))
            await self.voice_interface.initialize()
            
            # Load command history
            await self._load_history()
            
            self.initialized = True
            logger.info("Jarvis Orchestrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Jarvis: {e}")
            raise
            
    async def shutdown(self):
        """Shutdown all components"""
        try:
            # Save history
            await self._save_history()
            
            # Shutdown components
            if self.voice_interface:
                await self.voice_interface.shutdown()
            if self.plugin_manager:
                await self.plugin_manager.shutdown()
            if self.agent_coordinator:
                await self.agent_coordinator.shutdown()
            if self.task_planner:
                await self.task_planner.shutdown()
            if self.memory_manager:
                await self.memory_manager.shutdown()
                
            logger.info("Jarvis Orchestrator shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            
    def _load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            self.config = self._get_default_config()
            
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'agents': {
                'max_concurrent': 5,
                'timeout': 60,
                'retry_attempts': 3
            },
            'planner': {
                'max_steps': 10,
                'enable_reflection': True,
                'planning_model': 'tinyllama'
            },
            'voice': {
                'enable_speech_recognition': True,
                'enable_text_to_speech': True,
                'language': 'en-US',
                'voice_engine': 'pyttsx3'
            },
            'plugins': {
                'plugin_dir': '/opt/sutazaiapp/services/jarvis/plugins',
                'auto_load': True,
                'enabled_plugins': []
            },
            'memory': {
                'type': 'redis',
                'max_history': 1000,
                'ttl': 86400
            }
        }
        
    async def execute_task(self, command: str, context: Dict[str, Any] = None, 
                          voice_enabled: bool = False, plugins: List[str] = None,
                          session_id: str = None) -> Dict[str, Any]:
        """Execute a task with full orchestration"""
        start_time = time.time()
        execution_id = f"exec_{datetime.now().timestamp()}"
        
        try:
            # Record command
            command_record = {
                'id': execution_id,
                'command': command,
                'context': context or {},
                'timestamp': datetime.now().isoformat(),
                'session_id': session_id
            }
            self.command_history.append(command_record)
            
            # Retrieve relevant memory
            memory_context = await self.memory_manager.retrieve_context(command, context)
            
            # Plan the task
            plan = await self.task_planner.create_plan(command, {
                **memory_context,
                **(context or {})
            })
            
            # Execute plugins if specified
            plugin_results = {}
            if plugins:
                for plugin_name in plugins:
                    if self.plugin_manager.is_enabled(plugin_name):
                        plugin_results[plugin_name] = await self.plugin_manager.execute_plugin(
                            plugin_name, command, context
                        )
                        
            # Execute the plan using agents
            result = await self.agent_coordinator.execute_plan(plan, plugin_results)
            
            # Store in memory
            await self.memory_manager.store_interaction(command, result)
            
            # Generate voice response if enabled
            voice_response = None
            if voice_enabled and result.get('success'):
                voice_response = await self.voice_interface.text_to_speech(
                    result.get('summary', str(result.get('result')))
                )
                
            # Update command record
            command_record.update({
                'duration': time.time() - start_time,
                'success': result.get('success', True),
                'agents_used': result.get('agents_used', []),
                'plugins_used': list(plugin_results.keys())
            })
            
            return {
                'result': result.get('result'),
                'status': 'success' if result.get('success') else 'failed',
                'agents_used': result.get('agents_used', []),
                'voice_response': voice_response,
                'execution_id': execution_id,
                'plan': plan,
                'plugin_results': plugin_results
            }
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            return {
                'result': None,
                'status': 'error',
                'error': str(e),
                'agents_used': [],
                'execution_id': execution_id
            }
            
    async def process_voice_command(self, audio_path: str) -> Dict[str, Any]:
        """Process voice command from audio file"""
        try:
            # Convert speech to text
            command = await self.voice_interface.speech_to_text(audio_path)
            
            if not command:
                return {
                    'status': 'error',
                    'error': 'Could not understand audio'
                }
                
            # Execute the command
            return await self.execute_task(command, voice_enabled=True)
            
        except Exception as e:
            logger.error(f"Voice processing failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
            
    async def register_session(self, session_id: str, websocket: Any):
        """Register a WebSocket session"""
        self.active_sessions[session_id] = {
            'websocket': websocket,
            'created_at': datetime.now(),
            'commands': 0
        }
        logger.info(f"Registered session: {session_id}")
        
    async def unregister_session(self, session_id: str):
        """Unregister a WebSocket session"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info(f"Unregistered session: {session_id}")
            
    async def broadcast_to_sessions(self, message: Dict[str, Any], exclude: str = None):
        """Broadcast message to all active sessions"""
        for session_id, session_data in self.active_sessions.items():
            if session_id != exclude:
                try:
                    await session_data['websocket'].send_json(message)
                except Exception as e:
                    logger.error(f"Failed to send to session {session_id}: {e}")
                    
    async def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            'initialized': self.initialized,
            'agents_available': await self.agent_coordinator.get_available_agents(),
            'plugins_loaded': self.plugin_manager.list_plugins(),
            'voice_enabled': self.voice_interface.is_available(),
            'active_sessions': len(self.active_sessions),
            'memory_stats': await self.memory_manager.get_stats()
        }
        
    async def list_available_agents(self) -> List[Dict[str, Any]]:
        """List all available agents"""
        return await self.agent_coordinator.get_agent_info()
        
    async def list_plugins(self) -> List[Dict[str, Any]]:
        """List all plugins"""
        return self.plugin_manager.get_plugin_info()
        
    async def enable_plugin(self, plugin_name: str):
        """Enable a plugin"""
        await self.plugin_manager.enable_plugin(plugin_name)
        
    async def disable_plugin(self, plugin_name: str):
        """Disable a plugin"""
        await self.plugin_manager.disable_plugin(plugin_name)
        
    async def get_command_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get command history"""
        return self.command_history[-limit:]
        
    async def record_feedback(self, session_id: str, rating: int, comment: str = None):
        """Record user feedback"""
        feedback = {
            'session_id': session_id,
            'rating': rating,
            'comment': comment,
            'timestamp': datetime.now().isoformat()
        }
        await self.memory_manager.store_feedback(feedback)
        
    async def _load_history(self):
        """Load command history from storage"""
        try:
            history_file = Path('/opt/sutazaiapp/data/jarvis/history.json')
            if history_file.exists():
                with open(history_file, 'r') as f:
                    self.command_history = json.load(f)
                logger.info(f"Loaded {len(self.command_history)} commands from history")
        except Exception as e:
            logger.error(f"Failed to load history: {e}")
            
    async def _save_history(self):
        """Save command history to storage"""
        try:
            history_file = Path('/opt/sutazaiapp/data/jarvis/history.json')
            history_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(history_file, 'w') as f:
                json.dump(self.command_history[-1000:], f, indent=2)  # Keep last 1000 commands
                
            logger.info("Saved command history")
        except Exception as e:
            logger.error(f"Failed to save history: {e}")