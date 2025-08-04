#!/bin/bash

# Jarvis AI System Setup Script
# Purpose: Initialize Jarvis unified AI interface
# Usage: ./setup-jarvis.sh [--env dev|prod] [--enable-voice]

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
JARVIS_DIR="$PROJECT_ROOT/services/jarvis"
CONFIG_DIR="$PROJECT_ROOT/config/jarvis"
LOG_DIR="$PROJECT_ROOT/logs/jarvis"

# Default values
ENVIRONMENT="dev"
ENABLE_VOICE=false
JARVIS_PORT="8888"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --enable-voice)
            ENABLE_VOICE=true
            shift
            ;;
        --port)
            JARVIS_PORT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--env dev|prod] [--enable-voice] [--port PORT]"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create directory structure
create_directories() {
    log_info "Creating directory structure..."
    
    mkdir -p "$JARVIS_DIR"/{core,plugins,static}
    mkdir -p "$CONFIG_DIR"
    mkdir -p "$LOG_DIR"
    mkdir -p "$PROJECT_ROOT/data/jarvis"
    
    # Create __init__.py files
    touch "$JARVIS_DIR/__init__.py"
    touch "$JARVIS_DIR/core/__init__.py"
    touch "$JARVIS_DIR/plugins/__init__.py"
}

# Create core components
create_core_components() {
    log_info "Creating core components..."
    
    # Create task planner
    cat > "$JARVIS_DIR/core/task_planner.py" << 'EOF'
#!/usr/bin/env python3
"""Task Planner - Creates execution plans for complex tasks"""

import logging
from typing import Dict, Any, List
import aiohttp

logger = logging.getLogger(__name__)

class TaskPlanner:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_steps = config.get('max_steps', 10)
        self.planning_model = config.get('planning_model', 'tinyllama')
        self.enable_reflection = config.get('enable_reflection', True)
        self.ollama_url = "http://localhost:11434"
        
    async def initialize(self):
        """Initialize task planner"""
        logger.info("Task planner initialized")
        
    async def shutdown(self):
        """Shutdown task planner"""
        pass
        
    async def create_plan(self, command: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create execution plan for a command"""
        try:
            # Analyze command intent
            intent = await self._analyze_intent(command, context)
            
            # Decompose into steps
            steps = await self._decompose_task(command, intent, context)
            
            # Add dependencies
            steps = self._add_dependencies(steps)
            
            return {
                'goal': command,
                'intent': intent,
                'steps': steps,
                'estimated_duration': len(steps) * 10  # seconds
            }
            
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            return {
                'goal': command,
                'steps': [{
                    'id': 1,
                    'type': 'direct',
                    'description': command,
                    'input': context
                }]
            }
            
    async def _analyze_intent(self, command: str, context: Dict[str, Any]) -> str:
        """Analyze user intent"""
        # Simple intent classification
        command_lower = command.lower()
        
        if any(word in command_lower for word in ['create', 'build', 'make', 'generate']):
            return 'create'
        elif any(word in command_lower for word in ['analyze', 'check', 'review', 'inspect']):
            return 'analyze'
        elif any(word in command_lower for word in ['fix', 'debug', 'solve', 'repair']):
            return 'fix'
        elif any(word in command_lower for word in ['find', 'search', 'locate', 'get']):
            return 'search'
        else:
            return 'general'
            
    async def _decompose_task(self, command: str, intent: str, 
                             context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decompose task into steps"""
        steps = []
        
        # Create steps based on intent
        if intent == 'create':
            steps = [
                {
                    'id': 1,
                    'type': 'planning',
                    'description': f'Plan structure for: {command}',
                    'required_capabilities': ['planning']
                },
                {
                    'id': 2,
                    'type': 'implementation',
                    'description': f'Implement: {command}',
                    'required_capabilities': ['coding', 'creation']
                },
                {
                    'id': 3,
                    'type': 'validation',
                    'description': 'Validate implementation',
                    'required_capabilities': ['testing', 'validation']
                }
            ]
        elif intent == 'analyze':
            steps = [
                {
                    'id': 1,
                    'type': 'data_gathering',
                    'description': f'Gather data for: {command}',
                    'required_capabilities': ['search', 'data']
                },
                {
                    'id': 2,
                    'type': 'analysis',
                    'description': f'Analyze: {command}',
                    'required_capabilities': ['analysis', 'reasoning']
                }
            ]
        else:
            steps = [
                {
                    'id': 1,
                    'type': 'execution',
                    'description': command,
                    'required_capabilities': ['general']
                }
            ]
            
        # Add context to each step
        for step in steps:
            step['input'] = context
            
        return steps
        
    def _add_dependencies(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add step dependencies"""
        for i, step in enumerate(steps):
            if i > 0:
                step['dependencies'] = [steps[i-1]['id']]
            else:
                step['dependencies'] = []
        return steps
EOF

    # Create voice interface
    cat > "$JARVIS_DIR/core/voice_interface.py" << 'EOF'
#!/usr/bin/env python3
"""Voice Interface - Speech recognition and text-to-speech"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Import voice libraries conditionally
try:
    import speech_recognition as sr
    import pyttsx3
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False
    logger.warning("Voice libraries not installed")

class VoiceInterface:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = VOICE_AVAILABLE and config.get('enable_speech_recognition', False)
        self.recognizer = None
        self.tts_engine = None
        
    async def initialize(self):
        """Initialize voice components"""
        if not self.enabled:
            logger.info("Voice interface disabled")
            return
            
        try:
            self.recognizer = sr.Recognizer()
            self.tts_engine = pyttsx3.init()
            
            # Configure TTS
            self.tts_engine.setProperty('rate', self.config.get('text_to_speech', {}).get('rate', 150))
            self.tts_engine.setProperty('volume', self.config.get('text_to_speech', {}).get('volume', 0.9))
            
            logger.info("Voice interface initialized")
        except Exception as e:
            logger.error(f"Failed to initialize voice: {e}")
            self.enabled = False
            
    async def shutdown(self):
        """Shutdown voice interface"""
        if self.tts_engine:
            self.tts_engine.stop()
            
    def is_available(self) -> bool:
        """Check if voice interface is available"""
        return self.enabled
        
    async def speech_to_text(self, audio_path: str) -> Optional[str]:
        """Convert speech to text"""
        if not self.enabled:
            return None
            
        try:
            with sr.AudioFile(audio_path) as source:
                audio = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio)
                return text
        except Exception as e:
            logger.error(f"Speech recognition failed: {e}")
            return None
            
    async def text_to_speech(self, text: str) -> Optional[str]:
        """Convert text to speech"""
        if not self.enabled:
            return None
            
        try:
            # Generate audio file
            output_path = f"/tmp/jarvis_tts_{os.getpid()}.wav"
            self.tts_engine.save_to_file(text, output_path)
            self.tts_engine.runAndWait()
            return output_path
        except Exception as e:
            logger.error(f"Text to speech failed: {e}")
            return None
EOF

    # Create plugin manager
    cat > "$JARVIS_DIR/core/plugin_manager.py" << 'EOF'
#!/usr/bin/env python3
"""Plugin Manager - Dynamic plugin loading and execution"""

import logging
import importlib
import os
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class PluginManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.plugin_dir = config.get('plugin_dir', '/opt/sutazaiapp/services/jarvis/plugins')
        self.plugins = {}
        self.enabled_plugins = set(config.get('enabled_plugins', []))
        
    async def initialize(self):
        """Initialize plugin manager"""
        if self.config.get('auto_load', True):
            await self._load_plugins()
        logger.info(f"Plugin manager initialized with {len(self.plugins)} plugins")
        
    async def shutdown(self):
        """Shutdown plugins"""
        for plugin_name, plugin in self.plugins.items():
            if hasattr(plugin, 'shutdown'):
                await plugin.shutdown()
                
    async def _load_plugins(self):
        """Load all plugins from plugin directory"""
        plugin_path = Path(self.plugin_dir)
        if not plugin_path.exists():
            logger.warning(f"Plugin directory not found: {self.plugin_dir}")
            return
            
        for file_path in plugin_path.glob("*.py"):
            if file_path.name.startswith("_"):
                continue
                
            plugin_name = file_path.stem
            try:
                # Import plugin module
                spec = importlib.util.spec_from_file_location(plugin_name, file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Get plugin class
                plugin_class = getattr(module, 'Plugin', None)
                if plugin_class:
                    plugin = plugin_class()
                    self.plugins[plugin_name] = plugin
                    logger.info(f"Loaded plugin: {plugin_name}")
                    
            except Exception as e:
                logger.error(f"Failed to load plugin {plugin_name}: {e}")
                
    def is_enabled(self, plugin_name: str) -> bool:
        """Check if plugin is enabled"""
        return plugin_name in self.enabled_plugins
        
    async def enable_plugin(self, plugin_name: str):
        """Enable a plugin"""
        if plugin_name in self.plugins:
            self.enabled_plugins.add(plugin_name)
            logger.info(f"Enabled plugin: {plugin_name}")
        else:
            raise ValueError(f"Plugin not found: {plugin_name}")
            
    async def disable_plugin(self, plugin_name: str):
        """Disable a plugin"""
        self.enabled_plugins.discard(plugin_name)
        logger.info(f"Disabled plugin: {plugin_name}")
        
    async def execute_plugin(self, plugin_name: str, command: str, 
                           context: Dict[str, Any]) -> Any:
        """Execute a plugin"""
        if plugin_name not in self.plugins:
            raise ValueError(f"Plugin not found: {plugin_name}")
            
        if not self.is_enabled(plugin_name):
            raise ValueError(f"Plugin not enabled: {plugin_name}")
            
        plugin = self.plugins[plugin_name]
        return await plugin.execute(command, context)
        
    def list_plugins(self) -> List[str]:
        """List all loaded plugins"""
        return list(self.plugins.keys())
        
    def get_plugin_info(self) -> List[Dict[str, Any]]:
        """Get information about all plugins"""
        info = []
        for name, plugin in self.plugins.items():
            info.append({
                'name': name,
                'enabled': self.is_enabled(name),
                'description': getattr(plugin, 'description', 'No description'),
                'version': getattr(plugin, 'version', '1.0.0')
            })
        return info
EOF

    # Create memory manager
    cat > "$JARVIS_DIR/core/memory_manager.py" << 'EOF'
#!/usr/bin/env python3
"""Memory Manager - Handles context and history"""

import logging
import json
import time
from typing import Dict, Any, List
import redis
from datetime import datetime

logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.type = config.get('type', 'redis')
        self.max_history = config.get('max_history', 1000)
        self.ttl = config.get('ttl', 86400)
        self.redis_client = None
        
    async def initialize(self):
        """Initialize memory storage"""
        if self.type == 'redis':
            try:
                self.redis_client = redis.Redis(
                    host='localhost',
                    port=6379,
                    decode_responses=True
                )
                self.redis_client.ping()
                logger.info("Memory manager connected to Redis")
            except Exception as e:
                logger.warning(f"Redis not available: {e}, using in-memory storage")
                self.type = 'memory'
                self.memory_store = {}
                
    async def shutdown(self):
        """Shutdown memory manager"""
        if self.redis_client:
            self.redis_client.close()
            
    async def store_interaction(self, command: str, result: Any):
        """Store command and result"""
        key = f"jarvis:interaction:{time.time()}"
        data = {
            'command': command,
            'result': result,
            'timestamp': datetime.now().isoformat()
        }
        
        if self.type == 'redis' and self.redis_client:
            self.redis_client.setex(key, self.ttl, json.dumps(data))
        else:
            self.memory_store[key] = data
            
    async def retrieve_context(self, command: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve relevant context for command"""
        # Simple implementation - can be enhanced with embeddings
        recent_interactions = []
        
        if self.type == 'redis' and self.redis_client:
            keys = self.redis_client.keys("jarvis:interaction:*")
            for key in keys[-10:]:  # Last 10 interactions
                data = self.redis_client.get(key)
                if data:
                    recent_interactions.append(json.loads(data))
        else:
            for key in list(self.memory_store.keys())[-10:]:
                recent_interactions.append(self.memory_store[key])
                
        return {
            'recent_interactions': recent_interactions,
            'user_context': context
        }
        
    async def store_feedback(self, feedback: Dict[str, Any]):
        """Store user feedback"""
        key = f"jarvis:feedback:{time.time()}"
        
        if self.type == 'redis' and self.redis_client:
            self.redis_client.setex(key, self.ttl, json.dumps(feedback))
        else:
            self.memory_store[key] = feedback
            
    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        if self.type == 'redis' and self.redis_client:
            return {
                'total_interactions': len(self.redis_client.keys("jarvis:interaction:*")),
                'total_feedback': len(self.redis_client.keys("jarvis:feedback:*"))
            }
        else:
            return {
                'total_interactions': len([k for k in self.memory_store.keys() if 'interaction' in k]),
                'total_feedback': len([k for k in self.memory_store.keys() if 'feedback' in k])
            }
EOF
}

# Create basic plugins
create_plugins() {
    log_info "Creating basic plugins..."
    
    # Web search plugin
    cat > "$JARVIS_DIR/plugins/web_search.py" << 'EOF'
#!/usr/bin/env python3
"""Web Search Plugin"""

import logging

logger = logging.getLogger(__name__)

class Plugin:
    description = "Search the web for information"
    version = "1.0.0"
    
    async def execute(self, command: str, context: dict) -> dict:
        """Execute web search"""
        # Placeholder implementation
        return {
            'plugin': 'web_search',
            'query': command,
            'results': [
                {'title': 'Result 1', 'url': 'http://example.com/1'},
                {'title': 'Result 2', 'url': 'http://example.com/2'}
            ]
        }
        
    async def shutdown(self):
        pass
EOF

    # Calculator plugin
    cat > "$JARVIS_DIR/plugins/calculator.py" << 'EOF'
#!/usr/bin/env python3
"""Calculator Plugin"""

import logging
import re

logger = logging.getLogger(__name__)

class Plugin:
    description = "Perform mathematical calculations"
    version = "1.0.0"
    
    async def execute(self, command: str, context: dict) -> dict:
        """Execute calculation"""
        try:
            # Extract mathematical expression
            expr = re.sub(r'[^0-9+\-*/().\s]', '', command)
            result = eval(expr)
            
            return {
                'plugin': 'calculator',
                'expression': expr,
                'result': result
            }
        except Exception as e:
            return {
                'plugin': 'calculator',
                'error': str(e)
            }
            
    async def shutdown(self):
        pass
EOF

    # System monitor plugin
    cat > "$JARVIS_DIR/plugins/system_monitor.py" << 'EOF'
#!/usr/bin/env python3
"""System Monitor Plugin"""

import psutil
import logging

logger = logging.getLogger(__name__)

class Plugin:
    description = "Monitor system resources"
    version = "1.0.0"
    
    async def execute(self, command: str, context: dict) -> dict:
        """Get system status"""
        return {
            'plugin': 'system_monitor',
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'processes': len(psutil.pids())
        }
        
    async def shutdown(self):
        pass
EOF
}

# Create web interface
create_web_interface() {
    log_info "Creating web interface..."
    
    cat > "$JARVIS_DIR/static/index.html" << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Jarvis AI Assistant</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 0;
            background: #0a0a0a;
            color: #e0e0e0;
            display: flex;
            flex-direction: column;
            height: 100vh;
        }
        .header {
            background: linear-gradient(135deg, #1e3c72, #2a5298);
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        .container {
            flex: 1;
            display: flex;
            max-width: 1200px;
            margin: 0 auto;
            width: 100%;
            padding: 20px;
            gap: 20px;
        }
        .chat-area {
            flex: 2;
            display: flex;
            flex-direction: column;
            background: #1a1a1a;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }
        .message {
            margin-bottom: 15px;
            padding: 10px 15px;
            border-radius: 8px;
            animation: fadeIn 0.3s ease-in;
        }
        .user-message {
            background: #2a5298;
            margin-left: 20%;
            text-align: right;
        }
        .jarvis-message {
            background: #2a2a2a;
            margin-right: 20%;
        }
        .input-area {
            display: flex;
            padding: 20px;
            border-top: 1px solid #333;
        }
        .input-area input {
            flex: 1;
            padding: 12px;
            border: none;
            border-radius: 25px;
            background: #2a2a2a;
            color: #e0e0e0;
            font-size: 16px;
            outline: none;
        }
        .input-area button {
            margin-left: 10px;
            padding: 12px 25px;
            border: none;
            border-radius: 25px;
            background: #2a5298;
            color: white;
            cursor: pointer;
            font-size: 16px;
            transition: background 0.3s;
        }
        .input-area button:hover {
            background: #1e3c72;
        }
        .sidebar {
            flex: 1;
            background: #1a1a1a;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        .status {
            margin-bottom: 20px;
            padding: 15px;
            background: #2a2a2a;
            border-radius: 8px;
        }
        .plugins {
            margin-top: 20px;
        }
        .plugin {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            background: #2a2a2a;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .toggle {
            width: 50px;
            height: 25px;
            background: #666;
            border-radius: 25px;
            position: relative;
            cursor: pointer;
            transition: background 0.3s;
        }
        .toggle.active {
            background: #2a5298;
        }
        .toggle::after {
            content: '';
            position: absolute;
            width: 20px;
            height: 20px;
            background: white;
            border-radius: 50%;
            top: 2.5px;
            left: 2.5px;
            transition: left 0.3s;
        }
        .toggle.active::after {
            left: 27.5px;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 Jarvis AI Assistant</h1>
        <p>Your intelligent companion powered by 131 AI agents</p>
    </div>
    
    <div class="container">
        <div class="chat-area">
            <div class="messages" id="messages">
                <div class="message jarvis-message">
                    Hello! I'm Jarvis, your AI assistant. How can I help you today?
                </div>
            </div>
            <div class="input-area">
                <input type="text" id="input" placeholder="Type your message..." autofocus>
                <button onclick="sendMessage()">Send</button>
                <button onclick="startVoice()" id="voiceBtn">🎤</button>
            </div>
        </div>
        
        <div class="sidebar">
            <div class="status">
                <h3>System Status</h3>
                <div id="status">
                    <p>🟢 Connected</p>
                    <p>Agents: <span id="agentCount">131</span></p>
                    <p>Plugins: <span id="pluginCount">6</span></p>
                </div>
            </div>
            
            <div class="plugins">
                <h3>Plugins</h3>
                <div id="pluginList"></div>
            </div>
        </div>
    </div>
    
    <script>
        let ws = null;
        let sessionId = null;
        
        // Initialize WebSocket connection
        function initWebSocket() {
            ws = new WebSocket('ws://localhost:8888/ws');
            
            ws.onopen = () => {
                console.log('Connected to Jarvis');
                sessionId = Date.now().toString();
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                displayMessage(data.result || data.error || 'Processing...', false);
            };
            
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                displayMessage('Connection error. Please refresh.', false);
            };
            
            ws.onclose = () => {
                console.log('Disconnected from Jarvis');
                setTimeout(initWebSocket, 3000);
            };
        }
        
        // Send message
        function sendMessage() {
            const input = document.getElementById('input');
            const message = input.value.trim();
            
            if (!message) return;
            
            displayMessage(message, true);
            
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    command: message,
                    context: {},
                    voice_enabled: false
                }));
            }
            
            input.value = '';
        }
        
        // Display message in chat
        function displayMessage(text, isUser) {
            const messages = document.getElementById('messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user-message' : 'jarvis-message'}`;
            messageDiv.textContent = text;
            messages.appendChild(messageDiv);
            messages.scrollTop = messages.scrollHeight;
        }
        
        // Voice input
        function startVoice() {
            if ('webkitSpeechRecognition' in window) {
                const recognition = new webkitSpeechRecognition();
                recognition.lang = 'en-US';
                recognition.onresult = (event) => {
                    const transcript = event.results[0][0].transcript;
                    document.getElementById('input').value = transcript;
                    sendMessage();
                };
                recognition.start();
            } else {
                alert('Speech recognition not supported in your browser');
            }
        }
        
        // Load plugins
        async function loadPlugins() {
            try {
                const response = await fetch('/api/plugins');
                const plugins = await response.json();
                
                const pluginList = document.getElementById('pluginList');
                pluginList.innerHTML = '';
                
                plugins.forEach(plugin => {
                    const pluginDiv = document.createElement('div');
                    pluginDiv.className = 'plugin';
                    pluginDiv.innerHTML = `
                        <span>${plugin.name}</span>
                        <div class="toggle ${plugin.enabled ? 'active' : ''}" 
                             onclick="togglePlugin('${plugin.name}', this)"></div>
                    `;
                    pluginList.appendChild(pluginDiv);
                });
                
                document.getElementById('pluginCount').textContent = plugins.length;
            } catch (error) {
                console.error('Failed to load plugins:', error);
            }
        }
        
        // Toggle plugin
        async function togglePlugin(pluginName, element) {
            const isActive = element.classList.contains('active');
            const endpoint = isActive ? 'disable' : 'enable';
            
            try {
                await fetch(`/api/plugins/${pluginName}/${endpoint}`, { method: 'POST' });
                element.classList.toggle('active');
            } catch (error) {
                console.error('Failed to toggle plugin:', error);
            }
        }
        
        // Keyboard shortcuts
        document.getElementById('input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
        
        // Initialize
        initWebSocket();
        loadPlugins();
    </script>
</body>
</html>
EOF
}

# Create Python virtual environment
setup_python_env() {
    log_info "Setting up Python environment..."
    
    cd "$JARVIS_DIR"
    
    # Create virtual environment
    python3 -m venv venv
    source venv/bin/activate
    
    # Create requirements file
    cat > requirements.txt << EOF
fastapi==0.104.1
uvicorn[standard]==0.24.0
websockets==12.0
aiohttp==3.9.1
pydantic==2.5.0
python-consul==1.1.0
prometheus-client==0.19.0
redis==5.0.1
psutil==5.9.6
PyYAML==6.0.1
python-multipart==0.0.6
EOF

    # Add voice dependencies if enabled
    if [ "$ENABLE_VOICE" = true ]; then
        echo "SpeechRecognition==3.10.0" >> requirements.txt
        echo "pyttsx3==2.90" >> requirements.txt
        echo "pyaudio==0.2.13" >> requirements.txt
    fi
    
    # Install dependencies
    pip install -r requirements.txt
    
    log_info "Python environment ready"
}

# Create systemd service (production only)
create_systemd_service() {
    if [ "$ENVIRONMENT" != "prod" ]; then
        return
    fi
    
    log_info "Creating systemd service..."
    
    sudo tee /etc/systemd/system/jarvis.service > /dev/null << EOF
[Unit]
Description=Jarvis AI Assistant
After=network.target ollama-manager.service
Requires=network.target

[Service]
Type=simple
User=$(whoami)
Group=$(id -gn)
WorkingDirectory=$JARVIS_DIR
Environment="PATH=$JARVIS_DIR/venv/bin:/usr/local/bin:/usr/bin:/bin"
Environment="JARVIS_CONFIG=$CONFIG_DIR/config.yaml"
Environment="JARVIS_PORT=$JARVIS_PORT"
ExecStart=$JARVIS_DIR/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable jarvis
    
    log_info "Systemd service created"
}

# Main execution
main() {
    log_info "Setting up Jarvis AI System..."
    
    create_directories
    create_core_components
    create_plugins
    create_web_interface
    setup_python_env
    create_systemd_service
    
    log_info "Jarvis setup complete!"
    log_info ""
    log_info "To start Jarvis:"
    log_info "  cd $JARVIS_DIR"
    log_info "  source venv/bin/activate"
    log_info "  python main.py"
    log_info ""
    log_info "Access the web interface at: http://localhost:$JARVIS_PORT"
    
    if [ "$ENVIRONMENT" = "prod" ]; then
        log_info ""
        log_info "Production commands:"
        log_info "  sudo systemctl start jarvis"
        log_info "  sudo systemctl status jarvis"
        log_info "  sudo journalctl -u jarvis -f"
    fi
}

# Run main
main