# JARVIS - Unified Voice Interface for SutazAI

**J**ust **A** **R**ather **V**ery **I**ntelligent **S**ystem

A sophisticated voice-controlled AI assistant that provides unified access to all 69+ AI agents in the SutazAI ecosystem.

## 🎯 Overview

JARVIS serves as the central voice interface for SutazAI, enabling natural language interactions with a comprehensive multi-agent AI system. It combines advanced speech recognition, intelligent task planning, and seamless agent coordination to provide a JARVIS-like experience for users.

### Key Features

- **🎤 Advanced Voice Recognition**: Real-time speech-to-text with wake word detection
- **🗣️ Natural Speech Synthesis**: High-quality text-to-speech responses
- **🤖 Intelligent Agent Routing**: Automatically routes commands to the most suitable AI agents
- **🧠 Smart Task Planning**: Uses local LLM (TinyLlama) for intelligent task decomposition
- **💾 Contextual Memory**: Maintains conversation history and learns from interactions
- **🔌 Plugin System**: Extensible architecture for custom functionality
- **🌐 Web Interface**: Modern web-based control panel
- **📊 Real-time Monitoring**: Comprehensive metrics and health monitoring

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Voice Input    │    │   Web Interface │    │   WebSocket     │
│  (Microphone)   │    │   (Browser)     │    │   (Real-time)   │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │      JARVIS CORE          │
                    │   (FastAPI + AsyncIO)     │
                    └─────────────┬─────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
    ┌─────────▼─────────┐ ┌──────▼──────┐ ┌─────────▼─────────┐
    │  Voice Interface  │ │Task Planner │ │ Agent Coordinator │
    │ (Speech/TTS/VAD)  │ │  (Ollama)   │ │  (69+ Agents)     │
    └───────┬───────────┘ └──────┬──────┘ └─────────┬─────────┘
            │                    │                  │
    ┌───────▼───────┐   ┌────────▼────────┐ ┌──────▼──────┐
    │ Plugin Manager│   │ Memory Manager  │ │Service Mesh │
    │               │   │ (Redis/SQLite)  │ │   (Kong)    │
    └───────────────┘   └─────────────────┘ └─────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- 4GB+ RAM recommended
- Audio devices (for voice features)
- Linux/macOS/Windows with WSL2

### Installation

1. **Clone and navigate to the project:**
   ```bash
   cd /opt/sutazaiapp
   ```

2. **Deploy JARVIS:**
   ```bash
   ./scripts/deploy-jarvis.sh
   ```

3. **Access the interface:**
   - Web Interface: http://localhost:8888
   - API Documentation: http://localhost:8888/docs
   - Health Check: http://localhost:8888/health

### Quick Test

```bash
# Test voice interface
curl -X POST "http://localhost:8888/api/task" \
  -H "Content-Type: application/json" \
  -d '{"command": "Hello JARVIS, what agents are available?", "voice_enabled": true}'

# Check system status
curl http://localhost:8888/health
```

## 🎮 Usage

### Voice Commands

JARVIS responds to natural language commands. Here are some examples:

#### Wake Words
- "Hey JARVIS"
- "JARVIS"
- "Computer"
- "Assistant"

#### Example Commands

**Development Tasks:**
- "JARVIS, create a new React component"
- "Build a REST API for user management"
- "Review the code quality of my project"
- "Run the test suite and show me the results"

**Infrastructure Operations:**
- "Deploy the application to staging"
- "Check the system performance metrics"
- "Scale up the database containers"
- "Show me the security scan results"

**Data Analysis:**
- "Analyze the user behavior data"
- "Generate a performance report"
- "Find anomalies in the system logs"
- "Create a visualization of the metrics"

**AI/ML Tasks:**
- "Train a sentiment analysis model"
- "Optimize the neural network architecture"
- "Process the image dataset"
- "Run inference on the latest model"

### Web Interface

Access the modern web interface at `http://localhost:8888`:

- **Voice Control Panel**: Start/stop voice recognition
- **Chat Interface**: Type commands or view conversation history
- **Agent Monitor**: See active agents and their status
- **System Status**: Real-time health and performance metrics

### API Integration

JARVIS provides a comprehensive REST API:

```python
import requests

# Send a command
response = requests.post("http://localhost:8888/api/task", json={
    "command": "Create a Python script to analyze log files",
    "context": {"project": "data-analysis"},
    "voice_enabled": False
})

result = response.json()
print(f"Result: {result['result']}")
print(f"Agents used: {result['agents_used']}")
```

### WebSocket Real-time Interface

```javascript
const ws = new WebSocket('ws://localhost:8888/ws');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('JARVIS response:', data.result);
};

// Send a command
ws.send(JSON.stringify({
    command: "Show me the system status",
    voice_enabled: true
}));
```

## ⚙️ Configuration

### Main Configuration

Edit `/opt/sutazaiapp/config/jarvis/config.yaml`:

```yaml
# Voice interface settings
voice:
  enable_speech_recognition: true
  enable_text_to_speech: true
  enable_wake_word_detection: true
  
  wake_words:
    - "jarvis"
    - "hey jarvis"
    - "computer"
    
  # Audio processing
  audio:
    sample_rate: 16000
    vad_aggressiveness: 2
    noise_reduction: true

# Agent coordination
agents:
  max_concurrent: 5
  timeout: 60
  retry_attempts: 3

# Task planning with Ollama
planner:
  ollama_url: "http://ollama:11434"
  planning_model: "tinyllama"
  max_steps: 10
  enable_reflection: true
```

### Environment Variables

```bash
# Core settings
JARVIS_HOST=0.0.0.0
JARVIS_PORT=8888
JARVIS_DEBUG=false

# Ollama integration
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_API_KEY=local

# Database connections
DATABASE_URL=postgresql://user:pass@postgres:5432/sutazai
REDIS_URL=redis://redis:6379/1

# Service discovery
CONSUL_HOST=consul
CONSUL_PORT=8500
```

## 🔌 Plugin Development

JARVIS supports custom plugins for extended functionality:

### Creating a Plugin

```python
#!/usr/bin/env python3
"""
Example JARVIS Plugin
"""

import asyncio
from typing import Dict, Any

# Plugin metadata
PLUGIN_INFO = {
    'name': 'Weather Plugin',
    'description': 'Provides weather information',
    'version': '1.0.0',
    'author': 'Your Name',
    'commands': ['weather', 'forecast', 'temperature']
}

class WeatherPlugin:
    def __init__(self):
        self.initialized = False
        
    async def initialize(self):
        """Initialize the plugin"""
        self.initialized = True
        
    async def execute(self, command: str, context: Dict[str, Any]) -> Any:
        """Execute plugin command"""
        if 'weather' in command.lower():
            return await self._get_weather(context)
        elif 'forecast' in command.lower():
            return await self._get_forecast(context)
        else:
            return {"error": f"Unknown command: {command}"}
            
    async def _get_weather(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get current weather"""
        # Implementation here
        return {"weather": "sunny", "temperature": "22°C"}
```

### Installing Plugins

1. Place plugin file in `/opt/sutazaiapp/services/jarvis/plugins/`
2. Enable in configuration:
   ```yaml
   plugins:
     enabled_plugins:
       - "weather_plugin"
   ```
3. Restart JARVIS

## 🤖 Agent Integration

JARVIS intelligently routes commands to 69+ specialized AI agents:

### Agent Categories

- **Development**: Backend, Frontend, Full-stack developers
- **AI/ML**: Deep learning architects, ML engineers, data scientists
- **Infrastructure**: DevOps managers, container orchestrators
- **Security**: Penetration testers, security analyzers
- **Testing**: QA validators, test automation specialists
- **Data**: Data analysts, pipeline engineers
- **Monitoring**: Observability managers, metrics collectors

### Smart Routing

JARVIS uses pattern matching and semantic analysis to route commands:

```python
# Command: "Deploy my application to production"
# Routes to: deployment-automation-master, infrastructure-devops-manager

# Command: "Analyze user behavior patterns"  
# Routes to: private-data-analyst, data-analysis-engineer

# Command: "Test the security of my API"
# Routes to: security-pentesting-specialist, semgrep-security-analyzer
```

## 📊 Monitoring and Metrics

### Prometheus Metrics

JARVIS exposes comprehensive metrics at `/metrics`:

- Request rates and response times
- Agent usage statistics
- Voice recognition performance
- Memory and system resource usage
- Error rates and success metrics

### Health Monitoring

```bash
# Check JARVIS health
curl http://localhost:8888/health

# Response
{
  "status": "healthy",
  "version": "1.0.0",
  "agents_available": 69,
  "plugins_loaded": 4,
  "voice_enabled": true
}
```

### Logging

Structured logging with multiple levels:

```bash
# View logs
docker-compose -f docker-compose.jarvis.yml logs -f jarvis

# Log locations
/opt/sutazaiapp/logs/jarvis/jarvis.log        # Application logs
/opt/sutazaiapp/logs/jarvis/access.log        # Access logs
/opt/sutazaiapp/logs/jarvis/error.log         # Error logs
```

## 🔐 Security

### Authentication (Production)

```yaml
security:
  authentication:
    enabled: true
    jwt_secret: "${JWT_SECRET}"
    token_expiry: 3600
```

### Rate Limiting

```yaml
security:
  rate_limiting:
    enabled: true
    requests_per_minute: 60
    burst_limit: 10
```

### Input Validation

All inputs are sanitized and validated:
- Maximum command length: 1000 characters
- Allowed character filtering
- SQL injection prevention
- XSS protection

## 🚨 Troubleshooting

### Common Issues

**Voice Recognition Not Working:**
```bash
# Check audio devices
ls -la /dev/snd/

# Test microphone
arecord -l

# Check JARVIS audio permissions
docker exec sutazai-jarvis ls -la /dev/snd/
```

**Ollama Connection Issues:**
```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# Restart Ollama
docker-compose restart ollama

# Check JARVIS logs
docker logs sutazai-jarvis
```

**Agent Communication Problems:**
```bash
# Check agent registry
curl http://localhost:8888/api/agents

# Test backend connection
curl http://localhost:8000/api/v1/agents

# Verify Kong routes
curl http://localhost:8001/routes
```

### Performance Tuning

**Memory Optimization:**
```yaml
memory:
  max_history: 500      # Reduce for lower memory usage
  ttl: 3600            # Shorter TTL for less storage
```

**Voice Processing:**
```yaml
voice:
  audio:
    sample_rate: 8000   # Lower quality, better performance
    vad_aggressiveness: 3  # More aggressive VAD
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black services/jarvis/
flake8 services/jarvis/
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- Documentation: [SutazAI Docs](https://docs.sutazai.com)
- Issues: [GitHub Issues](https://github.com/sutazai/issues)
- Discord: [SutazAI Community](https://discord.gg/sutazai)
- Email: support@sutazai.com

## 🗺️ Roadmap

### Version 1.1
- [ ] Multi-language support
- [ ] Voice biometrics authentication
- [ ] Mobile app integration
- [ ] Custom wake word training

### Version 1.2
- [ ] Emotional analysis
- [ ] Advanced conversation summarization
- [ ] Video processing capabilities
- [ ] Integration with external APIs

### Version 2.0
- [ ] Distributed deployment
- [ ] Edge computing support
- [ ] Advanced AI reasoning
- [ ] Autonomous task execution

---

**JARVIS** - *Making AI accessible through natural voice interaction* 🎤✨