# 🚀 SutazAI - Local AI Task Automation

> Practical task automation using local AI with GPT-OSS model. No cloud dependencies, no API costs, just working automation tools for developers.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org)
[![Docker](https://img.shields.io/badge/docker-20.0+-blue.svg)](https://www.docker.com)
[![Status](https://img.shields.io/badge/status-development-yellow.svg)](https://github.com)

## 🎯 What It Does

- **Code Review**: Automated code analysis and improvement suggestions
- **Security Scanning**: Find vulnerabilities and security issues
- **Test Generation**: Create unit tests automatically
- **Deployment Automation**: CI/CD pipeline automation
- **Documentation**: Generate and maintain documentation
- **All Local**: Runs entirely on your machine, no external APIs

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/sutazai.git
cd sutazai

# 2. Start the system
docker-compose up -d

# 3. System will be available at:
#    - Backend API: http://localhost:10010/docs
#    - Frontend UI: http://localhost:10011
#    - Health: http://localhost:10010/health
```

The system will start the core infrastructure services.

## 💻 Example Usage

### Code Review Workflow
```python
# Review your Python code
python workflows/simple_code_review.py
```

### Security Scan
```python
# Scan for vulnerabilities
python workflows/security_scan_workflow.py
```

### Deployment Automation
```python
# Automate your deployment
python workflows/deployment_automation.py
```

## 🛠️ Requirements

- **Docker**: 20.0+ 
- **Docker Compose**: 2.0+
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 10GB for models and data
- **CPU**: 4+ cores recommended

## 📦 What's Included

### Currently Running Services (13 containers)
**IMPORTANT**: Most are STUB IMPLEMENTATIONS with basic "Hello World" endpoints
- `ai-agent-orchestrator` - Basic HTTP service (stub)
- `multi-agent-coordinator` - Basic HTTP service (stub) 
- `task-assignment-coordinator` - Basic HTTP service (stub)
- `ai-senior-engineer-phase1` - Basic HTTP service (stub)
- Plus 9 other basic container services
- **ACTUAL AI FUNCTIONALITY**: Limited to Ollama with GPT-OSS model

### Local AI Model
- **GPT-OSS**: The exclusive model for all AI operations
- **Ollama**: Local GPT-OSS model serving, no internet required
- **100% Private**: Your code never leaves your machine

## 🎮 Commands

```bash
# Start the system
docker-compose up -d

# Stop the system
docker-compose down

# View logs
docker-compose logs -f

# Check status
curl http://localhost:10010/health
```

## 📚 Documentation

- [Project Overview](docs/overview.md) - Complete project summary and architecture
- [Setup Guide](docs/setup/) - Installation and configuration instructions
- [API Documentation](docs/backend/api_reference.md) - Complete API reference
- [Frontend Guide](docs/frontend/) - UI components and styling
- [Deployment Guide](docs/ci-cd/) - CI/CD and deployment processes
- [Working Agents List](docs/PRACTICAL_AGENTS_LIST.md) - Available AI agents
- [Example Workflows](workflows/) - Automation workflow examples
- [Live API Docs](http://localhost:10010/docs) - Interactive API documentation (after starting)

## 🔒 Privacy & Security

- **No External APIs**: Everything runs locally
- **No Data Collection**: Your data stays on your machine
- **No Internet Required**: Can run completely offline
- **Open Source**: Audit the code yourself

## 🤝 Contributing

Contributions are welcome! Please focus on:
- Practical automation workflows
- Performance improvements
- Bug fixes
- Documentation improvements

## 📝 License

MIT License - See [LICENSE](LICENSE) file

---

**CRITICAL WARNING**: 
- This system is ~90% documentation and ~10% implementation
- Most "AI agents" are placeholder HTTP services returning stub responses
- NO AGI, quantum computing, or advanced AI orchestration exists
- Documentation claiming 149 agents is FALSE - only 13 basic containers run
- Following old documentation will lead to system failure
