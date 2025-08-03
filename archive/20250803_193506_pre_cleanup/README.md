# 🚀 SutazAI - Local AI Task Automation

> Practical task automation using local AI models. No cloud dependencies, no API costs, just working automation tools for developers.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org)
[![Docker](https://img.shields.io/badge/docker-20.0+-blue.svg)](https://www.docker.com)
[![Status](https://img.shields.io/badge/status-production_ready-green.svg)](https://github.com)

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

# 2. Start the system (one command!)
./start.sh

# 3. System will be available at:
#    - API: http://localhost:8000/docs
#    - Health: http://localhost:8000/health
```

That's it! The system will pull all necessary images and start automatically.

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

### Working AI Agents (34 total)
- `senior-ai-engineer` - Code implementation and optimization
- `code-generation-improver` - Code quality analysis
- `testing-qa-validator` - Automated testing
- `security-pentesting-specialist` - Security scanning
- `deployment-automation-master` - CI/CD automation
- [See full list](docs/PRACTICAL_AGENTS_LIST.md)

### Local AI Models
- **TinyLlama (637MB)**: Fast, efficient general-purpose model
- **Ollama**: Local model serving, no internet required
- **100% Private**: Your code never leaves your machine

## 🎮 Commands

```bash
# Start the system
./start.sh

# Stop the system
./stop.sh

# View logs
docker-compose -f docker-compose.tinyllama.yml logs -f

# Check status
curl http://localhost:8000/health
```

## 📚 Documentation

- [Project Overview](docs/overview.md) - Complete project summary and architecture
- [Setup Guide](docs/setup/) - Installation and configuration instructions
- [API Documentation](docs/backend/api_reference.md) - Complete API reference
- [Frontend Guide](docs/frontend/) - UI components and styling
- [Deployment Guide](docs/ci-cd/) - CI/CD and deployment processes
- [Working Agents List](docs/PRACTICAL_AGENTS_LIST.md) - Available AI agents
- [Example Workflows](workflows/) - Automation workflow examples
- [Live API Docs](http://localhost:8000/docs) - Interactive API documentation (after starting)

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

**Note**: This is a practical tool for developers. No automation system, no system state, just useful automation.
