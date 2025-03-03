# SutazAI - AI-Powered Code Analysis and Generation System

## Overview
SutazAI is a comprehensive system for analyzing, maintaining, and generating code. It includes monitoring, maintenance, and deployment capabilities to ensure robust operation.

## Features
- Code analysis and quality checks
- Security scanning and vulnerability detection
- System monitoring and health checks
- Automated maintenance and optimization
- CI/CD pipeline integration
- Comprehensive logging and alerting

## Prerequisites
- Python 3.11+
- PostgreSQL 13+
- Node.js 16+ (for web UI)
- Systemd (for service management)

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/sutazai.git
cd sutazai
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the project root:
```env
DATABASE_URL=postgresql://sutazai:sutazai@localhost:5432/sutazai
SMTP_USER=your_smtp_user
SMTP_PASSWORD=your_smtp_password
```

### 5. Install System Services
```bash
sudo ./scripts/install_services.sh
```

## Usage

### Monitoring
The monitoring service runs automatically and provides:
- Resource usage monitoring (CPU, memory, disk)
- Service health checks
- Performance metrics collection
- Alert management

### Maintenance
The maintenance service runs daily at 2:00 AM and performs:
- System optimization
- Security validation
- Dependency management
- Log rotation
- Backup management

### Code Analysis
Run code analysis manually:
```bash
python scripts/code_audit.py
```

### Deployment
Deploy to production:
```bash
python scripts/deploy.py
```

## Development

### Running Tests
```bash
pytest
```

### Code Quality Checks
```bash
pylint .
mypy .
bandit -r .
safety check
```

### Pre-commit Hooks
The repository includes pre-commit hooks for:
- Security scanning
- Code quality checks
- Type checking

## Monitoring and Maintenance

### Logs
Logs are stored in `/opt/sutazaiapp/logs/`:
- `monitoring.log`: System monitoring logs
- `maintenance.log`: Maintenance task logs
- `code_audit.log`: Code analysis logs
- `deploy.log`: Deployment logs

### Metrics
Metrics are stored in `/opt/sutazaiapp/metrics/` and include:
- Resource usage metrics
- Performance metrics
- Service health status

### Alerts
Alerts are sent via email when:
- Resource usage exceeds thresholds
- Services become unhealthy
- Critical issues are detected

## Security

### Access Control
- All services run under the `sutazaiapp_dev` user
- SSH keys required for deployment
- Environment variables for sensitive data

### Security Scanning
- Bandit for Python code security
- Safety for dependency security
- Regular security audits

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Support
For support, please contact:
- Email: chrissuta01@gmail.com
- GitHub Issues: [Project Issues](https://github.com/yourusername/sutazai/issues)

# SutazAI Application

A comprehensive AI development platform featuring autonomous agents, model management, and advanced system integration.

## Project Structure

```
/opt/sutazaiapp/
├── ai_agents/               # AI Agent implementations
│   ├── auto_gpt/           # AutoGPT agent
│   ├── configs/            # Agent configurations
│   └── schemas/            # JSON schemas for validation
├── model_management/       # Model lifecycle management
├── backend/               # Core backend services
├── web_ui/               # Frontend application
├── scripts/              # Utility scripts
├── packages/             # Custom packages
│   └── wheels/          # Python wheel files
├── logs/                # Application logs
├── doc_data/            # Documentation data
└── docs/               # Project documentation
```

## Setup

1. Create and activate virtual environment:
```bash
python3.11 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

3. Configure environment:
- Copy `.env.example` to `.env`
- Update configuration values as needed

4. Initialize directories:
```bash
mkdir -p logs doc_data
```

## Development

### Running Tests
```bash
pytest
```

### Code Quality
- Linting: `pylint backend/ ai_agents/ model_management/`
- Type checking: `mypy .`
- Security scan: `bandit -r .`

### Documentation
- API docs available at `/docs` endpoint when running backend
- Additional documentation in `/docs` directory

## Components

### AI Agents
- Base agent framework in `ai_agents/base_agent.py`
- AutoGPT implementation in `ai_agents/auto_gpt/`
- Configuration management via JSON schemas

### Backend Services
- FastAPI-based REST API
- Database migrations using Alembic
- Comprehensive error handling and logging

### Model Management
- Model versioning and deployment
- Performance monitoring
- Resource optimization

## Security

- All sensitive data must be stored in `.env`
- API keys and credentials never committed to repo
- Regular security audits with Bandit
- Input validation using Pydantic

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes following style guide
4. Add tests for new functionality
5. Submit pull request

## License

Copyright (c) 2024 SutazAI. All rights reserved.

# Sutazaiapp: Comprehensive AI-Powered Development System

## Project Overview

Sutazaiapp is an advanced, offline-first AI development system designed to provide comprehensive code generation, management, and deployment capabilities.

### System Architecture

- **Code Server**: 192.168.100.28
- **Deployment Server**: 192.168.100.100
- **Owner**: Florin Cristian Suta
  - Phone: +48517716005
  - Email: chrissuta01@gmail.com

### Key Components

1. **AI Agents** (`ai_agents/`)
   - Integrates SuperAGI, AutoGPT
   - Modular agent management system

2. **Model Management** (`model_management/`)
   - Supports GPT4All, DeepSeek-Coder-33B
   - Offline model handling

3. **Backend** (`backend/`)
   - FastAPI-based services
   - Strict security measures
   - OTP-based external call authorization

4. **Web UI** (`web_ui/`)
   - React frontend
   - Responsive design

5. **Deployment Scripts** (`scripts/`)
   - Automated deployment
   - Rollback capabilities
   - System configuration

### Setup Instructions

#### Prerequisites

- Python 3.11
- Ubuntu/Debian-based Linux
- Root access for initial setup

#### Installation Steps

1. Clone the repository
```bash
git clone https://github.com/chrissuta/sutazaiapp.git /opt/sutazaiapp
```

2. Run Setup Script
```bash
sudo bash /opt/sutazaiapp/scripts/setup_sutazaiapp.sh
```

### Security Features

- OTP-based authorization
- Offline-first design
- Comprehensive code auditing
- Strict permission management

### Development Workflow

1. Use `scripts/code_audit.sh` for regular code quality checks
2. Commit changes to the appropriate module directory
3. Run tests before deployment

### Logging

Centralized logging available in `logs/` directory

### Documentation

Comprehensive documentation located in `docs/` directory

## License

[To be determined - contact project owner]

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

# SutazAI: Autonomous AI Development Platform

## 🚀 Project Overview

SutazAI is an advanced, self-improving AI development platform designed to push the boundaries of artificial intelligence through comprehensive, secure, and autonomous systems.

## 🌟 Key Features (Version 1.1.9)

### 1. Autonomous System Orchestration
- Dynamic component management
- Self-healing system architecture
- Intelligent resource allocation
- Centralized component coordination

### 2. Advanced Dependency Management
- Intelligent vulnerability scanning
- Comprehensive dependency tracking
- Automated update mechanisms
- Dependency graph generation

- Zero-trust network design
- Multi-tier threat detection
- Adaptive access control

### 4. Performance Optimization
- Real-time resource monitoring
- Predictive system health assessment
- Intelligent bottleneck detection
- Autonomous performance tuning

### 5. System Maintenance & Self-Repair
- Comprehensive system checkup
- Automatic syntax error fixing
- Dependency resolution
- Empty file detection
- Duplicate code identification

## 🏗️ System Architecture

### Core Components
1. **System Orchestrator**: Coordinates system-wide operations
2. **Dependency Manager**: Advanced dependency tracking
4. **Performance Optimizer**: Continuous system tuning
5. **System Integrator**: Component discovery and synchronization

## 📦 Project Structure

```
/opt/sutazaiapp/
├── ai_agents/
│   ├── superagi/
│   ├── auto_gpt/
│   ├── langchain_agents/
│   ├── tabbyml/
│   └── semgrep/
├── model_management/
│   ├── GPT4All/
│   ├── DeepSeek-Coder-33B/
│   └── ...
├── backend/
│   ├── main.py
│   ├── services/
│   ├── api_routes.py
│   └── ...
├── web_ui/
│   ├── package.json
│   ├── src/
│   └── ...
├── scripts/
│   ├── deploy.sh
│   ├── trigger_deploy.sh
│   ├── otp_override.py
│   ├── syntax_fixer.py
│   └── ...
├── packages/
│   ├── wheels/  (offline .whl files)
│   └── node/    (offline node modules)
├── logs/
├── doc_data/
└── docs/


- Zero-trust architecture
- Granular access control
- Continuous threat assessment

## 🚀 Performance Strategy
- Intelligent resource allocation
- Predictive scaling
- Real-time performance monitoring
- Autonomous optimization

## 📋 Prerequisites

### Hardware Requirements
- **CPU**: 8+ cores
- **RAM**: 32GB+
- **Storage**: 256GB SSD
- **OS**: Ubuntu 20.04+ LTS

### Software Requirements
- Python 3.11
- pip 23.3+
- Node.js 16+

## 🛠️ Installation

### 1. Clone Repository
```bash
git clone https://github.com/sutazai/sutazaiapp.git
cd sutazaiapp
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Initialize System
```bash
python core_system/system_orchestrator.py
```

## 🧪 Testing

### Running Tests
```bash
pytest tests/
```

### Test Coverage
- Unit Tests: 90%+
- Integration Tests: Comprehensive
- Performance Benchmarks: Included

## 📊 Monitoring

- Centralized logging
- Real-time performance tracking
- Automated alerting
- Distributed tracing support

## 🔍 Debugging

- Comprehensive error logging
- Performance profiling
- Autonomous error recovery

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create pull request

## 📜 Documentation

- [SutazAI Master Plan](/docs/SUTAZAI_MASTER_PLAN.md)
- [System Architecture](/docs/SYSTEM_ARCHITECTURE.md)
- [Dependency Management](/docs/DEPENDENCY_MANAGEMENT.md)
- [Contribution Guidelines](/docs/CONTRIBUTION_GUIDELINES.md)

## 📞 Contact

**Creator**: Florin Cristian Suta
- **Email**: chrissuta01@gmail.com
- **Phone**: +48517716005

## 📜 License

Proprietary - All Rights Reserved

---

*Empowering Autonomous Intelligence*

## 🛠️ System Maintenance Tools

SutazAI includes powerful self-maintenance tools to ensure optimal system health:

### System Checkup
Run a comprehensive system checkup to identify issues:
```bash
python system_checkup.py
```

This tool checks for:
- Syntax errors in Python files
- Import errors and missing dependencies
- Empty files
- Duplicate code
- Uninstalled required packages

### Syntax Fixer
Automatically fix syntax errors in Python files:
```bash
python scripts/syntax_fixer.py <directory>
```

### Fix All Issues
To run a complete system repair that addresses all detected issues:
```bash
python fix_all_issues.py
```

This script orchestrates multiple repair tools to fix:
- Syntax errors
- Import problems
- Dependency issues
- System configuration problems
- Performance bottlenecks

# Updated on Tue Feb 25 03:38:01 PM UTC 2025

# SutazAI Application

A Python 3.11 compatible application with advanced features.

## Requirements

- Python 3.11 or later
- Dependencies listed in requirements.txt

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/sutazai/sutazaiapp.git
   cd sutazaiapp
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the application:

```

# Project Development Guidelines

## Code Quality Standards

### Python Code
- We use `black` for code formatting
- `pylint` for static code analysis
- `mypy` for type checking
- `isort` for import sorting

### TypeScript/React Code
- ESLint for static code analysis
- Prettier for consistent code formatting
- TypeScript for type safety

## Development Setup

### Python Environment
1. Create a virtual environment:
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   ```

2. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. Run code quality checks:
   ```bash
   black .
   pylint **/*.py
   mypy .
   ```

### Web UI Setup
1. Install Node.js dependencies:
   ```bash
   cd web_ui
   npm install
   ```

2. Run code quality checks:
   ```bash
   npm run lint
   npm run format
   ```

## Commit Guidelines
- Always run code quality checks before committing
- Write clear, descriptive commit messages
- Reference issue numbers when applicable

## Continuous Integration
- All pull requests must pass automated code quality checks
- Maintain 80% test coverage for new code
