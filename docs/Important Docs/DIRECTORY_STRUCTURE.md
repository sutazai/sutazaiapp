# 🌐 SutazAI Project Structure

## Generated: 2025-02-19T20:45:02.094616

### Project Overview
- **Total Components**: 28
- **Total Directories**: 22
- **Total Files**: 6

## Detailed Structure

### AI AGENTS
- 📁 **supreme_ai/** {'description': 'Supreme AI (non-root orchestrator)', 'path': '/opt/SutazAI/ai_agents/supreme_ai', 'contents': ['orchestrator.py']}
  - 📄 orchestrator.py
- 📁 **auto_gpt/** {'description': 'Autonomous GPT Agent', 'path': '/opt/SutazAI/ai_agents/auto_gpt', 'contents': ['schemas', 'configs', 'src', 'logs']}
  - 📄 schemas
  - 📄 configs
  - 📄 src
  - 📄 logs
- 📁 **superagi/** {'description': 'SuperAGI Agent Framework', 'path': '/opt/SutazAI/ai_agents/superagi', 'contents': []}
- 📁 **langchain_agents/** {'description': 'LangChain-based Agents', 'path': '/opt/SutazAI/ai_agents/langchain_agents', 'contents': []}
- 📁 **tabbyml/** {'description': 'TabbyML Integration', 'path': '/opt/SutazAI/ai_agents/tabbyml', 'contents': []}
- 📁 **semgrep/** {'description': 'Code Analysis Agents', 'path': '/opt/SutazAI/ai_agents/semgrep', 'contents': []}
- 📁 **gpt_engineer/** {'description': 'GPT-based Code Generation', 'path': '/opt/SutazAI/ai_agents/gpt_engineer', 'contents': ['schemas', 'configs', 'src', 'logs']}
  - 📄 schemas
  - 📄 configs
  - 📄 src
  - 📄 logs
- 📁 **aider/** {'description': 'AI Collaborative Coding Assistant', 'path': '/opt/SutazAI/ai_agents/aider', 'contents': []}

### MODEL MANAGEMENT
- 📁 **GPT4All/** {'description': 'Open-source Language Model', 'path': '/opt/SutazAI/model_management/GPT4All', 'contents': []}
- 📁 **DeepSeek-R1/** {'description': 'Research-grade Language Model', 'path': '/opt/SutazAI/model_management/DeepSeek-R1', 'contents': []}
- 📁 **DeepSeek-Coder/** {'description': 'Code Generation Model', 'path': '/opt/SutazAI/model_management/DeepSeek-Coder', 'contents': []}
- 📁 **Llama2/** {'description': "Meta's Language Model", 'path': '/opt/SutazAI/model_management/Llama2', 'contents': []}
- 📁 **Molmo/** {'description': 'Diagram Recognition Model', 'path': '/opt/SutazAI/model_management/Molmo', 'contents': []}

### BACKEND
- 📁 **main.py/** {'description': 'Application Entry Point', 'path': '/opt/SutazAI/backend/main.py', 'size': 1217}
- 📁 **api_routes.py/** {'description': 'API Endpoint Definitions', 'path': '/opt/SutazAI/backend/api_routes.py', 'size': 4558}
- 📁 **services/** {'description': 'Business Logic Implementations', 'path': '/opt/SutazAI/backend/services', 'contents': []}
- 📁 **config/** {'description': 'Backend Configuration', 'path': '/opt/SutazAI/backend/config', 'contents': ['__init__.py', 'database.py']}
  - 📄 __init__.py
  - 📄 database.py
- 📁 **tests/** {'description': 'Backend Test Suite', 'path': '/opt/SutazAI/backend/tests', 'contents': []}

### WEB UI
- 📄 **package.json**: Node.js Dependencies
- 📄 **node_modules**: Installed NPM Packages
- 📁 **src/** {'description': 'Frontend Source Code', 'path': '/opt/SutazAI/web_ui/src', 'contents': []}
- 📁 **public/** {'description': 'Static Assets', 'path': '/opt/SutazAI/web_ui/public', 'contents': []}
- 📄 **build_or_dist**: Compiled Frontend

### SCRIPTS
- 📁 **deploy.sh/** {'description': 'Main Online Deployment Script', 'path': '/opt/SutazAI/scripts/deploy.sh', 'size': 3016}
- 📄 **setup_repos.sh**: Manual Repository Synchronization
- 📁 **test_pipeline.py/** {'description': 'Comprehensive Testing Pipeline', 'path': '/opt/SutazAI/scripts/test_pipeline.py', 'size': 4624}

### PACKAGES
- 📁 **wheels/** {'description': 'Pinned Python Wheel Packages', 'path': '/opt/SutazAI/packages/wheels', 'contents': []}
- 📁 **node/** {'description': 'Cached Node.js Modules', 'path': '/opt/SutazAI/packages/node', 'contents': []}

### LOGS
- 📄 **deploy.log**: Deployment Logs
- 📄 **pipeline.log**: CI/CD Pipeline Logs
- 📄 **online_calls.log**: External API Call Logs

### DOC DATA
- 📁 **pdfs/** {'description': 'PDF Document Storage', 'path': '/opt/SutazAI/doc_data/pdfs', 'contents': []}
- 📁 **diagrams/** {'description': 'Project Diagrams and Visualizations', 'path': '/opt/SutazAI/doc_data/diagrams', 'contents': []}

### ROOT FILES
- 📄 requirements.txt
- 📄 venv
- 📄 README.md

# SutazAI Directory Structure

This document outlines the layout of /opt/SutazAI:
- ai_agents/: Contains AI-specific frameworks and sub-agents
- model_management/: Local models...
- backend/: The primary Python backend (FastAPI)...
- web_ui/: Code for the front-end UI...
- scripts/: Shell and Python scripts for deployments, OTP checks...
- packages/: Offline packages (wheels, node modules)...
- logs/: System and application logs...
- doc_data/: Supporting data for docs...
- docs/: Project documentation...

(Include any additional details or rationale as needed.)

Provide a bullet list explaining each top-level directory’s purpose, e.g.:
ai_agents/: code for sub‑agents (SuperAGI, AutoGPT, etc.)
model_management/: LLM binaries, local model files
backend/: FastAPI or Python backend
web_ui/: React or similar front‑end
scripts/: all deployment or maintenance scripts
packages/: offline dependencies (Python wheels, Node modules)
logs/: rotating logs for each service
doc_data/: raw data files (PDF, DOCX for testing)
docs/: all documentation
