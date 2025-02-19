# 🌐 SutazAI Project Structure

## Generated: 2025-02-19T07:30:02.088077

### Project Overview
- **Total Components**: 28
- **Total Directories**: 22
- **Total Files**: 6

## Detailed Structure

### AI AGENTS
- 📁 **supreme_ai/** {'description': 'Supreme AI (non-root orchestrator)', 'path': '/opt/sutazai_project/SutazAI/ai_agents/supreme_ai', 'contents': ['orchestrator.py']}
  - 📄 orchestrator.py
- 📁 **auto_gpt/** {'description': 'Autonomous GPT Agent', 'path': '/opt/sutazai_project/SutazAI/ai_agents/auto_gpt', 'contents': ['schemas', 'configs', 'src', 'logs']}
  - 📄 schemas
  - 📄 configs
  - 📄 src
  - 📄 logs
- 📁 **superagi/** {'description': 'SuperAGI Agent Framework', 'path': '/opt/sutazai_project/SutazAI/ai_agents/superagi', 'contents': []}
- 📁 **langchain_agents/** {'description': 'LangChain-based Agents', 'path': '/opt/sutazai_project/SutazAI/ai_agents/langchain_agents', 'contents': []}
- 📁 **tabbyml/** {'description': 'TabbyML Integration', 'path': '/opt/sutazai_project/SutazAI/ai_agents/tabbyml', 'contents': []}
- 📁 **semgrep/** {'description': 'Code Analysis Agents', 'path': '/opt/sutazai_project/SutazAI/ai_agents/semgrep', 'contents': []}
- 📁 **gpt_engineer/** {'description': 'GPT-based Code Generation', 'path': '/opt/sutazai_project/SutazAI/ai_agents/gpt_engineer', 'contents': ['schemas', 'configs', 'src', 'logs']}
  - 📄 schemas
  - 📄 configs
  - 📄 src
  - 📄 logs
- 📁 **aider/** {'description': 'AI Collaborative Coding Assistant', 'path': '/opt/sutazai_project/SutazAI/ai_agents/aider', 'contents': []}

### MODEL MANAGEMENT
- 📁 **GPT4All/** {'description': 'Open-source Language Model', 'path': '/opt/sutazai_project/SutazAI/model_management/GPT4All', 'contents': []}
- 📁 **DeepSeek-R1/** {'description': 'Research-grade Language Model', 'path': '/opt/sutazai_project/SutazAI/model_management/DeepSeek-R1', 'contents': []}
- 📁 **DeepSeek-Coder/** {'description': 'Code Generation Model', 'path': '/opt/sutazai_project/SutazAI/model_management/DeepSeek-Coder', 'contents': []}
- 📁 **Llama2/** {'description': "Meta's Language Model", 'path': '/opt/sutazai_project/SutazAI/model_management/Llama2', 'contents': []}
- 📁 **Molmo/** {'description': 'Diagram Recognition Model', 'path': '/opt/sutazai_project/SutazAI/model_management/Molmo', 'contents': []}

### BACKEND
- 📁 **main.py/** {'description': 'Application Entry Point', 'path': '/opt/sutazai_project/SutazAI/backend/main.py', 'size': 1217}
- 📁 **api_routes.py/** {'description': 'API Endpoint Definitions', 'path': '/opt/sutazai_project/SutazAI/backend/api_routes.py', 'size': 4558}
- 📁 **services/** {'description': 'Business Logic Implementations', 'path': '/opt/sutazai_project/SutazAI/backend/services', 'contents': []}
- 📁 **config/** {'description': 'Backend Configuration', 'path': '/opt/sutazai_project/SutazAI/backend/config', 'contents': ['__init__.py', 'database.py']}
  - 📄 __init__.py
  - 📄 database.py
- 📁 **tests/** {'description': 'Backend Test Suite', 'path': '/opt/sutazai_project/SutazAI/backend/tests', 'contents': []}

### WEB UI
- 📄 **package.json**: Node.js Dependencies
- 📄 **node_modules**: Installed NPM Packages
- 📁 **src/** {'description': 'Frontend Source Code', 'path': '/opt/sutazai_project/SutazAI/web_ui/src', 'contents': []}
- 📁 **public/** {'description': 'Static Assets', 'path': '/opt/sutazai_project/SutazAI/web_ui/public', 'contents': []}
- 📄 **build_or_dist**: Compiled Frontend

### SCRIPTS
- 📁 **deploy.sh/** {'description': 'Main Online Deployment Script', 'path': '/opt/sutazai_project/SutazAI/scripts/deploy.sh', 'size': 3016}
- 📄 **setup_repos.sh**: Manual Repository Synchronization
- 📁 **test_pipeline.py/** {'description': 'Comprehensive Testing Pipeline', 'path': '/opt/sutazai_project/SutazAI/scripts/test_pipeline.py', 'size': 4624}

### PACKAGES
- 📁 **wheels/** {'description': 'Pinned Python Wheel Packages', 'path': '/opt/sutazai_project/SutazAI/packages/wheels', 'contents': []}
- 📁 **node/** {'description': 'Cached Node.js Modules', 'path': '/opt/sutazai_project/SutazAI/packages/node', 'contents': []}

### LOGS
- 📄 **deploy.log**: Deployment Logs
- 📄 **pipeline.log**: CI/CD Pipeline Logs
- 📄 **online_calls.log**: External API Call Logs

### DOC DATA
- 📁 **pdfs/** {'description': 'PDF Document Storage', 'path': '/opt/sutazai_project/SutazAI/doc_data/pdfs', 'contents': []}
- 📁 **diagrams/** {'description': 'Project Diagrams and Visualizations', 'path': '/opt/sutazai_project/SutazAI/doc_data/diagrams', 'contents': []}

### ROOT FILES
- 📄 requirements.txt
- 📄 venv
- 📄 README.md
