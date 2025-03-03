# Directory Structure

This document outlines the directory structure of the SutazAI application.

## Root Directory

```
/opt/sutazaiapp/
├── ai_agents/           # AI agent modules and components
├── model_management/    # Model management and monitoring
├── backend/            # Backend API and services
├── web_ui/            # Frontend web interface
├── scripts/           # Utility and deployment scripts
├── packages/          # Local package dependencies
├── logs/             # Application logs
├── doc_data/         # Document processing data
└── docs/             # Documentation
```

## Directory Details

### ai_agents/
- Contains AI agent modules and components
- Includes AutoGPT integration
- Test files for each module

### model_management/
- Model management utilities
- Performance monitoring
- System optimization
- Dependency management

### backend/
- FastAPI application
- API routes and endpoints
- Service implementations
- Configuration files

### web_ui/
- React frontend application
- UI components
- Static assets
- Frontend configuration

### scripts/
- Deployment scripts
- Utility scripts
- Configuration scripts
- Maintenance scripts

### packages/
- Local package dependencies
- Wheels directory for offline installation
- Custom packages

### logs/
- Application logs
- Error logs
- Audit logs
- Performance logs

### doc_data/
- Document processing data
- Parsed documents
- Generated diagrams
- Temporary files

### docs/
- Project documentation
- API documentation
- Deployment guides
- User guides

## Configuration Files

- Backend configuration: `backend/config/`
- Script configuration: `scripts/config/`
- Model configuration: `model_management/config/`
- AI agent configuration: `ai_agents/config/`

## Naming Conventions

1. Python Files:
   - Use lowercase with underscores
   - End with `.py`
   - Example: `file_utils.py`

2. Directories:
   - Use lowercase with underscores
   - Example: `model_management`

3. Test Files:
   - Prefix with `test_`
   - Match the name of the file being tested
   - Example: `test_utils.py`

4. Configuration Files:
   - Use lowercase with underscores
   - Use appropriate extension (`.json`, `.yaml`, `.toml`)
   - Example: `config.toml` 