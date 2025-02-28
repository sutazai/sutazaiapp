# SutazAI Project Directory Structure

## Overview
This document outlines the organization and purpose of each directory in the SutazAI project.

## Root Directory Structure

```
/opt/sutazaiapp/
├── ai_agents/         # AI agent implementations and related modules
├── model_management/  # ML model handling and versioning
├── backend/          # FastAPI backend server and API endpoints
├── web_ui/          # Frontend web interface
├── scripts/         # Utility and maintenance scripts
├── packages/        # Custom Python packages and wheels
├── logs/           # Application and system logs
├── doc_data/       # Documentation data and assets
└── docs/           # Project documentation
```

## Directory Details

### ai_agents/
- Core AI agent implementations
- Agent configuration files
- Agent-specific utilities and helpers
- Test suites for AI functionality

### model_management/
- Model versioning and tracking
- Model configuration files
- Model deployment scripts
- Model performance metrics

### backend/
- FastAPI application
- API route definitions
- Database models and schemas
- Middleware components
- Configuration files
- Unit tests

### web_ui/
- React/Vue.js frontend application
- Static assets (images, fonts)
- Component libraries
- Frontend build configuration
- Frontend tests

### scripts/
- Deployment scripts
- System maintenance utilities
- Database migration scripts
- Health check tools
- Performance monitoring tools

### packages/
- Custom Python package source code
- Wheel files for offline installation
- Package documentation
- Package tests

### logs/
- Application logs
- Error logs
- Audit logs
- Performance metrics logs

### doc_data/
- Documentation images
- API documentation data
- Training data documentation
- Sample configurations

### docs/
- Project documentation
- API documentation
- Setup guides
- Contributing guidelines
- Architecture diagrams
- Deployment guides

## File Naming Conventions
- Python files: lowercase with underscores (e.g., `file_name.py`)
- Class files: PascalCase (e.g., `UserManager.py`)
- Test files: prefix with `test_` (e.g., `test_user_manager.py`)
- Configuration files: lowercase with descriptive names (e.g., `production_config.yaml`)

## Important Files
- `requirements.txt`: Python package dependencies
- `setup.py`: Package installation configuration
- `.env`: Environment variables (not in version control)
- `.gitignore`: Git ignore patterns
- `README.md`: Project overview and quick start
- `CHANGELOG.md`: Version history and changes
- `LICENSE`: Project license information

## Notes
- Keep configuration files in `backend/config/` or `scripts/config/`
- Store sensitive data in environment variables, not in code
- Follow the established directory structure when adding new components
- Document any deviations from this structure in this file 