# Phase 1: Codebase Audit & Repository Reorganization

## Progress Tracking

- [x] Verify existing directory structure
- [x] Confirm sutazaiapp_dev user setup
- [x] Check Python virtual environment
- [ ] Update directory ownership and permissions
- [ ] Fix security issues in scripts
- [ ] Organize dependencies (requirements files)
- [ ] Complete automated code audits
- [ ] Document directory structure

## Directory Structure

Current structure (as of initial audit):
```
/opt/sutazaiapp/
├── ai_agents/         # AI agent implementations
├── backend/           # Backend API and server logic
├── config/            # Configuration files
├── core_system/       # Core system components
├── data/              # Data storage
├── doc_data/          # Document-related data
├── docs/              # Documentation
├── logs/              # Log files
├── model_management/  # ML model management
├── packages/          # Package resources
├── scripts/           # Utility scripts
├── types/             # Type definitions
├── uploads/           # Upload storage
├── venv/              # Python virtual environment
└── web_ui/            # Web user interface
```

## Security Issues

Key security issues identified from bandit scan:
1. Subprocess calls with shell=True in scripts/fix_performance_issues.py
2. Hardcoded /tmp directory usage in scripts/fix_performance_issues.py
3. Starting processes with partial paths in scripts/fix_performance_issues.py
4. Subprocess calls without proper sanitization

## Dependencies

Multiple requirements files found:
1. requirements.txt - Main project requirements
2. requirements_current.txt - Current installed packages (from pip freeze)
3. packages/requirements.txt - Optimized requirements with Python version constraints 