# SutazAI Application Directory Structure

## Core Directories

### `/ai_agents`
- Contains AI agent implementations
- Includes base agent classes, factory patterns, and specific agent types
- Configuration management for agents

### `/model_management`
- Model lifecycle management
- Model versioning and updates
- Performance monitoring
- Configuration: `model_management/config/`

### `/backend`
- Core application backend services
- API endpoints and routing
- Database models and migrations
- Configuration: `backend/config/`

### `/web_ui`
- Frontend application code
- React/Vue components
- Static assets
- Build configurations

## Support Directories

### `/packages`
- Custom Python packages
- Wheel files: `packages/wheels/`
- Internal utilities and shared code

### `/scripts`
- Deployment scripts
- Maintenance utilities
- Development tools
- System management scripts

### `/logs`
- Application logs
- Performance metrics: `logs/performance/`
- Security scans: `logs/security/`
- Audit logs: `logs/audit/`

### `/doc_data`
- Training data: `doc_data/training/`
- Documentation assets
- Sample data files
- Test fixtures

### `/docs`
- Project documentation
- API documentation
- Architecture diagrams
- Deployment guides

## Development Directories

### `/migrations`
- Database migration scripts
- Schema version control
- Migration history

### `/tests`
- Unit tests
- Integration tests
- Test configurations
- Test utilities

### `/venv`
- Python virtual environment
- Isolated dependencies
- Development tools

## Configuration Files

### Root Level
- `requirements.txt`: Python dependencies
- `pyproject.toml`: Project configuration
- `alembic.ini`: Database migration config
- `.env`: Environment variables (gitignored)
- `.pylintrc`: Python linting rules
- `.pre-commit-config.yaml`: Git hooks
- `.flake8`: Flake8 configuration

## Version Control

### `/.git`
- Git repository data
- Version history
- Branch information

### `/.github`
- GitHub workflows
- CI/CD configurations
- Issue templates

## Security

### `/secrets`
- Secure credential storage
- API keys (gitignored)
- Certificates

### `/backups`
- Database backups
- Configuration backups
- System state backups

## Notes

1. All sensitive data should be stored in `/secrets` and properly gitignored
2. Logs are automatically rotated in `/logs` subdirectories
3. Configuration files should be environment-specific
4. Documentation should be kept up-to-date with code changes 