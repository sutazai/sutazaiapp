# CHANGELOG INDEX - SutazAI System

## Overview
This index provides quick access to all CHANGELOG files across the system.
Generated: 2025-08-19

## CHANGELOG Files by Category

### Core System
- [Main CHANGELOG](/CHANGELOG.md) - System-wide changes
- [Backend CHANGELOG](/backend/CHANGELOG.md) - Backend service changes
- [Frontend CHANGELOG](/frontend/CHANGELOG.md) - Frontend application changes

### Backend Components
- [Backend App](/backend/app/CHANGELOG.md) - Application core
- [API](/backend/app/api/CHANGELOG.md) - API endpoints
- [Service Mesh](/backend/app/mesh/CHANGELOG.md) - Mesh implementation
- [Services](/backend/app/services/CHANGELOG.md) - Service components

### Docker & Infrastructure
- [Docker](/docker/CHANGELOG.md) - Docker configurations
- [Base Images](/docker/base/CHANGELOG.md) - Base Docker images
- [DinD](/docker/dind/CHANGELOG.md) - Docker-in-Docker
- [MCP Services](/docker/mcp-services/CHANGELOG.md) - MCP containers

### Scripts & Automation
- [Scripts](/scripts/CHANGELOG.md) - All scripts
- [Enforcement](/scripts/enforcement/CHANGELOG.md) - Rule enforcement
- [Deployment](/scripts/deployment/CHANGELOG.md) - Deployment scripts
- [Monitoring](/scripts/monitoring/CHANGELOG.md) - Monitoring tools
- [MCP Scripts](/scripts/mcp/CHANGELOG.md) - MCP management

### Configuration & Documentation
- [Agents](/agents/CHANGELOG.md) - AI agent configurations
- [Claude Config](/.claude/CHANGELOG.md) - Claude configuration
- [Claude Agents](/.claude/agents/CHANGELOG.md) - Agent definitions
- [Documentation](/docs/CHANGELOG.md) - Documentation changes
- [Reports](/docs/reports/CHANGELOG.md) - Investigation reports
- [Important Rules](/IMPORTANT/CHANGELOG.md) - Critical rules

### Testing
- [Tests](/tests/CHANGELOG.md) - Test infrastructure

## Quick Links
- Search all CHANGELOGs: `grep -r "search_term" */CHANGELOG.md`
- Recent changes: `find . -name "CHANGELOG.md" -mtime -7`
- Today's changes: `find . -name "CHANGELOG.md" -mtime -1`

## Maintenance
Run `/opt/sutazaiapp/scripts/enforcement/create_changelogs.sh` to ensure all directories have CHANGELOG files.
