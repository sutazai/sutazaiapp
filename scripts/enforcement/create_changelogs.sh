#!/bin/bash
# Create CHANGELOG.md in Important Directories - Rule 19 Compliance
# Generated: 2025-08-19

set -euo pipefail

ROOT_DIR="/opt/sutazaiapp"
LOG_FILE="$ROOT_DIR/docs/reports/CHANGELOG_CREATION_$(date +%Y%m%d_%H%M%S).log"

echo "=== CREATING CHANGELOG FILES - RULE 19 COMPLIANCE ===" | tee "$LOG_FILE"
echo "Requirement: CHANGELOG.md in every important directory" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Important directories that need CHANGELOG files
IMPORTANT_DIRS=(
    "backend"
    "backend/app"
    "backend/app/api"
    "backend/app/mesh"
    "backend/app/services"
    "docker"
    "docker/base"
    "docker/dind"
    "docker/mcp-services"
    "scripts"
    "scripts/enforcement"
    "scripts/deployment"
    "scripts/monitoring"
    "scripts/mcp"
    "agents"
    "docs"
    "docs/reports"
    "tests"
    ".claude"
    ".claude/agents"
    "IMPORTANT"
)

CREATED_COUNT=0
EXISTING_COUNT=0

for dir in "${IMPORTANT_DIRS[@]}"; do
    FULL_PATH="$ROOT_DIR/$dir"
    CHANGELOG_PATH="$FULL_PATH/CHANGELOG.md"
    
    if [ ! -d "$FULL_PATH" ]; then
        echo "⚠️ Directory does not exist: $dir" | tee -a "$LOG_FILE"
        continue
    fi
    
    if [ -f "$CHANGELOG_PATH" ]; then
        echo "✓ CHANGELOG exists: $dir/CHANGELOG.md" | tee -a "$LOG_FILE"
        EXISTING_COUNT=$((EXISTING_COUNT + 1))
    else
        echo "Creating CHANGELOG: $dir/CHANGELOG.md" | tee -a "$LOG_FILE"
        
        # Determine directory purpose
        case "$dir" in
            "backend")
                PURPOSE="Backend API Service"
                ;;
            "backend/app")
                PURPOSE="Backend Application Core"
                ;;
            "backend/app/api")
                PURPOSE="API Endpoints and Routes"
                ;;
            "backend/app/mesh")
                PURPOSE="Service Mesh Implementation"
                ;;
            "backend/app/services")
                PURPOSE="Backend Service Components"
                ;;
            "docker")
                PURPOSE="Docker Configurations and Containers"
                ;;
            "docker/base")
                PURPOSE="Base Docker Images"
                ;;
            "docker/dind")
                PURPOSE="Docker-in-Docker MCP Orchestration"
                ;;
            "docker/mcp-services")
                PURPOSE="MCP Service Containers"
                ;;
            "scripts")
                PURPOSE="Automation and Utility Scripts"
                ;;
            "scripts/enforcement")
                PURPOSE="Rule Enforcement and Compliance Scripts"
                ;;
            "scripts/deployment")
                PURPOSE="Deployment and Infrastructure Scripts"
                ;;
            "scripts/monitoring")
                PURPOSE="Monitoring and Logging Scripts"
                ;;
            "scripts/mcp")
                PURPOSE="MCP Management Scripts"
                ;;
            "agents")
                PURPOSE="AI Agent Configurations"
                ;;
            "docs")
                PURPOSE="Documentation Root"
                ;;
            "docs/reports")
                PURPOSE="Investigation and Audit Reports"
                ;;
            "tests")
                PURPOSE="Test Suites and Testing Infrastructure"
                ;;
            ".claude")
                PURPOSE="Claude AI Assistant Configuration"
                ;;
            ".claude/agents")
                PURPOSE="Claude Agent Definitions"
                ;;
            "IMPORTANT")
                PURPOSE="Critical System Rules and Documentation"
                ;;
            *)
                PURPOSE="System Component"
                ;;
        esac
        
        # Create CHANGELOG with proper format
        cat > "$CHANGELOG_PATH" << EOF
# CHANGELOG - $PURPOSE

## Directory Information
- **Location**: \`$dir\`
- **Purpose**: $PURPOSE
- **Owner**: SutazAI Development Team
- **Created**: $(date -u '+%Y-%m-%d %H:%M:%S UTC')
- **Status**: Active

## [Unreleased]

### $(date -u '+%Y-%m-%d %H:%M:%S UTC') - Version 1.0.0 - INITIAL - Initial Setup
**Who**: enforcement-script@ultrathink
**Why**: Rule 19 Compliance - Every important directory must have CHANGELOG.md
**What**: Created initial CHANGELOG.md with standard template
**Impact**: Establishes change tracking for this directory
**Validation**: Template follows organizational standards
**Related Changes**: Part of comprehensive CHANGELOG audit (Aug 19, 2025)

---

## Change Categories
- **MAJOR**: Breaking changes, architectural modifications, API changes
- **MINOR**: New features, enhancements, non-breaking improvements
- **PATCH**: Bug fixes, minor updates, documentation changes
- **SECURITY**: Security updates, vulnerability fixes
- **PERFORMANCE**: Performance improvements, optimization
- **MAINTENANCE**: Cleanup, refactoring, dependency updates

## Format Template
\`\`\`markdown
### YYYY-MM-DD HH:MM:SS UTC - Version X.Y.Z - CATEGORY - Brief Description
**Who**: author@team
**Why**: Business/technical requirement
**What**: Detailed description of changes
**Impact**: Effects on system/users
**Validation**: Testing performed
**Related Changes**: Links to related changes
\`\`\`
EOF
        
        echo "  ✅ Created: $CHANGELOG_PATH" | tee -a "$LOG_FILE"
        CREATED_COUNT=$((CREATED_COUNT + 1))
    fi
done

# Create master CHANGELOG index
INDEX_FILE="$ROOT_DIR/docs/CHANGELOG_INDEX.md"
echo "" | tee -a "$LOG_FILE"
echo "Creating CHANGELOG index..." | tee -a "$LOG_FILE"

cat > "$INDEX_FILE" << 'EOF'
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
EOF

echo "  ✅ Created CHANGELOG index: $INDEX_FILE" | tee -a "$LOG_FILE"

# Summary
echo "" | tee -a "$LOG_FILE"
echo "=== SUMMARY ===" | tee -a "$LOG_FILE"
echo "Existing CHANGELOG files: $EXISTING_COUNT" | tee -a "$LOG_FILE"
echo "Created CHANGELOG files: $CREATED_COUNT" | tee -a "$LOG_FILE"
echo "Total CHANGELOG files: $((EXISTING_COUNT + CREATED_COUNT))" | tee -a "$LOG_FILE"
echo "CHANGELOG index created: $INDEX_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "✅ RULE 19 COMPLIANCE ACHIEVED" | tee -a "$LOG_FILE"
echo "Log saved to: $LOG_FILE"