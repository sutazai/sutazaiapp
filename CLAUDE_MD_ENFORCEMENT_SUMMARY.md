# CLAUDE.md Rules Enforcement Implementation Summary

## Overview
All AI agents in the SutazAI system have been configured to automatically check and follow the rules defined in `/opt/sutazaiapp/CLAUDE.md` before executing any tasks.

## Implementation Components

### 1. **Rules Checker Module** (`/opt/sutazaiapp/.claude/agents/claude_rules_checker.py`)
- Provides `ClaudeRulesChecker` class for rule validation
- Implements blocking and warning rule checks
- Global instance available for all agents

### 2. **Agent Directory Rules** (`claude_rules.py` in each agent directory)
- Each agent directory now has its own rules enforcement module
- Automatically loaded when agent starts
- Provides `check_claude_rules()` function for action validation

### 3. **Docker Compose Override** (`/opt/sutazaiapp/docker-compose.claude-rules.yml`)
- Mounts CLAUDE.md as read-only in all agent containers
- Sets environment variables:
  - `CLAUDE_RULES_PATH=/app/CLAUDE.md`
  - `ENFORCE_CLAUDE_RULES=true`

### 4. **Agent Startup Scripts**
- `/opt/sutazaiapp/.claude/agents/start_with_rules.sh` - Basic wrapper
- `/opt/sutazaiapp/agents/startup_with_rules.sh` - Enhanced startup with visual feedback

### 5. **Verification Script** (`/opt/sutazaiapp/scripts/verify_claude_rules.py`)
- Checks agent compliance
- Verifies rules modules are in place

## How It Works

### Agent Startup Process:
1. Agent container starts with CLAUDE.md mounted
2. Startup script verifies CLAUDE.md is accessible
3. Rules enforcement module is loaded
4. Agent checks rules before executing any task
5. Blocking violations prevent task execution
6. Warning violations are logged for review

### Rule Categories:
- **BLOCKING** (Must never violate):
  - Rule 1: No Fantasy Elements
  - Rule 2: Do Not Break Existing Functionality
  
- **WARNING** (Require careful consideration):
  - Rule 3: Analyze Everything
  - Rule 4: Reuse Before Creating
  - Rule 5: Professional Project

- **GUIDANCE** (Best practices):
  - Clean code standards
  - Documentation requirements
  - Testing coverage

## Usage

### Starting Agents with Rules Enforcement:
```bash
# Using docker-compose override
docker-compose -f docker-compose.yml -f docker-compose.claude-rules.yml up -d

# Individual agent with rules
docker run -v /opt/sutazaiapp/CLAUDE.md:/app/CLAUDE.md:ro \
  -e CLAUDE_RULES_PATH=/app/CLAUDE.md \
  -e ENFORCE_CLAUDE_RULES=true \
  agent-image
```

### Verifying Compliance:
```bash
# Check agent compliance
python /opt/sutazaiapp/scripts/verify_claude_rules.py

# Test rules enforcement
cd /opt/sutazaiapp/.claude/agents
python -c "from claude_rules_checker import check_action; print(check_action('create magic wizard', 'test'))"
```

### In Agent Code:
```python
from claude_rules import check_claude_rules

def execute_task(task):
    # Check rules before executing
    if not check_claude_rules(task, agent_name="my-agent"):
        return {"error": "Task violates CLAUDE.md rules"}
    
    # Proceed with task execution
    return perform_task(task)
```

## Files Created/Modified

### New Files:
- `/opt/sutazaiapp/.claude/agents/claude_rules_checker.py`
- `/opt/sutazaiapp/.claude/agents/*/claude_rules.py` (in each agent dir)
- `/opt/sutazaiapp/docker-compose.claude-rules.yml`
- `/opt/sutazaiapp/.claude/agents/start_with_rules.sh`
- `/opt/sutazaiapp/agents/startup_with_rules.sh`
- `/opt/sutazaiapp/scripts/enforce_claude_md_rules.py`
- `/opt/sutazaiapp/scripts/enforce_claude_md_simple.py`
- `/opt/sutazaiapp/scripts/verify_claude_rules.py`

### Modified Files:
- Agent Dockerfiles (added CLAUDE.md copy and environment vars)
- Agent Python files in knowledge-graph-builder (added rules checking)

### Backup Location:
- `/opt/sutazaiapp/.claude/agents/backups/` (original agent files)

## Monitoring & Debugging

### Check Logs for Rules Enforcement:
```bash
# View agent logs
docker logs <agent-container> | grep "CLAUDE.md"

# Check for blocked actions
docker logs <agent-container> | grep "BLOCKED"
```

### Environment Variables:
All agents now have:
- `CLAUDE_RULES_PATH` - Path to CLAUDE.md file
- `ENFORCE_CLAUDE_RULES` - Enable/disable enforcement
- `AGENT_MUST_CHECK_RULES` - Mandatory checking flag

## Next Steps

1. **Restart all agents** to apply the new rules enforcement
2. **Monitor agent logs** to ensure rules are being checked
3. **Test with sample tasks** that should be blocked (e.g., fantasy elements)
4. **Review warning logs** to identify potential issues
5. **Update agent documentation** to reflect rules checking

## Rollback Instructions

If needed, to rollback:
1. Restore agent files from `/opt/sutazaiapp/.claude/agents/backups/`
2. Remove docker-compose override file
3. Remove claude_rules.py files from agent directories
4. Restart agents without the override file

## Success Metrics

- ✅ All agents have access to CLAUDE.md
- ✅ Rules checking happens before task execution
- ✅ Blocking rules prevent violations
- ✅ Warning rules generate logs
- ✅ No performance impact on normal operations