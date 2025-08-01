# Agent Access Troubleshooting Guide

## Current Status

✅ **Agents are properly configured** - All 38 agents have correct YAML frontmatter and system prompts
✅ **Agents appear in Claude UI** - You can see all agents in the agent selection interface
❌ **Task tool not available** - Cannot invoke agents programmatically without the Task tool

## The Issue

You're seeing the agents in the Claude UI but can't use them because:

1. **The Task tool is missing** - This tool is required to invoke agents programmatically
2. **MCP server connection issue** - The task-master-ai MCP server provides the Task tool but isn't properly connected
3. **Session limitation** - The current Claude session doesn't have the Task tool loaded

## Solutions

### Option 1: Restart Claude (Recommended)
```bash
# 1. Close Claude completely
# 2. Ensure MCP servers are configured in .mcp.json (✓ Already done)
# 3. Restart Claude
# 4. The Task tool should be available
```

### Option 2: Use Agents Through UI
- Click on an agent in the Claude UI agent selector
- This will switch to that agent's context
- You can then interact with that specialized agent directly

### Option 3: Manual Agent Selection
When you can't use the Task tool, you can still request specific agents by saying:
- "I need the ai-agent-orchestrator to help coordinate this task"
- "Please use the deployment-automation-master agent for this deployment"
- "Can the testing-qa-validator agent review this code?"

## What the Task Tool Does

The Task tool allows Claude to:
```python
# Programmatically invoke agents
Task(
    subagent_type="ai-agent-orchestrator",
    description="Coordinate multi-agent workflow", 
    prompt="Design a system for autonomous agent collaboration"
)
```

Without it, agents must be selected manually through the UI.

## Verification Steps

1. Check if Task tool is available:
   - Try using: `Task()` in Claude
   - If it says "Unknown tool: Task", the MCP server isn't connected

2. Check MCP server status:
   - The task-master-ai server should be running
   - It requires Ollama (✓ Running on port 11434)

3. Check agent availability:
   - All 38 agents are in `/opt/sutazaiapp/.claude/agents/`
   - They all have proper YAML configuration

## Summary

Your agents are properly configured and visible in the UI. The issue is that the Task tool (provided by task-master-ai MCP server) isn't available in the current session. Restarting Claude should resolve this and allow programmatic agent invocation.