# Claude Code Optimization Guide

## Token Usage Optimization

### Subagent Management
- The warning "Large cumulative agent descriptions will impact performance (~21.7k tokens)" indicates too many subagents are loaded
- Use `/agents` command to manage subagent access
- Limit parallel subagent execution to 3-4 maximum

### Best Practices for Token Efficiency

1. **Selective Subagent Usage**
   - Only use subagents for complex tasks requiring specialized expertise
   - For simple tasks, handle directly without subagents
   - Be specific: use `python-pro` instead of `fullstack-developer` when appropriate

2. **Context Management**
   - Use `/clear` between major task switches
   - Keep CLAUDE.md files concise (<500 lines)
   - Avoid loading unnecessary documentation

3. **Parallel Processing Control**
   - Limit concurrent subagents to prevent token multiplication
   - Each parallel subagent multiplies token usage
   - Monitor usage with awareness of rate limits

4. **MCP Tool Optimization**
   - The `.mcpignore` file prevents traversing large directories
   - Avoid operations that generate massive outputs
   - Be specific with file paths and search patterns

## Current Optimizations Applied

1. **Created .mcpignore** - Prevents MCP tools from traversing:
   - node_modules/ directories (3,059 found!)
   - venv/ and .venv/ directories
   - Build outputs and caches
   - Large binary files

2. **System Cleanup Performed**
   - Removed problematic Kong container
   - Killed runaway processes
   - Optimized memory usage from 72% to 44%

## Quick Commands

```bash
# Check system health
docker stats --no-stream
free -h
ps aux | grep -c claude

# Clean up if needed
pkill -f "code-index-mcp"
docker system prune -a --volumes

# Monitor token-heavy processes
ps aux --sort=-%mem | head -15
```

## When to Use Subagents vs Direct Implementation

### Use Subagents For:
- Complex architectural analysis
- Multi-file refactoring
- Security audits
- Performance optimization
- System design

### Handle Directly For:
- Simple file edits
- Configuration changes
- Documentation updates
- Basic debugging
- Package installation

## Token Budget Guidelines

- Main context: Keep under 50k tokens
- Subagent descriptions: 15k token warning threshold
- Per-task budget: 10-20k tokens typical
- Parallel execution: Multiply by number of concurrent agents

Remember: Efficiency > Comprehensiveness. Focus on specific, targeted solutions rather than loading everything at once.