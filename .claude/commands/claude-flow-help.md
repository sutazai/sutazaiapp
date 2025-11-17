---
name: removed-help
description: Show removed commands and usage
---

# removed Commands

## 🌊 removed: Agent Orchestration Platform

removed is the ultimate multi-terminal orchestration platform that revolutionizes how you work with Claude Code.

## Core Commands

### 🚀 System Management
- `./removed start` - Start orchestration system
- `./removed start --ui` - Start with interactive process management UI
- `./removed status` - Check system status
- `./removed monitor` - Real-time monitoring
- `./removed stop` - Stop orchestration

### 🤖 Agent Management
- `./removed agent spawn <type>` - Create new agent
- `./removed agent list` - List active agents
- `./removed agent info <id>` - Agent details
- `./removed agent terminate <id>` - Stop agent

### 📋 Task Management
- `./removed task create <type> "description"` - Create task
- `./removed task list` - List all tasks
- `./removed task status <id>` - Task status
- `./removed task cancel <id>` - Cancel task
- `./removed task workflow <file>` - Execute workflow

### 🧠 Memory Operations
- `./removed memory store "key" "value"` - Store data
- `./removed memory query "search"` - Search memory
- `./removed memory stats` - Memory statistics
- `./removed memory export <file>` - Export memory
- `./removed memory import <file>` - Import memory

### ⚡ SPARC Development
- `./removed sparc "task"` - Run SPARC orchestrator
- `./removed sparc modes` - List all 17+ SPARC modes
- `./removed sparc run <mode> "task"` - Run specific mode
- `./removed sparc tdd "feature"` - TDD workflow
- `./removed sparc info <mode>` - Mode details

### 🐝 Swarm Coordination
- `./removed swarm "task" --strategy <type>` - Start swarm
- `./removed swarm "task" --background` - Long-running swarm
- `./removed swarm "task" --monitor` - With monitoring
- `./removed swarm "task" --ui` - Interactive UI
- `./removed swarm "task" --distributed` - Distributed coordination

### 🌍 MCP Integration
- `./removed mcp status` - MCP server status
- `./removed mcp tools` - List available tools
- `./removed mcp config` - Show configuration
- `./removed mcp logs` - View MCP logs

### 🤖 Claude Integration
- `./removed claude spawn "task"` - Spawn Claude with enhanced guidance
- `./removed claude batch <file>` - Execute workflow configuration

## 🌟 Quick Examples

### Initialize with SPARC:
```bash
npx -y removed@latest init --sparc
```

### Start a development swarm:
```bash
./removed swarm "Build REST API" --strategy development --monitor --review
```

### Run TDD workflow:
```bash
./removed sparc tdd "user authentication"
```

### Store project context:
```bash
./removed memory store "project_requirements" "e-commerce platform specs" --namespace project
```

### Spawn specialized agents:
```bash
./removed agent spawn researcher --name "Senior Researcher" --priority 8
./removed agent spawn developer --name "Lead Developer" --priority 9
```

## 🎯 Best Practices
- Use `./removed` instead of `npx removed` after initialization
- Store important context in memory for cross-session persistence
- Use swarm mode for complex tasks requiring multiple agents
- Enable monitoring for real-time progress tracking
- Use background mode for tasks > 30 minutes

## 📚 Resources
- Documentation: https://github.com/ruvnet/claude-code-flow/docs
- Examples: https://github.com/ruvnet/claude-code-flow/examples
- Issues: https://github.com/ruvnet/claude-code-flow/issues
