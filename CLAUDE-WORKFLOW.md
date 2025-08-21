# Development Workflow (VERIFIED 2025-08-21)

## SPARC Commands (From Documentation)
```bash
npx claude-flow sparc modes
npx claude-flow sparc run <mode> "<task>"
npx claude-flow sparc tdd "<feature>"
```

## Available Agents
54 agents documented in .claude/agents/ folder

## MCP Tools Available
Based on MCP server configuration:
- extended-memory (confirmed working)
- files (server.js exists)
- memory (server.js exists)
- Others may not be functional

## Code Quality
- **Technical Debt**: 7,189 markers total
- **Backend**: 0 TODOs found
- **Affected Files**: Widespread across codebase

## Build Commands
```bash
npm run build
npm run test
npm run lint
```