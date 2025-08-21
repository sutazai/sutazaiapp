# Development Workflow

## SPARC Commands
```bash
npx claude-flow sparc modes
npx claude-flow sparc run <mode> "<task>"
npx claude-flow sparc tdd "<feature>"
npx claude-flow sparc batch <modes> "<task>"
```

## Phases
1. **Specification** - Requirements
2. **Pseudocode** - Algorithm
3. **Architecture** - Design
4. **Refinement** - TDD
5. **Completion** - Integration

## Available Agents (54)
- **Core**: coder, reviewer, tester, planner, researcher
- **Swarm**: hierarchical/mesh/adaptive coordinators
- **GitHub**: pr-manager, code-review-swarm, issue-tracker
- **Specialized**: backend-dev, mobile-dev, ml-developer

## MCP Tools
- **Coordination**: swarm_init, agent_spawn, task_orchestrate
- **Monitoring**: swarm_status, agent_metrics, task_results
- **Memory**: memory_usage, neural_status, neural_train