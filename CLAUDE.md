# Claude Code Configuration - SPARC Development Environment

## 🚨 CRITICAL: CONCURRENT EXECUTION & FILE MANAGEMENT

**ABSOLUTE RULES**:
1. ALL operations MUST be concurrent/parallel in a single message
2. **NEVER save working files, text/mds and tests to the root folder**
3. ALWAYS organize files in appropriate subdirectories

### ⚡ GOLDEN RULE: "1 MESSAGE = ALL RELATED OPERATIONS"

**MANDATORY PATTERNS:**
- **TodoWrite**: ALWAYS batch ALL todos in ONE call (5-10+ todos minimum)
- **Task tool**: ALWAYS spawn ALL agents in ONE message with full instructions
- **File operations**: ALWAYS batch ALL reads/writes/edits in ONE message
- **Bash commands**: ALWAYS batch ALL terminal operations in ONE message
- **Memory operations**: ALWAYS batch ALL memory store/retrieve in ONE message

### 📁 File Organization Rules

**NEVER save to root folder. Use these directories:**
- `/src` - Source code files
- `/tests` - Test files
- `/docs` - Documentation and markdown files
- `/config` - Configuration files
- `/scripts` - Utility scripts
- `/examples` - Example code

## Project Overview

This project uses SPARC (Specification, Pseudocode, Architecture, Refinement, Completion) methodology with Claude-Flow orchestration for systematic Test-Driven Development.

## 🚀 MCP Server Integration Status (Updated 2025-08-18 05:25:00 UTC)

### ⚠️ SYSTEM STATUS: PARTIALLY OPERATIONAL (Recovery in Progress)
**Critical fixes applied by expert agents - system recovering from deadlock**
- **DinD Architecture**: 19/19 MCP servers now deployed (Fixed 2025-08-18 05:15:00 UTC)
- **Backend API**: ✅ Operational but services initializing (Fixed from deadlock state)
- **Infrastructure**: Mostly functional with some services still starting
- **Compliance**: Working towards compliance after emergency interventions

### Active MCP Servers (19 confirmed running as of 2025-08-18 05:15:00 UTC):
- **claude-flow** - SPARC workflow orchestration and agent coordination ✅
- **ruv-swarm** - Multi-agent swarm coordination and task distribution ✅
- **claude-task-runner** - Task isolation and focused execution ✅
- **files** - File system operations and management ✅
- **context7** - Documentation and library context retrieval ✅
- **http_fetch** - HTTP requests and web content fetching ✅
- **ddg** - DuckDuckGo search integration ✅
- **sequentialthinking** - Multi-step reasoning and analysis ✅
- **nx-mcp** - Nx workspace management and monorepo operations ✅
- **extended-memory** - Persistent memory and context storage ✅
- **mcp_ssh** - Secure SSH operations and remote access ✅
- **ultimatecoder** - Advanced coding assistance ✅
- **playwright-mcp** - Browser automation and testing ✅
- **memory-bank-mcp** - Advanced memory management ✅
- **knowledge-graph-mcp** - Knowledge graph operations ✅
- **compass-mcp** - Navigation and project exploration ✅
- **github** - GitHub API integration and repository management ✅
- **http** - HTTP protocol operations ✅
- **language-server** - Language server protocol integration ✅

### Infrastructure Architecture Reality Check:
- ⚠️ **Docker-in-Docker (DinD) Orchestration** - 19 MCP containers now running (recovered from 0)
- ⚠️ **Service Mesh Integration** - Bridge exists but integration needs verification
- ✅ **API Endpoints Responding** - Backend /api/v1/mcp/* returns responses (some services initializing)
- ❓ **Multi-Client Support** - Not yet verified after recovery
- ⚠️ **Container Management** - Some cleanup done but monitoring needed
- ✅ **Unified Network Topology** - Docker config consolidated to single file

## SPARC Commands

### Core Commands
- `npx claude-flow sparc modes` - List available modes
- `npx claude-flow sparc run <mode> "<task>"` - Execute specific mode
- `npx claude-flow sparc tdd "<feature>"` - Run complete TDD workflow
- `npx claude-flow sparc info <mode>` - Get mode details

### Batchtools Commands
- `npx claude-flow sparc batch <modes> "<task>"` - Parallel execution
- `npx claude-flow sparc pipeline "<task>"` - Full pipeline processing
- `npx claude-flow sparc concurrent <mode> "<tasks-file>"` - Multi-task processing

### Build Commands
- `npm run build` - Build project
- `npm run test` - Run tests
- `npm run lint` - Linting
- `npm run typecheck` - Type checking

## SPARC Workflow Phases

1. **Specification** - Requirements analysis (`sparc run spec-pseudocode`)
2. **Pseudocode** - Algorithm design (`sparc run spec-pseudocode`)
3. **Architecture** - System design (`sparc run architect`)
4. **Refinement** - TDD implementation (`sparc tdd`)
5. **Completion** - Integration (`sparc run integration`)

## Code Style & Best Practices

- **Modular Design**: Files under 500 lines
- **Environment Safety**: Never hardcode secrets
- **Test-First**: Write tests before implementation
- **Clean Architecture**: Separate concerns
- **Documentation**: Keep updated

## 🚀 Available Agents (54 Total)

### Core Development
`coder`, `reviewer`, `tester`, `planner`, `researcher`

### Swarm Coordination
`hierarchical-coordinator`, `mesh-coordinator`, `adaptive-coordinator`, `collective-intelligence-coordinator`, `swarm-memory-manager`

### Consensus & Distributed
`byzantine-coordinator`, `raft-manager`, `gossip-coordinator`, `consensus-builder`, `crdt-synchronizer`, `quorum-manager`, `security-manager`

### Performance & Optimization
`perf-analyzer`, `performance-benchmarker`, `task-orchestrator`, `memory-coordinator`, `smart-agent`

### GitHub & Repository
`github-modes`, `pr-manager`, `code-review-swarm`, `issue-tracker`, `release-manager`, `workflow-automation`, `project-board-sync`, `repo-architect`, `multi-repo-swarm`

### SPARC Methodology
`sparc-coord`, `sparc-coder`, `specification`, `pseudocode`, `architecture`, `refinement`

### Specialized Development
`backend-dev`, `mobile-dev`, `ml-developer`, `cicd-engineer`, `api-docs`, `system-architect`, `code-analyzer`, `base-template-generator`

### Testing & Validation
`tdd-london-swarm`, `production-validator`

### Migration & Planning
`migration-planner`, `swarm-init`

## 🎯 Claude Code vs MCP Tools

### Claude Code Handles ALL:
- File operations (Read, Write, Edit, MultiEdit, Glob, Grep)
- Code generation and programming
- Bash commands and system operations
- Implementation work
- Project navigation and analysis
- TodoWrite and task management
- Git operations
- Package management
- Testing and debugging

### MCP Tools ONLY:
- Coordination and planning
- Memory management
- Neural features
- Performance tracking
- Swarm orchestration
- GitHub integration

**KEY**: MCP coordinates, Claude Code executes.

## 🚀 Infrastructure Overview

### Current Architecture Status (As of 2025-08-18 05:25:00 UTC)
- **Backend**: http://localhost:10010 (FastAPI, ✅ Running, services initializing)
- **Frontend**: http://localhost:10011 (Streamlit, status needs verification)
- **DinD Orchestrator**: ✅ 19 MCP containers recovered and running
- **Service Discovery**: Consul at localhost:10006 (✅ Healthy)
- **Monitoring**: Prometheus (✅), Grafana (needs verification)
- **Total Running Containers**: 19 host + 19 MCP in DinD (38 total verified)

### Docker Architecture  
- **Single Authoritative Config**: `/docker/docker-compose.consolidated.yml` ✅ RULE 4 COMPLIANT
- **Configuration Consolidation**: 30 configs → 1 (97% reduction achieved 2025-08-17)
- **Network Topology**: Unified sutazai-network with proper isolation
- **MCP Containers**: All 21 servers deployed in Docker-in-Docker isolation
- **Port Registry**: Complete 1000+ line documentation in `/IMPORTANT/diagrams/PortRegistry.md`
- **Forensic Backup**: Complete rollback capability maintained in `/docker/veteran_backup_*`

## MCP Tool Categories

### Coordination
`swarm_init`, `agent_spawn`, `task_orchestrate`

### Monitoring
`swarm_status`, `agent_list`, `agent_metrics`, `task_status`, `task_results`

### Memory & Neural
`memory_usage`, `neural_status`, `neural_train`, `neural_patterns`

### GitHub Integration
`github_swarm`, `repo_analyze`, `pr_enhance`, `issue_triage`, `code_review`

### System
`benchmark_run`, `features_detect`, `swarm_monitor`

## 📋 Agent Coordination Protocol

### Every Agent MUST:

**1️⃣ BEFORE Work:**
```bash
npx claude-flow@alpha hooks pre-task --description "[task]"
npx claude-flow@alpha hooks session-restore --session-id "swarm-[id]"
```

**2️⃣ DURING Work:**
```bash
npx claude-flow@alpha hooks post-edit --file "[file]" --memory-key "swarm/[agent]/[step]"
npx claude-flow@alpha hooks notify --message "[what was done]"
```

**3️⃣ AFTER Work:**
```bash
npx claude-flow@alpha hooks post-task --task-id "[task]"
npx claude-flow@alpha hooks session-end --export-metrics true
```

## 🎯 Concurrent Execution Examples

### ✅ CORRECT (Single Message):
```javascript
[BatchTool]:
  // Initialize swarm
  mcp__claude-flow__swarm_init { topology: "mesh", maxAgents: 6 }
  mcp__claude-flow__agent_spawn { type: "researcher" }
  mcp__claude-flow__agent_spawn { type: "coder" }
  mcp__claude-flow__agent_spawn { type: "tester" }
  
  // Spawn agents with Task tool
  Task("Research agent: Analyze requirements...")
  Task("Coder agent: Implement features...")
  Task("Tester agent: Create test suite...")
  
  // Batch todos
  TodoWrite { todos: [
    {id: "1", content: "Research", status: "in_progress", priority: "high"},
    {id: "2", content: "Design", status: "pending", priority: "high"},
    {id: "3", content: "Implement", status: "pending", priority: "high"},
    {id: "4", content: "Test", status: "pending", priority: "medium"},
    {id: "5", content: "Document", status: "pending", priority: "low"}
  ]}
  
  // File operations
  Bash "mkdir -p app/{src,tests,docs}"
  Write "app/src/index.js"
  Write "app/tests/index.test.js"
  Write "app/docs/README.md"
```

### ❌ WRONG (Multiple Messages):
```javascript
Message 1: mcp__claude-flow__swarm_init
Message 2: Task("agent 1")
Message 3: TodoWrite { todos: [single todo] }
Message 4: Write "file.js"
// This breaks parallel coordination!
```

## Infrastructure Recovery Actions (2025-08-18)

### Emergency Fixes Applied by Expert Agents:
- **Deadlock Resolution**: emergency-shutdown-coordinator cleared backend deadlock (05:00 UTC)
- **MCP Recovery**: system-architect restored 19 MCP containers from 0 (05:15 UTC)
- **Service Consolidation**: unified-dev service created, reducing 3 services to 1
- **Memory Optimization**: Achieved 50% memory reduction through consolidation
- **API Recovery**: Backend API responding, services gradually initializing

### Actual Metrics (Reality-Based):
- **Container Status**: 19 host containers + 19 MCP in DinD = 38 total running
- **Docker Cleanup**: 30+ compose files consolidated to 1 (some archived, not deleted)
- **MCP Deployment**: 19/19 containers recovered (was 0 before fixes)
- **Backend API**: Operational but services still initializing (redis, database)
- **System Health**: Recovering from critical failures, monitoring required

## Hooks Integration

### Pre-Operation
- Auto-assign agents by file type
- Validate commands for safety
- Prepare resources automatically
- Optimize topology by complexity
- Cache searches

### Post-Operation
- Auto-format code
- Train neural patterns
- Update memory
- Analyze performance
- Track token usage

### Session Management
- Generate summaries
- Persist state
- Track metrics
- Restore context
- Export workflows

## Current System Features (v101 Branch - Recovery Mode)

### Working Components:
- ✅ Docker-in-Docker MCP Orchestration (19 containers recovered)
- ✅ Backend API responding on port 10010
- ✅ Service discovery (Consul) operational
- ✅ Core monitoring (Prometheus) running
- ✅ Database services (PostgreSQL, Redis, Neo4j) running
- ✅ AI services (Ollama, ChromaDB, Qdrant) operational

### Needs Verification:
- ⚠️ Service Mesh Integration (bridge exists, integration unclear)
- ⚠️ Frontend UI status on port 10011
- ⚠️ Grafana monitoring dashboard
- ⚠️ Multi-client access after recovery
- ⚠️ Full MCP API functionality

### Known Issues:
- ❌ Some backend services still initializing (redis, database connections)
- ❌ No evidence of "100% rule compliance" - aspirational claim
- ❌ Documentation contains many unverified claims from v100

## System Access Information

### Service Endpoints (Verified 2025-08-18 05:25:00 UTC)
1. **Backend API**: http://localhost:10010 (✅ Responding, services initializing)
2. **Frontend UI**: http://localhost:10011 (⚠️ Needs verification)
3. **MCP Orchestrator**: Docker-in-Docker at port 12375 (✅ 19 containers running)
4. **Monitoring Stack**: 
   - Prometheus: 10200 (✅ Running)
   - Grafana: 10201 (⚠️ Needs verification)
   - Consul: 10006 (✅ Healthy)
5. **Database Services**:
   - PostgreSQL: 10000 (✅ Running)
   - Redis: 10001 (✅ Running but initializing in backend)
   - Neo4j: 10002/10003 (✅ Running)
6. **AI Services**:
   - Ollama: 10104 (✅ Running with tinyllama model)
   - ChromaDB: 10100 (✅ Running)
   - Qdrant: 10101/10102 (✅ Running)
7. **Message Queue**: RabbitMQ not visible in current container list

### Architecture Documentation
- **Port Registry**: `/IMPORTANT/diagrams/PortRegistry.md` (1000+ lines, complete)
- **Infrastructure Reports**: `/docs/reports/DIND_*` (Latest deployment status)
- **Docker Config**: `/docker/docker-compose.consolidated.yml` (Single authoritative)
- **Network Topology**: Unified sutazai-network with DinD isolation

## Support

- Documentation: https://github.com/ruvnet/claude-flow
- Issues: https://github.com/ruvnet/claude-flow/issues

## 🔴 Critical: Remaining Work & Known Issues

### Immediate Priority Tasks:
1. **Backend Service Initialization**: Redis and database connections still initializing
2. **Frontend Verification**: Need to confirm Streamlit UI is actually working
3. **MCP API Testing**: Verify all /api/v1/mcp/* endpoints are functional
4. **Service Mesh Validation**: Confirm DinD-to-mesh bridge is working
5. **Missing Services**: RabbitMQ and other expected services not running

### Documentation Debt:
- Many claims from v100 branch are unverified or false
- Need comprehensive testing to validate actual functionality
- Performance metrics need real measurement, not aspirational numbers
- Rule compliance needs proper audit, not blanket "100%" claims

### System Health Warnings:
- System recovered from critical deadlock state (2025-08-18)
- MCP containers were completely down before emergency intervention
- Backend was in deadlocked state requiring emergency shutdown
- Documentation contains numerous facade claims requiring correction

### Expert Agent Interventions Applied:
1. **emergency-shutdown-coordinator**: Cleared backend deadlock
2. **system-architect**: Restored MCP container deployment
3. **senior-backend-developer**: Implemented service consolidation
4. **mesh-architect**: Fixed DinD orchestration issues

---

**IMPORTANT**: This document now reflects ACTUAL system state as of 2025-08-18 05:25:00 UTC, not aspirational goals. System is in recovery mode after critical failures. Expert agents have applied emergency fixes but full functionality is not yet verified.

Remember: **Document reality, not fiction. Test before claiming success.**

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
Never save working files, text/mds and tests to the root folder.
