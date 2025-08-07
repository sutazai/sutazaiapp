# CLAUDE.md - SutazAI System Truth Document

This file provides the SINGLE SOURCE OF TRUTH for Claude Code (claude.ai/code) when working with this repository.

**Last Modified:** December 19, 2024  
**Document Sections:** System Reality Check + Comprehensive Codebase Rules

## ⚠️ CRITICAL REALITY CHECK ⚠️

**Last System Verified:** August 6, 2025  
**Verification Method:** Direct container inspection and endpoint testing  
**Major Cleanup Completed:** v56 cleanup operation removed 200+ fantasy documentation files  
**Rules Added:** December 19, 2024 - 19 comprehensive codebase standards

### The Truth About This System
- **59 services defined** in docker-compose.yml
- **28 containers actually running** 
- **7 agent services running** (all Flask stubs with health endpoints only)
- **Model Reality**: TinyLlama 637MB loaded (NOT gpt-oss as docs claim)
- **No AI Logic**: Agents return hardcoded JSON responses, no actual AI processing

## 🔴 What ACTUALLY Works (Verified by Testing)

### Core Infrastructure (All Verified Healthy)
| Service | Port | Status | Reality Check |
|---------|------|--------|---------------|
| PostgreSQL | 10000 | ✅ HEALTHY | Database has 14 tables (users, agents, tasks, etc.) |
| Redis | 10001 | ✅ HEALTHY | Cache layer functional |
| Neo4j | 10002/10003 | ✅ HEALTHY | Graph database available |
| Ollama | 10104 | ✅ HEALTHY | TinyLlama model loaded and working |

### Application Layer
| Service | Port | Status | Notes |
|---------|------|--------|-------|
| Backend API | 10010 | ✅ HEALTHY | FastAPI v17.0.0 - Ollama connected, all services operational |
| Frontend | 10011 | ⚠️ STARTING | Streamlit UI - takes time to initialize |

### Service Mesh (Actually Running)
| Service | Port | Status | Usage |
|---------|------|--------|-------|
| Kong Gateway | 10005/8001 | ✅ RUNNING | API gateway (no routes configured) |
| Consul | 10006 | ✅ RUNNING | Service discovery (minimal usage) |
| RabbitMQ | 10007/10008 | ✅ RUNNING | Message queue (not actively used) |

### Vector Databases
| Service | Port | Status | Integration |
|---------|------|--------|-------------|
| Qdrant | 10101/10102 | ✅ HEALTHY | Not integrated with app |
| FAISS | 10103 | ✅ HEALTHY | Not integrated with app |
| ChromaDB | 10100 | ⚠️ STARTING | Connection issues |

### Monitoring Stack (All Operational)
| Service | Port | Purpose |
|---------|------|---------|
| Prometheus | 10200 | Metrics collection |
| Grafana | 10201 | Visualization dashboards |
| Loki | 10202 | Log aggregation |
| AlertManager | 10203 | Alert routing |
| Node Exporter | 10220 | System metrics |
| cAdvisor | 10221 | Container metrics |

## 🟡 What Are STUBS (Return Fake Responses)

### Running Agent Services (Flask Apps with Health Endpoints Only)
| Agent | Port | Actual Functionality |
|-------|------|---------------------|
| AI Agent Orchestrator | 8589 | Returns `{"status": "healthy"}` |
| Multi-Agent Coordinator | 8587 | Basic coordination stub |
| Resource Arbitration | 8588 | Resource allocation stub |
| Task Assignment | 8551 | Task routing stub |
| Hardware Optimizer | 8002 | Hardware monitoring stub |
| Ollama Integration | 11015 | Ollama wrapper (may work) |
| AI Metrics Exporter | 11063 | Metrics collection (UNHEALTHY) |

**Reality**: These agents have `/health` endpoints but `/process` endpoints return hardcoded JSON regardless of input.

## ❌ What is PURE FANTASY (Doesn't Exist)

### Never Deployed Services
- HashiCorp Vault (secrets management)
- Jaeger (distributed tracing)  
- Elasticsearch (search/analytics)
- Kubernetes orchestration
- Terraform infrastructure
- 60+ additional AI agents mentioned in docs

### Fictional Capabilities
- Quantum computing modules (deleted `/backend/quantum_architecture/`)
- AGI/ASI orchestration (complete fiction)
- Complex agent communication (agents don't talk to each other)
- Inter-agent message passing (not implemented)
- Advanced ML pipelines (not present)
- Self-improvement capabilities (stub only)

## 📍 Accurate Port Registry

```yaml
# Core Services (Running)
10000: PostgreSQL database
10001: Redis cache
10002: Neo4j browser interface
10003: Neo4j bolt protocol
10005: Kong API Gateway
10006: Consul service discovery
10007: RabbitMQ AMQP
10008: RabbitMQ management UI
10010: Backend FastAPI
10011: Frontend Streamlit
10104: Ollama LLM server

# Vector Databases (Running)
10100: ChromaDB (connection issues)
10101: Qdrant HTTP
10102: Qdrant gRPC
10103: FAISS vector service

# Monitoring (Running)
10200: Prometheus metrics
10201: Grafana dashboards
10202: Loki logs
10203: AlertManager
10220: Node Exporter
10221: cAdvisor
10229: Blackbox Exporter

# Agent Services (Stubs)
8002: Hardware Resource Optimizer
8551: Task Assignment Coordinator
8587: Multi-Agent Coordinator
8588: Resource Arbitration Agent
8589: AI Agent Orchestrator
11015: Ollama Integration Specialist
11063: AI Metrics Exporter
```

## 🛠️ Working Commands (Tested & Verified)

### System Management
```bash
# Check what's ACTUALLY running
docker ps --format "table {{.Names}}\t{{.Ports}}\t{{.Status}}"

# Start system (only method that works)
docker network create sutazai-network 2>/dev/null
docker-compose up -d

# View logs for debugging
docker-compose logs -f backend
docker-compose logs -f ollama

# Restart a service
docker-compose restart backend

# Stop everything
docker-compose down
```

### Testing Endpoints
```bash
# Backend API (returns degraded status)
curl http://127.0.0.1:10010/health

# Frontend UI
open http://localhost:10011

# Ollama - check loaded model (shows tinyllama, not gpt-oss)
curl http://127.0.0.1:10104/api/tags

# Test text generation with ACTUAL model
curl http://127.0.0.1:10104/api/generate \
  -d '{"model": "tinyllama", "prompt": "Hello world"}'

# Agent health check (stub response)
curl http://127.0.0.1:8589/health

# Monitoring
open http://localhost:10200  # Prometheus
open http://localhost:10201  # Grafana (admin/admin)
```

### Database Access
```bash
# PostgreSQL (no tables exist yet!)
docker exec -it sutazai-postgres psql -U sutazai -d sutazai
# \dt to list tables (will be empty)

# Redis
docker exec -it sutazai-redis redis-cli

# Neo4j browser
open http://localhost:10002
```

## ⚠️ Common Documentation LIES to Ignore

### Lie #1: "69 Agents Deployed"
**Truth**: 7 Flask stubs with health endpoints

### Lie #2: "GPT-OSS Migration Complete"  
**Truth**: TinyLlama is loaded, no gpt-oss model present

### Lie #3: "Complex Service Mesh with Advanced Routing"
**Truth**: Kong/Consul/RabbitMQ running but not configured or integrated

### Lie #4: "Quantum Computing Capabilities"
**Truth**: Complete fiction, code was deleted

### Lie #5: "Self-Improving AI System"
**Truth**: Hardcoded stub responses

### Lie #6: "Production Ready"
**Truth**: No database tables, agents are stubs, basic PoC at best

## 📂 Code Structure Reality

### Working Code Locations
```
/backend/app/          # FastAPI application (partially implemented)
  ├── main.py         # Entry point with feature flags
  ├── api/            # API endpoints (many stubs)
  └── core/           # Core utilities

/frontend/            # Streamlit UI (basic implementation)

/agents/*/app.py     # Flask stub applications
  └── All return:    {"status": "healthy", "result": "processed"}
```

### Post-Cleanup Clean Locations (v56)
```
/backend/app/            # FastAPI application (partially implemented)
  ├── main.py           # Entry point with feature flags
  ├── api/              # API endpoints (many stubs)
  └── core/             # Core utilities

/frontend/              # Streamlit UI (basic implementation)
/agents/core/           # Consolidated agent base classes
/config/                # Clean configuration files
/docker/                # Service container definitions
/scripts/               # Utility and deployment scripts
/tests/                 # Test suite (updated to test real functionality)
```

### Cleaned Up (Removed in v56)
```
❌ /IMPORTANT/*.md       # Fantasy docs removed
❌ /archive/             # Backup directories cleaned
❌ Root-level *_test.py  # Analysis scripts deleted
❌ Root-level *_audit.py # Compliance files removed
❌ Fantasy agent dirs    # Non-functional agent services
❌ Duplicate base agents # Multiple BaseAgent implementations
```

## 🚀 Quick Start (What Actually Works)

### 1. Start the System
```bash
# Ensure network exists
docker network create sutazai-network 2>/dev/null

# Start all services
docker-compose up -d

# Wait for initialization
sleep 30

# Check status
docker-compose ps
```

### 2. Verify Core Services
```bash
# Backend should return degraded (Ollama disconnected)
curl http://127.0.0.1:10010/health | jq

# Ollama should show tinyllama model
curl http://127.0.0.1:10104/api/tags | jq
```

### 3. Test Basic Functionality
```bash
# Generate text with TinyLlama
curl -X POST http://127.0.0.1:10104/api/generate \
  -d '{"model": "tinyllama", "prompt": "What is Docker?"}' | jq

# Access frontend
open http://localhost:10011

# View monitoring
open http://localhost:10201  # Grafana dashboards
```

## 🔧 Common Issues & REAL Solutions

### Backend shows "degraded" status
**Cause**: Ollama connection issue  
**Solution**: Backend expects gpt-oss but only tinyllama is loaded
```bash
# Either load gpt-oss model:
docker exec sutazai-ollama ollama pull gpt-oss

# Or update backend to use tinyllama:
# Edit backend config to use "tinyllama" instead of "gpt-oss"
```

### Agents don't do anything intelligent
**Cause**: They're stubs  
**Solution**: Implement actual logic in `/agents/*/app.py` or accept they're placeholders

### PostgreSQL has no tables
**Cause**: Migrations never run  
**Solution**: 
```bash
# Create tables manually or run migrations if they exist
docker exec -it sutazai-backend python -m app.db.init_db
```

### ChromaDB keeps restarting
**Cause**: Connection/initialization issues  
**Solution**: Check logs and ensure proper config
```bash
docker-compose logs -f chromadb
```

## 📋 What This System Can ACTUALLY Do

### ✅ Can Do:
- Local LLM text generation with TinyLlama (637MB model)
- Container monitoring via Prometheus/Grafana
- Basic web UI via Streamlit
- Store data in PostgreSQL/Redis/Neo4j (once tables created)
- Vector similarity search (once integrated)

### ❌ Cannot Do:
- Complex AI agent orchestration (stubs only)
- Distributed AI processing (no real implementation)
- Advanced NLP pipelines (not present)
- Production workloads (too many stubs)
- Any quantum computing (pure fiction)
- AGI/ASI features (marketing fiction)
- Inter-agent communication (not implemented)

## 🎯 Realistic Next Steps

Instead of chasing fantasy features:

1. **Fix Model Mismatch**: Either load gpt-oss or update code to use tinyllama
2. **Create Database Schema**: PostgreSQL has no tables
3. **Implement One Real Agent**: Pick one agent and add actual logic
4. **Fix ChromaDB**: Resolve connection issues
5. **Configure Service Mesh**: Kong has no routes defined
6. **Consolidate Requirements**: Still need to merge 75+ files into 3  
7. **Update Docker Compose**: Remove 31 non-running service definitions

## 🚨 Developer Warning

**Before starting work:**
1. Run `docker ps` to see what's actually running
2. Test endpoints directly with curl
3. Check agent code in `/agents/*/app.py` for actual logic
4. Ignore most documentation files
5. Verify claims before implementing features

**Trust only:**
- Container logs: `docker-compose logs [service]`
- Direct endpoint testing: `curl http://127.0.0.1:[port]/health`
- Actual code files (not documentation)
- This CLAUDE.md file

**After v56 cleanup, you can now trust:**
- This CLAUDE.md file (reflects reality)
- CLEANUP_COMPLETE_REPORT.md (accurate system state)
- Container logs and direct endpoint testing
- Actual code files (fantasy removed)

**Still be wary of:**
- Old documentation in git history
- Claims about complex agent features (agents are stubs)
- GPT-OSS migration completion (still using TinyLlama)
- Production readiness claims (still PoC with missing pieces)

---

## Post-Cleanup Status (v56)

### ✅ Cleanup Achievements  
- **Removed 200+ fantasy documentation files** (quantum, AGI/ASI, non-existent features)
- **Eliminated duplicate code** across BaseAgent implementations
- **Cleaned root directory** of temporary analysis/audit scripts
- **Preserved all working functionality** during cleanup
- **Created truthful documentation** reflecting actual system state

### 🔄 Still Needs Work
- **Requirements consolidation** (75+ files need merging to 3)
- **Docker compose cleanup** (31 non-running services to remove)
- **Database schema creation** (PostgreSQL empty)
- **Model configuration fix** (gpt-oss vs TinyLlama mismatch)
- **Agent logic implementation** (replace stubs with real processing)

## Summary: Clean Foundation for Real Development

This system is now a **clean Docker Compose setup** with:
- Basic web services (FastAPI + Streamlit)  
- Standard databases (PostgreSQL, Redis, Neo4j)
- Local LLM via Ollama (TinyLlama, not gpt-oss)
- Full monitoring stack (Prometheus, Grafana, Loki)
- 7 Flask stub "agents" ready for real implementation
- **Honest documentation** that won't mislead developers

**Post-cleanup, this is a solid foundation for building real AI agent functionality.**

### Development Priority
1. **Fix immediate issues** (model mismatch, database schema)
2. **Implement one real agent** to establish patterns
3. **Build incrementally** on working foundation
4. **Add complexity only after basics work**

See `CLEANUP_COMPLETE_REPORT.md` for full details of what was cleaned and what remains.

---

## 📋 COMPREHENSIVE CODEBASE RULES

**Added:** December 19, 2024  
**Purpose:** Establish firm engineering standards and discipline for this codebase

These rules are MANDATORY for all contributors. They ensure codebase hygiene, prevent regression, and maintain professional standards.

### 🔧 Codebase Hygiene
A clean, consistent, and organized codebase is non-negotiable. It reflects engineering discipline and enables scalability, team velocity, and fault tolerance.

Every contributor is accountable for maintaining and improving hygiene—not just avoiding harm.

🧼 Enforce Consistency Relentlessly
✅ Follow the existing structure, naming patterns, and conventions. Never introduce your own style or shortcuts.

✅ Centralize logic — do not duplicate code across files, modules, or services.

🚫 Avoid multiple versions of:

APIs doing the same task (REST + GraphQL duplicating effort, for example)

UI components or CSS/SCSS modules with near-identical logic or styling

Scripts that solve the same problem in slightly different ways

Requirements files scattered across environments with conflicting dependencies

Documentation split across folders with different levels of accuracy

📂 Project Structure Discipline
📌 Never dump files or code in random or top-level folders.

📌 Place everything intentionally, following modular boundaries:

components/ for reusable UI parts

services/ or api/ for network interactions

utils/ for pure logic or helpers

hooks/ for reusable frontend logic

schemas/ or types/ for data validation

If the ideal location doesn't exist, propose a clear structure and open a small RFC (Request for Comments) before proceeding.

🗑️ Dead Code is Debt
🔥 Regularly delete unused code, legacy assets, stale test files, or experimental stubs.

❌ "Just in case" or "might be useful later" is not a valid reason to keep clutter.

🧪 Temporary test code must be removed or clearly gated (e.g. with feature flags or development-only checks).

🧪 Use Tools to Automate Discipline
✅ Mandatory for all contributors:

Linters: ESLint, Flake8, RuboCop

Formatters: Prettier, Black, gofmt

Static analysis: TypeScript, mypy, SonarQube, Bandit

Dependency managers: pip-tools, Poetry, pnpm, npm lockfiles

Schema enforcement: JSON schema, Pydantic, zod

Test coverage tooling: Jest, pytest-cov, Istanbul

🔄 Integrate these tools in pre-commit, pre-push, and CI/CD workflows:

No code gets into production branches without passing hygiene checks.

Every PR should be green and self-explanatory.

✍️ Commits Are Contracts
✅ Write atomic commits—one logical change per commit.

🧾 Follow conventional commit patterns or similar style guides (feat:, fix:, refactor:, etc.).

🧪 No skipping reviews or tests for "quick fixes." These introduce long-term chaos.

🧠 Execution Mindset: Act Like a Top-Level Engineer
🛠️ Think like an Architect, Engineer, QA, and PM—all at once.

🔬 Examine the full context of any change before writing code.

🧭 Prioritize long-term clarity over short-term speed.

🧱 Every change should make the codebase easier to maintain for someone else later.

🚩 Red Flags (Anti-Patterns to Avoid)
🔴 "I'll just put this here for now" — No, there is no "for now."

🔴 "It's just a tiny change" — That's how tech debt begins.

🔴 "We can clean this up later" — "Later" rarely comes.

🔴 Duplicate modules named utils.js, helper.py, or service.ts across packages.

🔴 PRs that include: unrelated changes, commented-out code, unreviewed temporary logs.

🧭 Final Reminder
A healthy codebase is a shared responsibility.
Every line of code you touch should be better than you found it.

🚫 Rules to Follow

-------
📌 Rule 1: No Fantasy Elements
✨ Only real, production-ready implementations are allowed.
Do not write speculative, placeholder, "in-theory," or overly abstract code unless it's been fully validated and grounded in current platform constraints.

✨ Avoid overengineering or unnecessary abstraction.
No fictional components, fake classes, dream APIs, or imaginary infrastructure. All code must reflect actual, working systems.

✨ No 'someday' solutions.
Avoid comments like // TODO: magically scale this later or // imagine this uses a future AI module. If it doesn't exist now, it doesn't go in the codebase.

✨ Be honest with the present limitations.
Code must work today, not in a hypothetical perfect setup. Assume real-world constraints like flaky hardware, latency, cold starts, and limited memory.
All code and documentation must use real, grounded constructs—no metaphors, magic terms, or hypothetical "black-box" AI.

✨ Forbidden:

Terms like wizardService, magicHandler, teleportData(), or comments such as // TODO: add telekinesis here.

Pseudo-functions that don't map to an actual library or API (e.g. superIntuitiveAI.optimize()).

✨ Mandated Practices:

Name things concretely: emailSender, not magicMailer.

Use real libraries: import from nodemailer, not from "the built-in mailer."

Link to docs in comments or README—every external API or framework must be verifiable.

✅ Pre-Commit Checks:

 Search for banned keywords (magic, wizard, black-box, etc.) in your diff.

 Verify every new dependency is in package.json (or requirements.txt) with a valid version.

 Ensure code examples in docs actually compile or run.

----
 Rule 2: Do Not Break Existing Functionality
✨ Every change must respect what already works.
Before modifying any file, component, or flow, verify exactly what it currently does and why. Don't assume anything.

✨ Regression = failure.
If your change breaks or downgrades existing features—even temporarily—it's considered a critical issue. Stability comes first.

✨ Backwards compatibility is a must.
If your refactor or feature update changes existing behavior, either support legacy use cases or migrate them gracefully.

✨ Always test before merging.
Write or update test cases to explicitly cover both new logic and old logic. Nothing ships unless it's verified to not break production behavior.

✨ Communicate impact clearly.
If there's any risk of side effects, escalate and document. Silent changes are forbidden.

🔍 Before modifying any file, investigate the full functionality and behavior of the existing code—understand what it does, how it's used, and whether it's actively supporting a feature or integration.

🧪 If a change is required, test the full end-to-end flow before and after. Confirm the logic is preserved or improved—never regressed.

🔁 Refactor only when necessary and with proper safeguards. If existing advanced functionality is present (e.g., dynamic routing, lazy loading, caching, etc.), it must be preserved or enhanced, not removed.

📊 Maintain proper version control and rollback strategies in case a new change introduces instability or conflict.

💡 Document what was changed, why, and what was verified to ensure that others won't unknowingly override or disrupt a critical flow later.

❗ Breaking changes must never be merged without full validation across all dependent systems and deployment scenarios.
Every change must preserve or improve current behavior—no regressions, ever.

✨ Investigation Steps:

Trace usage:

grep -R "functionName" .

Check import graphs or IDE "Find Usages."

Run baseline tests:

npm test, pytest, or your CI suite.

Manual sanity check of any affected UI or API endpoints.

Review consumers:

Frontend pages that call an endpoint

Cron jobs or scripts that rely on a helper

✨ Testing & Safeguards:

 Write or update tests covering both old and new behavior.

 Gate big changes behind feature flags until fully validated.

 Prepare a rollback plan—document the exact revert commit or steps.

✅ Merge Criteria:

 Green build with 100% test pass rate

 No new lint/type errors

 Explicit sign-off from the original feature owner or lead

❗ Do Not merge breaking changes without:

A clear "Breaking Change" section in the PR description

A migration or upgrade guide in CHANGELOG.md or docs

-------
📌 Rule 3: Analyze Everything—Every Time
✨ A thorough, deep review of the entire application is required before any change is made.

✨ Check all files, folders, scripts, directories, configuration files, pipelines, logs, and documentation without exception.

✨ Do not rely on assumptions—validate every piece of code logic, every dependency, every API interaction, and every test.

✨ Document what you find, and do not move forward until you have a complete understanding of the system.

-------
📌 Rule 4: Reuse Before Creating
✨ Always check if a script or piece of code already exists before creating a new one.

✨ If it exists, use it or improve it—don't duplicate it. No more script chaos where there are five different versions of the same functionality scattered across the codebase.

-------
📌 Rule 5: Treat This as a Professional Project — Not a Playground
✨ This is not a testing ground or experimental repository. Every change must be done with a professional mindset—no trial-and-error, no haphazard additions, and no skipping steps.

✨ Respect the structure, follow established standards, and treat this like you would a high-stakes production system.

-------
📌 Rule 6: Clear, Centralized, and Structured Documentation
✨ All documentation must be in a central /docs/ directory with a logical folder structure.

✨ Update documentation as part of every change—no exceptions.

✨ Do not leave outdated documentation lying around. Remove it immediately or update it to reflect the current state.

✨ Ownership and collaboration: Make it clear what each document is for, who owns it, and when it was last updated.

-------
📌 Rule 7: Eliminate Script Chaos — Clean, Consolidate, and Control
✨ We will not tolerate script sprawl. All scripts must be:

• Centralized in a single, well-organized /scripts/ directory.

• Categorized clearly (e.g., /scripts/deployment/, /scripts/testing/, /scripts/utils/).

• Named descriptively and purposefully.

• Documented with headers explaining their purpose, usage, and dependencies.

✨ Remove all unused scripts. If you find duplicates, consolidate them into one.

✨ Scripts should have one purpose and do it well. No monolithic, do-everything scripts.

-------
📌 Rule 8: Python Script Sanity — Structure, Purpose, and Cleanup
✨ Python scripts must:

• Be organized into a clear location (e.g., /scripts/python/ or within specific module directories).

• Include proper headers: purpose, author, date, usage instructions.

• Use argparse or similar for CLI arguments—no hardcoded values.

• Handle errors gracefully with logging.

• Be production-ready, not quick hacks.

✨ Delete all test scripts, debugging scripts, and one-off experiments from the repository. If you need them temporarily, use a separate branch or local environment.

-------
📌 Rule 9: Backend & Frontend Version Control — No More Duplication Chaos
✨ There should be one and only one source of truth for the backend and frontend.

✨ Remove all v1, v2, v3, old, backup, deprecated versions immediately.

✨ If you need to experiment, use branches and feature flags—not duplicate directories.

-------
📌 Rule 10: Functionality-First Cleanup — Never Delete Blindly
✨ Before removing any code, script, or file:

• Verify all references and dependencies.

• Understand its purpose and usage.

• Test the system without it to ensure nothing breaks.

• Archive before deletion if there's any doubt.

✨ Do not delete advanced functionality that works (e.g., caching, optimization, monitoring) just because you don't understand it immediately. Investigate first.

-------
📌 Rule 11: Docker Structure Must Be Clean, Modular, and Predictable
✨ All Docker-related files must follow a consistent structure:

• Dockerfiles should be optimized, multi-stage where appropriate, and well-commented.

• docker-compose.yml files must be modular and environment-specific (dev, staging, prod).

• Use .dockerignore properly to exclude unnecessary files.

• Version-pin all base images and dependencies.

-------
📌 Rule 12: One Self-Updating, Intelligent, End-to-End Deployment Script
✨ Create and maintain a single deploy.sh script that:

• Is self-sufficient and comprehensive.

• Handles all environments (dev, staging, production) with appropriate flags.

• Is self-updating—pulls the latest changes and updates itself before running.

• Provides clear logging, error handling, and rollback capabilities.

• Is documented inline and in /docs/deployment/.

✨ No more scattered deployment scripts. One script to rule them all.

-------
📌 Rule 13: No Garbage, No Rot
✨ Abandoned code, TODO comments older than 30 days, commented-out blocks, and unused imports/variables must be removed.

✨ If it's not being used, it doesn't belong in the codebase.

✨ Regular cleanup sprints will be enforced.

-------
📌 Rule 14: Engage the Correct AI Agent for Every Task
✨ We have specialized AI agents. Use them appropriately:

• Backend tasks → Backend specialist

• Frontend tasks → Frontend specialist

• DevOps tasks → DevOps specialist

• Documentation → Documentation specialist

✨ Do not use a generalist agent for specialized work when a specialist is available.

✨ Document which agent was used for which task in commit messages.

-------
📌 Rule 15: Keep Documentation Clean, Clear, and Deduplicated
✨ Documentation must be:

• Clear and concise—no rambling or redundancy.

• Up-to-date—reflects the current state of the system.

• Structured—follows a consistent format and hierarchy.

• Actionable—provides clear next steps, not just descriptions.

✨ Remove all duplicate documentation immediately. There should be one source of truth for each topic.

-------
📌 Rule 16: Use Local LLMs Exclusively via Ollama, Default to TinyLlama
✨ All AI/LLM operations must use Ollama with locally hosted models.

✨ Default model: TinyLlama (fast, efficient, sufficient for most tasks).

✨ Document any model overrides clearly in configuration and code comments.

✨ No external API calls to OpenAI, Anthropic, or other cloud providers without explicit approval and documentation.

-------
📌 Rule 17: Review and Follow All Documents in /opt/sutazaiapp/IMPORTANT
✨ The /opt/sutazaiapp/IMPORTANT directory contains canonical documentation that must be reviewed before making any changes.

✨ These documents represent the source of truth and override any conflicting information elsewhere.

✨ If you find discrepancies, the IMPORTANT/ documents win.

-------
📌 Rule 18: Absolute, Line-by-Line Deep Review of Core Documentation
✨ Before starting any work, you must perform a line-by-line review of:

• /opt/sutazaiapp/CLAUDE.md

• /opt/sutazaiapp/IMPORTANT/*

• Project README files

• Architecture documentation

✨ This is not optional. Zero tolerance for skipping this step.

✨ Document your understanding and any discrepancies found.

-------
📌 Rule 19: Mandatory Change Tracking in /opt/sutazaiapp/docs/CHANGELOG.md
✨ Every single change, no matter how small, must be documented in the CHANGELOG.

✨ Format: [Date] - [Version] - [Component] - [Change Type] - [Description]

✨ Include:

• What was changed

• Why it was changed

• Who made the change (AI agent or human)

• Potential impact or dependencies

✨ No exceptions. Undocumented changes will be reverted.