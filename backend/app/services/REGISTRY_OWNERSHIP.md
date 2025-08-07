Registry Ownership and Usage

Canonical guidance based on IMPORTANT documentation and current code roles.

- `backend/app/services/agent_registry.py`: Runtime process/activation manager for agent subprocesses. Responsibilities:
  - Discover agents under `agents/`
  - Start/stop processes and perform health checks
  - Produce system-wide activation/health summaries

- `backend/app/agents/registry.py`: In-memory API-facing registry. Responsibilities:
  - Track registered agents and capabilities in-memory
  - Provide lookups for API handlers and orchestration components

Guidance:
- New code should use `app.services.agent_registry.AgentRegistry` for lifecycle management and `app.agents.registry.AgentRegistry` for in-process lookups.
- Do not remove either until interface mapping and tests converge; consolidation will follow with adapters if needed.

