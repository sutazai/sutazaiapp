# Agent Cleanup Program — 200 Claude Agents (Local Profiles)

Purpose: Execute a disciplined, end-to-end cleanup of the entire codebase using 200 approved local agent role profiles. All work references only `/opt/sutazaiapp/IMPORTANT/` as the authoritative source of truth (ASoT). No external AI APIs.

## Leadership & Roles
- System Architect (lead): arbitration, integrity, phase gates
- Backend Architect: services, data, integrations
- Frontend Architect: UX, components, client integrations

## Coordination Bus
- Channel: `coordination_bus/` (local) with append-only logs per task ID
- Events: `status`, `diff_ready`, `error`, `approval_request`, `integration_report`
- Heartbeat: 60s per agent; auto-escalate after 3 missed beats

## Workstreams & Agent Allocation (200 total)
- Architecture analysis: 20 (static analysis, dependency graphs)
- Backend refactoring: 40 (API, services, data access)
- Frontend refactoring: 20 (components, state, API clients)
- Integrations & gateway: 20 (routes, authn/authz, rate limits)
- Data & migrations: 20 (UUID migration, indices, retention)
- Testing: 30 (unit/integration, fixtures, coverage ≥ 80%)
- Security & compliance: 15 (threat model, secrets hygiene)
- Observability & SRE: 15 (logs/metrics/traces, alerts)
- Docs & governance: 20 (ASoT alignment, ADRs, CHANGELOG)

Each agent tags changes with `TID-<epic>-<seq>` and writes a per-file changelog.

## Cleanup Phases
P1 – Discovery
- Static/code analysis across all modules; produce conflict map (duplicate classes, deprecated APIs, circular imports)
- Deliver `conflict_map.json` and `integration_report_P1.md`

P2 – Resolution
- Remove/refactor issues by priority; maintain per-file changelogs
- Gate risky changes with feature flags; preserve behavior (R2)

P3 – Integration Testing
- Run `pytest -v backend/tests --cov=backend --cov-fail-under=80`
- Execute E2E where applicable; fix or document deviations with ADRs

P4 – Validation & Optimization
- Benchmarks, security scans, linters; finalize docs and dependency graphs

## Quality & Compliance
- Traceability: Each PR links to Issue Card and backlog line
- Audit trail: Immutable logs under `coordination_bus/ledger/`
- Zero assumptions: Cite sources as `/opt/sutazaiapp/IMPORTANT/<path>#Lx-Ly`

## Success Criteria
- Conflict-free build; all tests green; coverage ≥ 80%
- Meets/exceeds baseline performance metrics (CPU, memory, p95 latency)
- Updated diagrams and final cleanup report signed by System Architect

## Immediate Actions (Quick Wins)
- Align model defaults with TinyLlama
- Introduce JWT auth middleware and basic RBAC
- Add UUID migrations and FK indexes
- Define Kong routes and health checks

Sources: [ASoT INDEX] /opt/sutazaiapp/IMPORTANT/10_canonical/INDEX.md#L1-L100; [Backlog] /opt/sutazaiapp/IMPORTANT/20_plan/remediation_backlog.csv#L1-L50

