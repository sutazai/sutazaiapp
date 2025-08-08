# Migration Plan

- Phase 1 (Stabilize):
  - UUID migrations, auth scaffolding, model default alignment
  - Checkpoint: tests green; health checks pass; routes defined
  - Rollback: revert migrations; toggle feature flags

- Phase 2 (RAG & Orchestration):
  - ChromaDB integration; orchestrator + 3 agents
  - Checkpoint: E2E flows for RAG and agent exec pass
  - Rollback: disable orchestrator; fallback simple exec

- Phase 3 (Hardening):
  - Observability dashboards, rate limits, retries, backup verification
  - Checkpoint: SLO dashboards at p95 targets
  - Rollback: config rollback

Exit Criteria: SLOs met; security checks pass; ADRs accepted.