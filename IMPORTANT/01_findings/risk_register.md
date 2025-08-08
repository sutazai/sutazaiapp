# Risk Register

- R1: Inconsistent DB schema (UUID vs SERIAL)
  - Impact: High; breaks data model consistency and migrations
  - Likelihood: High
  - Mitigation: Migrate to UUID PKs, add FK indexes, regenerate migrations

- R2: Orchestration and agents incomplete
  - Impact: High; core functionality blocked
  - Likelihood: High
  - Mitigation: Implement agent orchestrator and priority agents with CPU/GPU fallback

- R3: Model mismatch (gpt-oss vs TinyLlama)
  - Impact: Medium; degraded UX/perf
  - Likelihood: High
  - Mitigation: Align backend model list with Ollama; preload target models

- R4: Missing AuthN/AuthZ
  - Impact: High; security exposure
  - Likelihood: High
  - Mitigation: Introduce JWT-based auth, RBAC, secrets via env

- R5: Service mesh routes missing
  - Impact: Medium; integration blocked
  - Likelihood: Medium
  - Mitigation: Define Kong routes/services, health checks

- R6: Vector DB not integrated
  - Impact: Medium; RAG features blocked
  - Likelihood: High
  - Mitigation: Implement embedding pipelines and query endpoints

- R7: Documentation drift
  - Impact: Medium; misalignment risk
  - Likelihood: High
  - Mitigation: Canonicalize docs (ASoT), ADRs, deprecate outdated copies