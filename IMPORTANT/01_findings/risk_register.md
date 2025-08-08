# Risk Register

| ID | Title | Severity (H/M/L) | Likelihood (H/M/L) | Owner | Mitigation | Status |
|----|-------|-------------------|--------------------|-------|------------|--------|
| RISK-0001 | Inconsistent DB schema (UUID vs SERIAL) | H | H | Backend Architect | Migrate to UUID PKs; add FK indexes; regenerate migrations | Open |
| RISK-0002 | Orchestration and agents incomplete | H | H | System Architect | Implement orchestrator and priority agents; CPU/GPU fallback | Open |
| RISK-0003 | Model mismatch (gpt-oss vs TinyLlama) | M | H | Backend Architect | Align backend model list; preload TinyLlama; env overrides | Open |
| RISK-0004 | Missing AuthN/AuthZ | H | H | Backend Architect | JWT auth + RBAC; env secrets; middleware | Open |
| RISK-0005 | Service mesh routes missing | M | M | Backend Architect | Define Kong services/routes; health checks; CI validation | Open |
| RISK-0006 | Vector DB not integrated | M | H | Backend Architect | Implement embeddings pipeline and query endpoints | Open |
| RISK-0007 | Documentation drift/duplication | M | H | System Architect | Canonicalize ASoT; ADR coverage; deprecate duplicates | Open |
