# SutazAI Codebase Cleanup — System Architect Plan

Source of truth: IMPORTANT/*.md, *.sql, Archives/* (no external sources). No backups created; all decisions derive from IMPORTANT.

## Phases

1) Dependency Analysis and Conflict Identification
- Build Python import graph for backend, services, agents, frontend, scripts
- Detect circular imports and orphaned modules
- Identify duplicate class names across modules
- Surface files with "deprecated" markers or TODO debt

2) Module-by-Module Cleanup
- Prioritize by blast radius: services -> backend -> agents -> frontend -> scripts
- Resolve circular imports via dependency inversion or local refactors
- Consolidate duplicate classes into single authoritative modules
- Remove dead files per IMPORTANT blueprint alignment

3) Integration Testing Checkpoints
- Execute existing unit/integration tests in stages per component folder
- Capture junit, coverage to test-results/ and htmlcov/
- Gate progression on zero failures per phase

4) Final Validation & Stabilization
- Run linting (ruff), typing (mypy), security (bandit) if available offline
- Verify compose variants start locally (dry-run config validation)
- Update diagrams and docs to match cleaned dependency graph

## Governance & Traceability
- Unique change IDs: CLN-YYYYMMDD-####
- Immutable ledger: reports/ledger/ledger.jsonl (JSON Lines)
- Agent communications: coordination_bus/messages/*.jsonl
- Architect approvals recorded in reports/ledger/approvals.jsonl

## Checkpoints
- Phase 1: Discovery report and conflict map produced
- Phase 2: All high-priority conflicts resolved, change logs complete
- Phase 3: All tests green, coverage >= 80%
- Phase 4: Validation bundle (lint/type/security) and docs updated

## Risk Controls
- No destructive deletes unless corroborated by IMPORTANT documents
- All removals logged with rationale and document references

