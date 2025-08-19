Changes - 2025-08-19

Summary
- Added `Agents.md` consolidating agent role locations and policies.
- Created `.codex/agents/` entrypoint referencing canonical `.claude/agents/`.
- Fixed indentation in `config/port-registry.yaml` for `system` port range.
- Added `scripts/tools/generate_index.py` and regenerated `IMPORTANT/INDEX.md`.

Impact
- Improves consistency (single source of truth for roles).
- Restores YAML validity for port registry.
- Provides an automated, reproducible system index for audits.

Follow-ups
- Consolidate duplicate Dockerfiles and compose manifests (guided by index).
- Validate MCP mesh registration vs. STDIO wrappers and align.
