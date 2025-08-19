Codex Agents Directory

Purpose
- Provide a stable entrypoint for OpenAI/Codex agent role profiles without duplicating content.

Structure
- Source of truth for role definitions lives in `.claude/agents/`.
- This directory references the same roles for Codex/OpenAI agents to ensure one canonical set of specifications.

How To Use
- Use the role profiles in `.claude/agents/` for all agent setup and orchestration.
- If an OpenAI/Codex-specific override is needed, add a minimal file here named after the role (e.g., `backend-architect.codex.md`) that documents only the delta from the canonical role.

Rules
- Do not copy or fork full role files here. Reference canonical files to avoid drift.
- Keep any overrides minimal and clearly scoped.

Cross-References
- Canonical roles: `../.claude/agents/`
- Orchestration docs: `Agents.md` at repo root
- Port Registry: `config/port-registry.yaml`
