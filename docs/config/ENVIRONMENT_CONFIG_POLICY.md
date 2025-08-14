# Environment Configuration Policy

This repository standardizes environment configuration to eliminate duplication and reduce risk.

## Principles

- Single source of truth for environment variables per environment
- No plaintext secrets committed; use `detect-secrets` baseline and secret managers
- Keep `.env.example` as the authoritative developer reference

## Current State (root)

Observed env files in the repo root:
- `.env` (local overrides)
- `.env.example` (reference — keep)
- `.env.production`
- `.env.production.secure`
- `.env.secure`
- `.env.secure.generated`
- `.env.agents`
- `.env.ollama`

Backups moved to `backups/env/`:
- `.env.backup.20250811_193726`
- `.env.secure.backup.20250811_193052`
- `.env.secure.backup.20250813_092537`

## Target State (Phase 2)

- Keep only `.env.example` in root
- Move environment‑specific files under `environments/`:
  - `environments/development.env`
  - `environments/production.env`
  - `environments/staging.env`
  - `environments/agents.env`
- Update compose/Make to load from `environments/` consistently
- Document required variables in `.env.example` and service READMEs

No active env files were moved in Phase 1 to avoid breaking workflows.

