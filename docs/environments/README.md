# Environments Layout

This repository standardizes env files under `environments/`.

Files (templates):
- `development.env.template` — local development defaults (no secrets)
- `production.env.template` — production keys with required placeholders (no secrets)
- `agents.env.template` — agent layer defaults
- `ollama.env.template` — Ollama performance defaults

Usage:
- Copy a template to `*.env` (same name without `.template`) and set values locally.
- For Compose, prefer `--env-file environments/<name>.env` rather than root `.env`.
- Do NOT commit filled `.env` files; use a secrets manager in production.

Note: Original root env files remain untouched to avoid breakage while we migrate scripts/compose to this layout.
