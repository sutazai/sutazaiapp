# Infrastructure Setup Documentation

Baseline
- Containers via `docker-compose.yml` (canonical). Health checks validate readiness.
- Local-only LLMs through Ollama; TinyLlama default.

Setup Steps
- Configure `.env` using `operations/env_template.md`.
- Start stack with root compose and verify health endpoints.

Citations
- Compose definitions: /opt/sutazaiapp/docker-compose.yml#L1-L600
- Health checks: /opt/sutazaiapp/README.md#L1-L200

