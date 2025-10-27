# Environments

- Local (WSL2): Intel i7-12700H, 48GB RAM, 4GB GPU
- Dockerized microservices; service names not localhost.
- GPU detection with CPU fallback; Ollama GPU if available.
- Secrets via `.env`, no hardcoded passwords.

Assumptions: Based on provided specs; validate GPU passthrough.